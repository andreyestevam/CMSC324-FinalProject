#load_dataset.py
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import nibabel as nib
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader


############################################################
###############     SET HYPERPARAMETERS      ###############
############################################################
# Load file path
with open('./config.json', 'r') as f:
    config = json.load(f)
    dataset_path = config['dataset_path']
ROOT = dataset_path
TRAIN_DIR = Path(os.path.join(ROOT, "Training"))

# Load hyperparameters
with open('./hparams.json', 'r') as f:
    hparams = json.load(f)
    swin_hparams = hparams['swin_transformer']['hparam_grid']
    unet_hparams = hparams['unet']['hparam_grid']
    
    if swin_hparams["patch_size"] != unet_hparams["patch_size"]:
        raise RuntimeError(f"Patch size mismatch")
    else:
        PATCH_SIZE = swin_hparams["patch_size"]
    
    if swin_hparams["batch_size"] != unet_hparams["batch_size"]:
        raise RuntimeError(f"Batch size mismatch")
    else:
        BATCH_SIZE = swin_hparams["batch_size"]
        
# Other fixed hyperparameters
POSITIVE_PATCH_RATIO = 0.75  # 75% of patches are centered on tumor voxels, 25% are random     
PATCHES_PER_CASE = 8

# Random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)


############################################################
###############          LOAD CASES          ###############
############################################################
def _locate_case_files(case_dir: Path) -> Dict[str, Optional[Path]]:
    
    mapping = {
        "-t1n.nii.gz": "t1",
        "-t1c.nii.gz": "t1ce",
        "-t2w.nii.gz": "t2",
        "-t2f.nii.gz": "flair",
        "-seg.nii.gz": "seg"
    }
    
    files = list(case_dir.glob("*.nii.gz"))

    result = {}
    for suffix, key in mapping.items():
        found_file = None
        for f in files:
            if f.name.lower().endswith(suffix):
                found_file = f
                break
        
        result[key] = found_file

    return result

def collect_cases(training_dir: Path = TRAIN_DIR) -> List[Dict[str, Path]]:
    cases = []
    for case_dir in sorted(training_dir.iterdir()):
        if not case_dir.is_dir() or "BraTS" not in case_dir.name:
            continue

        # Remove data with issues
        if "BraTS-PED-00024-000" in case_dir.name or "BraTS-PED-00098-000" in case_dir.name:
            continue
        
        files = _locate_case_files(case_dir)

        # Remove data with issues
        if "BraTS-PED-00024-000" in case_dir.name or "BraTS-PED-00098-000" in case_dir.name:
            continue
        
        files = _locate_case_files(case_dir)

        if all(files[k] is not None for k in ["t1", "t1ce", "t2", "flair", "seg"]):
            files["case_id"] = case_dir.name
            cases.append(files)
    if not cases:
        raise RuntimeError(f"No complete cases found in {training_dir}.")
    
    
    return cases


############################################################
###############       LOAD TO PATCHES        ###############
############################################################
###### Volume Preprocessing ######
def _zscore_nonzero(x: np.ndarray) -> np.ndarray:
    """Normalize only non-zero voxels to avoid background skewing stats. (mean=0; std=1)"""
    x = x.astype(np.float32)    # type casting
    mask = x != 0
    if np.any(mask):
        vals = x[mask]
        mean, std = vals.mean(), vals.std()
        if std < 1e-6:
            std = 1.0
        x[mask] = (x[mask] - mean) / std
    return x
###### Volume Preprocessing ######

######### Down Sampling ##########
def _sample_patch_centers(mask_3d: np.ndarray,
                          n_patches: int,
                          pos_ratio: float = 0.75) -> List[Tuple[int, int, int]]:
    """
    Choose patch centers — 75% centered on tumor voxels, 25% random.
    This prevents the model from only seeing tumor and ignoring background.
    """
    tumor_voxels = np.argwhere(mask_3d > 0)
    D, H, W = mask_3d.shape
    centers = []
    for _ in range(n_patches):
        if len(tumor_voxels) > 0 and random.random() < pos_ratio:
            cz, cy, cx = tumor_voxels[random.randrange(len(tumor_voxels))]
        else:
            cz, cy, cx = random.randrange(D), random.randrange(H), random.randrange(W)
        centers.append((int(cz), int(cy), int(cx)))
    return centers

def _extract_patch(volume: np.ndarray,
                  mask: np.ndarray,
                  patch_size: Tuple[int, int, int],
                  center: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Cut a fixed-size 3D cube around a center coordinate. Pads if near boundary."""
    dz, dy, dx = patch_size
    D, H, W    = volume.shape[:3]
    cz, cy, cx = int(center[0]), int(center[1]), int(center[2])

    # Compute start indices and clamp to valid range
    z1 = max(0, cz - dz // 2);  z2 = min(D, z1 + dz);  z1 = max(0, z2 - dz)
    y1 = max(0, cy - dy // 2);  y2 = min(H, y1 + dy);  y1 = max(0, y2 - dy)
    x1 = max(0, cx - dx // 2);  x2 = min(W, x1 + dx);  x1 = max(0, x2 - dx)

    vol_patch  = volume[z1:z2, y1:y2, x1:x2, :]
    mask_patch = mask[z1:z2, y1:y2, x1:x2, :]

    # Zero-pad if the patch was cut off at the brain boundary
    pd = dz - vol_patch.shape[0]
    ph = dy - vol_patch.shape[1]
    pw = dx - vol_patch.shape[2]
    if pd > 0 or ph > 0 or pw > 0:
        pad = ((0, max(0, pd)), (0, max(0, ph)), (0, max(0, pw)), (0, 0))
        vol_patch  = np.pad(vol_patch,  pad, mode="constant")
        mask_patch = np.pad(mask_patch, pad, mode="constant")

    return vol_patch.astype(np.float32), mask_patch.astype(np.float32)
######### Down Sampling ##########


def _to_dhw(vol_xyz: np.ndarray) -> np.ndarray:
    """Reorder [H, W, Z] → [D, H, W]."""
    return np.transpose(vol_xyz, (2, 0, 1)).astype(np.float32)

def load_case_as_patches(case: Dict[str, Path],
                          patch_size: Tuple[int, int, int],
                          patches_per_case: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load one patient, normalize all 4 modalities, extract 3D patches."""
    modalities = []
    for key in ["t1", "t1ce", "t2", "flair"]:
        vol = nib.load(str(case[key])).get_fdata().astype(np.float32)
        modalities.append(_zscore_nonzero(_to_dhw(vol)))

    seg = nib.load(str(case["seg"])).get_fdata().astype(np.float32)
    seg = _to_dhw(seg)
    seg = (seg > 0).astype(np.float32)    # whole-tumor binary mask

    image_4d = np.stack(modalities, axis=-1)   
    mask_4d  = seg[..., np.newaxis]           

    centers = _sample_patch_centers(seg, patches_per_case, POSITIVE_PATCH_RATIO)

    xs, ys = [], []
    for center in centers:
        xp, yp = _extract_patch(image_4d, mask_4d, patch_size, center)
        xs.append(xp)
        ys.append(yp)

    return np.stack(xs), np.stack(ys)


############################################################
###############    Build & Split Datasets    ###############
############################################################
def _load_locked_splits(cases: List[Dict[str, Path]]) -> Tuple[List, List]:
    with open("dataset_splits.json", "r") as f:
        splits = json.load(f)
        
    train_cases = [c for c in cases if c["case_id"] in splits["train"]]
    test_cases = [c for c in cases if c["case_id"] in splits["test"]]
    
    return train_cases, test_cases


def _build_numpy_dataset(cases: List[Dict], 
                        patch_size: Tuple,
                        patches_per_case: int) -> Tuple[np.ndarray, np.ndarray]:
    all_x, all_y = [], []
    for i, case in enumerate(cases, 1):
        print(f"  Loading {i}/{len(cases)}: {case['case_id']}")
        x, y = load_case_as_patches(case, patch_size, patches_per_case)
        all_x.append(x)
        all_y.append(y)
    return np.concatenate(all_x).astype(np.float32), np.concatenate(all_y).astype(np.float32)


def build_dataset(training: bool,
                  n_train: int = 204,
                  n_test: int = 51) -> Tuple[np.ndarray, np.ndarray]:
    all_cases = collect_cases()
    train_cases, test_cases = _load_locked_splits(all_cases)

    if training:
        x_train, y_train = _build_numpy_dataset(train_cases, PATCH_SIZE, PATCHES_PER_CASE)
        return x_train[0:n_train], y_train[0:n_train]
    else:
        x_test, y_test = _build_numpy_dataset(test_cases, PATCH_SIZE, PATCHES_PER_CASE)
        return x_test[0:n_test], y_test[0:n_test]


############################################################
###############       Data Augmentation      ###############
############################################################
def _augment_patch(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random flips along all 3 spatial axes.
    x shape: [D, H, W, 4]
    y shape: [D, H, W, 1]
    """
    # Flip along depth axis (axial planes)
    if random.random() < 0.5:
        x = x[::-1, :, :, :].copy()
        y = y[::-1, :, :, :].copy()
    # Flip along height axis (coronal planes)
    if random.random() < 0.5:
        x = x[:, ::-1, :, :].copy()
        y = y[:, ::-1, :, :].copy()
    # Flip along width axis (sagittal planes)
    if random.random() < 0.5:
        x = x[:, :, ::-1, :].copy()
        y = y[:, :, ::-1, :].copy()
    return x, y

def _augment_tf(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Wraps numpy augmentation for use inside tf.data pipeline."""
    x_aug, y_aug = tf.py_function(
        lambda xi, yi: _augment_patch(xi.numpy(), yi.numpy()),
        [x, y],
        [tf.float32, tf.float32]
    )
    x_aug.set_shape(x.shape)
    y_aug.set_shape(y.shape)
    return x_aug, y_aug

def augment_torch_batch(x_batch: torch.Tensor,
                        y_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Wraps numpy augmentation for use inside PyTorch pipeline."""
    x_np = x_batch.numpy()
    y_np = y_batch.numpy()
    
    for i in range(len(x_np)):
        x_np[i], y_np[i] = _augment_patch(x_np[i], y_np[i])
        
    x_tensor = torch.from_numpy(x_np)
    y_tensor = torch.from_numpy(y_np)
    
    # Permute axes to [B, C, D, H, W]
    x_tensor = x_tensor.permute(0, 4, 1, 2, 3)
    y_tensor = y_tensor.permute(0, 4, 1, 2, 3)
    
    return x_tensor, y_tensor

def format_torch_test_batch(x_batch: torch.Tensor,
                            y_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Permutes testing batches to PyTorch format [B, C, D, H, W] without augmenting."""
    x_tensor = x_batch.permute(0, 4, 1, 2, 3)
    y_tensor = y_batch.permute(0, 4, 1, 2, 3)
    return x_tensor, y_tensor

############################################################
###############         Make Datasets        ###############
############################################################
def make_tf_dataset(x: np.ndarray, y: np.ndarray, training: bool) -> TensorDataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y)) # treats each patch as one element in the dataset
    if training: 
        ds = ds.shuffle(min(len(x), 256), seed=SEED, reshuffle_each_iteration=True) # randomly reorders 
        ds = ds.map(_augment_tf, num_parallel_calls=tf.data.AUTOTUNE)  
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def make_torch_dataloader(x: np.ndarray, y: np.ndarray, training: bool) -> DataLoader:

    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()
    
    dataset = TensorDataset(x_tensor, y_tensor)

    # Reproducibilefor the shuffling
    g = torch.Generator()
    g.manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=training,
        generator=g if training else None,
        pin_memory=True
    )