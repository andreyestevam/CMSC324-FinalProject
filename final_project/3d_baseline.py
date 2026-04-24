# 3D u-net deep learning pipeline,  valuates performance using Dice coefficient and Hausdorff distance

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from skimage.metrics import hausdorff_distance as skimage_hausdorff

PATCH_SIZE          = (64, 64, 64) #one 3D cube fed to the model, depth x height x width
BATCH_SIZE          = 1              
EPOCHS              = 5  # keep at low to save time           
MAX_TRAIN_CASES     = 8             
MAX_VAL_CASES       = 2
PATCHES_PER_CASE    = 8            
POSITIVE_PATCH_RATIO = 0.75  # 75% of patches are centered on tumor voxels, 25% are random      
USE_WHOLE_TUMOR     = True           
BASE_FILTERS        = 16            
DROPOUT_RATE        = 0.20  # enables MC uncertainty estimation later
SEED                = 42
MODEL_OUT           = "unet3d_baseline.keras"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# subject to change (identical to the pervious 2d version)
from google.colab import drive
drive.mount('/content/drive')

ROOT      = "/content/drive/MyDrive/CMSC 324 Final project dataset/BraTS-PEDs-v1/"
TRAIN_DIR = os.path.join(ROOT, "Training")

required_suffixes = ["-t1n.nii.gz", "-t1c.nii.gz", "-t2w.nii.gz", "-t2f.nii.gz", "-seg.nii.gz"]

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

case_ids = sorted([
    name for name in os.listdir(TRAIN_DIR)
    if not name.startswith(".") and os.path.isdir(os.path.join(TRAIN_DIR, name))
])
print(f"Found {len(case_ids)} case folders.")

# Sanity check first 5 cases
for case_id in case_ids[:5]:
    case_dir = os.path.join(TRAIN_DIR, case_id)
    files = [f for f in os.listdir(case_dir) if not f.startswith(".")]
    print(f"\n{case_id}")
    for suf in required_suffixes:
        found = any(name.endswith(suf) for name in files)
        print(f"  {suf}: {'✓' if found else '✗ MISSING'}")

def locate_case_files(case_dir: Path) -> Dict[str, Optional[Path]]:
    files = list(case_dir.glob("*.nii.gz"))
    lower = {f.name.lower(): f for f in files}

    def match_any(suffixes):
        for name, path in lower.items():
            for suffix in suffixes:
                if name.endswith(suffix):
                    return path
        return None

    return {
        "t1":    match_any(["-t1n.nii.gz", "_t1.nii.gz",    "-t1.nii.gz"]),
        "t1ce":  match_any(["-t1c.nii.gz", "-t1ce.nii.gz",  "_t1ce.nii.gz"]),
        "t2":    match_any(["-t2w.nii.gz", "-t2.nii.gz",    "_t2.nii.gz"]),
        "flair": match_any(["-t2f.nii.gz", "-flair.nii.gz", "_flair.nii.gz"]),
        "seg":   match_any(["-seg.nii.gz", "_seg.nii.gz"]),
    }


def collect_cases(training_dir: Path) -> List[Dict[str, Path]]:
    cases = []
    for case_dir in sorted(training_dir.iterdir()):
        if not case_dir.is_dir() or "BraTS" not in case_dir.name:
            continue
        files = locate_case_files(case_dir)
        if all(files[k] is not None for k in ["t1", "t1ce", "t2", "flair", "seg"]):
            files["case_id"] = case_dir.name
            cases.append(files)
    if not cases:
        raise RuntimeError(f"No complete cases found in {training_dir}.")
    return cases


all_cases = collect_cases(Path(TRAIN_DIR))
print(f"Complete cases (all 5 files present): {len(all_cases)}")


##############
# different from the 2d version
# Instead of extracting flat slices, it extracts 3D cubes

# same as 2d
def zscore_nonzero(x: np.ndarray) -> np.ndarray:
    """Normalize only non-zero voxels to avoid background skewing stats,; zero voxels are background/air"""
    x = x.astype(np.float32)
    mask = x != 0
    if np.any(mask):
        vals = x[mask]
        mean, std = vals.mean(), vals.std()
        if std < 1e-6:
            std = 1.0
        x[mask] = (x[mask] - mean) / std
    return x

# Reorders the axes to match what the model expects
# I am not very sure about what this part is doing. This part comes from AI suggestion
def to_dhw(vol_xyz: np.ndarray) -> np.ndarray:
    """Reorder [H, W, Z] → [D, H, W]."""
    return np.transpose(vol_xyz, (2, 0, 1)).astype(np.float32)

# core 3D patch extraction function
def extract_patch(volume: np.ndarray,
                  mask: np.ndarray,
                  patch_size: Tuple[int, int, int],
                  center: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Cut a fixed-size 3D cube around a center coordinate. Pads if near boundary."""
    dz, dy, dx = patch_size
    D, H, W    = volume.shape[:3]
    # Given a center coordinate (cz, cy, cx) inside the brain volume, it cuts out a 64×64×64 cube centered there
    cz, cy, cx = int(center[0]), int(center[1]), int(center[2])

    # Compute start indices and clamp to valid range
    # Start z1, y1, x1; end z2, y2, x2
    # handles edge cases where the center is near the brain boundary 
    # shifts the patch inward rather than going out of bounds
    z1 = max(0, cz - dz // 2);  z2 = min(D, z1 + dz);  z1 = max(0, z2 - dz)
    y1 = max(0, cy - dy // 2);  y2 = min(H, y1 + dy);  y1 = max(0, y2 - dy)
    x1 = max(0, cx - dx // 2);  x2 = min(W, x1 + dx);  x1 = max(0, x2 - dx)

    vol_patch  = volume[z1:z2, y1:y2, x1:x2, :]
    mask_patch = mask[z1:z2, y1:y2, x1:x2, :]

    # Zero-pad if the patch was cut off at the very edge of the brain (smaller than 64x64x64)
    pd = dz - vol_patch.shape[0]
    ph = dy - vol_patch.shape[1]
    pw = dx - vol_patch.shape[2]
    if pd > 0 or ph > 0 or pw > 0:
        pad = ((0, max(0, pd)), (0, max(0, ph)), (0, max(0, pw)), (0, 0))
        vol_patch  = np.pad(vol_patch,  pad, mode="constant")
        mask_patch = np.pad(mask_patch, pad, mode="constant")

    return vol_patch.astype(np.float32), mask_patch.astype(np.float32)


def sample_patch_centers(mask_3d: np.ndarray,
                          n_patches: int,
                          pos_ratio: float = 0.75) -> List[Tuple[int, int, int]]:
    """
    Choose patch centers — 75% centered on tumor voxels, 25% random.
    This prevents the model from only seeing tumor and ignoring background.
    """
    tumor_voxels = np.argwhere(mask_3d > 0) # finds all voxel coordinates that contain tumor
    D, H, W = mask_3d.shape
    centers = []
    for _ in range(n_patches):
        if len(tumor_voxels) > 0 and random.random() < pos_ratio:
            cz, cy, cx = tumor_voxels[random.randrange(len(tumor_voxels))]
        else:
            cz, cy, cx = random.randrange(D), random.randrange(H), random.randrange(W)
        centers.append((int(cz), int(cy), int(cx)))
    return centers

# the full pipeline for one patient
def load_case_as_patches(case: Dict[str, Path],
                          patch_size: Tuple[int, int, int],
                          patches_per_case: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load one patient, normalize all 4 modalities, extract 3D patches."""
    modalities = []
    # loads all 4 volumes, reorder and do normalization
    for key in ["t1", "t1ce", "t2", "flair"]:
        vol = nib.load(str(case[key])).get_fdata().astype(np.float32)
        modalities.append(zscore_nonzero(to_dhw(vol)))

    # Loads the segmentation mask, reorders axes, and collapses all tumor labels to binary (0=background, 1=tumor)
    seg = nib.load(str(case["seg"])).get_fdata().astype(np.float32)
    seg = to_dhw(seg)
    seg = (seg > 0).astype(np.float32)    # whole-tumor binary mask

    image_4d = np.stack(modalities, axis=-1)   
    mask_4d  = seg[..., np.newaxis]           

    centers = sample_patch_centers(seg, patches_per_case, POSITIVE_PATCH_RATIO)

    xs, ys = [], []
    for center in centers:
        xp, yp = extract_patch(image_4d, mask_4d, patch_size, center) # cut out the actual 3D cubes
        xs.append(xp)
        ys.append(yp)

    return np.stack(xs), np.stack(ys) # return a stacked array of shape for images and masks

# new for 3d baseline
# multiplies the dataset by up to 8× with zero extra data collection, prevent overfitting
# only applied only during training (not validation)
def augment_patch(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random flips along all 3 spatial axes.
    This triples the effective dataset size with almost zero compute cost.
    Critical when training on only 12 cases to prevent overfitting.
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


def augment_tf(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Wraps numpy augmentation for use inside tf.data pipeline."""
    x_aug, y_aug = tf.py_function(
        lambda xi, yi: augment_patch(xi.numpy(), yi.numpy()),
        [x, y],
        [tf.float32, tf.float32]
    )
    x_aug.set_shape(x.shape)
    y_aug.set_shape(y.shape)
    return x_aug, y_aug

# loops over all assigned cases, call load_case_as_patches on each one
def build_numpy_dataset(cases, patch_size, patches_per_case):
    all_x, all_y = [], []
    for i, case in enumerate(cases, 1):
        print(f"  Loading {i}/{len(cases)}: {case['case_id']}")
        x, y = load_case_as_patches(case, patch_size, patches_per_case)
        all_x.append(x)
        all_y.append(y)
    return np.concatenate(all_x).astype(np.float32), np.concatenate(all_y).astype(np.float32)

# wraps the numpy arrays into a TensorFlow pipeline
def make_tf_dataset(x, y, batch_size, training):
    ds = tf.data.Dataset.from_tensor_slices((x, y)) # treats each patch as one element in the dataset
    if training:
        ds = ds.shuffle(min(len(x), 256), seed=SEED, reshuffle_each_iteration=True) # randomly reorders
        ds = ds.map(augment_tf, num_parallel_calls=tf.data.AUTOTUNE)  
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# Build splits
random.shuffle(all_cases)
train_cases = all_cases[:MAX_TRAIN_CASES]
val_cases   = all_cases[MAX_TRAIN_CASES:MAX_TRAIN_CASES + MAX_VAL_CASES]

if len(val_cases) == 0:
    raise RuntimeError("Not enough cases. Lower MAX_TRAIN_CASES.")

print("Loading training patches...")
x_train, y_train = build_numpy_dataset(train_cases, PATCH_SIZE, PATCHES_PER_CASE)

print("Loading validation patches...")
x_val, y_val = build_numpy_dataset(val_cases, PATCH_SIZE, PATCHES_PER_CASE)

print(f"\nx_train: {x_train.shape}  y_train: {y_train.shape}")
print(f"x_val:   {x_val.shape}  y_val:   {y_val.shape}")

train_ds = make_tf_dataset(x_train, y_train, BATCH_SIZE, training=True)
val_ds   = make_tf_dataset(x_val,   y_val,   BATCH_SIZE, training=False)

# 3d model
# There are many references in github open sources

def conv3d_block(x, filters, dropout_rate=0.0):
    """
    Two Conv3D layers with BatchNormalization and ReLU.
    BatchNorm added vs original — stabilizes training significantly
    with small datasets and 3D convolutions.
    """
    x = tf.keras.layers.Conv3D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv3D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

# This is a 3D model!
def build_unet3d(input_shape=(64, 64, 64, 4), base_filters=32, dropout_rate=0.20):
    """
    3D U-Net with skip connections.
    base_filters=32 (increased from 16) gives the model enough capacity
    to learn meaningful 3D features.

    Filter progression: 32 → 64 → 128 → 256 (bottleneck)
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    c1 = conv3d_block(inputs, base_filters)
    p1 = tf.keras.layers.MaxPooling3D(2)(c1)           # 64→32

    c2 = conv3d_block(p1, base_filters * 2)
    p2 = tf.keras.layers.MaxPooling3D(2)(c2)           # 32→16

    c3 = conv3d_block(p2, base_filters * 4)
    p3 = tf.keras.layers.MaxPooling3D(2)(c3)           # 16→8

    # Bottleneck
    bn = conv3d_block(p3, base_filters * 8, dropout_rate=dropout_rate)

    # Decoder
    u3 = tf.keras.layers.UpSampling3D(2)(bn)           # 8→16
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c4 = conv3d_block(u3, base_filters * 4, dropout_rate=dropout_rate)

    u2 = tf.keras.layers.UpSampling3D(2)(c4)           # 16→32
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c5 = conv3d_block(u2, base_filters * 2)

    u1 = tf.keras.layers.UpSampling3D(2)(c5)           # 32→64
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c6 = conv3d_block(u1, base_filters)

    # Output: one sigmoid value per voxel
    outputs = tf.keras.layers.Conv3D(1, 1, activation="sigmoid")(c6)

    return tf.keras.Model(inputs, outputs, name="unet_3d_brats_peds")


model = build_unet3d(input_shape=(*PATCH_SIZE, 4), base_filters=BASE_FILTERS, dropout_rate=DROPOUT_RATE)
model.summary()

@tf.function
def dice_coef(y_true, y_pred, smooth=1.0):
    """
    Binary Dice coefficient — the primary segmentation metric.
    Thresholds predictions at 0.5 before computing.
    Range: 0 (no overlap) → 1 (perfect overlap).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3, 4])
    denom        = tf.reduce_sum(y_true, axis=[1, 2, 3, 4]) + tf.reduce_sum(y_pred, axis=[1, 2, 3, 4])
    return tf.reduce_mean((2.0 * intersection + smooth) / (denom + smooth))


@tf.function
def soft_dice_loss(y_true, y_pred, smooth=1.0):
    """
    Differentiable version — uses raw probabilities (not thresholded)
    so gradients can flow during backprop.
    """
    y_true       = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3, 4])
    denom        = tf.reduce_sum(y_true, axis=[1, 2, 3, 4]) + tf.reduce_sum(y_pred, axis=[1, 2, 3, 4])
    return 1.0 - tf.reduce_mean((2.0 * intersection + smooth) / (denom + smooth))


@tf.function
def bce_dice_loss(y_true, y_pred):
    """
    Combined loss: BCE handles per-voxel accuracy early in training,
    Dice handles region-level overlap and class imbalance.
    """
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + soft_dice_loss(y_true, y_pred)


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=bce_dice_loss,
    metrics=[dice_coef],
)

callbacks = [
    # saves the model to disk whenever validation Dice improves (critical to colab)
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_OUT, save_best_only=True, monitor="val_dice_coef", mode="max"
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_coef", mode="max", patience=5, restore_best_weights=True
    ),
    # if Dice hasn't improved for 3 epochs, cuts the learning rate in half
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_dice_coef", mode="max", factor=0.5, patience=3, min_lr=1e-6
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

best_val = max(history.history.get("val_dice_coef", [float("nan")]))
print(f"\nBest validation Dice: {best_val:.4f}")

# Post-Training evaluation starts here
# Hausdorff is computed once after training on the full val set
def compute_hausdorff(y_true_batch: np.ndarray, y_pred_batch: np.ndarray) -> float:
    """
    Compute mean Hausdorff distance across a batch of 3D volumes.
    Handles edge cases: both empty → 0, one empty → max possible distance.
    """
    scores = []
    for yt, yp in zip(y_true_batch, y_pred_batch):
        yt = (yt.squeeze() > 0.5)
        yp = (yp.squeeze() > 0.5)
        if not yt.any() and not yp.any():
            scores.append(0.0)
        elif not yt.any() or not yp.any():
            scores.append(float(max(yt.shape)))
        else:
            scores.append(float(skimage_hausdorff(yt, yp)))
    return float(np.mean(scores))

print("Running post-training evaluation on validation set")

all_dice, all_hd = [], []
for xb, yb in val_ds:
    preds = model.predict(xb, verbose=0)
    batch_dice = dice_coef(yb, tf.constant(preds)).numpy()
    batch_hd   = compute_hausdorff(yb.numpy(), preds)
    all_dice.append(batch_dice)
    all_hd.append(batch_hd)

print(f"\n{'='*40}")
print(f"  Mean Dice coefficient : {np.mean(all_dice):.4f}") # the higher the better, 0->1
print(f"  Mean Hausdorff distance: {np.mean(all_hd):.2f} voxels") # measures boundary accuracy in voxels (lower is better, 0 is perfect)
print(f"{'='*40}")


# Sample 3D prediction visual
# Since we can't display a 3D volume directly, 
# This takes a single 2D slice from the middle of the depth dimension of the predicted 3D patch to give a representative cross-section
def show_prediction_3d(model, x, y, out_png="sample_prediction_3d.png"):
    idx  = np.random.randint(0, len(x))
    pred = model.predict(x[idx:idx+1], verbose=0)[0, ..., 0]  
    mid  = pred.shape[0] // 2                                  

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(x[idx, mid, :, :, 0], cmap="gray");  axes[0].set_title("T1 (mid slice)")
    axes[1].imshow(x[idx, mid, :, :, 3], cmap="gray");  axes[1].set_title("FLAIR (mid slice)")
    axes[2].imshow(y[idx, mid, :, :, 0], cmap="gray");  axes[2].set_title("Ground Truth")
    axes[3].imshow(pred[mid] > 0.5,      cmap="gray");  axes[3].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"Saved to {out_png}")

show_prediction_3d(model, x_val, y_val)