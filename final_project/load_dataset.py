#load_dataset.py

import os
import json
# import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

with open('./config.json', 'r') as f:
    config = json.load(f)
    dataset_path = config['dataset_path']

ROOT = dataset_path
TRAIN_DIR = os.path.join(ROOT, "Training")

required_suffixes = ["-t1n.nii.gz", "-t1c.nii.gz", "-t2w.nii.gz", "-t2f.nii.gz", "-seg.nii.gz"]

# if not os.path.exists(TRAIN_DIR):
#     raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
# else:
#     case_ids = sorted([
#         name for name in os.listdir(TRAIN_DIR)
#         if not name.startswith(".") and os.path.isdir(os.path.join(TRAIN_DIR, name))
#     ])
#     print(f"Found {len(case_ids)} case folders.")

#     # check the first 5 cases
#     for case_id in case_ids[:5]:
#         case_dir = os.path.join(TRAIN_DIR, case_id)
#         files = [f for f in os.listdir(case_dir) if not f.startswith(".")]
#         print(f"\n{case_id}")
#         for suf in required_suffixes:
#             found = any(name.endswith(suf) for name in files)
#             print(f"  {suf}: {'✓' if found else '✗ MISSING'}")

# helper 
def locate_case_files(case_dir: Path) -> Dict[str, Optional[Path]]:
    
    mapping = {
        "-t1n.nii.gz": "t1",
        "-t1c.nii.gz": "t1ce",
        "-t2w.nii.gz": "t2",
        "-t2f.nii.gz": "flair",
        "-seg.nii.gz": "seg"
    }
    
    files = list(case_dir.glob("*.nii.gz"))
    lower = {f.name.lower(): f for f in files}

    result = {}
    for suffix, key in mapping.items():
        found_file = None
        for f in files:
            if f.name.lower().endswith(suffix):
                found_file = f
                break
        
        result[key] = found_file

    return result


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

# def __main__():
#     all_cases = collect_cases(Path(TRAIN_DIR))
#     print(f"all 5 files present: {len(all_cases)}")

# Volume preprocessing
# #############################################################
# def zscore_nonzero(x: np.ndarray) -> np.ndarray:
#     x = x.astype(np.float32)
#     mask = x != 0
#     if np.any(mask):
#         vals = x[mask]
#         mean, std = vals.mean(), vals.std()
#         if std < 1e-6:
#             std = 1.0
#         x[mask] = (x[mask] - mean) / std
#     return x

# def resize_slice(arr2d: np.ndarray, target_hw: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
#     arr = arr2d[..., np.newaxis].astype(np.float32)
#     method = "nearest" if is_mask else "bilinear"
#     arr = tf.image.resize(arr, target_hw, method=method).numpy()[..., 0]
#     if is_mask:
#         arr = (arr > 0.5).astype(np.float32)
#     return arr
# #############################################################