# just a very simple 2D unet segemenation model 

import os
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# can change
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 5
MAX_TRAIN_CASES = 12
MAX_VAL_CASES = 3
SLICES_PER_CASE = 24
USE_WHOLE_TUMOR = True
SEED = 42
MODEL_OUT = "unet_baseline.keras"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# I ran this in google colab using the shared data in google drive
from google.colab import drive
drive.mount('/content/drive')

ROOT = "/content/drive/MyDrive/CMSC 324 Final project dataset/BraTS-PEDs-v1/"
TRAIN_DIR = os.path.join(ROOT, "Training")

required_suffixes = ["-t1n.nii.gz", "-t1c.nii.gz", "-t2w.nii.gz", "-t2f.nii.gz", "-seg.nii.gz"]

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
else:
    case_ids = sorted([
        name for name in os.listdir(TRAIN_DIR)
        if not name.startswith(".") and os.path.isdir(os.path.join(TRAIN_DIR, name))
    ])
    print(f"Found {len(case_ids)} case folders.")

    # check the first 5 cases
    for case_id in case_ids[:5]:
        case_dir = os.path.join(TRAIN_DIR, case_id)
        files = [f for f in os.listdir(case_dir) if not f.startswith(".")]
        print(f"\n{case_id}")
        for suf in required_suffixes:
            found = any(name.endswith(suf) for name in files)
            print(f"  {suf}: {'✓' if found else '✗ MISSING'}")

# helper 
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
        "t1":    match_any(["-t1n.nii.gz", "_t1.nii.gz",   "-t1.nii.gz"]),
        "t1ce":  match_any(["-t1c.nii.gz", "-t1ce.nii.gz", "_t1ce.nii.gz"]),
        "t2":    match_any(["-t2w.nii.gz", "-t2.nii.gz",   "_t2.nii.gz"]),
        "flair": match_any(["-t2f.nii.gz", "-flair.nii.gz","_flair.nii.gz"]),
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
print(f"all 5 files present: {len(all_cases)}")

# Volume preprocessing
def zscore_nonzero(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mask = x != 0
    if np.any(mask):
        vals = x[mask]
        mean, std = vals.mean(), vals.std()
        if std < 1e-6:
            std = 1.0
        x[mask] = (x[mask] - mean) / std
    return x


def resize_slice(arr2d: np.ndarray, target_hw: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
    arr = arr2d[..., np.newaxis].astype(np.float32)
    method = "nearest" if is_mask else "bilinear"
    arr = tf.image.resize(arr, target_hw, method=method).numpy()[..., 0]
    if is_mask:
        arr = (arr > 0.5).astype(np.float32)
    return arr


def load_case_as_slices(case: Dict[str, Path], img_size: int, slices_per_case: int) -> Tuple[np.ndarray, np.ndarray]:
    modalities = []
    for key in ["t1", "t1ce", "t2", "flair"]:
        vol = nib.load(str(case[key])).get_fdata().astype(np.float32)
        modalities.append(zscore_nonzero(vol))

    seg = nib.load(str(case["seg"])).get_fdata().astype(np.float32)
    seg = (seg > 0).astype(np.float32)  # whole-tumor binary mask

    image_4d = np.stack(modalities, axis=-1)  # [H, W, Z, 4]

    tumor_per_slice = seg.sum(axis=(0, 1))
    valid = np.where(tumor_per_slice > 0)[0]
    if len(valid) == 0:
        valid = np.arange(seg.shape[2])
    if len(valid) > slices_per_case:
        chosen = np.linspace(0, len(valid) - 1, slices_per_case).astype(int)
        valid = valid[chosen]

    xs, ys = [], []
    for z in valid:
        img_slice = image_4d[:, :, z, :]
        seg_slice = seg[:, :, z]
        img_r = tf.image.resize(img_slice, (img_size, img_size), method="bilinear").numpy().astype(np.float32)
        seg_r = resize_slice(seg_slice, (img_size, img_size), is_mask=True)[..., np.newaxis]
        xs.append(img_r)
        ys.append(seg_r)

    return np.stack(xs), np.stack(ys)


def build_numpy_dataset(cases, img_size, slices_per_case):
    all_x, all_y = [], []
    for i, case in enumerate(cases, 1):
        print(f"  Loading {i}/{len(cases)}: {case['case_id']}")
        x, y = load_case_as_slices(case, img_size, slices_per_case)
        all_x.append(x)
        all_y.append(y)
    return np.concatenate(all_x).astype(np.float32), np.concatenate(all_y).astype(np.float32)

# train/val split 
random.shuffle(all_cases)
train_cases = all_cases[:MAX_TRAIN_CASES]
val_cases   = all_cases[MAX_TRAIN_CASES:MAX_TRAIN_CASES + MAX_VAL_CASES]

if len(val_cases) == 0:
    raise RuntimeError("Not enough cases for a validation split. Lower MAX_TRAIN_CASES.")

print("Loading training cases...")
x_train, y_train = build_numpy_dataset(train_cases, IMG_SIZE, SLICES_PER_CASE)
print("Loading validation cases...")
x_val, y_val = build_numpy_dataset(val_cases, IMG_SIZE, SLICES_PER_CASE)

print(f"\nx_train: {x_train.shape}  y_train: {y_train.shape}")
print(f"x_val:   {x_val.shape}  y_val:   {y_val.shape}")


def make_tf_dataset(x, y, batch_size, training):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(min(len(x), 1024), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


train_ds = make_tf_dataset(x_train, y_train, BATCH_SIZE, training=True)
val_ds   = make_tf_dataset(x_val,   y_val,   BATCH_SIZE, training=False)

# model
def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def build_unet(input_shape=(128, 128, 4)):
    inputs = tf.keras.Input(shape=input_shape)

    c1 = conv_block(inputs, 32);  p1 = tf.keras.layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64);      p2 = tf.keras.layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128);     p3 = tf.keras.layers.MaxPooling2D()(c3)

    bn = conv_block(p3, 256)

    u3 = tf.keras.layers.UpSampling2D()(bn)
    u3 = tf.keras.layers.Concatenate()([u3, c3]);  c4 = conv_block(u3, 128)

    u2 = tf.keras.layers.UpSampling2D()(c4)
    u2 = tf.keras.layers.Concatenate()([u2, c2]);  c5 = conv_block(u2, 64)

    u1 = tf.keras.layers.UpSampling2D()(c5)
    u1 = tf.keras.layers.Concatenate()([u1, c1]);  c6 = conv_block(u1, 32)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c6)
    return tf.keras.Model(inputs, outputs, name="unet_2d_brats_peds")


model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 4))
model.summary()

# loss function
@tf.function
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    denom = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + smooth) / (denom + smooth)


@tf.function
def soft_dice_loss(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denom = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return 1.0 - tf.reduce_mean((2.0 * intersection + smooth) / (denom + smooth))


@tf.function
def bce_dice_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + soft_dice_loss(y_true, y_pred)


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=bce_dice_loss,
    metrics=[dice_coef],
)

# training 
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="val_dice_coef", mode="max"),
    tf.keras.callbacks.EarlyStopping(monitor="val_dice_coef", mode="max", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_dice_coef", mode="max", factor=0.5, patience=2),
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

# visual
def show_prediction(model, x, y, out_png="sample_prediction.png"):
    idx = np.random.randint(0, len(x))
    pred = model.predict(x[idx:idx+1], verbose=0)[0, :, :, 0]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(x[idx, :, :, 0], cmap="gray"); axes[0].set_title("T1")
    axes[1].imshow(x[idx, :, :, 3], cmap="gray"); axes[1].set_title("FLAIR")
    axes[2].imshow(y[idx, :, :, 0], cmap="gray"); axes[2].set_title("Ground Truth")
    axes[3].imshow(pred > 0.5,       cmap="gray"); axes[3].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"Saved to {out_png}")

show_prediction(model, x_val, y_val)