import tensorflow as tf
import os, json
from load_dataset import build_dataset, make_tf_dataset
from model_unet import build_unet3d
from metric_tf import dice_coef, bce_dice_loss

with open('./hparams.json', 'r') as f:
    config = json.load(f)
    hparams = config['unet']['hparam_grid'] 

# Check GPU 
gpus = tf.config.list_physical_devices('GPU')
print("Using device:", "GPU" if gpus else "CPU")

x_train, y_train = build_dataset(dataset='train', n_train=42)
x_val,   y_val   = build_dataset(dataset='val', n_val=6)

train_ds = make_tf_dataset(x_train, y_train, training=True)
val_ds = make_tf_dataset(x_val,   y_val,   training=False)

model = build_unet3d(
    input_shape=tuple(hparams["input_shape"]),
    dropout_rate=hparams["dropout_rate"]    # passes 0.2 from hparams.json
)
model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(hparams["model_out"], save_best_only=True, monitor="val_dice_coef", mode="max"),
    tf.keras.callbacks.EarlyStopping(monitor="val_dice_coef", mode="max", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_dice_coef", mode="max", factor=0.5, patience=2),
]

optim = tf.keras.optimizers.Adam(learning_rate=hparams["learning_rate"]) 

model.compile(
    optimizer=optim,
    loss=bce_dice_loss,
    metrics=[dice_coef],
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=hparams["n_epochs"],
    callbacks=callbacks,
    verbose=1,
)

# Best validation dice
best_val = max(history.history.get("val_dice_coef", [float("nan")]))
print(f"Best validation Dice: {best_val:.4f}")

# save model
model.save(hparams["model_out"])
print(f"Saved model to: {hparams['model_out']}")

# Save history as unet_history.json
with open("unet_history.json", "w") as f:
    json.dump(history.history, f)
print("Saved history to: unet_history.json")

# MC dropout variance check 
from metric import mc_prediction # the name may change

sample_x = x_val[:1]   # single patch to test uncertainty
pred_mean, pred_std, pred_all = mc_prediction(
    model, sample_x, num_passes=100
)
mc_variance = float(pred_std.mean())
print(f"MC dropout variance mean (single sample): {mc_variance:.6f}")
