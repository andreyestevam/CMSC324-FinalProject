import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from load_dataset import (
    build_dataset,
    make_torch_dataset,
)
from metric_torch import bce_dice_loss, dice_coef
# from metric import bce_dice_loss_torch, dice_coef_torch # TODO: maybe merge metric and metric_torch
from model_swin_unetr import build_swin_unetr_mc

with open("./hparams.json", "r") as f:
    config = json.load(f)
    swin_cfg = config["swin_transformer"]
    hparams = swin_cfg["hparam_grid"] # TODO: will change once the json structure is updated

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Hparams:", swin_cfg["hparam_grid"]) # TODO: will change once the json structure is updated

x_train, y_train = build_dataset(dataset='train', n_train=42)
x_val, y_val = build_dataset(dataset='val', n_val=6)

# TODO: Replace this temporary test-as-validation setup with a proper validation split.
train_loader = make_torch_dataset(x_train, y_train, training=True)
val_loader = make_torch_dataset(x_val, y_val, training=False)

print("Train samples:", len(x_train), "Val samples:", len(x_val))
print("Train batches:", len(train_loader), "Val batches:", len(val_loader))

model = build_swin_unetr_mc(
    input_shape=(*hparams["patch_size"], 4),
    out_channels=1,
    feature_size=hparams["feature_size"], # TODO: what does this hparameter do?
    drop_rate=hparams["dropout_rate"],
    attn_drop_rate=hparams["attn_drop_rate"],
    dropout_path_rate=hparams["dropout_path_rate"],
    force_mc_dropout=True,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])

model_out = Path(hparams["model_out"])

best_val_dice = float("-inf")
history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": []}

print(model.__class__.__name__)
print("Checkpoint path:", model_out)

for epoch in range(1, hparams["n_epochs"] + 1):
    model.train()
    train_losses = []
    train_dices = []

    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        preds = torch.sigmoid(model(xb))
        loss = bce_dice_loss(yb, preds)
        loss.backward()
        optimizer.step()

        train_losses.append(float(loss.detach().cpu()))
        train_dices.append(float(dice_coef(yb.detach(), preds.detach()).cpu()))

    model.eval()
    val_losses = []
    val_dices = []
    with torch.no_grad():
        for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch} val", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            preds = torch.sigmoid(model(xb))
            loss = bce_dice_loss(yb, preds)
            val_losses.append(float(loss.cpu()))
            val_dices.append(float(dice_coef(yb, preds).cpu()))

    mean_train_loss = sum(train_losses) / max(len(train_losses), 1)
    mean_train_dice = sum(train_dices) / max(len(train_dices), 1)
    mean_val_loss = sum(val_losses) / max(len(val_losses), 1)
    mean_val_dice = sum(val_dices) / max(len(val_dices), 1)

    history["train_loss"].append(mean_train_loss)
    history["train_dice"].append(mean_train_dice)
    history["val_loss"].append(mean_val_loss)
    history["val_dice"].append(mean_val_dice)

    if mean_val_dice > best_val_dice:
        best_val_dice = mean_val_dice
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "hparams": hparams,
                "best_val_dice": best_val_dice,
            },
            model_out,
        )

    print(
        f"Epoch {epoch}/{hparams['n_epochs']} | "
        f"train_loss={mean_train_loss:.4f} train_dice={mean_train_dice:.4f} | "
        f"val_loss={mean_val_loss:.4f} val_dice={mean_val_dice:.4f}"
    )

# Save history as swin_history.json
with open("swin_history.json", "w") as f:
    json.dump(history, f)
print("Saved history to: swin_history.json")


print(f"Best validation Dice: {best_val_dice:.4f}")
print(f"Saved checkpoint: {model_out}")

checkpoint = torch.load(model_out, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    sample_x, sample_y = next(iter(val_loader))
    sample_x, sample_y = sample_x.to(device), sample_y.to(device)
    sample_x = sample_x.to(device)

    mc_preds = [torch.sigmoid(model(sample_x[:1])) for _ in tqdm(range(100))]
    mc_stack = torch.stack(mc_preds, dim=0)
    mc_variance = float(mc_stack.var(dim=0).mean().cpu())

print(f"MC dropout variance mean (single sample): {mc_variance:.6f}")

from torchinfo import summary

summary(model, input_size=(1, 4, 64, 64, 64))