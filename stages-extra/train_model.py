import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import SIORSemanticDataset
from unet import UNet


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IGNORE_INDEX = 255


# =========================
# Determine NUM_CLASSES safely
# =========================
mask_dir = os.path.join(PROJECT_ROOT, "output", "masks_cleaned_remap")
all_masks = sorted(os.listdir(mask_dir))
num_classes = 0
for m in all_masks:
    mask_path = os.path.join(mask_dir, m)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue
    mask = mask.astype("int32")
    num_classes = max(num_classes, mask.max() + 1)

NUM_CLASSES = num_classes
print(f"[INFO] NUM_CLASSES set to {NUM_CLASSES}")


# =========================
# Datasets
# =========================
train_ds = SIORSemanticDataset(
    os.path.join(PROJECT_ROOT, "data", "trainval_images"),
    os.path.join(PROJECT_ROOT, "output", "masks_cleaned_remap"),
    os.path.join(PROJECT_ROOT, "output", "datasets", "train.txt"),
)
val_ds = SIORSemanticDataset(
    os.path.join(PROJECT_ROOT, "data", "trainval_images"),
    os.path.join(PROJECT_ROOT, "output", "masks_cleaned_remap"),
    os.path.join(PROJECT_ROOT, "output", "datasets", "val.txt"),
)


train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

model = UNet(3, NUM_CLASSES, base_c=16).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10
use_amp = True if DEVICE == "cuda" else False
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


# =========================
# Training process
# =========================
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch")
    for img, mask in pbar:
        img, mask = img.to(DEVICE), mask.to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(img)
            loss = criterion(pred, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch + 1}] Train Avg Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "model.pth")
    print("✔ Model saved to model.pth")
