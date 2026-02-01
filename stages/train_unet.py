"""
    Deprecated: This script is no longer in use.
"""
import os
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
LABEL_MAP_DIR = os.path.join(PROJECT_ROOT, "output", "label_maps")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
BATCH_SIZE = 2
LR = 1e-3
IMG_SIZE = 512


# =========================
# Infer number of classes
# =========================
def infer_num_classes(label_root):
    max_id = 0
    for name in os.listdir(label_root):
        mask = cv2.imread(os.path.join(label_root, name), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        max_id = max(max_id, int(mask.max()))
    return max_id + 1  # include background

NUM_CLASSES = infer_num_classes(LABEL_MAP_DIR)
print(f"[INFO] NUM_CLASSES = {NUM_CLASSES}")


# =========================
# Dataset
# =========================
class SegDataset(Dataset):
    def __init__(self, split):
        self.img_dir = os.path.join(DATASET_DIR, "images", split)
        self.mask_dir = os.path.join(DATASET_DIR, "masks", split)
        self.files = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = cv2.imread(os.path.join(self.img_dir, name))
        mask = cv2.imread(
            os.path.join(self.mask_dir, name),
            cv2.IMREAD_GRAYSCALE
        )

        # resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(
            mask,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1) / 255.0
        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.long)
        )


# =========================
# UNet (multi-class)
# =========================
class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def C(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = C(3, 32)
        self.enc2 = C(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.dec1 = C(64 + 32, 32)
        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))

        u = torch.nn.functional.interpolate(
            c2, scale_factor=2, mode="bilinear", align_corners=False
        )
        u = torch.cat([u, c1], dim=1)

        return self.out(self.dec1(u))


# =========================
# Training process
# =========================
def train():
    model = UNet(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        SegDataset("train"),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, masks in pbar:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(imgs)          # [B, C, H, W]
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Mean Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "unet.pth"))
    print("[SUCCESS] Model saved.")


if __name__ == "__main__":
    train()
