import os
import cv2
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
BATCH_SIZE = 4
LR = 1e-3


# =========================
# 1. Dataset
# =========================
class SegDataset(Dataset):
    def __init__(self, split):
        self.img_dir = os.path.join(DATASET_DIR, "images", split)
        self.mask_dir = os.path.join(DATASET_DIR, "masks", split)
        self.files = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = cv2.imread(os.path.join(self.img_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1) / 255.0

        mask = cv2.imread(
            os.path.join(self.mask_dir, name),
            cv2.IMREAD_GRAYSCALE
        )
        mask = (mask > 0).astype(np.float32)

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        )


# =========================
# 2. U-net
# =========================
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def C(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU()
            )

        self.d1 = C(3, 64)
        self.d2 = C(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.u1 = C(128 + 64, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        u = torch.nn.functional.interpolate(c2, scale_factor=2)
        u = self.u1(torch.cat([u, c1], dim=1))
        return self.out(u)


# =========================
# 3. Training
# =========================
def train():
    model = UNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        SegDataset("train"),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0

        for x, y in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            x, y = x.to(DEVICE), y.to(DEVICE)

            pred = model(x)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total / len(loader):.4f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "unet.pth"))
    print("[SUCCESS] Model saved.")


if __name__ == "__main__":
    train()
