import os
import cv2
import torch
import numpy as np
from train_unet import UNet, SegDataset, DEVICE
from torch.utils.data import DataLoader
from tqdm import tqdm


def miou(pred, mask):
    pred = (pred > 0.5)
    mask = (mask > 0.5)
    inter = (pred & mask).sum()
    union = (pred | mask).sum()
    return (inter / (union + 1e-6)).item()


def main():
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load("unet.pth", map_location=DEVICE))
    model.eval()

    loader = DataLoader(SegDataset("test"), batch_size=1)

    scores = []
    with torch.no_grad():
        for x, y in tqdm(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = torch.sigmoid(model(x))
            scores.append(miou(pred, y))

    print(f"Test mIoU: {np.mean(scores):.4f}")


if __name__ == "__main__":
    main()
