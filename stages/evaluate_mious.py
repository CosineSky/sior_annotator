import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_unet import UNet, SegDataset, DEVICE


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "unet.pth")


def calc_iou(pred, mask):
    """
    pred, mask: [1, 1, H, W] or [H, W]
    already binary
    """
    inter = (pred & mask).sum().item()
    union = (pred | mask).sum().item()
    if union == 0:
        return None
    return inter / union


def main():
    model = UNet().to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    loader = DataLoader(
        SegDataset("test"),
        batch_size=1,
        shuffle=False
    )
    ious = []
    skipped = 0

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = torch.sigmoid(model(x))
            pred_bin = (pred > 0.5).int()
            mask_bin = (y > 0.5).int()

            if mask_bin.sum() == 0:
                skipped += 1
                continue

            iou = calc_iou(pred_bin, mask_bin)
            if iou is not None:
                ious.append(iou)

    print(f"Valid samples: {len(ious)}")
    print(f"Skipped empty masks: {skipped}")
    print(f"Test mIoU: {np.mean(ious):.4f}")


if __name__ == "__main__":
    main()
