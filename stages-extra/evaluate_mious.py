import os
import torch
import numpy as np
from tqdm import tqdm
from datasets import SIORSemanticDataset
from torch.utils.data import DataLoader
from unet import UNet
import cv2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IGNORE_INDEX = 255
BASE_C = 16
NUM_CLASSES = 256


def compute_miou(pred, gt, num_classes):
    ious = []
    for cls in range(num_classes):
        if cls == IGNORE_INDEX:
            continue
        pred_i = (pred == cls)
        gt_i = (gt == cls)
        mask_valid = (gt != IGNORE_INDEX)
        inter = np.logical_and(pred_i, gt_i) & mask_valid
        union = np.logical_or(pred_i, gt_i) & mask_valid
        if union.sum() > 0:
            ious.append(inter.sum() / union.sum())
    return np.mean(ious) if ious else 0


def main():
    test_ds = SIORSemanticDataset(
        os.path.join(PROJECT_ROOT, "data", "test_images"),
        os.path.join(PROJECT_ROOT, "output", "masks_cleaned_remap"),
        os.path.join(PROJECT_ROOT, "output", "datasets", "test.txt"),
    )
    loader = DataLoader(test_ds, batch_size=1)

    model = UNet(3, NUM_CLASSES, base_c=BASE_C).to(DEVICE)
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model.eval()

    miou_list = []
    with torch.no_grad():
        for img, mask in tqdm(loader):
            img = img.to(DEVICE)
            pred = model(img).argmax(1).cpu().numpy()
            gt = mask.numpy()
            miou_list.append(compute_miou(pred, gt, NUM_CLASSES))

    print("Test mIoU:", np.mean(miou_list))


if __name__ == "__main__":
    main()
