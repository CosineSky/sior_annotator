import os
import cv2
import numpy as np
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "semlabels", "gray")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "masks_cleaned_remap")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IGNORE_ID = 255
MIN_REGION_PIXELS = 10   # threshold for a valid semantic region
MAX_VALID_ID = 200       # currently useless


def clean_mask(mask):
    cleaned = mask.copy()
    unique_ids = np.unique(mask)
    for sid in unique_ids:
        if sid == IGNORE_ID:
            continue
        if sid < 0 or sid > MAX_VALID_ID:
            cleaned[mask == sid] = IGNORE_ID
            continue
        # 保留小目标，但可选择性清理过小噪声
        region_pixels = np.sum(mask == sid)
        if region_pixels < MIN_REGION_PIXELS:
            cleaned[mask == sid] = IGNORE_ID
    return cleaned


def remap_ids(mask):
    unique_ids = sorted([i for i in np.unique(mask) if i != IGNORE_ID])
    id_map = {old: new for new, old in enumerate(unique_ids)}
    remapped = np.full_like(mask, IGNORE_ID)
    for old, new in id_map.items():
        remapped[mask == old] = new
    return remapped, len(unique_ids)


def main():
    files = sorted(os.listdir(INPUT_DIR))
    total_classes = 0
    for file in tqdm(files, desc="Cleaning & remapping masks"):
        in_path = os.path.join(INPUT_DIR, file)
        out_path = os.path.join(OUTPUT_DIR, file)
        mask = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] Failed to read {file}")
            continue
        cleaned = clean_mask(mask)
        remapped, n_classes = remap_ids(cleaned)
        cv2.imwrite(out_path, remapped)
        total_classes = max(total_classes, n_classes)
    print(f"\n[OK] Cleaned & remapped masks saved to {OUTPUT_DIR}, total_classes={total_classes}")


if __name__ == "__main__":
    main()
