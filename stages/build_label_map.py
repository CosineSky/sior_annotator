import os
import cv2
import numpy as np
from tqdm import tqdm

from configs.semantic_map import SEMANTIC_MAP

# =========================
# Path config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "trainval_images")
MASK_DIR = os.path.join(PROJECT_ROOT, "output", "masks_final")
LABEL_DIR = os.path.join(PROJECT_ROOT, "output", "label_maps")

os.makedirs(LABEL_DIR, exist_ok=True)

# =========================
# Config
# =========================
IGNORE_LABEL = 255
VALID_CLASS_IDS = set(SEMANTIC_MAP.values()) | {0}  # include background


def main():
    for name in tqdm(sorted(os.listdir(MASK_DIR)), desc="Building label maps"):
        mask_path = os.path.join(MASK_DIR, name)
        img_path = os.path.join(IMAGE_DIR, name)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(img_path)

        if mask is None or image is None:
            continue

        h, w = image.shape[:2]

        # ---- size check ----
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        label_map = mask.copy().astype(np.uint8)

        # ---- semantic validity check ----
        unique_ids = np.unique(label_map)
        invalid_ids = [i for i in unique_ids if i not in VALID_CLASS_IDS]

        if invalid_ids:
            # map invalid labels to IGNORE
            for i in invalid_ids:
                label_map[label_map == i] = IGNORE_LABEL

        cv2.imwrite(
            os.path.join(LABEL_DIR, name),
            label_map
        )


if __name__ == "__main__":
    main()
