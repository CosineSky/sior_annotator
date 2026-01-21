import os
import cv2
import json
import numpy as np
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "trainval_images")
MASK_DIR = os.path.join(PROJECT_ROOT, "output", "masks_final")
ANN_DIR = os.path.join(PROJECT_ROOT, "output", "annotations_json")
LABEL_DIR = os.path.join(PROJECT_ROOT, "output", "label_maps")

os.makedirs(LABEL_DIR, exist_ok=True)


def main():
    for name in tqdm(os.listdir(ANN_DIR)):
        ann_path = os.path.join(ANN_DIR, name)

        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        image_name = ann["image"]
        class_id = ann["class_id"]

        img_path = os.path.join(IMAGE_DIR, image_name)
        mask_path = os.path.join(MASK_DIR, image_name)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            continue

        h, w = image.shape[:2]
        label_map = np.zeros((h, w), dtype=np.uint8)
        label_map[mask > 0] = class_id

        cv2.imwrite(os.path.join(LABEL_DIR, image_name), label_map)


if __name__ == "__main__":
    main()
