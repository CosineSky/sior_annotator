"""
    Deprecated: This script is no longer in use.
"""
import os
import shutil
import random
import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "trainval_images")
LABEL_DIR = os.path.join(PROJECT_ROOT, "output", "label_maps")
OUT_DIR = os.path.join(PROJECT_ROOT, "dataset")

TRAIN_NUM = 1000
VAL_NUM = 200
TEST_NUM = 200
IGNORE_LABEL = 255

random.seed(42)


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def main():
    images = sorted(os.listdir(LABEL_DIR))

    assert len(images) >= TRAIN_NUM + VAL_NUM + TEST_NUM, (
        f"Not enough samples: {len(images)} "
        f"(need {TRAIN_NUM + VAL_NUM + TEST_NUM})"
    )

    random.shuffle(images)
    train = images[:TRAIN_NUM]
    val = images[TRAIN_NUM:TRAIN_NUM + VAL_NUM]
    test = images[TRAIN_NUM + VAL_NUM:TRAIN_NUM + VAL_NUM + TEST_NUM]
    splits = {"train": train, "val": val, "test": test}

    for split, files in splits.items():
        img_out = os.path.join(OUT_DIR, "images", split)
        mask_out = os.path.join(OUT_DIR, "masks", split)
        mkdir(img_out)
        mkdir(mask_out)

        for name in files:
            img_src = os.path.join(IMAGE_DIR, name)
            mask_src = os.path.join(LABEL_DIR, name)

            if not os.path.exists(img_src) or not os.path.exists(mask_src):
                continue

            # ---- semantic sanity check ----
            mask = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            unique_ids = np.unique(mask)
            if len(unique_ids) <= 1:
                continue

            shutil.copy(img_src, os.path.join(img_out, name))
            shutil.copy(mask_src, os.path.join(mask_out, name))

    print("[SUCCESS] Dataset split done.")
    print(f"  Train: {len(train)}")
    print(f"  Val  : {len(val)}")
    print(f"  Test : {len(test)}")


if __name__ == "__main__":
    main()
