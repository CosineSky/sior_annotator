import os
import shutil
import random


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "trainval_images")
MASK_DIR = os.path.join(PROJECT_ROOT, "output", "masks_final")
OUT_DIR = os.path.join(PROJECT_ROOT, "dataset")

TRAIN_NUM = 1000
VAL_NUM = 200
TEST_NUM = 200

random.seed(42)


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def main():
    images = sorted(os.listdir(MASK_DIR))
    random.shuffle(images)

    train = images[:TRAIN_NUM]
    val = images[TRAIN_NUM:TRAIN_NUM + VAL_NUM]
    test = images[TRAIN_NUM + VAL_NUM:TRAIN_NUM + VAL_NUM + TEST_NUM]

    splits = {"train": train, "val": val, "test": test}

    for split, files in splits.items():
        mkdir(os.path.join(OUT_DIR, "images", split))
        mkdir(os.path.join(OUT_DIR, "masks", split))

        for name in files:
            shutil.copy(
                os.path.join(IMAGE_DIR, name),
                os.path.join(OUT_DIR, "images", split, name)
            )
            shutil.copy(
                os.path.join(MASK_DIR, name),
                os.path.join(OUT_DIR, "masks", split, name)
            )

    print("[SUCCESS] Dataset split done.")


if __name__ == "__main__":
    main()
