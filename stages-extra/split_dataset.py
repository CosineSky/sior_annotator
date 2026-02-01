import os
import random


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SIOR_ROOT = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "datasets")

TRAIN_NUM = 1000
VAL_NUM = 200
TEST_NUM = 200

random.seed(42)
os.makedirs(OUT_DIR, exist_ok=True)


def read_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def write_list(path, items):
    with open(path, "w") as f:
        for x in items:
            f.write(x + "\n")


def main():
    train_all = read_list(os.path.join(SIOR_ROOT, "train.txt"))
    val_all = read_list(os.path.join(SIOR_ROOT, "valid.txt"))

    assert len(train_all) >= TRAIN_NUM
    assert len(val_all) >= VAL_NUM

    train_split = train_all[:TRAIN_NUM]
    val_split = val_all[:VAL_NUM]

    test_imgs = sorted(os.listdir(os.path.join(SIOR_ROOT, "test_images")))
    test_ids = [os.path.splitext(x)[0] for x in test_imgs]
    random.shuffle(test_ids)
    test_split = test_ids[:TEST_NUM]

    write_list(os.path.join(OUT_DIR, "train.txt"), train_split)
    write_list(os.path.join(OUT_DIR, "val.txt"), val_split)
    write_list(os.path.join(OUT_DIR, "test.txt"), test_split)

    print("[OK] Dataset split finished:")
    print(f" Train: {len(train_split)}")
    print(f" Val: {len(val_split)}")
    print(f" Test: {len(test_split)}")


if __name__ == "__main__":
    main()
