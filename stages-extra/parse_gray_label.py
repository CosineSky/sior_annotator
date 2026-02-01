import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SEM_LABEL_DIR = os.path.join(PROJECT_ROOT, "data", "semlabels", "gray")
SAVE_REPORT = os.path.join(PROJECT_ROOT, "logs", "semantic_statistics.txt")
os.makedirs(os.path.dirname(SAVE_REPORT), exist_ok=True)
IGNORE_ID = 255


# =========================
# Main
# =========================
def main():
    pixel_counter = defaultdict(int)
    image_counter = defaultdict(int)

    label_files = sorted(os.listdir(SEM_LABEL_DIR))
    assert len(label_files) > 0, "No semantic label files found."

    total_pixels = 0
    for file in tqdm(label_files, desc="Parsing semantic labels"):
        path = os.path.join(SEM_LABEL_DIR, file)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] Failed to read {file}")
            continue

        h, w = mask.shape
        total_pixels += h * w
        unique_ids, counts = np.unique(mask, return_counts=True)

        for sid, cnt in zip(unique_ids, counts):
            pixel_counter[int(sid)] += int(cnt)
            image_counter[int(sid)] += 1

    semantic_ids = sorted(pixel_counter.keys())
    print("\n=== Semantic ID Summary ===")
    for sid in semantic_ids:
        ratio = pixel_counter[sid] / total_pixels
        print(f"ID {sid:3d} | pixels={pixel_counter[sid]:>10d} | ratio={ratio:.6f}")

    # Save to file
    with open(SAVE_REPORT, "w") as f:
        f.write("Semantic ID Statistics\n")
        f.write("======================\n")
        f.write(f"Total images: {len(label_files)}\n")
        f.write(f"Total pixels: {total_pixels}\n\n")

        for sid in semantic_ids:
            ratio = pixel_counter[sid] / total_pixels
            f.write(
                f"ID {sid:3d} | pixels={pixel_counter[sid]:>10d} "
                f"| ratio={ratio:.6f} | appears_in_images={image_counter[sid]}\n"
            )

    print(f"\n[OK] Statistics saved to {SAVE_REPORT}")

    # Suggested training config
    valid_ids = [sid for sid in semantic_ids if sid != IGNORE_ID]
    print("\n=== Suggested Training Config ===")
    print(f"num_classes = {max(valid_ids) + 1}")
    print(f"ignore_index = {IGNORE_ID}")


if __name__ == "__main__":
    main()
