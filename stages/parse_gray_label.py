import os
import cv2
import numpy as np
from tqdm import tqdm
import json


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
GRAY_DIR = os.path.join(PROJECT_ROOT, "data", "semlabels", "gray")
OUT_JSON = os.path.join(PROJECT_ROOT, "logs", "sior_gray_semantic_stats.json")
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)


# =========================
# Main
# =========================
def parse_gray_labels():
    semantic_counter = {}   # gray_id -> pixel count
    image_counter = {}      # gray_id -> number of images appeared

    for name in tqdm(sorted(os.listdir(GRAY_DIR)), desc="Parsing gray labels"):
        path = os.path.join(GRAY_DIR, name)

        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        unique, counts = np.unique(mask, return_counts=True)

        for k, c in zip(unique.tolist(), counts.tolist()):
            semantic_counter[k] = semantic_counter.get(k, 0) + c
            image_counter[k] = image_counter.get(k, 0) + 1

    sorted_stats = dict(
        sorted(semantic_counter.items(), key=lambda x: x[0])
    )

    result = {
        "num_semantic_ids": len(sorted_stats),
        "semantic_pixel_stats": sorted_stats,
        "semantic_image_stats": image_counter
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"✔ Parsed {len(sorted_stats)} semantic gray IDs")
    print(f"✔ Stats saved to: {OUT_JSON}")


if __name__ == "__main__":
    parse_gray_labels()
