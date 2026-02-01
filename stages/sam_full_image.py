"""
    Deprecated: This script is no longer in use.
"""
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor


# =========================
# 1. Paths & Hyper-params
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "trainval_images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "masks_raw")

SAM_CHECKPOINT = os.path.join(PROJECT_ROOT, "sam", "sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"
DEVICE = "cpu" if torch.cuda.is_available() else "cpu"

GRID_SIZE = 64
MIN_AREA_RATIO = 0.008
MAX_AREA_RATIO = 0.45
MIN_SCORE = 0.75

MIN_COMPONENT_AREA = 100      # small noise removal
MORPH_KERNEL = 5              # closing kernel size


# =========================
# 2. Load SAM
# =========================
def load_sam():
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    sam.eval()
    return SamPredictor(sam)


# =========================
# 3. Grid Point Generator
# =========================
def generate_grid_points(h, w, grid):
    points = []
    for y in range(grid // 2, h, grid):
        for x in range(grid // 2, w, grid):
            points.append([x, y])
    return np.array(points)


# =========================
# 4. Mask Quality Score
# =========================
def mask_quality(mask, image_area):
    """
    Lightweight heuristic quality score:
    - area ratio
    - connected component count
    """
    area = mask.sum()
    area_ratio = area / image_area

    num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8))
    components = num_labels - 1  # exclude background

    # score favors:
    # - reasonable size
    # - fewer fragments
    score = area_ratio * np.exp(-0.3 * max(0, components - 1))
    return score


# =========================
# 5. SAM Inference (Coarse)
# =========================
def run_sam_on_image(predictor, image):
    h, w, _ = image.shape
    image_area = h * w
    predictor.set_image(image)
    points = generate_grid_points(h, w, GRID_SIZE)
    candidates = []

    for (x, y) in points:
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=False
        )

        mask = masks[0]
        score = scores[0]
        area_ratio = mask.sum() / image_area

        if score < MIN_SCORE:
            continue
        if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
            continue

        q = mask_quality(mask, image_area)
        candidates.append((q, mask.astype(np.uint8)))

    return candidates


# =========================
# 6. Mask Cleaning & Merge (Fine)
# =========================
def clean_and_merge(candidates, shape):
    if len(candidates) == 0:
        return np.zeros(shape, dtype=np.uint8)

    # sort by quality (high → low)
    candidates.sort(key=lambda x: x[0], reverse=True)
    merged = np.zeros(shape, dtype=np.uint8)

    for _, mask in candidates:
        # remove tiny components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
                cleaned[labels == i] = 1

        merged |= cleaned

    # morphological closing to fill holes
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL)
    )
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)

    return merged


# =========================
# 7. Main Pipeline
# =========================
def process_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    predictor = load_sam()
    image_files = sorted(os.listdir(IMAGE_DIR))
    done = set(os.listdir(OUTPUT_DIR))

    for name in tqdm(image_files):
        if name in done:
            continue

        img_path = os.path.join(IMAGE_DIR, name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"[WARN] Cannot read image: {name}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        candidates = run_sam_on_image(predictor, image)
        label = clean_and_merge(candidates, (h, w))

        if label.sum() == 0:
            print(f"[WARN] {name}: empty after cleaning")

        save_path = os.path.join(OUTPUT_DIR, name)
        cv2.imwrite(save_path, label * 255)


if __name__ == "__main__":
    process_images()
