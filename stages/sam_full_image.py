import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor

# =========================
# 1. Paths and Hyper-params
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "trainval_images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "masks_raw")
SAM_CHECKPOINT = os.path.join(PROJECT_ROOT, "sam", "sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALES = [1.0]            # 单尺度
GRID_SIZE = 64            # grid 点间距
MIN_AREA_RATIO = 0.009    # 最小 mask 面积比例
MAX_AREA_RATIO = 0.4      # 最大 mask 面积比例（遥感图适当放宽）
MIN_SCORE = 0.75           # mask 置信度阈值（降低以避免全黑）


# =========================
# 2. Initializing SAM
# =========================
def load_sam():
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    return SamPredictor(sam)


# =========================
# 3. Grid Prompt
# =========================
def generate_grid_points(h, w, grid):
    pts = []
    for y in range(grid // 2, h, grid):
        for x in range(grid // 2, w, grid):
            pts.append([x, y])
    return np.array(pts)


# =========================
# 4. Single point SAM
# =========================
def run_sam_on_image(predictor, image):
    h, w, _ = image.shape
    predictor.set_image(image)
    points = generate_grid_points(h, w, GRID_SIZE)
    total_pixels = h * w
    collected = []

    for (x, y) in points:
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=False   # only one mask for one single point
        )

        m = masks[0]
        s = scores[0]
        area_ratio = m.sum() / total_pixels
        # print(f"[DEBUG] point ({x},{y}): score={s:.3f}, area_ratio={area_ratio:.3f}")

        if s < MIN_SCORE:
            continue
        if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
            continue
        collected.append(m.astype(np.uint8))

    print(f"[DEBUG] collected masks: {len(collected)}")
    if len(collected) == 0:
        return []

    # OR
    merged = np.zeros((h, w), dtype=np.uint8)
    for m in collected:
        merged |= m

    return [merged]


# =========================
# 5. Main process
# =========================
def process_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    predictor = load_sam()
    image_files = sorted(os.listdir(IMAGE_DIR))

    for name in tqdm(image_files):
        path = os.path.join(IMAGE_DIR, name)
        image = cv2.imread(path)
        if image is None:
            print(f"[WARN] cannot read image: {name}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        masks = run_sam_on_image(predictor, image)

        if len(masks) == 0:
            label = np.zeros((h, w), dtype=np.uint8)
            print(f"[WARN] {name}: empty mask")
        else:
            label = masks[0]

        save_path = os.path.join(OUTPUT_DIR, name)
        cv2.imwrite(save_path, label * 255)


if __name__ == "__main__":
    process_images()
