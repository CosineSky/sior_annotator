import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from agents.llm_agent import LLMAgent
from configs.api_key import LLM_ENDPOINT, LLM_API_KEY
from configs.semantic_map import SEMANTIC_MAP


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "trainval_images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "masks_raw")
SAM_CHECKPOINT = os.path.join(PROJECT_ROOT, "sam", "sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRID_SIZE = 16
MIN_AREA_RATIO = 0.0005
MAX_AREA_RATIO = 0.5
MIN_SCORE = 0.6
MIN_COMPONENT_AREA = 100
MORPH_KERNEL = 8
SEM_ID_MAP = {name: idx for idx, name in enumerate(SEMANTIC_MAP.keys())}


# =========================
# Load SAM
# =========================
def load_sam():
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    sam.eval()
    return SamPredictor(sam)


# =========================
# Generate grid points
# =========================
def generate_grid_points(h, w, grid):
    points = []
    for y in range(grid // 2, h, grid):
        for x in range(grid // 2, w, grid):
            points.append([x, y])
    return np.array(points)


# =========================
# Mask quality
# =========================
def mask_quality(mask, image_area):
    area = mask.sum()
    area_ratio = area / image_area
    num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8))
    components = num_labels - 1
    score = area_ratio * np.exp(-0.3 * max(0, components - 1))
    return score


# =========================
# SAM inference (coarse mask generation)
# =========================
def run_sam(predictor, image):
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

        if score < MIN_SCORE or area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
            continue

        q = mask_quality(mask, image_area)
        candidates.append((q, mask.astype(np.uint8)))

    return candidates


# =========================
# Merge masks
# =========================
def clean_and_merge(candidates, shape):
    if len(candidates) == 0:
        return np.zeros(shape, dtype=np.uint8)

    candidates.sort(key=lambda x: x[0], reverse=True)
    merged = np.zeros(shape, dtype=np.uint8)

    for _, mask in candidates:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
                cleaned[labels == i] = 1

        merged |= cleaned

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)

    return merged


# =========================
# Main processing
# =========================
def process_images(llm_enabled=True):
    llm_endpoint = LLM_ENDPOINT
    llm_api_key = LLM_API_KEY
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    predictor = load_sam()
    llm_agent = LLMAgent(endpoint=llm_endpoint, api_key=llm_api_key) if llm_enabled else None

    image_files = sorted(os.listdir(IMAGE_DIR))
    done = set(os.listdir(OUTPUT_DIR))

    for name in tqdm(image_files, desc="Generating multi-class masks"):
        if name in done:
            continue

        img_path = os.path.join(IMAGE_DIR, name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Cannot read image: {name}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        final_mask = np.full((h, w), 255, dtype=np.uint8)  # background=255

        candidates = run_sam(predictor, image)
        merged_mask = clean_and_merge(candidates, (h, w))

        # Save temporary mask for LLM prompt
        tmp_mask_path = os.path.join(OUTPUT_DIR, f"{name[:5]}.png")
        cv2.imwrite(tmp_mask_path, merged_mask*255)  # foreground=255, background=0 for visualization

        if llm_enabled:
            result = llm_agent.classify_mask(img_path, tmp_mask_path)
            if result["decision"] == "keep":
                semantic_id = SEM_ID_MAP.get(result["semantic"], 0)
                final_mask[merged_mask > 0] = semantic_id
        else:
            # fallback: assign all foreground to a default id (e.g., 1)
            final_mask[merged_mask > 0] = 1

        save_path = os.path.join(OUTPUT_DIR, name)
        cv2.imwrite(save_path, final_mask)


if __name__ == "__main__":
    process_images(llm_enabled=False)
