import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from agents.llm_agent import LLMAgent
from configs.api_key import LLM_ENDPOINT, LLM_API_KEY


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "semlabels", "gray")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "masks_cleaned_remap")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IGNORE_ID = 255
MIN_REGION_PIXELS = 10   # threshold for a valid semantic region
MAX_VALID_ID = 200       # currently unused

# Initialize LLM Agent
llm_agent = LLMAgent(endpoint=LLM_ENDPOINT, api_key=LLM_API_KEY)


def clean_mask(mask):
    cleaned = mask.copy()
    unique_ids = np.unique(mask)
    for sid in unique_ids:
        if sid == IGNORE_ID:
            continue
        if sid < 0 or sid > MAX_VALID_ID:
            cleaned[mask == sid] = IGNORE_ID
            continue
        region_pixels = np.sum(mask == sid)
        if region_pixels < MIN_REGION_PIXELS:
            cleaned[mask == sid] = IGNORE_ID
    return cleaned


# =========================
# Remap IDs to consecutive [0, num_classes-1]
# =========================
def remap_ids(mask):
    unique_ids = sorted([i for i in np.unique(mask) if i != IGNORE_ID])
    id_map = {old: new for new, old in enumerate(unique_ids)}
    remapped = np.full_like(mask, IGNORE_ID)
    for old, new in id_map.items():
        remapped[mask == old] = new
    return remapped, len(unique_ids)


# =========================
# LLM Agent mask cleaning
# =========================
def llm_clean_mask(image_path, mask_path):
    result = llm_agent.judge(image_path, mask_path)
    if result["decision"] == "discard":
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask[:, :] = IGNORE_ID
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    return mask, result


def main(use_llm=False):
    files = sorted(os.listdir(INPUT_DIR))
    total_classes = 0

    for file in tqdm(files, desc="Cleaning & remapping masks"):
        in_path = os.path.join(INPUT_DIR, file)
        out_path = os.path.join(OUTPUT_DIR, file)
        mask = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] Failed to read {file}")
            continue

        cleaned = clean_mask(mask)
        llm_result = None
        if use_llm:
            cleaned, llm_result = llm_clean_mask(
                image_path=in_path.replace("mask", "image"),  # assumes corresponding image
                mask_path=in_path
            )
            # Optionally save LLM result as JSON
            json_path = os.path.join(OUTPUT_DIR, file.replace(".png", "_llm.json"))
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(llm_result, f, ensure_ascii=False, indent=2)

        remapped, n_classes = remap_ids(cleaned)
        cv2.imwrite(out_path, remapped)
        total_classes = max(total_classes, n_classes)

    print(f"\n[OK] Cleaned & remapped masks saved to {OUTPUT_DIR}, total_classes={total_classes}")


if __name__ == "__main__":
    main(use_llm=False)
