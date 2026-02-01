import json
import os
import cv2
import numpy as np
from tqdm import tqdm
from agents.llm_agent import LLMAgent
from configs.api_key import LLM_ENDPOINT, LLM_API_KEY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# replace with masks_raw if SAM is manually run
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "semlabels", "gray")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "masks_cleaned_remap")
IGNORE_ID = 255
MIN_REGION_PIXELS = 100

llm_agent = LLMAgent(endpoint=LLM_ENDPOINT, api_key=LLM_API_KEY)


def rule_clean_mask(mask):
    unique_labels = np.unique(mask)
    cleaned_mask = mask.copy()
    for label in unique_labels:
        if label == IGNORE_ID:
            continue
        region = (mask == label).astype(np.uint8)
        if cv2.countNonZero(region) < MIN_REGION_PIXELS:
            cleaned_mask[mask == label] = 0
    return cleaned_mask


def llm_clean_mask(image_path, mask_path):
    result = llm_agent.judge(image_path, mask_path)
    if result["decision"] == "discard":
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask[:, :] = 0
        return mask, result
    return cv2.imread(mask_path, cv2.IMREAD_UNCHANGED), result


def process_masks(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, mode="rule_only"):
    os.makedirs(output_dir, exist_ok=True)
    mask_files = [f for f in os.listdir(input_dir) if f.endswith((".png", ".jpg", ".tif"))]

    for mask_file in tqdm(mask_files, desc="Processing masks"):
        mask_path = os.path.join(input_dir, mask_file)
        image_path = mask_path.replace("mask", "image")
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = rule_clean_mask(mask)

        if mode=='llm_agent':
            mask, llm_result = llm_clean_mask(image_path, mask_path)
            result_path = os.path.join(output_dir, mask_file.replace(".png", "_llm.json"))
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(llm_result, f, ensure_ascii=False, indent=2)

        out_path = os.path.join(output_dir, mask_file)
        cv2.imwrite(out_path, mask)

    print("Mask cleaning finished.")


if __name__ == "__main__":
    process_masks(mode='rule_only')
