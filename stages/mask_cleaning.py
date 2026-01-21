import os
import cv2
import csv
import json
import numpy as np
from tqdm import tqdm
import argparse

from agents.llm_agent import LLMAgent
from agents.mock_agent import MockLLMAgent
from configs.semantic_map import SEMANTIC_MAP


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

IMAGE_DIR = os.path.join(PROJECT_ROOT, "data/trainval_images")
RAW_MASK_DIR = os.path.join(PROJECT_ROOT, "output/masks_refined")

RULE_PASS_DIR = os.path.join(PROJECT_ROOT, "output/masks_rule_pass")
FINAL_MASK_DIR = os.path.join(PROJECT_ROOT, "output/masks_final")
FINAL_JSON_DIR = os.path.join(PROJECT_ROOT, "output/annotations_json")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(RULE_PASS_DIR, exist_ok=True)
os.makedirs(FINAL_MASK_DIR, exist_ok=True)
os.makedirs(FINAL_JSON_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

RULE_LOG = os.path.join(LOG_DIR, "removed_rule.txt")
AGENT_LOG = os.path.join(LOG_DIR, "removed_agent.txt")
SCORE_LOG = os.path.join(LOG_DIR, "agent_scores.csv")


# =========================
# 1. Rule params
# =========================
MIN_RATIO = 0.001
MAX_RATIO = 0.95
MIN_COMPONENT_AREA = 300


# =========================
# 2. Rule method
# =========================
def rule_filter(mask):
    ratio = mask.mean()
    if ratio < MIN_RATIO or ratio > MAX_RATIO:
        return False, "bad_ratio"

    num, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    valid = sum(
        stats[i, cv2.CC_STAT_AREA] > MIN_COMPONENT_AREA
        for i in range(1, num)
    )

    if valid == 0:
        return False, "no_valid_component"

    return True, "ok"


# =========================
# 3. Main process
# =========================
def run_pipeline(mode="rule_only"):
    """
    mode: "rule_only" | "mock_agent" | "llm_agent"
    """
    if mode not in ["rule_only", "mock_agent", "llm_agent"]:
        raise ValueError(f"Invalid mode: {mode}")

    agent = None
    if mode == "mock_agent":
        agent = MockLLMAgent()
    elif mode == "llm_agent":
        agent = LLMAgent()

    removed_rule = []
    removed_agent = []
    score_records = []

    for name in tqdm(sorted(os.listdir(RAW_MASK_DIR))):
        img_path = os.path.join(IMAGE_DIR, name)
        mask_path = os.path.join(RAW_MASK_DIR, name)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask > 0).astype(np.uint8)

        # -------- Rule Stage --------
        ok, reason = rule_filter(mask)
        if not ok:
            removed_rule.append(f"{name} {reason}")
            continue

        cv2.imwrite(os.path.join(RULE_PASS_DIR, name), mask * 255)

        # -------- Agent Stage --------
        if mode != "rule_only":
            result = agent.judge(image, mask)

            score_records.append([
                name,
                result.get("decision", ""),
                result.get("semantic", ""),
                result.get("confidence", 0),
                result.get("reason", "")
            ])

            if result.get("decision") == "discard":
                removed_agent.append(f"{name} {result['reason']}")
                continue

            semantic = result.get("semantic", "")
            class_id = SEMANTIC_MAP.get(semantic, 0)

            # Save JSON Annotation
            ann = {
                "image": name,
                "semantic": semantic,
                "class_id": class_id,
                "confidence": result.get("confidence", 0)
            }
            json_path = os.path.join(
                FINAL_JSON_DIR,
                name.replace(".png", ".json")
            )
            with open(json_path, "w") as f:
                json.dump(ann, f, indent=2)

        # -------- Save Final Mask --------
        cv2.imwrite(os.path.join(FINAL_MASK_DIR, name), mask * 255)

    # -------- Logs --------
    with open(RULE_LOG, "w") as f:
        f.write("\n".join(removed_rule))

    with open(AGENT_LOG, "w") as f:
        f.write("\n".join(removed_agent))

    with open(SCORE_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        if mode != "rule_only":
            writer.writerow(["image", "decision", "semantic", "confidence", "reason"])
            writer.writerows(score_records)
        else:
            writer.writerow(["image", "reason"])
            for r in removed_rule:
                writer.writerow([r.split()[0], r.split()[1]])


if __name__ == "__main__":
    run_pipeline(mode="mock_agent")
