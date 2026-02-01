import os
import cv2
import csv
import json
import numpy as np
from tqdm import tqdm

from configs.semantic_map import SEMANTIC_MAP

# =========================
# Path config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

IMAGE_DIR = os.path.join(PROJECT_ROOT, "data/trainval_images")
RAW_MASK_DIR = os.path.join(PROJECT_ROOT, "output/masks_raw")

FINAL_MASK_DIR = os.path.join(PROJECT_ROOT, "output/masks_final")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(FINAL_MASK_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SCORE_LOG = os.path.join(LOG_DIR, "semantic_quality.csv")

# =========================
# Config (semantic-level)
# =========================
MIN_FOREGROUND_RATIO = 0.01     # 至少 1% 前景
MAX_FOREGROUND_RATIO = 0.90     # 背景不能塌缩
MIN_VALID_CLASSES = 1           # 至少 1 个前景类
MAX_CLASS_FRAGMENT = 0.6        # 单类最大占比（防止一类吞全图）
IGNORE_LABEL = 255              # 可选

# =========================
# Semantic quality score
# =========================
def compute_semantic_quality(mask: np.ndarray):
    """
    mask: HxW, int semantic id
    """
    h, w = mask.shape
    img_area = h * w

    valid_mask = mask != IGNORE_LABEL
    if valid_mask.sum() == 0:
        return 0.0, {"reason": "all_ignore"}

    unique, counts = np.unique(mask[valid_mask], return_counts=True)
    stat = dict(zip(unique.tolist(), counts.tolist()))

    fg_pixels = sum(
        c for k, c in stat.items()
        if k != 0 and k != IGNORE_LABEL
    )

    fg_ratio = fg_pixels / img_area

    if fg_ratio < MIN_FOREGROUND_RATIO:
        return 0.0, {"reason": "too_little_foreground", "fg_ratio": fg_ratio}

    if fg_ratio > MAX_FOREGROUND_RATIO:
        return 0.0, {"reason": "background_missing", "fg_ratio": fg_ratio}

    fg_classes = [
        k for k in stat.keys()
        if k != 0 and k != IGNORE_LABEL
    ]

    if len(fg_classes) < MIN_VALID_CLASSES:
        return 0.0, {"reason": "no_valid_class"}

    max_class_ratio = max(
        stat[k] / fg_pixels for k in fg_classes
    )

    if max_class_ratio > MAX_CLASS_FRAGMENT:
        return 0.0, {
            "reason": "class_dominates",
            "max_class_ratio": max_class_ratio
        }

    # ---------- soft score ----------
    class_balance_score = 1.0 - max_class_ratio
    fg_score = min(fg_ratio / 0.2, 1.0)

    score = 0.6 * class_balance_score + 0.4 * fg_score
    score = float(np.clip(score, 0, 1))

    details = {
        "fg_ratio": fg_ratio,
        "num_classes": len(fg_classes),
        "max_class_ratio": max_class_ratio
    }

    return score, details


# =========================
# Main pipeline
# =========================
def run_cleaning():
    records = []

    for name in tqdm(sorted(os.listdir(RAW_MASK_DIR)), desc="Semantic checking"):
        mask_path = os.path.join(RAW_MASK_DIR, name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            continue

        score, details = compute_semantic_quality(mask)
        records.append((name, score, details))

    records.sort(key=lambda x: x[1], reverse=True)

    # ---- log ----
    with open(SCORE_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "quality_score",
            "fg_ratio", "num_classes", "max_class_ratio", "reason"
        ])

        for name, score, d in records:
            writer.writerow([
                name, score,
                d.get("fg_ratio"),
                d.get("num_classes"),
                d.get("max_class_ratio"),
                d.get("reason")
            ])

    # ---- save passed masks ----
    for name, score, d in tqdm(records, desc="Saving"):
        if score <= 0:
            continue

        src = os.path.join(RAW_MASK_DIR, name)
        dst = os.path.join(FINAL_MASK_DIR, name)
        cv2.imwrite(dst, cv2.imread(src, cv2.IMREAD_GRAYSCALE))


if __name__ == "__main__":
    run_cleaning()
