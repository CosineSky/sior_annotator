import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class SIORSemanticDataset(Dataset):
    def __init__(self, image_dir, mask_dir, split_file):
        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f]

        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img_path = os.path.join(self.image_dir, img_id + ".jpg")
        mask_path = os.path.join(self.mask_dir, img_id + ".png")

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise FileNotFoundError(
                f"[Dataset Error] Missing file: {img_path} or {mask_path}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()

        return image, mask

