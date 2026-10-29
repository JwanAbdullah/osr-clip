
import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class BirdDataset(Dataset):
    def __init__(self, csv_path, preprocess, known_ids):
        self.preprocess = preprocess

        # Keep class labels consistent with CLIP text prompts
        self.known_ids = sorted(known_ids)

        self.label_map = {
            category_id: index
            for index, category_id in enumerate(self.known_ids)
        }

        # Read image paths and species IDs
        with open(csv_path, newline="", encoding="utf-8") as f:
            self.records = list(csv.DictReader(f))

        # Verify all species belong to the known-class set
        for row in self.records:
            category_id = int(row["category_id"])

            if category_id not in self.label_map:
                raise ValueError(
                    f"Unknown category in training data: {category_id}"
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]

        image_path = Path(row["image_path"])
        category_id = int(row["category_id"])

        # Open the actual photograph
        with Image.open(image_path) as image:
            image_tensor = self.preprocess(image.convert("RGB"))

        # Convert original species ID to training label
        label = self.label_map[category_id]

        return image_tensor, label
