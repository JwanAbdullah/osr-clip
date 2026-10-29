
import csv
import json
from pathlib import Path
from collections import Counter

# 1. Configuration

DATASET_ROOT = Path(r"D:\datasets\inat2021")
ANNOTATIONS_PATH = DATASET_ROOT / "train_mini.json"

SPLIT_PATH = Path(
    "benchmark/datasets/inat21_osr_splits/"
    "inat21-osr-aves-id-1hop.json"
)

OUTPUT_PATH = Path("outputs")
OUTPUT_PATH.mkdir(exist_ok=True)


# 2. Load annotations and official known classes

with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
    annotations = json.load(f)

with open(SPLIT_PATH, encoding="utf-8") as f:
    split = json.load(f)

known_ids = set(split["train_categories"])


# 3. Map image IDs to species IDs

image_categories = {
    annotation["image_id"]: annotation["category_id"]
    for annotation in annotations["annotations"]
}


# 4. Collect known-species training images

training_records = []

for image_info in annotations["images"]:
    image_id = image_info["id"]
    category_id = image_categories.get(image_id)

    if category_id not in known_ids:
        continue

    image_path = DATASET_ROOT / image_info["file_name"]

    training_records.append({
        "image_path": str(image_path),
        "category_id": category_id
    })


# 5. Save training manifest

filename = OUTPUT_PATH / "aves_vpt_train.csv"

with open(filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["image_path", "category_id"]
    )
    writer.writeheader()
    writer.writerows(training_records)


# 6. Verify training distribution

counts = Counter(
    record["category_id"]
    for record in training_records
)

print("Known species:", len(counts))
print("Training images:", len(training_records))
print("Minimum images per species:", min(counts.values()))
print("Maximum images per species:", max(counts.values()))
print("Saved:", filename)
