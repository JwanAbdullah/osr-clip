
import csv
import random
from pathlib import Path
from collections import defaultdict

SEED = 42
TRAIN_PER_CLASS = 40

OUTPUT_PATH = Path("outputs")
INPUT_FILE = OUTPUT_PATH / "aves_vpt_train.csv"

random.seed(SEED)

# 1. Group images by species

species_images = defaultdict(list)

with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        species_images[int(row["category_id"])].append(row)


# 2. Create reproducible train/validation splits

train_records = []
val_records = []

for species_id in sorted(species_images):
    images = sorted(
        species_images[species_id],
        key=lambda row: row["image_path"]
    )

    random.shuffle(images)

    train_records.extend(images[:TRAIN_PER_CLASS])
    val_records.extend(images[TRAIN_PER_CLASS:])


# 3. Save split files

for name, records in [
    ("aves_vpt_train_split.csv", train_records),
    ("aves_vpt_val_split.csv", val_records)
]:
    with open(OUTPUT_PATH / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "category_id"]
        )
        writer.writeheader()
        writer.writerows(records)


# 4. Verify splits

train_paths = {row["image_path"] for row in train_records}
val_paths = {row["image_path"] for row in val_records}

assert train_paths.isdisjoint(val_paths)
assert len(species_images) == 745

print("Known species:", len(species_images))
print("Training images:", len(train_records))
print("Validation images:", len(val_records))
print("Train/validation overlap:", len(train_paths & val_paths))
print("Random seed:", SEED)
