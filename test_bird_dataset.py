
import json
from pathlib import Path

from bird_dataset import BirdDataset


# 1. Load official known classes

SPLIT_PATH = Path(
    "benchmark/datasets/inat21_osr_splits/"
    "inat21-osr-aves-id-1hop.json"
)

with open(SPLIT_PATH, encoding="utf-8") as f:
    split = json.load(f)

known_ids = set(split["train_categories"])


# 2. Load training and validation manifests

train_dataset = BirdDataset(
    csv_path="outputs/aves_vpt_train_split.csv",
    preprocess=None,
    known_ids=known_ids
)

val_dataset = BirdDataset(
    csv_path="outputs/aves_vpt_val_split.csv",
    preprocess=None,
    known_ids=known_ids
)


# 3. Verify metadata

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Known classes:", len(train_dataset.label_map))

assert len(train_dataset) == 29800
assert len(val_dataset) == 7450
assert len(train_dataset.label_map) == 745

assert train_dataset.label_map == val_dataset.label_map

# Confirm training labels are consecutive
assert set(train_dataset.label_map.values()) == set(range(745))

print("\nBird dataset metadata test passed!")
print("Actual photographs not loaded yet.")
