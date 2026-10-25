
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

# 1. Configuration

SEED = 42
DATA_ROOT = Path(r"D:\datasets\inat2021")
SPLIT_ROOT = Path("benchmark/datasets/inat21_osr_splits")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

# 2. Load annotations

with open(DATA_ROOT / "val.json", encoding="utf-8") as f:
    annotations = json.load(f)

image_categories = {
    a["image_id"]: a["category_id"]
    for a in annotations["annotations"]
}

# 3. Load all seven official splits

splits = {}

for hop in range(1, 8):
    path = SPLIT_ROOT / f"inat21-osr-aves-id-{hop}hop.json"

    with open(path, encoding="utf-8") as f:
        splits[hop] = json.load(f)

known_ids = set(splits[1]["train_categories"])

assert all(
    set(s["train_categories"]) == known_ids
    for s in splits.values()
)

# 4. Collect previously inspected pilot images

pilot_paths = set()

for filename in [
    "aves_1hop_pilot.csv",
    "aves_1hop_common_names_pilot.csv"
]:
    path = OUTPUT_DIR / filename

    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            pilot_paths.update(
                str(Path(row["image_path"]).resolve()).lower()
                for row in csv.DictReader(f)
            )

# Also protect images used in earlier development experiments
old_dev = OUTPUT_DIR / "aves_1hop_development.csv"

if old_dev.exists():
    with open(old_dev, newline="", encoding="utf-8") as f:
        pilot_paths.update(
            str(Path(row["image_path"]).resolve()).lower()
            for row in csv.DictReader(f)
        )

# 5. Collect images across all Aves splits

all_category_ids = set(known_ids)

for split in splits.values():
    all_category_ids.update(split["test_categories"])

images_by_species = defaultdict(list)

for image in annotations["images"]:
    category_id = image_categories.get(image["id"])

    if category_id not in all_category_ids:
        continue

    image_path = DATA_ROOT / image["file_name"]

    images_by_species[category_id].append(image_path)

# 6. Assign each image to development or test ONCE

rng = random.Random(SEED)

assignment = {}

for category_id, paths in sorted(images_by_species.items()):
    paths = sorted(paths)
    rng.shuffle(paths)

    for index, path in enumerate(paths):
        normalized = str(path.resolve()).lower()

        if normalized in pilot_paths:
            assignment[normalized] = "development"
        elif index < 2:
            assignment[normalized] = "development"
        else:
            assignment[normalized] = "test"

# 7. Create development/test CSVs for each hop

def save_csv(filename, records):
    with open(OUTPUT_DIR / filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "category_id", "ground_truth"])
        writer.writerows(records)

for hop in range(1, 8):
    split = splits[hop]

    known = set(split["train_categories"])
    unknown = set(split["test_categories"])

    development = []
    testing = []

    for category_id in sorted(known | unknown):
        status = "known" if category_id in known else "unknown"

        for path in images_by_species[category_id]:
            normalized = str(path.resolve()).lower()

            record = (str(path), category_id, status)

            if assignment[normalized] == "development":
                development.append(record)
            else:
                testing.append(record)

    save_csv(f"aves_{hop}hop_development.csv", development)
    save_csv(f"aves_{hop}hop_test.csv", testing)

    print(
        f"{hop}-hop | "
        f"Development: {len(development)} | "
        f"Test: {len(testing)}"
    )

# 8. Verify no cross-hop leakage

dev_paths = set()
test_paths = set()

for path, group in assignment.items():
    if group == "development":
        dev_paths.add(path)
    else:
        test_paths.add(path)

assert dev_paths.isdisjoint(test_paths)
assert not (pilot_paths & test_paths)

print("\nShared assignment verified!")
print("Development/test overlap:", len(dev_paths & test_paths))
print("Previously inspected images in test:", len(pilot_paths & test_paths))
print("Random seed:", SEED)
