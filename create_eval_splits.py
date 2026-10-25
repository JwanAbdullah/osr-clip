
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
DATA_ROOT = Path(r"D:\datasets\inat2021")
SPLIT_PATH = Path(
    "benchmark/datasets/inat21_osr_splits/"
    "inat21-osr-aves-id-1hop.json"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

with open(DATA_ROOT / "val.json", encoding="utf-8") as f:
    annotations = json.load(f)

with open(SPLIT_PATH, encoding="utf-8") as f:
    split = json.load(f)

known_ids = set(split["train_categories"])
unknown_ids = set(split["test_categories"])

# Previously inspected pilot images must not enter final testing
pilot_paths = set()

for filename in [
    "aves_1hop_pilot.csv",
    "aves_1hop_common_names_pilot.csv"
]:
    path = OUTPUT_DIR / filename

    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            pilot_paths.update(
                row["image_path"]
                for row in csv.DictReader(f)
            )

# Map images to category IDs
image_categories = {
    a["image_id"]: a["category_id"]
    for a in annotations["annotations"]
}

images_by_species = defaultdict(list)

for image in annotations["images"]:
    category_id = image_categories.get(image["id"])

    if category_id not in known_ids | unknown_ids:
        continue

    image_path = str(DATA_ROOT / image["file_name"])

    images_by_species[category_id].append(image_path)

rng = random.Random(SEED)

development = []
testing = []

for category_id, paths in sorted(images_by_species.items()):
    paths = sorted(paths)
    rng.shuffle(paths)

    # Put any previously inspected images in development
    inspected = [p for p in paths if p in pilot_paths]
    uninspected = [p for p in paths if p not in pilot_paths]

    # Reserve two development images where possible,
    # plus all previously inspected pilot images
    development_paths = inspected + uninspected[:max(0, 2 - len(inspected))]
    test_paths = uninspected[max(0, 2 - len(inspected)):]

    status = "known" if category_id in known_ids else "unknown"

    for p in development_paths:
        development.append((p, category_id, status))

    for p in test_paths:
        testing.append((p, category_id, status))

def save_csv(filename, records):
    with open(OUTPUT_DIR / filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "category_id", "ground_truth"])
        writer.writerows(records)

save_csv("aves_1hop_development.csv", development)
save_csv("aves_1hop_test.csv", testing)

dev_paths = {r[0] for r in development}
test_paths = {r[0] for r in testing}

assert dev_paths.isdisjoint(test_paths)
assert not (pilot_paths & test_paths)

print("Development images:", len(development))
print("Held-out test images:", len(testing))
print("Overlap:", len(dev_paths & test_paths))
print("Previously inspected images in test:", len(pilot_paths & test_paths))
print("Random seed:", SEED)
