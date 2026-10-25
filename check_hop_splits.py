
import json
from pathlib import Path

split_folder = Path("benchmark/datasets/inat21_osr_splits")

splits = []

for hop in range(1, 8):
    path = split_folder / f"inat21-osr-aves-id-{hop}hop.json"

    with open(path, encoding="utf-8") as f:
        split = json.load(f)

    splits.append(split)

    print(
        f"{hop}-hop | "
        f"Known: {len(split['train_categories'])} | "
        f"Unknown: {len(split['test_categories'])}"
    )

first_known = set(splits[0]["train_categories"])

same_known = all(
    set(split["train_categories"]) == first_known
    for split in splits
)

print("\nSame known classes across all hops:", same_known)
