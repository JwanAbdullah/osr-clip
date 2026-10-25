
import json
import csv
from pathlib import Path

import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# 1. Configuration

ANNOTATIONS_PATH = Path(r"D:\datasets\inat2021\val.json")

HOP = 7 #keep changing
PROMPT_TYPE = "combined"
EVALUATION_STAGE = "test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

SPLIT_PATH = Path(
    f"benchmark/datasets/inat21_osr_splits/"
    f"inat21-osr-aves-id-{HOP}hop.json"
)

EVALUATION_PATH = Path(
    f"outputs/aves_{HOP}hop_test.csv"
)

OUTPUT_PATH = Path("outputs")
OUTPUT_PATH.mkdir(exist_ok=True)


# 2. Load annotations and official split

with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
    annotations = json.load(f)

with open(SPLIT_PATH, encoding="utf-8") as f:
    split = json.load(f)

known_ids = set(split["train_categories"])
unknown_ids = set(split["test_categories"])

print(f"Evaluating Aves {HOP}-hop")
print("Known species:", len(known_ids))
print("Unknown species:", len(unknown_ids))


# 3. Load CLIP and prepare known-species prompts

model, preprocess = clip.load("ViT-B/16", device=DEVICE)
model.eval()

for parameter in model.parameters():
    parameter.requires_grad = False

category_names = {
    category["id"]: category["name"]
    for category in annotations["categories"]
}

common_names = {
    category["id"]: category.get("common_name") or category["name"]
    for category in annotations["categories"]
}

known_ids_ordered = sorted(known_ids)

if PROMPT_TYPE == "scientific":
    prompts = [
        f"a photo of {category_names[category_id]}"
        for category_id in known_ids_ordered
    ]

elif PROMPT_TYPE == "common":
    prompts = [
        f"a photo of {common_names[category_id]}"
        for category_id in known_ids_ordered
    ]

elif PROMPT_TYPE == "combined":
    prompts = [
        f"a photo of {common_names[category_id]} "
        f"({category_names[category_id]})"
        for category_id in known_ids_ordered
    ]

else:
    raise ValueError(
        "PROMPT_TYPE must be scientific, common, or combined"
    )

print("Prompt type:", PROMPT_TYPE)
print("Example prompt:", prompts[0])

with torch.inference_mode():
    tokens = clip.tokenize(prompts).to(DEVICE)
    text_features = model.encode_text(tokens).float()
    text_features = text_features / text_features.norm(
        dim=-1, keepdim=True
    )

print("Text embeddings:", text_features.shape)


# 4. Prepare images for batch processing

class BirdDataset(Dataset):
    def __init__(self, image_records, preprocess):
        self.image_records = image_records
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_records)

    def __getitem__(self, index):
        image_path, category_id = self.image_records[index]

        with Image.open(image_path) as image:
            image_tensor = self.preprocess(image.convert("RGB"))

        return image_tensor, category_id, str(image_path)


# 5. Run CLIP on image batches

def evaluate_images(loader, label):
    results = []

    with torch.inference_mode():
        for batch_number, (images, category_ids, paths) in enumerate(loader):
            images = images.to(DEVICE)

            image_features = model.encode_image(images).float()
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

            similarities = image_features @ text_features.T
            max_scores, predicted_indices = similarities.max(dim=1)

            for i in range(len(paths)):
                results.append({
                    "image_path": paths[i],
                    "actual_category": category_ids[i].item(),
                    "ground_truth": label,
                    "predicted_category": known_ids_ordered[
                        predicted_indices[i].item()
                    ],
                    "max_similarity": max_scores[i].item()
                })

            if (batch_number + 1) % 20 == 0:
                print(
                    f"{label}: processed "
                    f"{(batch_number + 1) * BATCH_SIZE} images"
                )

    return results


# 6. Load held-out test images

test_known = []
test_unknown = []

with open(EVALUATION_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        record = (
            Path(row["image_path"]),
            int(row["category_id"])
        )

        if row["ground_truth"] == "known":
            test_known.append(record)

        elif row["ground_truth"] == "unknown":
            test_unknown.append(record)

print("\nKnown test images:", len(test_known))
print("Unknown test images:", len(test_unknown))

assert all(
    category_id in known_ids
    for _, category_id in test_known
)

assert all(
    category_id in unknown_ids
    for _, category_id in test_unknown
)

known_test_loader = DataLoader(
    BirdDataset(test_known, preprocess),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

unknown_test_loader = DataLoader(
    BirdDataset(test_unknown, preprocess),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# 7. Evaluate known and unknown images

print("\nEvaluating known test images...")
known_results = evaluate_images(known_test_loader, "known")

print("\nEvaluating unknown test images...")
unknown_results = evaluate_images(unknown_test_loader, "unknown")


# 8. Save held-out test results

all_results = known_results + unknown_results

filename = f"aves_{HOP}hop_{PROMPT_TYPE}_{EVALUATION_STAGE}_results.csv"

with open(OUTPUT_PATH / filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)

print("\nEvaluation complete!")
print("Known images evaluated:", len(known_results))
print("Unknown images evaluated:", len(unknown_results))
print("Saved:", OUTPUT_PATH / filename)
