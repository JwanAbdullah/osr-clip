
import json
from pathlib import Path

# 1. Configuration
DATASET_PATH = Path(r"D:\datasets\inat2021\val")

ANNOTATIONS_PATH = Path(
    r"D:\datasets\inat2021\val.json"
)

SPLIT_PATH = Path(
    "benchmark/datasets/inat21_osr_splits/"
    "inat21-osr-aves-id-1hop.json"
)

DEVELOPMENT_PATH = Path("outputs/aves_1hop_development.csv")
PROMPT_TYPE = "combined"  # Change to "common" for second run

# 2. Load annotations and official split
with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
    annotations = json.load(f)

with open(SPLIT_PATH, encoding="utf-8") as f:
    split = json.load(f)

known_ids = set(split["train_categories"])
unknown_ids = set(split["test_categories"])

# 3. Map image records to species IDs
image_categories = {
    annotation["image_id"]: annotation["category_id"]
    for annotation in annotations["annotations"]
}

# 4. Collect known and unknown images
known_images = []
unknown_images = []

for image_info in annotations["images"]:
    image_id = image_info["id"]
    category_id = image_categories.get(image_id)

    if category_id is None:
        continue

    image_path = DATASET_PATH.parent / image_info["file_name"]

    if category_id in known_ids:
        known_images.append((image_path, category_id))

    elif category_id in unknown_ids:
        unknown_images.append((image_path, category_id))

# 5. Display dataset summary
print("Known species:", len(known_ids))
print("Unknown species:", len(unknown_ids))

print("Known images:", len(known_images))
print("Unknown images:", len(unknown_images))

print("\nFirst known image:", known_images[0] if known_images else "None")
print("First unknown image:", unknown_images[0] if unknown_images else "None")

print("\nFirst known image exists:",
      known_images[0][0].exists() if known_images else False)

print("First unknown image exists:",
      unknown_images[0][0].exists() if unknown_images else False)


# 6. load CLIP and prepare known-species prompts

import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

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


# 7. Prepare images for batch processing

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


known_loader = DataLoader(
    BirdDataset(known_images, preprocess),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

unknown_loader = DataLoader(
    BirdDataset(unknown_images, preprocess),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# 8. Run CLIP on image batches

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
                print(f"{label}: processed {(batch_number + 1) * BATCH_SIZE} images")

    return results



# 9. Load development images

import csv

development_known = []
development_unknown = []

with open(DEVELOPMENT_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        record = (
            Path(row["image_path"]),
            int(row["category_id"])
        )

        if row["ground_truth"] == "known":
            development_known.append(record)

        elif row["ground_truth"] == "unknown":
            development_unknown.append(record)

print("\nDevelopment known images:", len(development_known))
print("Development unknown images:", len(development_unknown))

known_dev_loader = DataLoader(
    BirdDataset(development_known, preprocess),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

unknown_dev_loader = DataLoader(
    BirdDataset(development_unknown, preprocess),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nEvaluating known development images...")
known_results = evaluate_images(known_dev_loader, "known")

print("\nEvaluating unknown development images...")
unknown_results = evaluate_images(unknown_dev_loader, "unknown")

# 10. Save development results

OUTPUT_PATH = Path("outputs")
OUTPUT_PATH.mkdir(exist_ok=True)

all_results = known_results + unknown_results

filename = f"aves_1hop_{PROMPT_TYPE}_development_results.csv"

with open(OUTPUT_PATH / filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)

print("\nEvaluation complete!")
print("Known images evaluated:", len(known_results))
print("Unknown images evaluated:", len(unknown_results))
print("Saved:", OUTPUT_PATH / filename)