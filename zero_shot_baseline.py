
import json
import torch
import clip

# 1. Configuration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/16"

ANNOTATIONS_PATH = r"D:\datasets\inat2021\val.json"
SPLIT_PATH = (
    "benchmark/datasets/inat21_osr_splits/"
    "inat21-osr-aves-id-1hop.json"
)

# 2. Load dataset metadata

with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
    annotations = json.load(f)

with open(SPLIT_PATH, encoding="utf-8") as f:
    split = json.load(f)

category_names = {
    category["id"]: category["name"]
    for category in annotations["categories"]
}

known_ids = split["train_categories"]
unknown_ids = split["test_categories"]

print("Known species:", len(known_ids))
print("Unknown species:", len(unknown_ids))

# 3. Create zero-shot text prompts

prompts = [
    f"a photo of {category_names[category_id]}"
    for category_id in known_ids
]

print("First 5 prompts:")
for prompt in prompts[:5]:
    print(prompt)

# 4. Load frozen CLIP

model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
model.eval()

for parameter in model.parameters():
    parameter.requires_grad = False

# 5. Encode text prompts

tokens = clip.tokenize(prompts).to(DEVICE)

with torch.inference_mode():
    text_features = model.encode_text(tokens)
    text_features = text_features / text_features.norm(
        dim=-1, keepdim=True
    )

print("Text embeddings shape:", text_features.shape)
print("Device:", DEVICE)
print("Text encoding completed!")


# 6. Select a species to test

from pathlib import Path
from PIL import Image

# CHANGE to test another species
SPECIES_ID = 3112  # Accipiter cooperii (UNKNOWN)
# SPECIES_ID = 3111  # Shikra (KNOWN)

DATASET_PATH = Path(r"D:\datasets\inat2021\val")

# Find the species folder
folders = list(DATASET_PATH.glob(f"{SPECIES_ID:05d}_*"))

if not folders:
    raise FileNotFoundError(
        f"No folder found for species {SPECIES_ID}"
    )

# Select the first image
image_path = next(folders[0].glob("*.jpg"))

# 7. Check known / unknown status

if SPECIES_ID in known_ids:
    status = "KNOWN"
elif SPECIES_ID in unknown_ids:
    status = "UNKNOWN (1-hop)"
else:
    status = "OUTSIDE THIS SPLIT"

print("\nActual species:", category_names[SPECIES_ID])
print("Ground-truth status:", status)
print("Image:", image_path.name)

# 8. Run zero-shot CLIP

image = Image.open(image_path).convert("RGB")
image_input = preprocess(image).unsqueeze(0).to(DEVICE)

with torch.inference_mode():
    image_features = model.encode_image(image_input)
    image_features = image_features / image_features.norm(
        dim=-1, keepdim=True
    )

    similarities = (
        image_features @ text_features.T
    ).squeeze(0)

# 9. Display predictions

top_scores, top_indices = similarities.topk(5)

print("\nTop 5 CLIP predictions:")

for score, index in zip(top_scores, top_indices):
    category_id = known_ids[index.item()]
    species_name = category_names[category_id]

    print(f"{species_name}: {score.item():.4f}")

print("\nHighest similarity:", top_scores[0].item())
