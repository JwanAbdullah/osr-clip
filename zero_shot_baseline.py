
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
