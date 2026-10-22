
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

model, preprocess = clip.load("ViT-B/16", device=device)
model.eval()

print("Model: ViT-B/16")
print("CLIP loaded successfully!")
print("Model device:", next(model.parameters()).device)
