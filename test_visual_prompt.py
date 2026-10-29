
import torch
import clip

from visual_prompt import VisualPromptCLIP


# 1. Configuration
print("\nVisual prompt architecture")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = VisualPromptCLIP(
    model_name="ViT-B/16",
    num_prompts=5,
    device=DEVICE
)

print("Device:", DEVICE)


# 2. Prepare synthetic images

images = torch.randn(
    2, 3, 224, 224,
    device=DEVICE
)

# 3. Prepare example text prompts

texts = [
    "a photo of Shikra (Accipiter badius)",
    "a photo of Cooper's Hawk (Accipiter cooperii)"
]

tokens = clip.tokenize(texts).to(DEVICE)

with torch.no_grad():
    text_features = model.clip_model.encode_text(tokens).float()
    text_features = text_features / text_features.norm(
        dim=-1, keepdim=True
    )


# 4. Forward pass

similarities = model(images, text_features)

print("Similarity shape:", similarities.shape)
print("Prompt shape:", model.prompts.shape)


# 5. Verify trainable parameters

trainable = [
    name for name, parameter in model.named_parameters()
    if parameter.requires_grad
]

print("Trainable parameters:", trainable)

assert trainable == ["prompts"], (
    "Unexpected trainable parameters detected!"
)


# 6. Verify gradient flow

loss = -similarities[:, 0].mean()
loss.backward()

print("Prompt gradient exists:", model.prompts.grad is not None)

assert model.prompts.grad is not None
assert torch.isfinite(model.prompts.grad).all()


# 7. Verify optimizer updates only visual prompts

optimizer = torch.optim.AdamW(
    [model.prompts],
    lr=1e-3
)

# Save parameters before the update
prompts_before = model.prompts.detach().clone()
clip_before = model.clip_model.visual.conv1.weight.detach().clone()

# Perform one optimizer step using a fresh forward pass
optimizer.zero_grad()

similarities = model(images, text_features)
loss = -similarities[:, 0].mean()

loss.backward()
optimizer.step()

# Compare parameters after the update
prompt_change = (
    model.prompts.detach() - prompts_before
).abs().max().item()

clip_change = (
    model.clip_model.visual.conv1.weight.detach() - clip_before
).abs().max().item()

print("\nOPTIMIZER TEST")
print("Maximum prompt change:", prompt_change)
print("Maximum CLIP weight change:", clip_change)

assert prompt_change > 0, "Visual prompts did not update!"
assert clip_change == 0, "Frozen CLIP weights changed!"

