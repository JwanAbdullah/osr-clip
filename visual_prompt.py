
import torch
import torch.nn as nn
import clip


class VisualPromptCLIP(nn.Module):
    def __init__(self, model_name="ViT-B/16", num_prompts=5, device="cuda"):
        super().__init__()

        # 1. Load pretrained CLIP
        self.clip_model, self.preprocess = clip.load(
            model_name,
            device=device,
            jit=False
        )

        # 2. Freeze all pretrained CLIP parameters
        self.clip_model.eval()

        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False

        # 3. Access CLIP's visual transformer
        self.visual = self.clip_model.visual

        # ViT-B/16 uses 768-dimensional visual tokens
        prompt_dim = self.visual.conv1.out_channels

        # 4. Create trainable visual prompt tokens
        self.prompts = nn.Parameter(
            torch.empty(num_prompts, prompt_dim, device=device)
        )

        nn.init.normal_(self.prompts, std=0.02)

        self.num_prompts = num_prompts

    def encode_image(self, images):
        visual = self.visual

        # 5. Convert images into patch embeddings
        images = images.to(
            device=visual.conv1.weight.device,
            dtype=visual.conv1.weight.dtype
        )

        x = visual.conv1(images)

        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # 6. Add the original class token
        class_token = (
            visual.class_embedding.to(x.dtype)
            + torch.zeros(
                x.shape[0], 1, x.shape[-1],
                dtype=x.dtype,
                device=x.device
            )
        )

        x = torch.cat([class_token, x], dim=1)

        # 7. Add CLIP's original positional embeddings
        x = x + visual.positional_embedding.to(x.dtype)

        # 8. Insert trainable visual prompts
        prompts = self.prompts.to(x.dtype)
        prompts = prompts.unsqueeze(0).expand(x.shape[0], -1, -1)

        x = torch.cat(
            [x[:, :1], prompts, x[:, 1:]],
            dim=1
        )

        # 9. Pass tokens through frozen transformer
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)

        x = visual.transformer(x)

        x = x.permute(1, 0, 2)

        # 10. Extract image representation
        x = visual.ln_post(x[:, 0, :])

        if visual.proj is not None:
            x = x @ visual.proj

        return x

    def forward(self, images, text_features):
        # 11. Encode images with visual prompts
        image_features = self.encode_image(images).float()

        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )

        # 12. Calculate image-text similarity
        similarities = image_features @ text_features.T

        return similarities
