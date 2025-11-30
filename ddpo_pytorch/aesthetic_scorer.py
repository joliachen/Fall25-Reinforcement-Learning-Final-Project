# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py

from importlib import resources
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

ASSETS_PATH = resources.files("ddpo_pytorch.assets")


class MLP(nn.Module):
    """Small MLP head for aesthetic score regression.

    This network maps CLIP image embeddings (dimension 768 for
    `openai/clip-vit-large-patch14`) to a single scalar aesthetic score using a
    few fully connected layers with dropout regularization.
    """
    def __init__(self):
        """Initialize the MLP architecture."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        """Forward pass through the aesthetic MLP.

        Args:
            embed: CLIP image embeddings of shape `(batch_size, 768)`.

        Returns:
            A tensor of shape `(batch_size, 1)` containing predicted aesthetic
            scores for each embedding.
        """
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    """Aesthetic score predictor using CLIP embeddings and a trained MLP head.

    This module wraps:
    * A CLIP image encoder (`openai/clip-vit-large-patch14`) to produce image
      embeddings.
    * A small MLP head trained on the SAC+LOGOS+AVA aesthetic dataset to
      predict scalar aesthetic scores from normalized CLIP embeddings.

    It exposes a callable interface that takes a batch of images and returns a
    1D tensor of predicted aesthetic scores.
    """
    def __init__(self, dtype):
        """Initialize the CLIP backbone and load the aesthetic MLP weights.

        Args:
            dtype: The floating-point dtype to use for CLIP inputs and
                embeddings (e.g., `torch.float16` or `torch.float32`).
        """
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        state_dict = torch.load(
            ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth")
        )
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        """Compute aesthetic scores for a batch of images.

        Args:
            images: A single image or a sequence of images. Each element should
                be compatible with `CLIPProcessor`, typically:
                * a `PIL.Image.Image`, or
                * a NumPy array / PyTorch tensor in HWC layout.

        Returns:
            `torch.FloatTensor` of shape `(batch_size,)` containing the
            predicted aesthetic score for each input image.
        """
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)
