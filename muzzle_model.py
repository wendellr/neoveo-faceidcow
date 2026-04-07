"""
muzzle_model.py — Definição do MuzzleEmbedder compartilhado entre
03_train_reid.py, 04_inference.py e 05_enroll.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MuzzleEmbedder(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,          # remove classification head
        )
        feat_dim = self.backbone.num_features   # 1280 para EfficientNet-B0
        self.head = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.head(features)
        return F.normalize(embeddings, dim=1)   # L2-normalized
