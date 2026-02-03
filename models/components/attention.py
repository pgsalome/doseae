"""Dimensionality-aware attention and spatial embedding utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialEmbeddingND(nn.Module):
    """Linear embedding for spatial coordinates (2D or 3D)."""

    def __init__(self, coord_dim: int, embed_dim: int, learnable_bias: bool = True):
        super().__init__()
        self.embedding = nn.Linear(coord_dim, embed_dim)
        if learnable_bias:
            self.bias = nn.Parameter(torch.zeros(1, embed_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(coords)
        if self.bias is not None:
            emb = emb + self.bias
        return emb


class HighDoseAttention(nn.Module):
    """Attention module biased toward high-dose regions."""

    def __init__(self, input_dim: int, dose_threshold: float = 0.5):
        super().__init__()
        self.threshold = dose_threshold
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, features: torch.Tensor, dose_map: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query(features)
        k = self.key(features)
        v = self.value(features)

        scores = torch.matmul(q, k.transpose(-2, -1)) / q.size(-1) ** 0.5
        if dose_map is not None:
            scores = scores + (dose_map > self.threshold).float()
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)
        return attended, weights


class MultiScaleSelfAttention(nn.Module):
    """Self-attention with multi-scale feature enrichment."""

    def __init__(self, feature_dim: int, num_heads: int = 4, dim: int = 3):
        super().__init__()
        self.num_heads = max(1, num_heads)
        self.head_dim = max(1, feature_dim // self.num_heads)
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        padding = 1
        self.scale_conv1 = conv(feature_dim, feature_dim, kernel_size=1)
        self.scale_conv2 = conv(feature_dim, feature_dim, kernel_size=3, padding=padding)
        self.scale_conv3 = conv(feature_dim, feature_dim, kernel_size=5, padding=2)
        self.dim = dim

    def forward(self, x: torch.Tensor, _: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, c = x.shape
        reshape_dim = (b * n, c, 1, 1, 1) if self.dim == 3 else (b * n, c, 1, 1)
        x_spatial = x.view(*reshape_dim)
        multi_scale = (self.scale_conv1(x_spatial) + self.scale_conv2(x_spatial) + self.scale_conv3(x_spatial)) / 3
        x_enhanced = multi_scale.view(b, n, c)

        q = self.q_proj(x_enhanced).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_enhanced).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_enhanced).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(b, n, c)
        output = self.out_proj(attended)
        return output, weights.mean(dim=1)


class ChannelAttentionND(nn.Module):
    """Squeeze-and-excitation style channel attention for 2D or 3D tensors."""

    def __init__(self, channels: int, reduction: int = 16, dim: int = 3):
        super().__init__()
        reduction = max(1, reduction)
        pool = nn.AdaptiveAvgPool3d if dim == 3 else nn.AdaptiveAvgPool2d
        self.pool = pool(1)

        hidden_dim = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels),
            nn.Sigmoid()
        )
        self.dim = dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels = x.shape[:2]
        pooled = self.pool(x).view(batch_size, channels)
        weights = self.mlp(pooled).view(batch_size, channels, *([1] * (x.dim() - 2)))
        attended = x * weights
        return attended, weights.squeeze()


class ContextGatedAttention(nn.Module):
    """Gates latent features using contextual embeddings."""

    def __init__(
        self,
        context_dim: int,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(feature_dim, context_dim)
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if context.dim() > 2:
            context = context.view(context.size(0), -1)
        weights = torch.sigmoid(self.net(context))
        gated = features * weights
        return gated, weights


class SpatialContextAttention(nn.Module):
    """Attention that conditions latent features on spatial coordinates."""

    def __init__(
        self,
        coord_dim: int,
        embed_dim: int,
        feature_dim: int,
        dropout: float = 0.0,
        learnable_bias: bool = True,
    ):
        super().__init__()
        self.embedding = SpatialEmbeddingND(coord_dim, embed_dim, learnable_bias=learnable_bias)
        self.attention = ContextGatedAttention(embed_dim, feature_dim, dropout=dropout)

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = coords.float()
        if coords.dim() > 2:
            coords = coords.view(coords.size(0), -1)
        embedding = self.embedding(coords)
        return self.attention(features, embedding)


class LobeContextAttention(nn.Module):
    """Attention mechanism that leverages lobe and side embeddings."""

    def __init__(
        self,
        feature_dim: int,
        *,
        num_lobes: int,
        num_side_types: int,
        num_anatomical_sides: int,
        lobe_embed_dim: int = 16,
        side_embed_dim: int = 8,
        anatomical_embed_dim: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lobe_embedding = nn.Embedding(num_lobes, lobe_embed_dim, padding_idx=0)
        self.side_embedding = nn.Embedding(num_side_types, side_embed_dim, padding_idx=0) if num_side_types > 0 else None
        self.anatomical_embedding = (
            nn.Embedding(num_anatomical_sides, anatomical_embed_dim, padding_idx=0)
            if num_anatomical_sides > 0 else None
        )

        context_dim = lobe_embed_dim
        if self.side_embedding is not None:
            context_dim += side_embed_dim
        if self.anatomical_embedding is not None:
            context_dim += anatomical_embed_dim

        self.attention = ContextGatedAttention(context_dim, feature_dim, dropout=dropout)

    def forward(
        self,
        features: torch.Tensor,
        lobe_index: torch.Tensor,
        side_type_index: Optional[torch.Tensor] = None,
        anatomical_side_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lobe_index = lobe_index.long().clamp(min=0, max=self.lobe_embedding.num_embeddings - 1)
        embeddings = [self.lobe_embedding(lobe_index)]

        if self.side_embedding is not None and side_type_index is not None:
            side_idx = side_type_index.long().clamp(min=0, max=self.side_embedding.num_embeddings - 1)
            embeddings.append(self.side_embedding(side_idx))

        if self.anatomical_embedding is not None and anatomical_side_index is not None:
            anatomical_idx = anatomical_side_index.long().clamp(min=0, max=self.anatomical_embedding.num_embeddings - 1)
            embeddings.append(self.anatomical_embedding(anatomical_idx))

        context = torch.cat(embeddings, dim=-1)
        return self.attention(features, context)
