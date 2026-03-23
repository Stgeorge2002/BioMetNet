"""Gene feature encoder for multi-organism models.

Projects organism-agnostic gene features (EC class, subsystem, etc.) into
d_model-dimensional embeddings, then contextualizes via transformer self-attention.

No positional encoding — gene ordering is arbitrary across organisms,
so the encoder treats genes as a SET rather than a sequence.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GeneFeatureEncoder(nn.Module):
    """Encode gene functional features into contextual gene embeddings."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

    def forward(
        self,
        gene_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            gene_features: (B, n_genes, n_features) gated by presence
            padding_mask: (B, n_genes) bool, True for absent/padded genes
        Returns:
            (B, n_genes, d_model) contextualized gene embeddings
        """
        x = self.feature_proj(gene_features)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return x
