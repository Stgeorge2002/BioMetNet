"""Multi-organism classifier with universal reaction prediction.

Uses organism-agnostic gene features → GeneFeatureEncoder → cross-attention
to predict which reactions from a universal reaction set are active.

Generalizes across organisms because gene features (EC, subsystem) are
universal, unlike per-gene ID embeddings.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from biometnet.model.feature_encoder import GeneFeatureEncoder
from biometnet.model.classifier import CrossAttentionBlock


class MultiOrganismClassifier(nn.Module):
    """Cross-organism classifier: predicts active reactions from gene features.

    Architecture:
      1. GeneFeatureEncoder: gene features -> contextualized embeddings
      2. Universal reaction query embeddings (one per reaction)
      3. Stacked cross-attention: reactions attend to gene embeddings
      4. Classification head: per-reaction logits
    """

    def __init__(
        self,
        n_features: int,
        n_reactions: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 2,
        n_cross_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_reactions = n_reactions
        self.encoder = GeneFeatureEncoder(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.reaction_queries = nn.Parameter(
            torch.randn(n_reactions, d_model) * (d_model ** -0.5)
        )
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_cross_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1),
        )

    def forward(
        self,
        gene_features: torch.Tensor,
        gene_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            gene_features: (B, n_genes, n_features) gated by presence
            gene_mask: (B, n_genes) bool, True for absent/padded genes
        Returns:
            logits: (B, n_reactions)
        """
        gene_enc = self.encoder(gene_features, padding_mask=gene_mask)
        B = gene_features.size(0)
        queries = self.reaction_queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.cross_layers:
            queries = layer(queries, gene_enc, key_padding_mask=gene_mask)
        return self.classifier(queries).squeeze(-1)

    @torch.no_grad()
    def predict(
        self,
        gene_features: torch.Tensor,
        gene_mask: torch.Tensor | None = None,
        threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict active reactions.

        Returns:
            probs: (B, n_reactions) sigmoid probabilities
            active: (B, n_reactions) boolean mask
        """
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(gene_features, gene_mask)
            probs = torch.sigmoid(logits)
            return probs, probs > threshold
        finally:
            if was_training:
                self.train()
