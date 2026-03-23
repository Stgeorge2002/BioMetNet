"""Multi-label classifier: genome vector -> active reactions via cross-attention."""
from __future__ import annotations

import torch
import torch.nn as nn

from biometnet.model.encoder import GenomeEncoder


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block: attn + add&norm + FFN + add&norm."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = self.cross_attn(queries, kv, kv, key_padding_mask=key_padding_mask)
        queries = self.cross_norm(queries + attended)
        queries = self.ffn_norm(queries + self.ffn(queries))
        return queries


class GenomeClassifier(nn.Module):
    """Cross-attention classifier: each reaction learns which genes matter.

    Architecture:
      1. GenomeEncoder with learned gene embeddings
      2. Learnable reaction query embeddings: (n_reactions, d_model)
      3. N stacked cross-attention blocks (attn + FFN + norms)
      4. Per-reaction classification head → (batch, n_reactions) logits
    """

    def __init__(
        self,
        n_genes: int,
        n_reactions: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        n_cross_layers: int = 1,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_reactions = n_reactions
        self.encoder = GenomeEncoder(
            n_genes=n_genes,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        # Learnable reaction queries — one embedding per reaction
        self.reaction_queries = nn.Parameter(
            torch.randn(n_reactions, d_model) * (d_model ** -0.5)
        )
        # Stacked cross-attention decoder blocks
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_cross_layers)
        ])

        # Per-reaction classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1),
        )

    def forward(self, genome: torch.Tensor) -> torch.Tensor:
        """
        Args:
            genome: (batch, n_genes) binary gene presence vector
        Returns:
            logits: (batch, n_reactions) raw logits (apply sigmoid for probs)
        """
        B = genome.size(0)
        gene_mask = genome == 0  # True for absent genes
        gene_enc = self.encoder(genome, padding_mask=gene_mask)

        # Expand reaction queries for the batch
        queries = self.reaction_queries.unsqueeze(0).expand(B, -1, -1)

        # Multi-layer cross-attention: reactions attend to gene encodings
        for layer in self.cross_layers:
            queries = layer(queries, gene_enc, key_padding_mask=gene_mask)

        # Classify each reaction independently
        logits = self.classifier(queries).squeeze(-1)  # (B, n_reactions)
        return logits

    @torch.no_grad()
    def predict(
        self,
        genome: torch.Tensor,
        threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict active reactions.

        Returns:
            probs: (batch, n_reactions) sigmoid probabilities
            active: (batch, n_reactions) boolean mask
        """
        self.eval()
        logits = self.forward(genome)
        probs = torch.sigmoid(logits)
        return probs, probs > threshold
