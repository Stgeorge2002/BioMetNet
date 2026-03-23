from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class GenomeEncoder(nn.Module):
    """Encode a binary gene-presence vector into contextual gene embeddings.

    Input:  (batch, n_genes)  float
    Output: (batch, n_genes, d_model) encoder hidden states
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gene_embedding = nn.Embedding(n_genes, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=n_genes, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self, genome: torch.Tensor, padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # genome: (batch, n_genes) binary gene-presence vector
        B, N = genome.shape
        gene_ids = torch.arange(N, device=genome.device)
        x = self.gene_embedding(gene_ids).unsqueeze(0).expand(B, -1, -1)
        x = x * genome.unsqueeze(-1)  # gate: absent genes → zero vector
        x = self.pos_enc(x)
        # padding_mask: True for absent genes → ignored in self-attention
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return x
