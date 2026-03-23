from __future__ import annotations

import torch
import torch.nn as nn

from biometnet.model.encoder import PositionalEncoding


class MetabolicDecoder(nn.Module):
    """Autoregressive transformer decoder for reaction token sequences.

    At each step, attends to previously generated tokens and cross-attends
    to the genome encoder output.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt_tokens: (batch, tgt_len) token IDs
            memory: (batch, src_len, d_model) encoder output
            tgt_mask: causal mask (tgt_len, tgt_len)
            tgt_key_padding_mask: (batch, tgt_len) True where padded
            memory_key_padding_mask: (batch, src_len) True where encoder padded
        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        x = self.token_emb(tgt_tokens)
        x = self.pos_enc(x)
        x = self.transformer(
            x,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(x)

    @staticmethod
    def generate_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask (True = masked)."""
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
