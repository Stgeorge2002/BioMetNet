from __future__ import annotations

import torch
import torch.nn as nn

from biometnet.model.encoder import GenomeEncoder
from biometnet.model.decoder import MetabolicDecoder


class Seq2SeqModel(nn.Module):
    """Encoder-decoder model: genome -> metabolic reaction sequence."""

    def __init__(
        self,
        n_genes: int,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.encoder = GenomeEncoder(
            n_genes=n_genes,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.decoder = MetabolicDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

    def forward(
        self,
        genome: torch.Tensor,
        tgt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Args:
            genome: (batch, n_genes)
            tgt_tokens: (batch, tgt_len) includes BOS, excludes final EOS for input
        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        gene_mask = genome == 0  # True where gene is absent
        memory = self.encoder(genome, padding_mask=gene_mask)
        tgt_len = tgt_tokens.size(1)
        tgt_mask = self.decoder.generate_causal_mask(tgt_len, tgt_tokens.device)
        tgt_pad_mask = tgt_tokens == 0  # PAD token id = 0
        logits = self.decoder(
            tgt_tokens,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=gene_mask,
        )
        return logits

    @torch.no_grad()
    def generate(
        self,
        genome: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 200,
    ) -> list[list[int]]:
        """Greedy autoregressive decoding.

        Args:
            genome: (batch, n_genes)
            bos_id: BOS token id
            eos_id: EOS token id
            max_len: maximum generation length
        Returns:
            List of token id lists, one per batch element.
        """
        was_training = self.training
        self.eval()
        try:
            return self._generate_inner(genome, bos_id, eos_id, max_len)
        finally:
            if was_training:
                self.train()

    def _generate_inner(
        self,
        genome: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
    ) -> list[list[int]]:
        batch_size = genome.size(0)
        gene_mask = genome == 0
        memory = self.encoder(genome, padding_mask=gene_mask)

        # Start with BOS
        generated = torch.full(
            (batch_size, 1), bos_id, dtype=torch.long, device=genome.device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=genome.device)

        for _ in range(max_len):
            tgt_mask = self.decoder.generate_causal_mask(
                generated.size(1), genome.device
            )
            logits = self.decoder(
                generated, memory, tgt_mask=tgt_mask,
                memory_key_padding_mask=gene_mask,
            )
            next_token = logits[:, -1, :].argmax(dim=-1)  # (batch,)
            next_token = next_token.masked_fill(finished, 0)  # pad finished seqs
            generated = torch.cat(
                [generated, next_token.unsqueeze(1)], dim=1
            )
            finished = finished | (next_token == eos_id)
            if finished.all():
                break

        return [generated[i].tolist() for i in range(batch_size)]
