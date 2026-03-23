from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from biometnet.data.metabolic_vocab import MetabolicVocab


class GenomeMetabolismDataset(Dataset):
    """PyTorch dataset for genome -> reaction-sequence pairs (seq2seq)."""

    def __init__(
        self,
        samples: list[dict],
        vocab: MetabolicVocab,
    ) -> None:
        self.samples = samples
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        genome = torch.tensor(sample["genome"], dtype=torch.float32)
        token_ids = self.vocab.encode(sample["reaction_ids"])
        tokens = torch.tensor(token_ids, dtype=torch.long)
        return {"genome": genome, "tokens": tokens}


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate with dynamic padding on token sequences."""
    genomes = torch.stack([b["genome"] for b in batch])
    tokens_list = [b["tokens"] for b in batch]
    # pad_sequence pads with 0, which is PAD_ID by construction
    tokens_padded = pad_sequence(tokens_list, batch_first=True, padding_value=0)
    return {"genome": genomes, "tokens": tokens_padded}


# ---------------------------------------------------------------------------
# Multi-label classification dataset
# ---------------------------------------------------------------------------


class MultiLabelDataset(Dataset):
    """Dataset for multi-label classification: genome -> binary reaction vector."""

    def __init__(
        self,
        samples: list[dict],
        reaction_list: list[str],
    ) -> None:
        self.samples = samples
        self.reaction_to_idx = {r: i for i, r in enumerate(reaction_list)}
        self.n_reactions = len(reaction_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        genome = torch.tensor(sample["genome"], dtype=torch.float32)
        labels = torch.zeros(self.n_reactions, dtype=torch.float32)
        for rid in sample["reaction_ids"]:
            if rid in self.reaction_to_idx:
                labels[self.reaction_to_idx[rid]] = 1.0
        return {"genome": genome, "labels": labels}


def multilabel_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate for multi-label dataset."""
    return {
        "genome": torch.stack([b["genome"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def compute_pos_weight(
    samples: list[dict],
    reaction_list: list[str],
    max_weight: float = 10.0,
) -> torch.Tensor:
    """Compute per-reaction positive weight for BCEWithLogitsLoss.

    pos_weight[i] = n_negative / n_positive for reaction i.
    Rare reactions get higher weight so the model learns to discriminate.
    Capped at max_weight to prevent extreme over-prediction of rare reactions.
    """
    n = len(samples)
    reaction_to_idx = {r: i for i, r in enumerate(reaction_list)}
    counts = torch.zeros(len(reaction_list))

    for sample in samples:
        for rid in sample["reaction_ids"]:
            if rid in reaction_to_idx:
                counts[reaction_to_idx[rid]] += 1

    pos = counts.clamp(min=1)
    neg = (n - counts).clamp(min=1)
    return (neg / pos).clamp(max=max_weight)


# ---------------------------------------------------------------------------
# Binary (tensor) dataset for fast loading
# ---------------------------------------------------------------------------


def save_binary_dataset(
    samples: list[dict],
    reaction_list: list[str],
    path: str | Path,
) -> None:
    """Save dataset as precomputed tensors for fast loading.

    Stores uint8 genome and label tensors + raw reaction_ids for evaluation.
    Typically 10-20× smaller and faster than JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rxn_idx = {r: i for i, r in enumerate(reaction_list)}
    n = len(samples)
    n_genes = len(samples[0]["genome"])
    n_rxns = len(reaction_list)

    genomes = torch.zeros(n, n_genes, dtype=torch.uint8)
    labels = torch.zeros(n, n_rxns, dtype=torch.uint8)
    for i, s in enumerate(samples):
        for j, g in enumerate(s["genome"]):
            genomes[i, j] = g
        for rid in s["reaction_ids"]:
            if rid in rxn_idx:
                labels[i, rxn_idx[rid]] = 1

    torch.save({"genomes": genomes, "labels": labels}, path)


class BinaryMultiLabelDataset(Dataset):
    """Dataset that loads from precomputed .pt tensors.

    Supports online noise augmentation: randomly flips a small fraction of
    genes each time a sample is accessed, acting as a denoising regularizer.
    """

    def __init__(self, path: str | Path, augment_noise: float = 0.0) -> None:
        data = torch.load(path, weights_only=True)
        self.genomes = data["genomes"].float()
        self.labels = data["labels"].float()
        self.augment_noise = augment_noise

    def __len__(self) -> int:
        return len(self.genomes)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        genome = self.genomes[idx]
        if self.augment_noise > 0:
            genome = genome.clone()
            noise_mask = torch.rand(genome.shape) < self.augment_noise
            genome = torch.where(noise_mask, 1.0 - genome, genome)
        return {"genome": genome, "labels": self.labels[idx]}


# ---------------------------------------------------------------------------
# E. coli strain dataset (cross-strain gene features)
# ---------------------------------------------------------------------------


class StrainDataset(Dataset):
    """Dataset for cross-strain training with per-gene features.

    Each sample stores which strain it comes from, a gene presence vector,
    and a universal reaction label vector.  Gene features are stored per-strain
    and gated by presence at access time.
    """

    def __init__(
        self,
        split_path: str | Path,
        organism_features_path: str | Path,
        augment_noise: float = 0.0,
    ) -> None:
        split = torch.load(split_path, weights_only=True)
        self.org_idx = split["organism_idx"]       # (N,) long
        self.presence = split["presence"].float()   # (N, max_genes) padded
        self.labels = split["labels"].float()       # (N, n_reactions)
        self.n_genes = split["n_genes"]             # (N,) int32

        org_data = torch.load(organism_features_path, weights_only=True)
        self.org_features = org_data["features"]    # list of (n_genes_i, n_feat)
        self.augment_noise = augment_noise

    def __len__(self) -> int:
        return len(self.org_idx)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        oi = self.org_idx[idx].item()
        ng = self.n_genes[idx].item()
        presence = self.presence[idx, :ng]  # (ng,)

        if self.augment_noise > 0:
            presence = presence.clone()
            flip = torch.rand(ng) < self.augment_noise
            presence = torch.where(flip, 1.0 - presence, presence)

        feats = self.org_features[oi]  # (ng, n_feat)
        gated = feats * presence.unsqueeze(-1)   # zero-out absent genes
        mask = presence == 0                      # True = absent (for attn mask)

        return {
            "gene_features": gated,
            "gene_mask": mask,
            "labels": self.labels[idx],
        }


def strain_collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate strain samples with dynamic gene-count padding."""
    max_genes = max(b["gene_features"].size(0) for b in batch)
    n_feat = batch[0]["gene_features"].size(1)
    B = len(batch)

    features = torch.zeros(B, max_genes, n_feat)
    masks = torch.ones(B, max_genes, dtype=torch.bool)  # True = padded
    labels = torch.stack([b["labels"] for b in batch])

    for i, b in enumerate(batch):
        ng = b["gene_features"].size(0)
        features[i, :ng] = b["gene_features"]
        masks[i, :ng] = b["gene_mask"]

    return {"gene_features": features, "gene_mask": masks, "labels": labels}
