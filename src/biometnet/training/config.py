from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    n_genes: int = 40
    d_model: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1
    max_seq_len: int = 128


@dataclass
class DataConfig:
    n_samples: int = 1000
    seed: int = 42
    gene_dropout: float = 0.05
    spurious_rate: float = 0.02
    train_split: float = 0.8
    val_split: float = 0.1
    # test_split is implicitly 1 - train - val
    data_dir: str = "data/processed/toy"
    vocab_path: str = "data/processed/toy/vocab.json"


@dataclass
class TrainingConfig:
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 50
    warmup_steps: int = 100
    grad_clip: float = 1.0
    device: str = "auto"  # "auto", "cpu", "cuda"
    checkpoint_dir: str = "checkpoints"
    log_every: int = 10
    val_every: int = 1  # validate every N epochs
    patience: int = 7  # early stopping: stop after N epochs without improvement

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
