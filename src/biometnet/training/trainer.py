from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from biometnet.model.seq2seq import Seq2SeqModel
from biometnet.model.classifier import GenomeClassifier
from biometnet.training.config import TrainingConfig


class Trainer:
    """Training loop with teacher forcing, LR scheduling, and checkpointing."""

    def __init__(
        self,
        model: Seq2SeqModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: TrainingConfig,
        vocab_size: int,
    ) -> None:
        self.device = config.resolve_device()
        self.model = model.to(self.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        total_steps = len(train_loader) * config.epochs
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._warmup_cosine(config.warmup_steps, total_steps),
        )

        self.best_val_loss = float("inf")
        self.global_step = 0

    @staticmethod
    def _warmup_cosine(warmup: int, total: int):
        def fn(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return fn

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            genome = batch["genome"].to(self.device)
            tokens = batch["tokens"].to(self.device)

            # Teacher forcing: input = tokens[:, :-1], target = tokens[:, 1:]
            tgt_input = tokens[:, :-1]
            tgt_output = tokens[:, 1:]

            logits = self.model(genome, tgt_input)
            # logits: (batch, seq_len, vocab), target: (batch, seq_len)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            if self.global_step % self.config.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"  step {self.global_step:5d} | "
                    f"loss {loss.item():.4f} | lr {lr:.2e}"
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        if self.val_loader is None:
            return float("inf")
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            genome = batch["genome"].to(self.device)
            tokens = batch["tokens"].to(self.device)
            tgt_input = tokens[:, :-1]
            tgt_output = tokens[:, 1:]

            logits = self.model(genome, tgt_input)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "global_step": self.global_step,
            },
            path,
        )

    def train(self) -> None:
        ckpt_dir = Path(self.config.checkpoint_dir)
        print(f"Training on {self.device} for {self.config.epochs} epochs")
        print(f"Train batches: {len(self.train_loader)}, "
              f"Val batches: {len(self.val_loader) if self.val_loader else 0}")

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch()
            print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f}", end="")

            if epoch % self.config.val_every == 0:
                val_loss = self._validate()
                print(f" | val_loss {val_loss:.4f}", end="")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(ckpt_dir / "best.pt", epoch, val_loss)
                    print(" *", end="")

            print()

        # Save final checkpoint
        self._save_checkpoint(ckpt_dir / "last.pt", self.config.epochs, self.best_val_loss)
        print(f"Done. Best val_loss: {self.best_val_loss:.4f}")


class FocalBCELoss(nn.Module):
    """Focal loss for multi-label classification with optional label smoothing.

    Down-weights easy examples so the model focuses on hard-to-classify reactions.
    gamma=0 recovers standard BCE; gamma=2 is a good default.
    Label smoothing shifts targets away from hard 0/1 to prevent overconfidence.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight,
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class ClassifierTrainer:
    """Training loop for multi-label GenomeClassifier with focal loss."""

    def __init__(
        self,
        model: GenomeClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: TrainingConfig,
        pos_weight: torch.Tensor | None = None,
        resume_checkpoint: dict | None = None,
        use_amp: bool = False,
        grad_accum_steps: int = 1,
    ) -> None:
        self.device = config.resolve_device()
        self.model = model.to(self.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_amp = use_amp and self.device == "cuda"
        self.grad_accum_steps = grad_accum_steps

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        pw = pos_weight.to(self.device) if pos_weight is not None else None
        self.criterion = FocalBCELoss(
            gamma=2.0, pos_weight=pw, label_smoothing=0.05,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        total_steps = (len(train_loader) // self.grad_accum_steps) * config.epochs
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=Trainer._warmup_cosine(config.warmup_steps, total_steps),
        )

        self.best_val_loss = float("inf")
        self.global_step = 0
        self.start_epoch = 1

        if resume_checkpoint is not None:
            self.optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in resume_checkpoint:
                self.scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            else:
                for _ in range(resume_checkpoint.get("global_step", 0)):
                    self.scheduler.step()
            self.global_step = resume_checkpoint.get("global_step", 0)
            self.best_val_loss = resume_checkpoint.get("val_loss", float("inf"))
            self.start_epoch = resume_checkpoint["epoch"] + 1

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            labels = batch["labels"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                if "gene_features" in batch:
                    logits = self.model(
                        batch["gene_features"].to(self.device, non_blocking=True),
                        batch["gene_mask"].to(self.device, non_blocking=True),
                    )
                else:
                    logits = self.model(batch["genome"].to(self.device, non_blocking=True))
                loss = self.criterion(logits, labels)
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            total_loss += loss.item() * self.grad_accum_steps
            n_batches += 1

            if self.global_step % self.config.log_every == 0 and ((batch_idx + 1) % self.grad_accum_steps == 0):
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"  step {self.global_step:5d} | "
                    f"loss {loss.item() * self.grad_accum_steps:.4f} | lr {lr:.2e}",
                    flush=True,
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        if self.val_loader is None:
            return float("inf")
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            labels = batch["labels"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                if "gene_features" in batch:
                    logits = self.model(
                        batch["gene_features"].to(self.device, non_blocking=True),
                        batch["gene_mask"].to(self.device, non_blocking=True),
                    )
                else:
                    logits = self.model(batch["genome"].to(self.device, non_blocking=True))
                loss = self.criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
                "global_step": self.global_step,
            },
            path,
        )

    def train(self) -> None:
        ckpt_dir = Path(self.config.checkpoint_dir)
        print(f"Training classifier on {self.device} for {self.config.epochs} epochs"
              f" (starting at epoch {self.start_epoch})", flush=True)
        print(f"Train batches: {len(self.train_loader)}, "
              f"Val batches: {len(self.val_loader) if self.val_loader else 0}", flush=True)

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            train_loss = self._train_epoch()
            msg = f"Epoch {epoch:3d} | train_loss {train_loss:.4f}"

            if epoch % self.config.val_every == 0:
                val_loss = self._validate()
                msg += f" | val_loss {val_loss:.4f}"

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(ckpt_dir / "best.pt", epoch, val_loss)
                    msg += " *"

            print(msg, flush=True)

        self._save_checkpoint(ckpt_dir / "last.pt", self.config.epochs, self.best_val_loss)
        print(f"Done. Best val_loss: {self.best_val_loss:.4f}", flush=True)
