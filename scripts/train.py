#!/usr/bin/env python
"""Train the BioMetNet genome-to-metabolism model."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from biometnet.data.metabolic_vocab import MetabolicVocab
from biometnet.data.toy_data import load_toy_dataset
from biometnet.data.dataset import (
    GenomeMetabolismDataset,
    MultiLabelDataset,
    BinaryMultiLabelDataset,
    StrainDataset,
    collate_fn,
    multilabel_collate_fn,
    strain_collate_fn,
)
from biometnet.model.seq2seq import Seq2SeqModel
from biometnet.model.classifier import GenomeClassifier
from biometnet.model.strain_classifier import EcoliStrainClassifier
from biometnet.training.config import TrainingConfig
from biometnet.training.trainer import Trainer, ClassifierTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BioMetNet")
    parser.add_argument("--dataset", default="toy", choices=["toy", "ecoli", "ecoli_strains"])
    parser.add_argument("--model", default="classifier", choices=["seq2seq", "classifier"])
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the best checkpoint")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # E. coli cross-strain path (BiGG strain training)
    # -----------------------------------------------------------------------
    if args.dataset == "ecoli_strains":
        data_dir = Path("data/processed/ecoli_strains")
        mo_config = json.loads((data_dir / "config.json").read_text())
        feature_vocabs = json.loads((data_dir / "feature_vocabs.json").read_text())
        universal_reactions = json.loads(
            (data_dir / "universal_reactions.json").read_text(),
        )

        config = TrainingConfig(
            batch_size=16,
            lr=1e-4,
            epochs=30,
            warmup_steps=500,
            weight_decay=5e-2,
            log_every=20,
            augment_noise=0.05,
        )
        grad_accum_steps = 2  # effective batch = 16 * 2 = 32

        print("Loading E. coli strain dataset...")
        org_feat_path = data_dir / "organism_features.pt"
        train_ds = StrainDataset(
            data_dir / "train.pt", org_feat_path, augment_noise=config.augment_noise,
        )
        val_ds = StrainDataset(data_dir / "val.pt", org_feat_path)
        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
        print(f"  Features/gene: {mo_config['n_features']}, "
              f"Universal reactions: {mo_config['n_universal_reactions']}")

        # Compute per-reaction positive weight from training labels.
        # Reactions that are rarely active get a lower pos_weight so the model
        # is penalised less for missing them, which substantially improves
        # precision without hurting recall on common reactions.
        all_labels = train_ds.labels  # (N, n_reactions) float
        pos_freq = all_labels.mean(dim=0).clamp(min=1e-3, max=1 - 1e-3)
        mo_pos_weight = (1.0 - pos_freq) / pos_freq  # inverse frequency
        mo_pos_weight = mo_pos_weight.clamp(max=10.0)  # cap extreme weights

        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size,
            shuffle=True, collate_fn=strain_collate_fn,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size,
            shuffle=False, collate_fn=strain_collate_fn,
            num_workers=2, pin_memory=True, persistent_workers=True,
        )

        # Load pre-computed reaction metadata features for query initialization
        rxn_feat_path = data_dir / "reaction_features.pt"
        rxn_features = None
        if rxn_feat_path.exists():
            rxn_features = torch.load(rxn_feat_path, map_location="cpu", weights_only=True)
            print(f"  Reaction features loaded: {rxn_features.shape}")

        model = EcoliStrainClassifier(
            n_features=mo_config["n_features"],
            n_reactions=mo_config["n_universal_reactions"],
            d_model=mo_config.get("d_model", 256),
            n_heads=mo_config.get("n_heads", 8),
            n_encoder_layers=mo_config.get("n_encoder_layers", 2),
            n_cross_layers=mo_config.get("n_cross_layers", 2),
            n_self_layers=mo_config.get("n_self_layers", 1),
            ff_dim=mo_config.get("ff_dim", 512),
            dropout=0.2,
            reaction_features=rxn_features,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"EcoliStrainClassifier parameters: {n_params:,}")

        resume_ckpt = None
        if args.resume:
            ckpt_path = Path(config.checkpoint_dir) / "best.pt"
            if ckpt_path.exists():
                resume_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                model.load_state_dict(resume_ckpt["model_state_dict"])
                print(f"Resuming from epoch {resume_ckpt['epoch'] + 1} "
                      f"(val_loss={resume_ckpt['val_loss']:.4f})", flush=True)
            else:
                print("No checkpoint found, starting from scratch.", flush=True)

        trainer = ClassifierTrainer(
            model, train_loader, val_loader, config,
            pos_weight=mo_pos_weight, resume_checkpoint=resume_ckpt,
            use_amp=True,
            grad_accum_steps=grad_accum_steps,
        )
        trainer.train()
        return

    if args.dataset == "ecoli":
        data_dir = Path("data/processed/ecoli")
        model_config = json.loads((data_dir / "model_config.json").read_text())
        config = TrainingConfig(
            batch_size=8,
            lr=3e-4,
            epochs=60,
            warmup_steps=500,
            log_every=100,
        )
    else:
        data_dir = Path("data/processed/toy")
        model_config = None
        config = TrainingConfig()

    # Load data
    print(f"Loading {args.dataset} data...")
    train_samples = load_toy_dataset(data_dir / "train.json")
    val_samples = load_toy_dataset(data_dir / "val.json")
    vocab = MetabolicVocab.load(data_dir / "vocab.json")
    reaction_list = vocab.itos[4:]  # strip special tokens
    print(f"Reactions: {len(reaction_list)}, Train: {len(train_samples)}, Val: {len(val_samples)}", flush=True)

    if args.model == "classifier":
        # Multi-label classification path
        # Prefer fast binary .pt datasets if available
        train_pt = data_dir / "train.pt"
        val_pt = data_dir / "val.pt"
        if train_pt.exists() and val_pt.exists():
            print("Loading binary tensor datasets...", flush=True)
            train_ds = BinaryMultiLabelDataset(train_pt, augment_noise=config.augment_noise)
            val_ds = BinaryMultiLabelDataset(val_pt)
            print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
        else:
            print("Building datasets from JSON...", flush=True)
            train_ds = MultiLabelDataset(train_samples, reaction_list)
            val_ds = MultiLabelDataset(val_samples, reaction_list)

        use_gpu = config.resolve_device() == "cuda"
        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size,
            shuffle=True, collate_fn=multilabel_collate_fn,
            num_workers=4 if use_gpu else 0,
            pin_memory=use_gpu,
            persistent_workers=use_gpu,
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size,
            shuffle=False, collate_fn=multilabel_collate_fn,
            num_workers=2 if use_gpu else 0,
            pin_memory=use_gpu,
            persistent_workers=use_gpu,
        )

        n_genes = model_config["n_genes"] if model_config else config.model.n_genes
        d_model = model_config.get("d_model", config.model.d_model) if model_config else config.model.d_model
        n_heads = model_config.get("n_heads", config.model.n_heads) if model_config else config.model.n_heads
        n_enc = model_config.get("n_encoder_layers", config.model.n_encoder_layers) if model_config else config.model.n_encoder_layers
        n_cross = model_config.get("n_cross_layers", config.model.n_cross_layers) if model_config else config.model.n_cross_layers
        ff_dim = model_config.get("ff_dim", config.model.ff_dim) if model_config else config.model.ff_dim

        model = GenomeClassifier(
            n_genes=n_genes,
            n_reactions=len(reaction_list),
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=n_enc,
            n_cross_layers=n_cross,
            ff_dim=ff_dim,
            dropout=config.model.dropout,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Classifier parameters: {n_params:,}")

        # Resume from checkpoint if requested
        resume_ckpt = None
        if args.resume:
            ckpt_path = Path(config.checkpoint_dir) / "best.pt"
            if ckpt_path.exists():
                resume_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                model.load_state_dict(resume_ckpt["model_state_dict"])
                print(f"Resuming from epoch {resume_ckpt['epoch'] + 1} "
                      f"(val_loss={resume_ckpt['val_loss']:.4f})", flush=True)
            else:
                print("No checkpoint found, starting from scratch.", flush=True)

        trainer = ClassifierTrainer(
            model, train_loader, val_loader, config,
            pos_weight=None, resume_checkpoint=resume_ckpt,
            use_amp=use_gpu,
        )
        trainer.train()

    else:
        # Legacy seq2seq path
        train_ds = GenomeMetabolismDataset(train_samples, vocab)
        val_ds = GenomeMetabolismDataset(val_samples, vocab)

        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size,
            shuffle=True, collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size,
            shuffle=False, collate_fn=collate_fn,
        )

        if model_config is not None:
            # Filter to only keys Seq2SeqModel accepts (config may
            # contain classifier-only keys like n_cross_layers)
            seq2seq_keys = {
                "n_genes", "d_model", "n_heads", "n_encoder_layers",
                "n_decoder_layers", "ff_dim", "max_seq_len",
            }
            s2s_config = {k: v for k, v in model_config.items() if k in seq2seq_keys}
            model = Seq2SeqModel(
                vocab_size=len(vocab),
                dropout=config.model.dropout,
                **s2s_config,
            )
        else:
            model = Seq2SeqModel(
                n_genes=config.model.n_genes,
                vocab_size=len(vocab),
                d_model=config.model.d_model,
                n_heads=config.model.n_heads,
                n_encoder_layers=config.model.n_encoder_layers,
                n_decoder_layers=config.model.n_decoder_layers,
                ff_dim=config.model.ff_dim,
                dropout=config.model.dropout,
                max_seq_len=config.model.max_seq_len,
            )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Seq2Seq parameters: {n_params:,}")

        trainer = Trainer(model, train_loader, val_loader, config, len(vocab))
        trainer.train()


if __name__ == "__main__":
    main()
