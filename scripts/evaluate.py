#!/usr/bin/env python
"""Evaluate a trained model on the test set."""
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
    MultiOrganismDataset,
    collate_fn,
    multilabel_collate_fn,
    multi_org_collate_fn,
)
from biometnet.model.seq2seq import Seq2SeqModel
from biometnet.model.classifier import GenomeClassifier
from biometnet.model.multi_org_classifier import MultiOrganismClassifier
from biometnet.training.config import TrainingConfig
from biometnet.evaluation.metrics import evaluate_predictions, per_pathway_breakdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BioMetNet")
    parser.add_argument("--dataset", default="toy", choices=["toy", "ecoli", "multi_organism"])
    parser.add_argument("--model", default="classifier", choices=["seq2seq", "classifier"])
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Max test samples to evaluate (default: all for toy, 20 for ecoli seq2seq)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid threshold for classifier predictions")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep thresholds to find optimal F1 (classifier only)")
    parser.add_argument("--pathway-breakdown", action="store_true",
                        help="Show per-pathway F1 breakdown (requires pathway_defs)")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Multi-organism evaluation path
    # -----------------------------------------------------------------------
    if args.dataset == "multi_organism":
        data_dir = Path("data/processed/multi_organism")
        mo_config = json.loads((data_dir / "config.json").read_text())
        uni_rxns = json.loads((data_dir / "universal_reactions.json").read_text())

        config = TrainingConfig()
        ckpt_path = Path(config.checkpoint_dir) / "best.pt"
        if not ckpt_path.exists():
            print(f"No checkpoint found at {ckpt_path}. Train first.")
            return

        device = config.resolve_device()
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
              f"(val_loss={checkpoint['val_loss']:.4f})")

        model = MultiOrganismClassifier(
            n_features=mo_config["n_features"],
            n_reactions=mo_config["n_universal_reactions"],
            d_model=mo_config.get("d_model", 256),
            n_heads=mo_config.get("n_heads", 8),
            n_encoder_layers=mo_config.get("n_encoder_layers", 2),
            n_cross_layers=mo_config.get("n_cross_layers", 2),
            ff_dim=mo_config.get("ff_dim", 512),
            dropout=0.0,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        org_feat_path = data_dir / "organism_features.pt"
        test_ds = MultiOrganismDataset(data_dir / "test.pt", org_feat_path)
        test_loader = DataLoader(
            test_ds, batch_size=16, shuffle=False, collate_fn=multi_org_collate_fn,
        )
        print(f"Evaluating {len(test_ds)} test samples, "
              f"Universal reactions: {len(uni_rxns)}")

        all_preds: list[list[str]] = []
        all_targets: list[list[str]] = []
        all_probs: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in test_loader:
                probs, active = model.predict(
                    batch["gene_features"].to(device),
                    batch["gene_mask"].to(device),
                    threshold=args.threshold,
                )
                all_probs.append(probs.cpu())
                labels = batch["labels"]
                for j in range(probs.size(0)):
                    pred = [uni_rxns[k] for k in active[j].nonzero(as_tuple=True)[0].tolist()]
                    tgt = [uni_rxns[k] for k in labels[j].nonzero(as_tuple=True)[0].tolist()]
                    all_preds.append(sorted(pred))
                    all_targets.append(sorted(tgt))

        results = evaluate_predictions(all_preds, all_targets)
        print(f"\n=== Multi-Organism Evaluation (threshold={args.threshold}) ===")
        print(f"  Reaction Precision: {results['precision']:.4f}")
        print(f"  Reaction Recall:    {results['recall']:.4f}")
        print(f"  Reaction F1:        {results['f1']:.4f}")
        print(f"  Pathway Accuracy:   {results['pathway_accuracy']:.4f}")
        print(f"  Pathway Jaccard:    {results['pathway_jaccard']:.4f}")
        print(f"  Metabolite Coverage: {results['metabolite_coverage']:.4f}")

        # Threshold sweep
        if args.sweep and all_probs:
            probs_tensor = torch.cat(all_probs, dim=0)
            print("\n=== Threshold Sweep ===")
            print(f"  {'t':>5s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}")
            for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]:
                sweep_preds = []
                for row in probs_tensor:
                    act = row > t
                    pred = [uni_rxns[k] for k in act.nonzero(as_tuple=True)[0].tolist()]
                    sweep_preds.append(sorted(pred))
                sr = evaluate_predictions(sweep_preds, all_targets)
                print(f"  {t:5.2f}  {sr['precision']:6.3f}  {sr['recall']:6.3f}  "
                      f"{sr['f1']:6.3f}")

        # Save results
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "eval_multi_organism.json"
        output_data = {
            "dataset": "multi_organism",
            "model_type": "multi_org_classifier",
            "epoch": checkpoint["epoch"],
            "val_loss": checkpoint["val_loss"],
            "metrics": results,
            "n_test_samples": len(test_ds),
        }
        output_file.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to {output_file}")
        return

    if args.dataset == "ecoli":
        data_dir = Path("data/processed/ecoli")
        model_config = json.loads((data_dir / "model_config.json").read_text())
        pw_path = data_dir / "pathway_defs.json"
        pathway_defs = json.loads(pw_path.read_text()) if pw_path.exists() else None
    else:
        data_dir = Path("data/processed/toy")
        model_config = None
        pathway_defs = None  # falls back to toy defs in metrics.py

    config = TrainingConfig()
    ckpt_path = Path(config.checkpoint_dir) / "best.pt"

    if not ckpt_path.exists():
        print(f"No checkpoint found at {ckpt_path}. Train first.")
        return

    # Load data & vocab
    test_samples = load_toy_dataset(data_dir / "test.json")
    vocab = MetabolicVocab.load(data_dir / "vocab.json")
    reaction_list = vocab.itos[4:]  # strip special tokens

    # Limit samples for speed (mostly matters for seq2seq)
    n_samples = args.n_samples
    if n_samples is None:
        if args.model == "seq2seq" and args.dataset == "ecoli":
            n_samples = 20  # autoregressive decoding is slow
        else:
            n_samples = len(test_samples)  # classifier is fast
    test_samples = test_samples[:n_samples]
    print(f"Evaluating {len(test_samples)} samples, Reactions: {len(reaction_list)}")

    device = config.resolve_device()
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f})")

    all_preds: list[list[str]] = []
    all_targets: list[list[str]] = []
    all_probs: list[torch.Tensor] = []  # for threshold sweep

    if args.model == "classifier":
        n_genes = model_config["n_genes"] if model_config else config.model.n_genes
        d_model = model_config.get("d_model", 256) if model_config else 256
        n_heads = model_config.get("n_heads", 8) if model_config else 8
        n_enc = model_config.get("n_encoder_layers", 4) if model_config else 4
        n_cross = model_config.get("n_cross_layers", 1) if model_config else 1
        ff_dim = model_config.get("ff_dim", 512) if model_config else 512

        model = GenomeClassifier(
            n_genes=n_genes,
            n_reactions=len(reaction_list),
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=n_enc,
            n_cross_layers=n_cross,
            ff_dim=ff_dim,
            dropout=0.0,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        test_ds = MultiLabelDataset(test_samples, reaction_list)
        test_loader = DataLoader(
            test_ds, batch_size=64, shuffle=False, collate_fn=multilabel_collate_fn,
        )

        with torch.no_grad():
            for batch in test_loader:
                genome = batch["genome"].to(device)
                probs, active = model.predict(genome, threshold=args.threshold)
                all_probs.append(probs.cpu())
                for row in active:
                    pred = [reaction_list[i] for i in row.nonzero(as_tuple=True)[0].tolist()]
                    all_preds.append(sorted(pred))

    else:
        # Legacy seq2seq path
        if model_config is not None:
            model = Seq2SeqModel(
                vocab_size=len(vocab), dropout=0.0, **model_config,
            )
        else:
            model = Seq2SeqModel(
                n_genes=config.model.n_genes, vocab_size=len(vocab),
                d_model=config.model.d_model, n_heads=config.model.n_heads,
                n_encoder_layers=config.model.n_encoder_layers,
                n_decoder_layers=config.model.n_decoder_layers,
                ff_dim=config.model.ff_dim, dropout=0.0,
                max_seq_len=config.model.max_seq_len,
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        test_ds = GenomeMetabolismDataset(test_samples, vocab)
        eval_batch = 4 if args.dataset == "ecoli" else 64
        max_gen_len = 2300 if args.dataset == "ecoli" else 200
        test_loader = DataLoader(
            test_ds, batch_size=eval_batch, shuffle=False, collate_fn=collate_fn,
        )

        n_batches = len(test_loader)
        for i, batch in enumerate(test_loader):
            print(f"  Generating batch {i+1}/{n_batches}...", flush=True)
            genome = batch["genome"].to(device)
            token_seqs = model.generate(
                genome, vocab.bos_id, vocab.eos_id, max_len=max_gen_len,
            )
            for seq in token_seqs:
                all_preds.append(vocab.decode(seq))

    for sample in test_samples:
        all_targets.append(sample["reaction_ids"])

    # Evaluate
    results = evaluate_predictions(all_preds, all_targets, pathway_defs=pathway_defs)
    print(f"\n=== Evaluation Results ({args.model}, threshold={args.threshold}) ===")
    print(f"  Reaction Precision: {results['precision']:.4f}")
    print(f"  Reaction Recall:    {results['recall']:.4f}")
    print(f"  Reaction F1:        {results['f1']:.4f}")
    print(f"  Pathway Accuracy:   {results['pathway_accuracy']:.4f}")
    print(f"  Pathway Jaccard:    {results['pathway_jaccard']:.4f}")
    print(f"  Metabolite Coverage: {results['metabolite_coverage']:.4f}")

    # Show a few examples (abbreviated)
    print("\n=== Sample Predictions ===")
    for i in range(min(3, len(all_preds))):
        t, p = all_targets[i], all_preds[i]
        tp = len(set(t) & set(p))
        print(f"\nSample {i}: target={len(t)} rxns, predicted={len(p)} rxns, overlap={tp}")
        print(f"  Target (first 10):    {t[:10]}")
        print(f"  Predicted (first 10): {p[:10]}")

    # Threshold sweep (classifier only)
    if args.model == "classifier" and args.sweep and all_probs:
        probs_tensor = torch.cat(all_probs, dim=0)
        print("\n=== Threshold Sweep ===")
        print(f"  {'t':>5s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  {'PwAcc':>6s}  {'PwJac':>6s}")
        for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]:
            sweep_preds = []
            for row in probs_tensor:
                active = row > t
                pred = [reaction_list[j] for j in active.nonzero(as_tuple=True)[0].tolist()]
                sweep_preds.append(sorted(pred))
            sr = evaluate_predictions(sweep_preds, all_targets, pathway_defs=pathway_defs)
            print(f"  {t:5.2f}  {sr['precision']:6.3f}  {sr['recall']:6.3f}  "
                  f"{sr['f1']:6.3f}  {sr['pathway_accuracy']:6.3f}  {sr['pathway_jaccard']:6.3f}")

    # Save results to file
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"eval_{args.dataset}.json"
    output_data = {
        "dataset": args.dataset,
        "model_type": args.model,
        "epoch": checkpoint["epoch"],
        "val_loss": checkpoint["val_loss"],
        "metrics": results,
        "n_test_samples": len(test_samples),
        "predictions": [
            {"target": t, "predicted": p}
            for t, p in zip(all_targets, all_preds)
        ],
    }
    output_file.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to {output_file}")

    # Per-pathway breakdown
    if args.pathway_breakdown and pathway_defs:
        pw_metrics = per_pathway_breakdown(all_preds, all_targets, pathway_defs)
        print(f"\n=== Per-Pathway Breakdown (worst first, {len(pw_metrics)} pathways) ===")
        print(f"  {'Pathway':<45s} {'#Rxn':>4s}  {'Prec':>5s}  {'Rec':>5s}  {'F1':>5s}  {'Jac':>5s}")
        for pw in pw_metrics[:15]:
            print(f"  {pw['name'][:44]:<45s} {pw['n_reactions']:4d}  "
                  f"{pw['precision']:5.3f}  {pw['recall']:5.3f}  "
                  f"{pw['f1']:5.3f}  {pw['jaccard']:5.3f}")
        if len(pw_metrics) > 15:
            print(f"  ... ({len(pw_metrics) - 15} more pathways omitted)")


if __name__ == "__main__":
    main()
