#!/usr/bin/env python
"""Predict metabolic profile from a GFF annotation file."""
import argparse
import json
from pathlib import Path

import torch

from biometnet.data.gff_parser import parse_gff_genes, map_gff_to_model_genes
from biometnet.data.metabolic_vocab import MetabolicVocab
from biometnet.model.seq2seq import Seq2SeqModel
from biometnet.model.classifier import GenomeClassifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict metabolic profile from a GFF annotation file"
    )
    parser.add_argument("--gff", required=True, help="Path to GFF3 file (.gff or .gff.gz)")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data-dir", default="data/processed/ecoli")
    parser.add_argument("--model", default="classifier", choices=["seq2seq", "classifier"])
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid threshold for classifier predictions")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load model info and vocab
    model_info = json.loads((data_dir / "model_info.json").read_text())
    model_config = json.loads((data_dir / "model_config.json").read_text())
    vocab = MetabolicVocab.load(data_dir / "vocab.json")
    reaction_list = vocab.itos[4:]  # strip special tokens
    genes = model_info["genes"]
    gene_names = model_info.get("gene_names", {})

    # Parse GFF
    print(f"Parsing {args.gff}...")
    gff_genes = parse_gff_genes(args.gff)
    total_parsed = len(gff_genes["locus_tags"] | gff_genes["gene_names"])

    # Map to model genes
    matched = map_gff_to_model_genes(gff_genes, genes, gene_names)
    print(f"  {total_parsed} gene features parsed")
    print(f"  {len(matched)}/{len(genes)} metabolic genes matched")

    # Build genome vector
    genome = [1 if g in matched else 0 for g in genes]
    genome_tensor = torch.tensor([genome], dtype=torch.float32)

    # Load model
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    if args.model == "classifier":
        from biometnet.training.config import ModelConfig
        _mc_defaults = ModelConfig()
        d_model = model_config.get("d_model", _mc_defaults.d_model)
        n_heads = model_config.get("n_heads", _mc_defaults.n_heads)
        n_enc = model_config.get("n_encoder_layers", _mc_defaults.n_encoder_layers)
        n_cross = model_config.get("n_cross_layers", _mc_defaults.n_cross_layers)
        ff_dim = model_config.get("ff_dim", _mc_defaults.ff_dim)

        model = GenomeClassifier(
            n_genes=model_config["n_genes"],
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

        print(f"Generating metabolic profile on {device} (classifier)...")
        genome_tensor = genome_tensor.to(device)
        probs, active = model.predict(genome_tensor, threshold=args.threshold)
        predicted = [reaction_list[i] for i in active[0].nonzero(as_tuple=True)[0].tolist()]
        predicted = sorted(predicted)
    else:
        seq2seq_keys = {
            "n_genes", "d_model", "n_heads", "n_encoder_layers",
            "n_decoder_layers", "ff_dim", "max_seq_len",
        }
        s2s_config = {k: v for k, v in model_config.items() if k in seq2seq_keys}
        model = Seq2SeqModel(
            vocab_size=len(vocab), dropout=0.0, **s2s_config,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        print(f"Generating metabolic profile on {device} (seq2seq)...")
        genome_tensor = genome_tensor.to(device)
        token_seqs = model.generate(
            genome_tensor, vocab.bos_id, vocab.eos_id, max_len=2000,
        )
        predicted = vocab.decode(token_seqs[0])

    print(f"\nPredicted {len(predicted)} active reactions:\n")
    for i, rxn in enumerate(predicted):
        print(f"  {rxn}", end="")
        if (i + 1) % 10 == 0:
            print()
    if len(predicted) % 10 != 0:
        print()


if __name__ == "__main__":
    main()
