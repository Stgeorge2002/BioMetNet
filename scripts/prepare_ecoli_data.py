#!/usr/bin/env python
"""Download E. coli iML1515 model and generate training data."""
import json
from pathlib import Path

from biometnet.data.ecoli_data import (
    download_ecoli_model,
    download_reference_gff,
    load_cobra_model,
    extract_model_info,
    extract_pathway_defs,
    generate_ecoli_training_data,
    resample_by_reaction_count,
    save_ecoli_data,
)
from biometnet.data.dataset import save_binary_dataset
from biometnet.data.metabolic_vocab import MetabolicVocab


# Model architecture defaults for E. coli classifier
ECOLI_MODEL_CONFIG = {
    "d_model": 256,
    "n_heads": 8,
    "n_encoder_layers": 4,
    "ff_dim": 512,
}


def main() -> None:
    raw_dir = Path("data/raw/ecoli")
    out_dir = Path("data/processed/ecoli")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download model
    print("Downloading iML1515 model from BiGG...")
    model_path = download_ecoli_model(raw_dir)
    print(f"  Saved to {model_path}")

    # Download reference GFF for prediction demo
    print("Downloading E. coli K-12 reference GFF...")
    gff_path = download_reference_gff(raw_dir)
    print(f"  Saved to {gff_path}")

    # Load and extract info
    print("Loading model with COBRApy...")
    model = load_cobra_model(model_path)
    info = extract_model_info(model)
    n_genes = len(info["genes"])
    n_rxns = len(info["reactions"])
    print(f"  Genes: {n_genes}, Reactions with GPR: {n_rxns}")

    # Save model info (gene list, name mappings, GPR rules)
    info_path = out_dir / "model_info.json"
    info_path.write_text(json.dumps(info, indent=2))

    # Extract and save pathway/subsystem definitions from iML1515
    pathway_defs = extract_pathway_defs(model)
    pw_path = out_dir / "pathway_defs.json"
    pw_path.write_text(json.dumps(pathway_defs, indent=2))
    print(f"  Pathways/subsystems: {len(pathway_defs)}")

    # Save model architecture config (includes n_genes from actual data)
    model_config = {"n_genes": n_genes, **ECOLI_MODEL_CONFIG}
    (out_dir / "model_config.json").write_text(json.dumps(model_config, indent=2))

    # Generate training data: large candidate pool with aggressive dropout,
    # then resample for uniform reaction-count distribution
    n_pool = 30000
    n_target = 12000
    print(f"Generating candidate pool ({n_pool} samples, mixed dropout strategies)...")
    pool = generate_ecoli_training_data(
        info, n_samples=n_pool, seed=42, pathway_defs=pathway_defs,
    )

    pool_rxn_counts = [len(s["reaction_ids"]) for s in pool]
    print(f"  Pool reaction counts: min={min(pool_rxn_counts)}, "
          f"max={max(pool_rxn_counts)}, mean={sum(pool_rxn_counts)//len(pool_rxn_counts)}")

    print(f"Resampling to {n_target} balanced samples (10 reaction-count bins)...")
    samples = resample_by_reaction_count(pool, n_target=n_target, n_bins=10, seed=42)

    # Split 80/10/10
    n = len(samples)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train = samples[:n_train]
    val = samples[n_train : n_train + n_val]
    test = samples[n_train + n_val :]

    save_ecoli_data(train, out_dir / "train.json", compact=True)
    save_ecoli_data(val, out_dir / "val.json", compact=True)
    save_ecoli_data(test, out_dir / "test.json", compact=True)

    # Save binary .pt for fast training loading
    save_binary_dataset(train, info["reactions"], out_dir / "train.pt")
    save_binary_dataset(val, info["reactions"], out_dir / "val.pt")
    print(f"  Saved: {len(train)} train, {len(val)} val, {len(test)} test")
    print(f"  Binary tensors: train.pt, val.pt")

    # Build and save vocabulary
    vocab = MetabolicVocab(info["reactions"])
    vocab.save(out_dir / "vocab.json")
    print(f"  Vocabulary: {len(vocab)} tokens ({n_rxns} reactions + 4 special)")

    # Dataset stats
    rxn_counts = [len(s["reaction_ids"]) for s in samples]
    gene_counts = [sum(s["genome"]) for s in samples]
    print(f"\nDataset stats:")
    print(f"  Reactions per sample: min={min(rxn_counts)}, max={max(rxn_counts)}, "
          f"mean={sum(rxn_counts)/len(rxn_counts):.0f}")
    print(f"  Genes present:       min={min(gene_counts)}, max={max(gene_counts)}, "
          f"mean={sum(gene_counts)/len(gene_counts):.0f}")
    print(f"\nReference GFF for prediction demo: {gff_path}")


if __name__ == "__main__":
    main()
