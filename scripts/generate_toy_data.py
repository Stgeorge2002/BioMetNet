#!/usr/bin/env python
"""Generate synthetic toy genome-metabolism training data."""
from pathlib import Path

from biometnet.data.toy_data import (
    _all_reaction_ids,
    generate_toy_dataset,
    save_toy_dataset,
)
from biometnet.data.metabolic_vocab import MetabolicVocab


def main() -> None:
    out_dir = Path("data/processed/toy")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating toy dataset (1000 samples)...")
    samples = generate_toy_dataset(n_samples=1000, seed=42)

    # Split 80/10/10
    n = len(samples)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train = samples[:n_train]
    val = samples[n_train : n_train + n_val]
    test = samples[n_train + n_val :]

    save_toy_dataset(train, out_dir / "train.json")
    save_toy_dataset(val, out_dir / "val.json")
    save_toy_dataset(test, out_dir / "test.json")
    print(f"Saved: {len(train)} train, {len(val)} val, {len(test)} test")

    # Build and save vocabulary
    vocab = MetabolicVocab(_all_reaction_ids())
    vocab.save(out_dir / "vocab.json")
    print(f"Vocabulary: {len(vocab)} tokens ({len(vocab) - 4} reactions + 4 special)")


if __name__ == "__main__":
    main()
