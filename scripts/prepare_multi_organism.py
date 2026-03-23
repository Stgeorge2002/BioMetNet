#!/usr/bin/env python
"""Download BiGG models and prepare multi-organism training dataset."""
import argparse

from biometnet.data.multi_organism import (
    download_all_bigg_models,
    prepare_multi_organism_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download BiGG models and prepare multi-organism dataset",
    )
    parser.add_argument(
        "--max-models", type=int, default=None,
        help="Limit number of BiGG models to download (default: all)",
    )
    parser.add_argument(
        "--samples-per-org", type=int, default=500,
        help="Training samples per organism (default: 500)",
    )
    parser.add_argument(
        "--eval-samples-per-org", type=int, default=50,
        help="Val/test samples per organism (default: 50)",
    )
    parser.add_argument(
        "--min-organisms", type=int, default=2,
        help="Min organisms a reaction must appear in to be included (default: 2)",
    )
    args = parser.parse_args()

    print("Step 1: Downloading BiGG models...")
    paths = download_all_bigg_models(max_models=args.max_models)
    print(f"\nDownloaded {len(paths)} models\n")

    print("Step 2: Preparing multi-organism dataset...")
    config = prepare_multi_organism_dataset(
        paths,
        samples_per_train_org=args.samples_per_org,
        samples_per_eval_org=args.eval_samples_per_org,
        min_rxn_organisms=args.min_organisms,
    )

    print(f"\n=== Dataset Summary ===")
    for k, v in config.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
