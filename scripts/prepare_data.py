#!/usr/bin/env python
"""Download E. coli strain models from BiGG and prepare training dataset."""
import argparse

from biometnet.data.strain_data import (
    download_all_bigg_models,
    prepare_strain_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download E. coli BiGG models and prepare strain dataset",
    )
    parser.add_argument(
        "--max-models", type=int, default=None,
        help="Limit number of BiGG models to download (default: all)",
    )
    parser.add_argument(
        "--samples-per-strain", type=int, default=1000,
        help="Training samples per strain (default: 1000)",
    )
    parser.add_argument(
        "--eval-samples-per-strain", type=int, default=200,
        help="Val/test samples per strain (default: 200)",
    )
    parser.add_argument(
        "--min-strains", type=int, default=2,
        help="Min strains a reaction must appear in to be included (default: 2)",
    )
    args = parser.parse_args()

    print("Step 1: Downloading E. coli strain models from BiGG...")
    paths = download_all_bigg_models(max_models=args.max_models)
    print(f"\nDownloaded {len(paths)} models\n")

    print("Step 2: Preparing E. coli strain dataset...")
    config = prepare_strain_dataset(
        paths,
        samples_per_train_org=args.samples_per_strain,
        samples_per_eval_org=args.eval_samples_per_strain,
        min_rxn_organisms=args.min_strains,
    )

    print(f"\n=== Dataset Summary ===")
    for k, v in config.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
