#!/usr/bin/env python
"""Download E. coli strain models from BiGG (and optionally CarveMe) and prepare training dataset."""
import argparse
from pathlib import Path

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
    parser.add_argument(
        "--carveme-dir", type=str, default=None,
        help="Directory containing CarveMe SBML models to include (default: None)",
    )
    args = parser.parse_args()

    print("Step 1: Downloading E. coli strain models from BiGG...")
    paths = download_all_bigg_models(max_models=args.max_models)
    print(f"\nDownloaded {len(paths)} BiGG models\n")

    # Optionally include CarveMe-generated SBML models
    if args.carveme_dir:
        carveme_path = Path(args.carveme_dir)
        if carveme_path.exists():
            sbml_files = sorted(carveme_path.glob("*.xml"))
            print(f"Found {len(sbml_files)} CarveMe models in {carveme_path}")
            paths.extend(sbml_files)
        else:
            print(f"WARNING: CarveMe directory not found: {carveme_path}")

    print(f"\nTotal models: {len(paths)}")

    print("\nStep 2: Preparing E. coli strain dataset...")
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
