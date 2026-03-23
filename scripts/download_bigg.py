#!/usr/bin/env python
"""Download BiGG models for future training."""
import argparse

from biometnet.data.bigg_loader import download_bigg_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BiGG metabolic models")
    parser.add_argument("--max-models", type=int, default=None, help="Limit models")
    parser.add_argument("--cache-dir", default="data/raw/bigg", help="Cache directory")
    args = parser.parse_args()

    print("Downloading BiGG models...")
    models = download_bigg_models(cache_dir=args.cache_dir, max_models=args.max_models)
    print(f"Downloaded {len(models)} models")
    for m in models[:5]:
        print(f"  {m['model_id']}: {len(m['gene_ids'])} genes, {len(m['reaction_ids'])} reactions")
    if len(models) > 5:
        print(f"  ... and {len(models) - 5} more")


if __name__ == "__main__":
    main()
