#!/usr/bin/env python
"""Download NCBI E. coli genomes and build CarveMe metabolic models.

This is the first step if you want to scale beyond BiGG's 34 models.
Run this BEFORE prepare_data.py, then point prepare_data.py at the output:

    python scripts/build_carveme_models.py --max-genomes 200
    python scripts/prepare_data.py --carveme-dir data/raw/carveme_models

Prerequisites:
    uv sync --extra carveme
"""
import argparse

from biometnet.data.ncbi_carveme import (
    build_ncbi_carveme_models,
    download_ncbi_genomes,
    run_carveme_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NCBI E. coli genomes and build CarveMe models",
    )
    parser.add_argument(
        "--max-genomes", type=int, default=None,
        help="Max genomes to download from NCBI (default: all unique strains)",
    )
    parser.add_argument(
        "--ncbi-dir", type=str, default="data/raw/ncbi",
        help="Directory for NCBI downloads (default: data/raw/ncbi)",
    )
    parser.add_argument(
        "--carveme-dir", type=str, default="data/raw/carveme_models",
        help="Directory for CarveMe SBML output (default: data/raw/carveme_models)",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Only download genomes, skip CarveMe reconstruction",
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="Skip strain deduplication (keep all assemblies)",
    )
    args = parser.parse_args()

    if args.download_only:
        paths = download_ncbi_genomes(
            out_dir=args.ncbi_dir,
            max_genomes=args.max_genomes,
            deduplicate=not args.no_dedup,
        )
        print(f"\nDownloaded {len(paths)} protein FASTAs to {args.ncbi_dir}")
        print("Run without --download-only to build CarveMe models.")
    else:
        model_paths = build_ncbi_carveme_models(
            ncbi_dir=args.ncbi_dir,
            carveme_dir=args.carveme_dir,
            max_genomes=args.max_genomes,
            deduplicate=not args.no_dedup,
        )
        print(f"\n{len(model_paths)} SBML models ready in {args.carveme_dir}")
        print(f"\nNext step:")
        print(f"  python scripts/prepare_data.py "
              f"--carveme-dir {args.carveme_dir}")


if __name__ == "__main__":
    main()
