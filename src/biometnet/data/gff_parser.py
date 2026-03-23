"""Parse GFF3 annotation files to extract gene information."""
from __future__ import annotations

import builtins
from pathlib import Path


def parse_gff_genes(gff_path: str | Path) -> dict[str, set[str]]:
    """Parse a GFF3 file and extract gene identifiers.

    Returns dict with keys: 'locus_tags', 'gene_names', 'old_locus_tags'
    """
    locus_tags: set[str] = set()
    gene_names: set[str] = set()
    old_locus_tags: set[str] = set()

    path = Path(gff_path)
    opener = _get_opener(path)

    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            feature_type = parts[2]
            if feature_type not in ("gene", "CDS"):
                continue
            attrs = _parse_attributes(parts[8])
            if "locus_tag" in attrs:
                locus_tags.add(attrs["locus_tag"])
            if "gene" in attrs:
                gene_names.add(attrs["gene"])
            if "Name" in attrs:
                gene_names.add(attrs["Name"])
            if "old_locus_tag" in attrs:
                for tag in attrs["old_locus_tag"].split(","):
                    old_locus_tags.add(tag.strip())

    return {
        "locus_tags": locus_tags,
        "gene_names": gene_names,
        "old_locus_tags": old_locus_tags,
    }


def map_gff_to_model_genes(
    gff_genes: dict[str, set[str]],
    model_genes: list[str],
    gene_name_map: dict[str, str] | None = None,
) -> set[str]:
    """Map GFF gene identifiers to BiGG model gene IDs.

    Tries matching by: locus_tag, old_locus_tag, then gene name.
    """
    model_gene_set = set(model_genes)
    matched: set[str] = set()

    # Direct locus_tag match (b-numbers for E. coli)
    for tag in gff_genes["locus_tags"]:
        if tag in model_gene_set:
            matched.add(tag)

    # Old locus tag match
    for tag in gff_genes["old_locus_tags"]:
        if tag in model_gene_set:
            matched.add(tag)

    # Gene name match via name-to-id mapping
    if gene_name_map:
        for name in gff_genes["gene_names"]:
            if name in gene_name_map:
                matched.add(gene_name_map[name])

    return matched


def _get_opener(path: Path):
    if path.suffix == ".gz":
        import gzip
        return gzip.open
    return builtins.open


def _parse_attributes(attr_str: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in attr_str.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key.strip()] = value.strip()
    return result
