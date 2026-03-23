"""Download and parse metabolic models from the BiGG Models database."""
from __future__ import annotations

import json
from pathlib import Path

import requests


BIGG_API = "https://bigg.ucsd.edu/api/v2"


def fetch_model_list() -> list[dict]:
    """Fetch the list of all models from BiGG."""
    resp = requests.get(f"{BIGG_API}/models", timeout=30)
    resp.raise_for_status()
    return resp.json()["results"]


def fetch_model_reactions(model_id: str) -> list[dict]:
    """Fetch reaction list for a specific model."""
    resp = requests.get(f"{BIGG_API}/models/{model_id}/reactions", timeout=30)
    resp.raise_for_status()
    return resp.json()["results"]


def fetch_model_genes(model_id: str) -> list[dict]:
    """Fetch gene list for a specific model."""
    resp = requests.get(f"{BIGG_API}/models/{model_id}/genes", timeout=30)
    resp.raise_for_status()
    return resp.json()["results"]


def download_bigg_models(
    cache_dir: str | Path = "data/raw/bigg",
    max_models: int | None = None,
) -> list[dict]:
    """Download BiGG models and return genome-reaction pairs.

    Each entry: {model_id, gene_ids, reaction_ids}.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached = cache_dir / "models.json"
    if cached.exists():
        return json.loads(cached.read_text())

    model_list = fetch_model_list()
    if max_models is not None:
        model_list = model_list[:max_models]

    results = []
    for info in model_list:
        mid = info["bigg_id"]
        try:
            reactions = fetch_model_reactions(mid)
            genes = fetch_model_genes(mid)
            results.append({
                "model_id": mid,
                "gene_ids": sorted(g["bigg_id"] for g in genes),
                "reaction_ids": sorted(r["bigg_id"] for r in reactions),
            })
        except requests.RequestException:
            continue  # skip models that fail to download

    cached.write_text(json.dumps(results, indent=2))
    return results
