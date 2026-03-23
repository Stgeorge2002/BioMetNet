"""E. coli metabolic model data pipeline using BiGG iML1515."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import requests


from biometnet.data.bigg_loader import BIGG_STATIC

BIGG_MODEL_URL = f"{BIGG_STATIC}/iML1515.json"

NCBI_ECOLI_GFF_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/"
    "GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.gff.gz"
)


def download_ecoli_model(cache_dir: str | Path = "data/raw/ecoli") -> Path:
    """Download iML1515 JSON from BiGG."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "iML1515.json"
    if path.exists():
        return path
    print(f"  Fetching {BIGG_MODEL_URL} ...")
    resp = requests.get(BIGG_MODEL_URL, timeout=120)
    resp.raise_for_status()
    path.write_text(resp.text)
    return path


def download_reference_gff(cache_dir: str | Path = "data/raw/ecoli") -> Path:
    """Download E. coli K-12 MG1655 reference GFF from NCBI."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "ecoli_k12_mg1655.gff.gz"
    if path.exists():
        return path
    print(f"  Fetching reference GFF from NCBI ...")
    resp = requests.get(NCBI_ECOLI_GFF_URL, timeout=120)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def load_cobra_model(json_path: str | Path):
    """Load a BiGG JSON model with COBRApy."""
    import cobra
    return cobra.io.load_json_model(str(json_path))


def extract_model_info(model) -> dict[str, Any]:
    """Extract genes, reactions, and GPR rules from a cobra model."""
    genes = sorted(g.id for g in model.genes)

    # Build gene name -> gene id mapping (e.g. "thrA" -> "b0002")
    gene_names: dict[str, str] = {}
    for g in model.genes:
        if g.name and g.name != g.id:
            gene_names[g.name] = g.id

    # Only include reactions that have GPR rules (enzyme-catalyzed)
    gpr_rules: list[dict[str, str]] = []
    for rxn in model.reactions:
        rule = rxn.gene_reaction_rule.strip()
        if rule:
            gpr_rules.append({"id": rxn.id, "gpr": rule})

    reaction_ids = sorted(set(r["id"] for r in gpr_rules))

    return {
        "genes": genes,
        "gene_names": gene_names,
        "reactions": reaction_ids,
        "gpr_rules": gpr_rules,
    }


def extract_pathway_defs(model) -> list[dict]:
    """Extract metabolic subsystem/pathway definitions from a cobra model.

    Groups reactions by their subsystem annotation. Only includes reactions
    that have GPR rules (enzyme-catalyzed) so pathway defs align with
    the reaction space used for training.
    """
    subsystems: dict[str, list[str]] = {}
    for rxn in model.reactions:
        rule = rxn.gene_reaction_rule.strip()
        if rule and rxn.subsystem:
            sub = rxn.subsystem.strip()
            if sub:
                subsystems.setdefault(sub, []).append(rxn.id)

    pathway_defs = []
    for name, reactions in sorted(subsystems.items()):
        pathway_defs.append({
            "name": name,
            "reactions": sorted(reactions),
        })
    return pathway_defs


# ---------------------------------------------------------------------------
# Safe GPR rule evaluation (recursive descent parser, no eval())
# ---------------------------------------------------------------------------

def _tokenize_gpr(rule: str) -> list[str]:
    return rule.replace("(", " ( ").replace(")", " ) ").split()


def _parse_or(tokens: list[str], pos: list[int], present: set[str]) -> bool:
    left = _parse_and(tokens, pos, present)
    while pos[0] < len(tokens) and tokens[pos[0]] == "or":
        pos[0] += 1
        right = _parse_and(tokens, pos, present)
        left = left or right
    return left


def _parse_and(tokens: list[str], pos: list[int], present: set[str]) -> bool:
    left = _parse_atom(tokens, pos, present)
    while pos[0] < len(tokens) and tokens[pos[0]] == "and":
        pos[0] += 1
        right = _parse_atom(tokens, pos, present)
        left = left and right
    return left


def _parse_atom(tokens: list[str], pos: list[int], present: set[str]) -> bool:
    if tokens[pos[0]] == "(":
        pos[0] += 1  # skip "("
        result = _parse_or(tokens, pos, present)
        pos[0] += 1  # skip ")"
        return result
    gene_id = tokens[pos[0]]
    pos[0] += 1
    return gene_id in present


def evaluate_gpr(rule: str, present_genes: set[str]) -> bool:
    """Safely evaluate a GPR boolean expression without eval()."""
    if not rule.strip():
        return False
    tokens = _tokenize_gpr(rule)
    return _parse_or(tokens, [0], present_genes)


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------


def _group_genes_into_blocks(genes: list[str], block_size: int = 5) -> list[list[int]]:
    """Group genes into blocks by genomic proximity (b-number ordering).

    Adjacent genes (similar b-numbers) are often in the same operon
    and should be dropped/retained together during data generation.
    """
    indexed = []
    for i, g in enumerate(genes):
        if g.startswith("b") and g[1:].isdigit():
            sort_key = int(g[1:])
        else:
            sort_key = 999_999 + i
        indexed.append((sort_key, i))
    indexed.sort(key=lambda x: x[0])

    blocks: list[list[int]] = []
    current: list[int] = []
    for _, gene_idx in indexed:
        current.append(gene_idx)
        if len(current) >= block_size:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _extract_genes_from_gpr(gpr_rule: str) -> set[str]:
    """Extract all gene IDs mentioned in a GPR rule string."""
    tokens = gpr_rule.replace("(", " ").replace(")", " ").split()
    return {t for t in tokens if t not in ("and", "or")}


def _build_pathway_gene_sets(
    pathway_defs: list[dict],
    gpr_rules: list[dict],
) -> dict[str, set[str]]:
    """Map each pathway to the set of genes involved in its reactions."""
    rxn_genes: dict[str, set[str]] = {}
    for entry in gpr_rules:
        rxn_genes[entry["id"]] = _extract_genes_from_gpr(entry["gpr"])

    result: dict[str, set[str]] = {}
    for pw in pathway_defs:
        genes: set[str] = set()
        for rxn_id in pw["reactions"]:
            if rxn_id in rxn_genes:
                genes.update(rxn_genes[rxn_id])
        result[pw["name"]] = genes
    return result


def _evaluate_active_reactions(
    gpr_rules: list[dict], present: set[str],
) -> list[str]:
    """Evaluate GPR rules and return sorted list of active reaction IDs."""
    return sorted({
        entry["id"] for entry in gpr_rules
        if evaluate_gpr(entry["gpr"], present)
    })


def _apply_noise(
    genome: list[int], genes: list[str], present: set[str],
    rng: random.Random, noise_rate: float,
) -> None:
    """Flip a small fraction of genes randomly (in-place)."""
    for i in range(len(genes)):
        if rng.random() < noise_rate:
            if genome[i] == 1:
                genome[i] = 0
                present.discard(genes[i])
            else:
                genome[i] = 1
                present.add(genes[i])


def _pathway_dropout_sample(
    genes: list[str],
    gpr_rules: list[dict],
    pathway_gene_sets: dict[str, set[str]],
    rng: random.Random,
    noise_rate: float,
) -> dict:
    """Generate a sample by dropping entire metabolic pathways.

    Drops 1 to N-1 pathways. All genes associated with dropped pathways
    are removed, creating biologically coherent pathway absence patterns.
    """
    pathway_names = list(pathway_gene_sets.keys())
    if len(pathway_names) <= 1:
        # Can't drop pathways if there are 0 or 1 — return full genome
        present = set(genes)
        genome = [1] * len(genes)
        _apply_noise(genome, genes, present, rng, noise_rate)
        return {"genome": genome, "reaction_ids": _evaluate_active_reactions(gpr_rules, present)}
    n_drop = rng.randint(1, len(pathway_names) - 1)
    to_drop = set(rng.sample(pathway_names, n_drop))

    dropped_genes: set[str] = set()
    for pw_name in to_drop:
        dropped_genes.update(pathway_gene_sets[pw_name])

    gene_set = set(genes)
    present = gene_set - dropped_genes
    genome = [1 if g in present else 0 for g in genes]
    _apply_noise(genome, genes, present, rng, noise_rate)
    return {"genome": genome, "reaction_ids": _evaluate_active_reactions(gpr_rules, present)}


def _independent_dropout_sample(
    genes: list[str],
    gpr_rules: list[dict],
    rng: random.Random,
    noise_rate: float,
) -> dict:
    """Generate a sample with independent per-gene dropout.

    Each gene is independently dropped with probability `dropout`.
    More effective than block dropout at disabling OR-gated reactions
    because it doesn't respect operon structure.
    """
    dropout = rng.random()  # uniform 0-100% dropout
    present: set[str] = set()
    genome: list[int] = [0] * len(genes)
    for i, g in enumerate(genes):
        if rng.random() > dropout:
            present.add(g)
            genome[i] = 1
    _apply_noise(genome, genes, present, rng, noise_rate)
    return {"genome": genome, "reaction_ids": _evaluate_active_reactions(gpr_rules, present)}


def _block_dropout_sample(
    genes: list[str],
    gpr_rules: list[dict],
    blocks: list[list[int]],
    rng: random.Random,
    noise_rate: float,
) -> dict:
    """Generate a sample with operon-correlated block dropout.

    Uses aggressive dropout tiers biased toward high dropout to ensure
    the pool covers the full range of reaction-activity levels.
    """
    r = rng.random()
    if r < 0.05:
        dropout = rng.uniform(0.00, 0.10)
    elif r < 0.10:
        dropout = rng.uniform(0.10, 0.30)
    elif r < 0.20:
        dropout = rng.uniform(0.30, 0.50)
    elif r < 0.35:
        dropout = rng.uniform(0.50, 0.70)
    elif r < 0.55:
        dropout = rng.uniform(0.70, 0.85)
    elif r < 0.75:
        dropout = rng.uniform(0.85, 0.95)
    elif r < 0.90:
        dropout = rng.uniform(0.95, 0.99)
    else:
        dropout = rng.uniform(0.99, 1.00)

    present: set[str] = set()
    genome: list[int] = [0] * len(genes)
    for block in blocks:
        if rng.random() > dropout:
            for idx in block:
                present.add(genes[idx])
                genome[idx] = 1

    _apply_noise(genome, genes, present, rng, noise_rate)
    return {"genome": genome, "reaction_ids": _evaluate_active_reactions(gpr_rules, present)}


def generate_ecoli_training_data(
    model_info: dict,
    n_samples: int = 30000,
    seed: int = 42,
    block_size: int = 5,
    noise_rate: float = 0.02,
    pathway_defs: list[dict] | None = None,
) -> list[dict]:
    """Generate training data using mixed dropout strategies.

    Three complementary strategies produce diverse reaction-count distributions:
    - Pathway-coherent dropout (30%): Drop entire metabolic subsystems
    - Independent gene dropout (30%): Per-gene random, hits OR-gated reactions
    - Block-level dropout (40%): Operon-correlated, aggressive tiers

    If pathway_defs is not provided, falls back to block + independent only.
    """
    genes = model_info["genes"]
    gpr_rules = model_info["gpr_rules"]
    rng = random.Random(seed)
    blocks = _group_genes_into_blocks(genes, block_size=block_size)

    pathway_gene_sets = None
    if pathway_defs:
        pathway_gene_sets = _build_pathway_gene_sets(pathway_defs, gpr_rules)

    samples = []
    for _ in range(n_samples):
        strategy = rng.random()

        if pathway_gene_sets and strategy < 0.30:
            sample = _pathway_dropout_sample(
                genes, gpr_rules, pathway_gene_sets, rng, noise_rate,
            )
        elif strategy < 0.60:
            sample = _independent_dropout_sample(
                genes, gpr_rules, rng, noise_rate,
            )
        else:
            sample = _block_dropout_sample(
                genes, gpr_rules, blocks, rng, noise_rate,
            )
        samples.append(sample)

    return samples


def resample_by_reaction_count(
    samples: list[dict],
    n_target: int,
    n_bins: int = 10,
    seed: int = 42,
) -> list[dict]:
    """Resample to get uniform distribution of active-reaction counts.

    Bins samples by number of active reactions and draws equally from each bin.
    Bins with too few samples are upsampled (with replacement).
    """
    rng = random.Random(seed + 1)
    rxn_counts = [len(s["reaction_ids"]) for s in samples]
    min_rxn = min(rxn_counts)
    max_rxn = max(rxn_counts) + 1
    bin_width = (max_rxn - min_rxn) / n_bins

    bins: dict[int, list[dict]] = {}
    for sample, count in zip(samples, rxn_counts):
        b = min(int((count - min_rxn) / bin_width), n_bins - 1)
        bins.setdefault(b, []).append(sample)

    per_bin = n_target // max(len(bins), 1)
    balanced: list[dict] = []
    for b in sorted(bins):
        pool = bins[b]
        if len(pool) >= per_bin:
            balanced.extend(rng.sample(pool, per_bin))
        else:
            # Upsample: take all + duplicate with replacement
            balanced.extend(pool)
            balanced.extend(rng.choices(pool, k=per_bin - len(pool)))

    # Fill any rounding shortfall
    while len(balanced) < n_target:
        b = rng.choice(list(bins.keys()))
        balanced.append(rng.choice(bins[b]))

    rng.shuffle(balanced)
    return balanced[:n_target]


def save_ecoli_data(samples: list[dict], path: str | Path, compact: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        path.write_text(json.dumps(samples, separators=(",", ":")))
    else:
        path.write_text(json.dumps(samples, indent=2))


def load_ecoli_data(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())
