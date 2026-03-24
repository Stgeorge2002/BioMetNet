"""E. coli strain data pipeline using BiGG genome-scale metabolic models.

Downloads full COBRA JSON models for E. coli strains, extracts strain-agnostic
gene features (EC numbers, metabolic subsystems), and generates cross-strain
training data with dropout augmentation.

Gene features are transferable across strains because they describe enzymatic
function (EC class, metabolic subsystem) rather than identity (strain-specific
gene IDs like b0002).
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import requests
import torch

from biometnet.data.ecoli_data import evaluate_gpr
from biometnet.data.bigg_loader import BIGG_API, BIGG_STATIC

# E. coli BiGG model IDs (K-12, B, and pathogenic strains).
# e_coli_core excluded: toy teaching model (137 genes), not representative.
_KNOWN_BIGG_MODELS = [
    "iAF1260", "iAF1260b",
    "iAPECO1_1312", "iB21_1397",
    "iEC1344_C", "iEC1349_Crooks", "iEC1356_Bl21DE3", "iEC1364_W", "iEC55989",
    "iECABU_c1320", "iECB_3114", "iECBD_1354", "iECED1_1282", "iECH74115_1262",
    "iECIAI1_1343", "iECIAI39_1322", "iECNA114_1301", "iECO103_1326",
    "iECO111_1330", "iECO26_1355", "iECO55EA1_1288", "iECOK1_1307",
    "iECsc47_1070", "iECUMN_1333", "iECW_1372", "iECW3110_1372",
    "iEKO11_1354", "iEcDH1_1363", "iEcDH1ME8569_1439", "iEcE24377_1341",
    "iEcHS_1320", "iEcSMS35_1347",
    "iJO1366", "iJR904", "iML1515",
    "iUMN146_1321", "iUMNK88_1353", "iUTI89_1310", "iZ_1308",
]


# ---------------------------------------------------------------------------
# Model downloading
# ---------------------------------------------------------------------------


def fetch_bigg_model_list() -> list[dict]:
    """Fetch list of all available BiGG model IDs.

    Falls back to a hardcoded list if the API is unreachable.
    """
    try:
        resp = requests.get(f"{BIGG_API}/models", timeout=15)
        resp.raise_for_status()
        return resp.json()["results"]
    except Exception as e:
        print(f"  Warning: BiGG API unreachable ({e}), using built-in model list.")
        return [{"bigg_id": mid} for mid in _KNOWN_BIGG_MODELS]


def download_bigg_model_json(
    model_id: str, cache_dir: Path, delay: float = 0.5,
) -> Path:
    """Download one BiGG model as full COBRA JSON."""
    path = cache_dir / f"{model_id}.json"
    if path.exists():
        return path
    url = f"{BIGG_STATIC}/{model_id}.json"
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    path.write_text(resp.text)
    time.sleep(delay)  # polite rate-limiting
    return path


def download_all_bigg_models(
    cache_dir: str | Path = "data/raw/bigg/models",
    max_models: int | None = None,
) -> list[Path]:
    """Download E. coli strain COBRA JSON models from BiGG.

    Returns paths to successfully downloaded model files.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_list = fetch_bigg_model_list()
    if max_models is not None:
        model_list = model_list[:max_models]

    paths = []
    for i, info in enumerate(model_list):
        mid = info["bigg_id"]
        print(f"  [{i+1}/{len(model_list)}] {mid}...", end=" ", flush=True)
        try:
            path = download_bigg_model_json(mid, cache_dir)
            paths.append(path)
            size_kb = path.stat().st_size // 1024
            print(f"OK ({size_kb}KB)")
        except requests.RequestException as e:
            print(f"FAILED ({e})")
    return paths


# ---------------------------------------------------------------------------
# Organism info extraction
# ---------------------------------------------------------------------------


def extract_organism_info(cobra_model) -> dict[str, Any]:
    """Extract genes, reactions, GPR rules, EC numbers, and subsystems."""
    genes = sorted(g.id for g in cobra_model.genes)

    gpr_rules: list[dict] = []
    reaction_ec: dict[str, list[str]] = {}
    reaction_subsystem: dict[str, str] = {}

    for rxn in cobra_model.reactions:
        rule = rxn.gene_reaction_rule.strip()
        if not rule:
            continue
        gpr_rules.append({"id": rxn.id, "gpr": rule})

        # EC numbers from reaction annotation
        ann = getattr(rxn, "annotation", None) or {}
        ec_codes = ann.get("ec-code", ann.get("EC Number", []))
        if isinstance(ec_codes, str):
            ec_codes = [ec_codes]
        reaction_ec[rxn.id] = [ec for ec in ec_codes if ec]

        # Subsystem
        sub = (rxn.subsystem or "").strip()
        if sub:
            reaction_subsystem[rxn.id] = sub

    reactions = sorted({r["id"] for r in gpr_rules})

    return {
        "genes": genes,
        "reactions": reactions,
        "gpr_rules": gpr_rules,
        "reaction_ec": reaction_ec,
        "reaction_subsystem": reaction_subsystem,
    }


# ---------------------------------------------------------------------------
# Feature vocabulary building (across all organisms)
# ---------------------------------------------------------------------------


def _extract_genes_from_gpr(rule: str) -> set[str]:
    tokens = rule.replace("(", " ").replace(")", " ").split()
    return {t for t in tokens if t not in ("and", "or")}


def build_feature_vocabs(organisms: list[dict]) -> dict[str, Any]:
    """Build EC level-2 and subsystem vocabularies.

    EC level-4 was removed: it adds ~966 sparse dimensions that hurt
    generalization with only 34 training organisms.
    """
    ec2_set: set[str] = set()
    sub_set: set[str] = set()

    for org in organisms:
        for ec_list in org["reaction_ec"].values():
            for ec in ec_list:
                parts = ec.split(".")
                if len(parts) >= 2:
                    ec2_set.add(f"{parts[0]}.{parts[1]}")
        for sub in org["reaction_subsystem"].values():
            sub_set.add(sub)

    ec2_vocab = {ec: i for i, ec in enumerate(sorted(ec2_set))}
    sub_vocab = {sub: i for i, sub in enumerate(sorted(sub_set))}
    # features: EC2 multi-hot + subsystem multi-hot + 4 scalars
    n_feat = len(ec2_vocab) + len(sub_vocab) + 4

    return {
        "ec_vocab": ec2_vocab,
        "ec4_vocab": {},
        "subsystem_vocab": sub_vocab,
        "n_ec": len(ec2_vocab),
        "n_ec4": 0,
        "n_subsystem": len(sub_vocab),
        "n_features": n_feat,
    }


def build_universal_reaction_list(
    organisms: list[dict], min_organisms: int = 2,
) -> list[str]:
    """Union of reactions appearing in >=min_organisms models."""
    counts: dict[str, int] = {}
    for org in organisms:
        for rxn in org["reactions"]:
            counts[rxn] = counts.get(rxn, 0) + 1
    return sorted(r for r, c in counts.items() if c >= min_organisms)


# ---------------------------------------------------------------------------
# Gene feature extraction
# ---------------------------------------------------------------------------


def extract_gene_features(
    organism: dict, feature_vocabs: dict,
) -> torch.Tensor:
    """Build (n_genes, n_features) feature matrix for one organism.

    Features per gene:
      - EC level-2 multi-hot (n_ec dims)
      - EC level-4 multi-hot (n_ec4 dims)
      - Subsystem multi-hot (n_sub dims)
      - log(1 + n_reactions) scalar
      - n_reactions / total_reactions fraction
      - n_unique_subsystems / total_subsystems (breadth)
      - is_multi_functional (participates in >1 subsystem)
    """
    genes = organism["genes"]
    gpr_rules = organism["gpr_rules"]
    rxn_ec = organism["reaction_ec"]
    rxn_sub = organism["reaction_subsystem"]
    ec2_vocab = feature_vocabs["ec_vocab"]
    ec4_vocab = feature_vocabs.get("ec4_vocab", {})
    sub_vocab = feature_vocabs["subsystem_vocab"]
    n_ec2 = feature_vocabs["n_ec"]
    n_ec4 = feature_vocabs.get("n_ec4", 0)
    n_sub = feature_vocabs["n_subsystem"]
    n_feat = feature_vocabs["n_features"]

    features = torch.zeros(len(genes), n_feat)

    # Map gene -> reactions it participates in
    gene_rxns: dict[str, list[str]] = {g: [] for g in genes}
    for entry in gpr_rules:
        for gid in _extract_genes_from_gpr(entry["gpr"]):
            if gid in gene_rxns:
                gene_rxns[gid].append(entry["id"])

    total_rxns = max(len(gpr_rules), 1)
    total_subs = max(len(sub_vocab), 1)

    for i, gene in enumerate(genes):
        rxns = gene_rxns[gene]
        offset = 0

        # EC level-2 multi-hot
        for rxn_id in rxns:
            for ec in rxn_ec.get(rxn_id, []):
                parts = ec.split(".")
                if len(parts) >= 2:
                    key = f"{parts[0]}.{parts[1]}"
                    if key in ec2_vocab:
                        features[i, offset + ec2_vocab[key]] = 1.0
        offset += n_ec2

        # EC level-4 multi-hot
        for rxn_id in rxns:
            for ec in rxn_ec.get(rxn_id, []):
                if ec in ec4_vocab:
                    features[i, offset + ec4_vocab[ec]] = 1.0
        offset += n_ec4

        # Subsystem multi-hot
        gene_subs: set[str] = set()
        for rxn_id in rxns:
            sub = rxn_sub.get(rxn_id)
            if sub and sub in sub_vocab:
                features[i, offset + sub_vocab[sub]] = 1.0
                gene_subs.add(sub)
        offset += n_sub

        # Scalar features
        features[i, offset] = math.log1p(len(rxns))
        features[i, offset + 1] = len(rxns) / total_rxns
        features[i, offset + 2] = len(gene_subs) / total_subs
        features[i, offset + 3] = 1.0 if len(gene_subs) > 1 else 0.0

    return features


# ---------------------------------------------------------------------------
# Training sample generation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fast compiled GPR evaluation
# ---------------------------------------------------------------------------


def _compile_gpr(rule: str, gene_to_idx: dict[str, int]):
    """Compile a GPR rule string into a fast evaluator operating on a bool array.

    Returns a callable: (present: np.ndarray[bool]) -> bool
    Parses once, evaluates in O(n_genes_in_rule) with no string ops.
    """
    import numpy as np

    tokens = rule.replace("(", " ( ").replace(")", " ) ").split()
    pos = [0]

    def parse_or():
        left = parse_and()
        while pos[0] < len(tokens) and tokens[pos[0]] == "or":
            pos[0] += 1
            right = parse_and()
            left = ("or", left, right)
        return left

    def parse_and():
        left = parse_atom()
        while pos[0] < len(tokens) and tokens[pos[0]] == "and":
            pos[0] += 1
            right = parse_atom()
            left = ("and", left, right)
        return left

    def parse_atom():
        if tokens[pos[0]] == "(":
            pos[0] += 1
            result = parse_or()
            pos[0] += 1  # skip ")"
            return result
        gene = tokens[pos[0]]
        pos[0] += 1
        return ("gene", gene_to_idx.get(gene, -1))

    tree = parse_or()

    def _eval_tree(node, present: np.ndarray) -> bool:
        if isinstance(node, bool):
            return node
        op = node[0]
        if op == "gene":
            idx = node[1]
            return idx >= 0 and bool(present[idx])
        elif op == "and":
            return _eval_tree(node[1], present) and _eval_tree(node[2], present)
        elif op == "or":
            return _eval_tree(node[1], present) or _eval_tree(node[2], present)
        return False

    return _eval_tree, tree


def _build_subsystem_gene_map(
    organism: dict,
) -> dict[str, list[int]]:
    """Map each subsystem to the gene indices that participate in it."""
    genes = organism["genes"]
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    sub_genes: dict[str, set[int]] = {}
    for entry in organism["gpr_rules"]:
        sub = organism["reaction_subsystem"].get(entry["id"], "")
        if not sub:
            continue
        for gid in _extract_genes_from_gpr(entry["gpr"]):
            if gid in gene_to_idx:
                sub_genes.setdefault(sub, set()).add(gene_to_idx[gid])
    return {s: sorted(gs) for s, gs in sub_genes.items()}


def generate_organism_samples(
    organism: dict,
    universal_reactions: list[str],
    n_samples: int = 200,
    seed: int = 42,
    noise_rate: float = 0.02,
) -> list[dict]:
    """Generate dropout-augmented samples using a mixture of strategies.

    Strategies (proportional allocation):
      - 30% moderate dropout (Beta(2,5) centered ~0.2-0.4)
      - 30% subsystem-level dropout (remove all genes in N random subsystems)
      - 20% block dropout (contiguous genomic segments removed)
      - 20% uniform dropout (original strategy, for diversity)

    Returns list of {presence: ByteTensor, labels: ByteTensor}.
    """
    import numpy as np

    genes = organism["genes"]
    gpr_rules = organism["gpr_rules"]
    n_genes = len(genes)
    uni_idx = {r: i for i, r in enumerate(universal_reactions)}
    n_uni = len(universal_reactions)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    # Pre-compile GPR rules
    compiled_rules: list[tuple[int, Any, Any]] = []
    for entry in gpr_rules:
        if entry["id"] in uni_idx:
            eval_fn, tree = _compile_gpr(entry["gpr"], gene_to_idx)
            compiled_rules.append((uni_idx[entry["id"]], eval_fn, tree))

    # Build subsystem -> gene index map for subsystem dropout
    sub_gene_map = _build_subsystem_gene_map(organism)
    sub_names = list(sub_gene_map.keys())

    rng = np.random.RandomState(seed)

    # Allocate samples to strategies
    n_moderate = int(n_samples * 0.30)
    n_subsystem = int(n_samples * 0.30)
    n_block = int(n_samples * 0.20)
    n_uniform = n_samples - n_moderate - n_subsystem - n_block

    samples = []

    def _make_sample(present: np.ndarray) -> dict:
        # Noise flips
        if noise_rate > 0:
            flip = rng.random(n_genes) < noise_rate
            present_noisy = present ^ flip
        else:
            present_noisy = present
        labels = np.zeros(n_uni, dtype=np.uint8)
        for rxn_idx, eval_fn, tree in compiled_rules:
            if eval_fn(tree, present_noisy):
                labels[rxn_idx] = 1
        return {
            "presence": torch.from_numpy(present_noisy.astype(np.uint8)),
            "labels": torch.from_numpy(labels),
        }

    # Strategy 1: Moderate dropout via Beta(2, 5) — peaks around 0.25
    for _ in range(n_moderate):
        dropout = rng.beta(2, 5)
        present = rng.random(n_genes) > dropout
        samples.append(_make_sample(present))

    # Strategy 2: Subsystem-level dropout — remove entire metabolic subsystems
    for _ in range(n_subsystem):
        if not sub_names:
            # Fallback to moderate if no subsystem info
            dropout = rng.beta(2, 5)
            present = rng.random(n_genes) > dropout
        else:
            present = np.ones(n_genes, dtype=bool)
            # Remove 1 to 40% of subsystems
            n_remove = max(1, rng.randint(1, max(2, len(sub_names) * 2 // 5)))
            removed = rng.choice(len(sub_names), size=min(n_remove, len(sub_names)), replace=False)
            for si in removed:
                for gi in sub_gene_map[sub_names[si]]:
                    present[gi] = False
            # Light additional dropout on remaining genes
            light_drop = rng.random(n_genes) < 0.05
            present = present & ~light_drop
        samples.append(_make_sample(present))

    # Strategy 3: Block dropout — contiguous genomic segments
    for _ in range(n_block):
        present = np.ones(n_genes, dtype=bool)
        # Remove 1-3 contiguous blocks
        n_blocks = rng.randint(1, 4)
        for _ in range(n_blocks):
            block_len = rng.randint(n_genes // 20, n_genes // 4 + 1)
            start = rng.randint(0, max(1, n_genes - block_len))
            present[start:start + block_len] = False
        samples.append(_make_sample(present))

    # Strategy 4: Uniform dropout (original, for diversity at extremes)
    for _ in range(n_uniform):
        dropout = rng.random()
        present = rng.random(n_genes) > dropout
        samples.append(_make_sample(present))

    # Shuffle to mix strategies within batches
    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Reaction feature builder (for A4: metadata-initialized queries)
# ---------------------------------------------------------------------------


def build_reaction_features(
    organisms: list[dict],
    universal_reactions: list[str],
    feature_vocabs: dict[str, Any],
) -> torch.Tensor:
    """Build (n_reactions, n_rxn_features) metadata tensor for query init.

    Per-reaction features:
      - EC level-2 multi-hot (n_ec dims)
      - Subsystem multi-hot (n_sub dims)
      - log(1 + n_organisms_with_reaction) scalar
      - mean_gene_count_per_organism scalar
    """
    ec_vocab = feature_vocabs["ec_vocab"]
    sub_vocab = feature_vocabs["subsystem_vocab"]
    n_ec = len(ec_vocab)
    n_sub = len(sub_vocab)
    n_feat = n_ec + n_sub + 2  # +2 scalars
    n_rxns = len(universal_reactions)

    features = torch.zeros(n_rxns, n_feat)
    rxn_to_idx = {r: i for i, r in enumerate(universal_reactions)}

    # Aggregate EC/subsystem info across all organisms
    rxn_org_count = torch.zeros(n_rxns)
    rxn_gene_count = torch.zeros(n_rxns)

    for org in organisms:
        org_rxns = set(org["reactions"])
        gene_to_idx = {g: i for i, g in enumerate(org["genes"])}

        for entry in org["gpr_rules"]:
            rid = entry["id"]
            if rid not in rxn_to_idx:
                continue
            ri = rxn_to_idx[rid]
            rxn_org_count[ri] += 1
            n_genes = len(_extract_genes_from_gpr(entry["gpr"]))
            rxn_gene_count[ri] += n_genes

        for rid, ec_list in org["reaction_ec"].items():
            if rid not in rxn_to_idx:
                continue
            ri = rxn_to_idx[rid]
            for ec in ec_list:
                parts = ec.split(".")
                if len(parts) >= 2:
                    key = f"{parts[0]}.{parts[1]}"
                    if key in ec_vocab:
                        features[ri, ec_vocab[key]] = 1.0

        for rid, sub in org["reaction_subsystem"].items():
            if rid not in rxn_to_idx:
                continue
            ri = rxn_to_idx[rid]
            if sub in sub_vocab:
                features[ri, n_ec + sub_vocab[sub]] = 1.0

    # Scalar features
    features[:, n_ec + n_sub] = torch.log1p(rxn_org_count)
    mean_genes = rxn_gene_count / rxn_org_count.clamp(min=1)
    features[:, n_ec + n_sub + 1] = mean_genes

    return features


# ---------------------------------------------------------------------------
# Stratified splitting by reaction-set dissimilarity
# ---------------------------------------------------------------------------


def _stratified_organism_split(
    organisms: list[dict],
    n_org: int,
    test_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[set[int], set[int], set[int]]:
    """Split organisms so that similar strains stay in the same split.

    Greedy farthest-first traversal picks test/val organisms that are
    maximally dissimilar to each other, reducing data leakage from
    near-duplicate strains.
    """
    import numpy as np

    # Build Jaccard distance matrix from reaction sets
    rxn_sets = [set(o["reactions"]) for o in organisms]
    dist = np.zeros((n_org, n_org), dtype=np.float32)
    for i in range(n_org):
        for j in range(i + 1, n_org):
            inter = len(rxn_sets[i] & rxn_sets[j])
            union = len(rxn_sets[i] | rxn_sets[j])
            d = 1.0 - (inter / union) if union > 0 else 1.0
            dist[i, j] = dist[j, i] = d

    n_test = max(1, int(n_org * test_frac))
    n_val = max(1, int(n_org * val_frac))
    n_pick = n_test + n_val

    rng = np.random.RandomState(seed)

    # Farthest-first traversal to pick maximally diverse eval organisms
    picked: list[int] = [rng.randint(0, n_org)]
    remaining = set(range(n_org)) - {picked[0]}
    for _ in range(n_pick - 1):
        if not remaining:
            break
        # For each remaining, compute min distance to any picked organism
        min_dists = {r: min(dist[r, p] for p in picked) for r in remaining}
        farthest = max(min_dists, key=min_dists.get)
        picked.append(farthest)
        remaining.discard(farthest)

    test_idx = set(picked[:n_test])
    val_idx = set(picked[n_test:n_test + n_val])
    train_idx = set(range(n_org)) - test_idx - val_idx
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Model loading (format-agnostic)
# ---------------------------------------------------------------------------


def _load_cobra_model(path: Path):
    """Load a COBRA model from JSON or SBML, detected by file extension."""
    import cobra

    suffix = path.suffix.lower()
    name = path.name.lower()

    if suffix == ".json":
        return cobra.io.load_json_model(str(path))
    elif suffix == ".xml" or name.endswith(".xml.gz") or suffix == ".sbml":
        return cobra.io.read_sbml_model(str(path))
    else:
        # Try JSON first, fall back to SBML
        try:
            return cobra.io.load_json_model(str(path))
        except Exception:
            return cobra.io.read_sbml_model(str(path))


# ---------------------------------------------------------------------------
# Full dataset preparation
# ---------------------------------------------------------------------------


def prepare_strain_dataset(
    model_paths: list[Path],
    out_dir: str | Path = "data/processed/ecoli_strains",
    samples_per_train_org: int = 1000,
    samples_per_eval_org: int = 200,
    min_rxn_organisms: int = 2,
    test_frac: float = 0.1,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    """Full pipeline: load models -> features -> samples -> save.

    Splits E. coli strains into train/val/test for cross-strain generalization.
    """
    import cobra

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: Load all organisms, extract info
    print("Loading COBRA models...")
    organisms: list[dict] = []
    model_ids: list[str] = []
    for i, path in enumerate(model_paths):
        mid = path.stem
        print(f"  [{i+1}/{len(model_paths)}] {mid}...", end=" ", flush=True)
        try:
            model = _load_cobra_model(path)
            info = extract_organism_info(model)
            del model  # free memory
            if len(info["genes"]) >= 200 and len(info["gpr_rules"]) >= 50:
                organisms.append(info)
                model_ids.append(mid)
                print(f"{len(info['genes'])} genes, "
                      f"{len(info['reactions'])} rxns")
            else:
                print("SKIPPED (too few genes/reactions)")
        except Exception as e:
            print(f"FAILED ({type(e).__name__}: {e})")

    n_org = len(organisms)
    print(f"\nLoaded {n_org} organisms")

    # Build vocabularies
    print("\nBuilding feature vocabularies...")
    feature_vocabs = build_feature_vocabs(organisms)
    uni_rxns = build_universal_reaction_list(
        organisms, min_organisms=min_rxn_organisms,
    )
    print(f"  EC level-2 classes: {feature_vocabs['n_ec']}")
    print(f"  EC level-4 classes: {feature_vocabs['n_ec4']}")
    print(f"  Subsystems: {feature_vocabs['n_subsystem']}")
    print(f"  Features per gene: {feature_vocabs['n_features']}")
    print(f"  Universal reactions (>={min_rxn_organisms} orgs): {len(uni_rxns)}")

    # Split organisms by reaction-set dissimilarity so train/val/test
    # contain maximally diverse strains (no near-duplicates leaking).
    train_idx, val_idx, test_idx = _stratified_organism_split(
        organisms, n_org, test_frac, val_frac, seed,
    )

    print(f"\nOrganism splits: {len(train_idx)} train, "
          f"{len(val_idx)} val, {len(test_idx)} test")

    # Pass 2: Extract features and generate samples
    print("\nGenerating samples...")
    org_features: list[torch.Tensor] = []
    org_n_genes: list[int] = []

    # Pre-compute features (fast)
    for idx in range(n_org):
        feats = extract_gene_features(organisms[idx], feature_vocabs)
        org_features.append(feats)
        org_n_genes.append(len(organisms[idx]["genes"]))

    # Build task list for sample generation
    tasks: list[tuple[int, str, int, int]] = []  # (idx, split, n_samp, seed)
    for idx in range(n_org):
        if idx in train_idx:
            tasks.append((idx, "train", samples_per_train_org, seed + idx))
        elif idx in val_idx:
            tasks.append((idx, "val", samples_per_eval_org, seed + idx + 10000))
        else:
            tasks.append((idx, "test", samples_per_eval_org, seed + idx + 20000))

    split_data: dict[str, dict[str, list]] = {
        "train": {"org_idx": [], "presence": [], "labels": []},
        "val": {"org_idx": [], "presence": [], "labels": []},
        "test": {"org_idx": [], "presence": [], "labels": []},
    }

    for idx, split, n_samp, s_seed in tasks:
        samples = generate_organism_samples(
            organisms[idx], uni_rxns, n_samples=n_samp, seed=s_seed,
        )
        for s in samples:
            split_data[split]["org_idx"].append(idx)
            split_data[split]["presence"].append(s["presence"])
            split_data[split]["labels"].append(s["labels"])
        print(f"  {model_ids[idx]}: {n_samp} {split} samples "
              f"({len(organisms[idx]['genes'])} genes, "
              f"{len(organisms[idx]['reactions'])} rxns)")

    # Save organism features (use weights_only=False compatible format)
    torch.save({
        "features": org_features,
        "n_genes": torch.tensor(org_n_genes, dtype=torch.int32),
    }, out_dir / "organism_features.pt")

    # Build and save reaction metadata features (for A4: query initialization)
    rxn_features = build_reaction_features(organisms, uni_rxns, feature_vocabs)
    torch.save(rxn_features, out_dir / "reaction_features.pt")
    print(f"  Reaction features: {rxn_features.shape}")

    # Save split data as padded tensors
    for split_name, data in split_data.items():
        n = len(data["org_idx"])
        if n == 0:
            continue

        max_g = max(data["presence"][i].shape[0] for i in range(n))
        padded_presence = torch.zeros(n, max_g, dtype=torch.uint8)
        for i in range(n):
            ng = data["presence"][i].shape[0]
            padded_presence[i, :ng] = data["presence"][i]

        torch.save({
            "organism_idx": torch.tensor(data["org_idx"], dtype=torch.long),
            "presence": padded_presence,
            "labels": torch.stack(data["labels"]),
            "n_genes": torch.tensor(
                [org_n_genes[oi] for oi in data["org_idx"]],
                dtype=torch.int32,
            ),
        }, out_dir / f"{split_name}.pt")
        print(f"\n  {split_name}: {n} samples saved")

    # Save config
    config = {
        "n_features": feature_vocabs["n_features"],
        "n_ec": feature_vocabs["n_ec"],
        "n_ec4": feature_vocabs["n_ec4"],
        "n_subsystem": feature_vocabs["n_subsystem"],
        "n_universal_reactions": len(uni_rxns),
        "n_organisms": n_org,
        "n_train_organisms": len(train_idx),
        "n_val_organisms": len(val_idx),
        "n_test_organisms": len(test_idx),
        "d_model": 256,
        "n_heads": 8,
        "n_encoder_layers": 2,
        "n_cross_layers": 2,
        "n_self_layers": 1,
        "ff_dim": 512,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "universal_reactions.json").write_text(
        json.dumps(uni_rxns, indent=2),
    )
    (out_dir / "feature_vocabs.json").write_text(json.dumps({
        "ec_vocab": feature_vocabs["ec_vocab"],
        "ec4_vocab": feature_vocabs["ec4_vocab"],
        "subsystem_vocab": feature_vocabs["subsystem_vocab"],
    }, indent=2))

    # Save organism summary
    org_summary = []
    for i in range(n_org):
        split = ("train" if i in train_idx
                 else ("val" if i in val_idx else "test"))
        org_summary.append({
            "model_id": model_ids[i],
            "n_genes": org_n_genes[i],
            "n_reactions": len(organisms[i]["reactions"]),
            "split": split,
        })
    (out_dir / "organisms.json").write_text(json.dumps(org_summary, indent=2))

    print(f"\nAll data saved to {out_dir}")
    return config
