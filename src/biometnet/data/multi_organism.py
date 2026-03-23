"""Multi-organism data pipeline using BiGG genome-scale metabolic models.

Downloads full COBRA JSON models for ~108 organisms, extracts organism-agnostic
gene features (EC numbers, metabolic subsystems), and generates cross-organism
training data with dropout augmentation.

Gene features are transferable across organisms because they describe enzymatic
function (EC class, metabolic subsystem) rather than identity (organism-specific
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

BIGG_API = "https://bigg.ucsd.edu/api/v2"
BIGG_STATIC = "http://bigg.ucsd.edu/static/models"

# Hardcoded fallback: all known BiGG model IDs as of 2024.
# Used when the API is unreachable (e.g. network restrictions on cloud pods).
_KNOWN_BIGG_MODELS = [
    "e_coli_core", "iAF1260", "iAF1260b", "iAF692", "iAM_Pb448", "iAM_Pk459",
    "iAM_Pf480", "iAM_Pv461", "iAM_Pf480", "iAPECO1_1312", "iB21_1397",
    "iBsu1103", "iCN718", "iCN900", "iEC1344_C", "iEC1349_Crooks",
    "iEC1356_Bl21DE3", "iEC1364_W", "iEC55989", "iECABU_c1320", "iECB_3114",
    "iECBD_1354", "iECED1_1282", "iECH74115_1262", "iECIAI1_1343",
    "iECIAI39_1322", "iECNA114_1301", "iECO103_1326", "iECO111_1330",
    "iECO26_1355", "iECO55EA1_1288", "iECOK1_1307", "iECsc47_1070",
    "iECUMN_1333", "iECW_1372", "iECW3110_1372", "iEKO11_1354", "iEcDH1_1363",
    "iEcDH1ME8569_1439", "iEcE24377_1341", "iEcHS_1320", "iEcSMS35_1347",
    "iHN637", "iIT341", "iJN678", "iJN746", "iJO1366", "iJR904",
    "iLB1027_lipid", "iLJ478", "iML1515", "iMM1415", "iMM904", "iND750",
    "iNF517", "iNJ661", "iNJ661m", "iNJ661v", "iPC815", "iPP668",
    "iPS189_WT", "iRC1080", "iRsp1095", "iS_1188", "iSB619", "iSF1195",
    "iSbBS512_1146", "iSynCJ816", "iUMN146_1321", "iUMNK88_1353", "iUTI89_1310",
    "iVS941_malaria", "iYL1228", "iYO844", "iYS1720", "iZ_1308",
    "STM_v1_0", "Recon1", "Recon3D", "AGORA_Lactobacillus_reuteri_JCM_1112",
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
    """Download all BiGG COBRA JSON models.

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
    """Build EC level-2 and subsystem vocabularies from all organisms."""
    ec_set: set[str] = set()
    sub_set: set[str] = set()

    for org in organisms:
        for ec_list in org["reaction_ec"].values():
            for ec in ec_list:
                parts = ec.split(".")
                if len(parts) >= 2:
                    ec_set.add(f"{parts[0]}.{parts[1]}")
        for sub in org["reaction_subsystem"].values():
            sub_set.add(sub)

    ec_vocab = {ec: i for i, ec in enumerate(sorted(ec_set))}
    sub_vocab = {sub: i for i, sub in enumerate(sorted(sub_set))}
    n_feat = len(ec_vocab) + len(sub_vocab) + 2  # +2 scalar features

    return {
        "ec_vocab": ec_vocab,
        "subsystem_vocab": sub_vocab,
        "n_ec": len(ec_vocab),
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
      - Subsystem multi-hot (n_sub dims)
      - log(1 + n_reactions) scalar
      - n_reactions / total_reactions fraction
    """
    genes = organism["genes"]
    gpr_rules = organism["gpr_rules"]
    rxn_ec = organism["reaction_ec"]
    rxn_sub = organism["reaction_subsystem"]
    ec_vocab = feature_vocabs["ec_vocab"]
    sub_vocab = feature_vocabs["subsystem_vocab"]
    n_ec = feature_vocabs["n_ec"]
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

    for i, gene in enumerate(genes):
        rxns = gene_rxns[gene]

        # EC level-2 multi-hot
        for rxn_id in rxns:
            for ec in rxn_ec.get(rxn_id, []):
                parts = ec.split(".")
                if len(parts) >= 2:
                    key = f"{parts[0]}.{parts[1]}"
                    if key in ec_vocab:
                        features[i, ec_vocab[key]] = 1.0

        # Subsystem multi-hot
        for rxn_id in rxns:
            sub = rxn_sub.get(rxn_id)
            if sub and sub in sub_vocab:
                features[i, n_ec + sub_vocab[sub]] = 1.0

        # Scalars
        features[i, n_ec + n_sub] = math.log1p(len(rxns))
        features[i, n_ec + n_sub + 1] = len(rxns) / total_rxns

    return features


# ---------------------------------------------------------------------------
# Training sample generation
# ---------------------------------------------------------------------------


def generate_organism_samples(
    organism: dict,
    universal_reactions: list[str],
    n_samples: int = 200,
    seed: int = 42,
    noise_rate: float = 0.02,
) -> list[dict]:
    """Generate dropout-augmented samples for one organism.

    Returns list of {presence: ByteTensor, labels: ByteTensor}.
    Gene features are gated by presence at DataLoader time, not here.
    """
    genes = organism["genes"]
    gpr_rules = organism["gpr_rules"]
    rng = random.Random(seed)
    uni_idx = {r: i for i, r in enumerate(universal_reactions)}
    n_uni = len(universal_reactions)

    samples = []
    for _ in range(n_samples):
        dropout = rng.random()

        presence = torch.zeros(len(genes), dtype=torch.uint8)
        present: set[str] = set()
        for i, g in enumerate(genes):
            if rng.random() > dropout:
                presence[i] = 1
                present.add(g)

        # Noise flips
        for i in range(len(genes)):
            if rng.random() < noise_rate:
                if presence[i]:
                    presence[i] = 0
                    present.discard(genes[i])
                else:
                    presence[i] = 1
                    present.add(genes[i])

        # Evaluate active reactions in universal space
        labels = torch.zeros(n_uni, dtype=torch.uint8)
        for entry in gpr_rules:
            if entry["id"] in uni_idx and evaluate_gpr(entry["gpr"], present):
                labels[uni_idx[entry["id"]]] = 1

        samples.append({"presence": presence, "labels": labels})

    return samples


# ---------------------------------------------------------------------------
# Full dataset preparation
# ---------------------------------------------------------------------------


def prepare_multi_organism_dataset(
    model_paths: list[Path],
    out_dir: str | Path = "data/processed/multi_organism",
    samples_per_train_org: int = 200,
    samples_per_eval_org: int = 50,
    min_rxn_organisms: int = 2,
    test_frac: float = 0.1,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    """Full pipeline: load models -> features -> samples -> save.

    Splits organisms into train/val/test for cross-organism generalization.
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
            model = cobra.io.load_json_model(str(path))
            info = extract_organism_info(model)
            del model  # free memory
            if len(info["genes"]) >= 10 and len(info["gpr_rules"]) >= 10:
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
    print(f"  Subsystems: {feature_vocabs['n_subsystem']}")
    print(f"  Features per gene: {feature_vocabs['n_features']}")
    print(f"  Universal reactions (>={min_rxn_organisms} orgs): {len(uni_rxns)}")

    # Split organisms into train/val/test
    rng = random.Random(seed)
    indices = list(range(n_org))
    rng.shuffle(indices)
    n_test = max(1, int(n_org * test_frac))
    n_val = max(1, int(n_org * val_frac))
    test_idx = set(indices[:n_test])
    val_idx = set(indices[n_test:n_test + n_val])
    train_idx = set(indices[n_test + n_val:])

    print(f"\nOrganism splits: {len(train_idx)} train, "
          f"{len(val_idx)} val, {len(test_idx)} test")

    # Pass 2: Extract features and generate samples
    print("\nGenerating samples...")
    org_features: list[torch.Tensor] = []
    org_n_genes: list[int] = []

    split_data: dict[str, dict[str, list]] = {
        "train": {"org_idx": [], "presence": [], "labels": []},
        "val": {"org_idx": [], "presence": [], "labels": []},
        "test": {"org_idx": [], "presence": [], "labels": []},
    }

    for idx in range(n_org):
        org = organisms[idx]
        feats = extract_gene_features(org, feature_vocabs)
        org_features.append(feats)
        org_n_genes.append(len(org["genes"]))

        if idx in train_idx:
            split, n_samp = "train", samples_per_train_org
            s_seed = seed + idx
        elif idx in val_idx:
            split, n_samp = "val", samples_per_eval_org
            s_seed = seed + idx + 10000
        else:
            split, n_samp = "test", samples_per_eval_org
            s_seed = seed + idx + 20000

        samples = generate_organism_samples(
            org, uni_rxns, n_samples=n_samp, seed=s_seed,
        )

        for s in samples:
            split_data[split]["org_idx"].append(idx)
            split_data[split]["presence"].append(s["presence"])
            split_data[split]["labels"].append(s["labels"])

        print(f"  {model_ids[idx]}: {n_samp} {split} samples "
              f"({len(org['genes'])} genes, {len(org['reactions'])} rxns)")

    # Save organism features (use weights_only=False compatible format)
    torch.save({
        "features": org_features,
        "n_genes": torch.tensor(org_n_genes, dtype=torch.int32),
    }, out_dir / "organism_features.pt")

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
        "n_subsystem": feature_vocabs["n_subsystem"],
        "n_universal_reactions": len(uni_rxns),
        "n_organisms": n_org,
        "n_train_organisms": len(train_idx),
        "n_val_organisms": len(val_idx),
        "n_test_organisms": len(test_idx),
        "d_model": 512,
        "n_heads": 8,
        "n_encoder_layers": 4,
        "n_cross_layers": 4,
        "ff_dim": 2048,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "universal_reactions.json").write_text(
        json.dumps(uni_rxns, indent=2),
    )
    (out_dir / "feature_vocabs.json").write_text(json.dumps({
        "ec_vocab": feature_vocabs["ec_vocab"],
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
