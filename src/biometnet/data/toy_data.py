from __future__ import annotations

import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Synthetic pathway definitions
# Each pathway: a set of required gene indices -> a set of produced reactions
# ---------------------------------------------------------------------------

PATHWAY_DEFS: list[dict] = [
    {"name": "glycolysis", "genes": [0, 1, 2, 3, 4], "reactions": ["PFK", "PGK", "PGM", "ENO", "PYK"]},
    {"name": "tca_cycle", "genes": [5, 6, 7, 8, 9], "reactions": ["CS", "ACONTa", "ACONTb", "ICDHyr", "AKGDH"]},
    {"name": "tca_cycle_2", "genes": [10, 11, 12], "reactions": ["SUCOAS", "FUM", "MDH"]},
    {"name": "pentose_phosphate", "genes": [13, 14, 15], "reactions": ["G6PDH2r", "PGL", "GND"]},
    {"name": "fatty_acid_syn", "genes": [16, 17, 18, 19], "reactions": ["ACCOAC", "KAS15", "KAS16", "MCOATA"]},
    {"name": "amino_acid_1", "genes": [20, 21, 22], "reactions": ["ALATA_L", "ASPTA", "PSERT"]},
    {"name": "amino_acid_2", "genes": [23, 24, 25], "reactions": ["GHMT2r", "TRPS1", "TRPS2"]},
    {"name": "nucleotide_syn", "genes": [26, 27, 28, 29], "reactions": ["PRPPS", "GLUPRT", "ADSS", "ADSL1r"]},
    {"name": "oxidative_phos", "genes": [30, 31, 32, 33], "reactions": ["NADH16", "CYTBD", "ATPS4r", "SUCDi"]},
    {"name": "transport", "genes": [34, 35, 36, 37], "reactions": ["GLCpts", "PIt2r", "NH4t", "CO2t"]},
]

N_GENES = 40  # total gene slots


def _all_reaction_ids() -> list[str]:
    """Return all unique reaction IDs across pathways."""
    rxns: set[str] = set()
    for pw in PATHWAY_DEFS:
        rxns.update(pw["reactions"])
    return sorted(rxns)


def generate_toy_sample(
    rng: random.Random,
    gene_dropout: float = 0.05,
    spurious_rate: float = 0.02,
) -> dict:
    """Generate one synthetic genome-metabolism sample.

    1. Randomly activate subsets of pathways.
    2. Set corresponding genes to 1 (with optional dropout).
    3. Collect reactions from fully-gated pathways.
    4. Optionally add spurious reactions.
    """
    all_rxns = _all_reaction_ids()

    # Randomly choose which pathways are active (each with ~75% chance)
    genome = [0] * N_GENES
    active_reactions: list[str] = []

    for pw in PATHWAY_DEFS:
        if rng.random() < 0.75:  # pathway active
            # Set genes (with possible dropout)
            all_genes_present = True
            for g in pw["genes"]:
                if rng.random() < gene_dropout:
                    all_genes_present = False
                else:
                    genome[g] = 1

            if all_genes_present:
                active_reactions.extend(pw["reactions"])

    # Add spurious reactions occasionally
    for rxn in all_rxns:
        if rxn not in active_reactions and rng.random() < spurious_rate:
            active_reactions.append(rxn)

    return {
        "genome": genome,
        "reaction_ids": sorted(set(active_reactions)),
    }


def generate_toy_dataset(
    n_samples: int = 1000,
    seed: int = 42,
    gene_dropout: float = 0.05,
    spurious_rate: float = 0.02,
) -> list[dict]:
    rng = random.Random(seed)
    return [
        generate_toy_sample(rng, gene_dropout, spurious_rate)
        for _ in range(n_samples)
    ]


def save_toy_dataset(samples: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(samples, indent=2))


def load_toy_dataset(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())
