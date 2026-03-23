"""Evaluation metrics for predicted metabolic programs."""
from __future__ import annotations


def reaction_metrics(predicted: set[str], target: set[str]) -> dict[str, float]:
    """Compute precision, recall, F1 at the reaction level."""
    if not predicted and not target:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(predicted & target)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(target) if target else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def pathway_accuracy(
    predicted: set[str],
    target: set[str],
    pathway_defs: list[dict] | None = None,
) -> dict[str, float]:
    """Fraction of pathways that are fully correctly predicted.

    A pathway is 'correct' if all its reactions that should be present are
    present and none of its reactions that shouldn't be present are present.
    If no pathway_defs provided, tries to import toy pathway defs.
    """
    if pathway_defs is None:
        try:
            from biometnet.data.toy_data import PATHWAY_DEFS
            pathway_defs = PATHWAY_DEFS
        except ImportError:
            return {"pathway_accuracy": float("nan")}

    correct = 0
    total = len(pathway_defs)

    for pw in pathway_defs:
        rxns = set(pw["reactions"])
        target_rxns = rxns & target
        pred_rxns = rxns & predicted

        if target_rxns == pred_rxns:
            correct += 1

    return {"pathway_accuracy": correct / total if total > 0 else 1.0}


def pathway_jaccard(
    predicted: set[str],
    target: set[str],
    pathway_defs: list[dict] | None = None,
) -> dict[str, float]:
    """Per-pathway Jaccard similarity (IoU), macro-averaged.

    Softer than exact-match pathway accuracy: gives partial credit when
    most reactions in a pathway are correctly predicted.
    """
    if pathway_defs is None:
        try:
            from biometnet.data.toy_data import PATHWAY_DEFS
            pathway_defs = PATHWAY_DEFS
        except ImportError:
            return {"pathway_jaccard": float("nan")}

    total = 0.0
    for pw in pathway_defs:
        rxns = set(pw["reactions"])
        target_pw = rxns & target
        pred_pw = rxns & predicted
        union = target_pw | pred_pw
        if not union:
            total += 1.0  # both empty = perfect agreement
        else:
            total += len(target_pw & pred_pw) / len(union)

    return {"pathway_jaccard": total / len(pathway_defs) if pathway_defs else 1.0}


def metabolite_coverage(predicted: set[str], target: set[str]) -> dict[str, float]:
    """Approximate metabolite coverage using reaction sets as proxy.

    For the toy data, we define each reaction as 'producing' a metabolite
    with the same name. This is a simplification — real evaluation would
    use stoichiometric matrices.
    """
    # In the toy setting, treat each reaction as producing one unique metabolite
    pred_metabolites = predicted
    target_metabolites = target

    if not target_metabolites:
        return {"metabolite_coverage": 1.0}

    covered = len(pred_metabolites & target_metabolites)
    return {"metabolite_coverage": covered / len(target_metabolites)}


def per_pathway_breakdown(
    predictions: list[list[str]],
    targets: list[list[str]],
    pathway_defs: list[dict],
) -> list[dict]:
    """Compute per-pathway precision, recall, F1, Jaccard averaged over samples.

    Returns a list of dicts sorted by F1 (ascending, worst pathways first).
    """
    n = len(predictions)
    pw_stats: dict[str, dict[str, float]] = {}

    for pw in pathway_defs:
        name = pw["name"]
        rxns = set(pw["reactions"])
        n_rxns = len(rxns)
        prec_sum = rec_sum = f1_sum = jac_sum = 0.0

        for pred, tgt in zip(predictions, targets):
            pred_pw = rxns & set(pred)
            tgt_pw = rxns & set(tgt)
            tp = len(pred_pw & tgt_pw)
            p = tp / len(pred_pw) if pred_pw else (1.0 if not tgt_pw else 0.0)
            r = tp / len(tgt_pw) if tgt_pw else (1.0 if not pred_pw else 0.0)
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else (1.0 if not tgt_pw and not pred_pw else 0.0)
            union = pred_pw | tgt_pw
            jac = len(pred_pw & tgt_pw) / len(union) if union else 1.0
            prec_sum += p
            rec_sum += r
            f1_sum += f1
            jac_sum += jac

        pw_stats[name] = {
            "name": name,
            "n_reactions": n_rxns,
            "precision": prec_sum / n,
            "recall": rec_sum / n,
            "f1": f1_sum / n,
            "jaccard": jac_sum / n,
        }

    return sorted(pw_stats.values(), key=lambda x: x["f1"])


def evaluate_predictions(
    predictions: list[list[str]],
    targets: list[list[str]],
    pathway_defs: list[dict] | None = None,
) -> dict[str, float]:
    """Aggregate all metrics over a dataset.

    Args:
        predictions: list of predicted reaction ID lists per sample
        targets: list of ground-truth reaction ID lists per sample
        pathway_defs: optional pathway definitions for pathway-level metrics
    """
    all_rxn = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    all_pw = {"pathway_accuracy": 0.0}
    all_pj = {"pathway_jaccard": 0.0}
    all_met = {"metabolite_coverage": 0.0}
    n = len(predictions)

    for pred, tgt in zip(predictions, targets):
        pred_set = set(pred)
        tgt_set = set(tgt)

        rxn = reaction_metrics(pred_set, tgt_set)
        pw = pathway_accuracy(pred_set, tgt_set, pathway_defs)
        pj = pathway_jaccard(pred_set, tgt_set, pathway_defs)
        met = metabolite_coverage(pred_set, tgt_set)

        for k in all_rxn:
            all_rxn[k] += rxn[k]
        for k in all_pw:
            all_pw[k] += pw[k]
        for k in all_pj:
            all_pj[k] += pj[k]
        for k in all_met:
            all_met[k] += met[k]

    result = {}
    for d in [all_rxn, all_pw, all_pj, all_met]:
        for k, v in d.items():
            result[k] = v / max(n, 1)
    return result
