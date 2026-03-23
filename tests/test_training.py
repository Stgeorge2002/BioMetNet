from biometnet.evaluation.metrics import (
    reaction_metrics,
    pathway_accuracy,
    evaluate_predictions,
)


def test_reaction_metrics_perfect():
    pred = {"PFK", "CS", "MDH"}
    tgt = {"PFK", "CS", "MDH"}
    m = reaction_metrics(pred, tgt)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_reaction_metrics_partial():
    pred = {"PFK", "CS"}
    tgt = {"PFK", "CS", "MDH"}
    m = reaction_metrics(pred, tgt)
    assert m["precision"] == 1.0
    assert abs(m["recall"] - 2 / 3) < 1e-6


def test_pathway_accuracy():
    # If all glycolysis reactions are predicted and target, that pathway is correct
    tgt = {"PFK", "PGK", "PGM", "ENO", "PYK"}
    pred = {"PFK", "PGK", "PGM", "ENO", "PYK"}
    pa = pathway_accuracy(pred, tgt)
    assert pa["pathway_accuracy"] == 1.0  # perfect prediction → all pathways correct


def test_evaluate_predictions():
    preds = [["PFK", "CS"], ["MDH"]]
    tgts = [["PFK", "CS"], ["MDH", "FUM"]]
    result = evaluate_predictions(preds, tgts)
    assert "precision" in result
    assert "f1" in result
    assert "pathway_accuracy" in result
    assert "metabolite_coverage" in result
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["f1"] <= 1.0
    assert result["recall"] < 1.0  # second sample misses FUM
