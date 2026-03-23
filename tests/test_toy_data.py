from biometnet.data.toy_data import (
    N_GENES,
    _all_reaction_ids,
    generate_toy_dataset,
)


def test_all_reaction_ids():
    rxns = _all_reaction_ids()
    assert len(rxns) > 30  # 10 pathways, ~3-5 reactions each
    assert rxns == sorted(rxns)  # should be sorted


def test_generate_dataset():
    samples = generate_toy_dataset(n_samples=50, seed=123)
    assert len(samples) == 50

    for s in samples:
        assert len(s["genome"]) == N_GENES
        assert all(g in (0, 1) for g in s["genome"])
        assert s["reaction_ids"] == sorted(s["reaction_ids"])


def test_deterministic():
    a = generate_toy_dataset(n_samples=20, seed=99)
    b = generate_toy_dataset(n_samples=20, seed=99)
    assert a == b
