import torch
from torch.utils.data import DataLoader

from biometnet.data.toy_data import _all_reaction_ids, generate_toy_dataset, N_GENES
from biometnet.data.dataset import (
    MultiLabelDataset,
    multilabel_collate_fn,
    compute_pos_weight,
)
from biometnet.model.classifier import GenomeClassifier


def _make_classifier_and_data():
    reaction_list = sorted(_all_reaction_ids())
    samples = generate_toy_dataset(n_samples=16, seed=0)
    ds = MultiLabelDataset(samples, reaction_list)
    loader = DataLoader(ds, batch_size=4, collate_fn=multilabel_collate_fn)
    model = GenomeClassifier(
        n_genes=N_GENES,
        n_reactions=len(reaction_list),
        d_model=32,
        n_heads=2,
        n_encoder_layers=1,
        ff_dim=64,
        dropout=0.0,
    )
    return model, loader, reaction_list, samples


def test_classifier_forward_shape():
    model, loader, reaction_list, _ = _make_classifier_and_data()
    batch = next(iter(loader))
    logits = model(batch["genome"])
    assert logits.shape == (4, len(reaction_list))


def test_classifier_predict():
    model, loader, reaction_list, _ = _make_classifier_and_data()
    batch = next(iter(loader))
    probs, active = model.predict(batch["genome"])
    assert probs.shape == (4, len(reaction_list))
    assert active.shape == (4, len(reaction_list))
    assert active.dtype == torch.bool


def test_multilabel_dataset():
    reaction_list = sorted(_all_reaction_ids())
    samples = generate_toy_dataset(n_samples=8, seed=0)
    ds = MultiLabelDataset(samples, reaction_list)
    item = ds[0]
    assert item["genome"].shape == (N_GENES,)
    assert item["labels"].shape == (len(reaction_list),)
    assert item["labels"].sum() > 0  # at least some reactions active


def test_pos_weight():
    reaction_list = sorted(_all_reaction_ids())
    samples = generate_toy_dataset(n_samples=100, seed=0)
    pw = compute_pos_weight(samples, reaction_list)
    assert pw.shape == (len(reaction_list),)
    assert (pw > 0).all()


def test_classifier_overfit_one_batch():
    """Classifier should easily overfit a single batch."""
    model, loader, reaction_list, samples = _make_classifier_and_data()
    batch = next(iter(loader))
    genome = batch["genome"]
    labels = batch["labels"]

    pw = compute_pos_weight(samples, reaction_list)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(200):
        logits = model(genome)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.3, f"Failed to overfit: loss={loss.item():.4f}"
