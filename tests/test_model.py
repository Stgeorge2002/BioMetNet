import torch
from torch.utils.data import DataLoader

from biometnet.data.metabolic_vocab import MetabolicVocab
from biometnet.data.toy_data import _all_reaction_ids, generate_toy_dataset
from biometnet.data.dataset import GenomeMetabolismDataset, collate_fn
from biometnet.model.seq2seq import Seq2SeqModel
from biometnet.data.toy_data import N_GENES


def _make_model_and_data():
    vocab = MetabolicVocab(_all_reaction_ids())
    samples = generate_toy_dataset(n_samples=16, seed=0)
    ds = GenomeMetabolismDataset(samples, vocab)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    model = Seq2SeqModel(
        n_genes=N_GENES, vocab_size=len(vocab),
        d_model=32, n_heads=2, n_encoder_layers=1, n_decoder_layers=1,
        ff_dim=64, dropout=0.0, max_seq_len=64,
    )
    return model, loader, vocab


def test_forward_shape():
    model, loader, vocab = _make_model_and_data()
    batch = next(iter(loader))
    tgt_input = batch["tokens"][:, :-1]
    logits = model(batch["genome"], tgt_input)
    assert logits.shape == (tgt_input.size(0), tgt_input.size(1), len(vocab))


def test_generate():
    model, loader, vocab = _make_model_and_data()
    batch = next(iter(loader))
    seqs = model.generate(batch["genome"], vocab.bos_id, vocab.eos_id, max_len=30)
    assert len(seqs) == batch["genome"].size(0)
    for seq in seqs:
        assert seq[0] == vocab.bos_id


def test_overfit_one_batch():
    """Model should be able to overfit a single batch."""
    model, loader, vocab = _make_model_and_data()
    batch = next(iter(loader))
    genome = batch["genome"]
    tokens = batch["tokens"]
    tgt_input = tokens[:, :-1]
    tgt_output = tokens[:, 1:]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    for _ in range(200):
        logits = model(genome, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.5, f"Failed to overfit: loss={loss.item():.4f}"
