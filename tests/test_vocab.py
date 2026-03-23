from biometnet.data.metabolic_vocab import MetabolicVocab


def test_vocab_roundtrip():
    rxns = ["PFK", "CS", "MDH", "ENO"]
    vocab = MetabolicVocab(rxns)

    assert len(vocab) == 4 + 4  # 4 special + 4 reactions
    assert vocab.pad_id == 0
    assert vocab.bos_id == 1
    assert vocab.eos_id == 2
    assert vocab.unk_id == 3

    encoded = vocab.encode(["MDH", "CS"])
    decoded = vocab.decode(encoded)
    assert decoded == ["CS", "MDH"]  # sorted


def test_vocab_unknown():
    vocab = MetabolicVocab(["PFK", "CS"])
    encoded = vocab.encode(["PFK", "UNKNOWN_RXN"])
    # UNKNOWN_RXN should map to UNK
    assert vocab.unk_id in encoded


def test_vocab_save_load(tmp_path):
    rxns = ["PFK", "CS", "MDH"]
    vocab = MetabolicVocab(rxns)
    path = tmp_path / "vocab.json"
    vocab.save(path)

    loaded = MetabolicVocab.load(path)
    assert len(loaded) == len(vocab)
    assert loaded.stoi == vocab.stoi
