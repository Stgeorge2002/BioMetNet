from __future__ import annotations

import json
from pathlib import Path


# Special tokens
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class MetabolicVocab:
    """Maps reaction IDs to integer token IDs and back."""

    def __init__(self, reaction_ids: list[str]) -> None:
        sorted_ids = sorted(set(reaction_ids))
        self.itos: list[str] = SPECIAL_TOKENS + sorted_ids
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, reaction_ids: list[str]) -> list[int]:
        """Encode a list of reaction IDs into a canonical token sequence with BOS/EOS."""
        tokens = [self.bos_id]
        for rid in sorted(reaction_ids):
            tokens.append(self.stoi.get(rid, self.unk_id))
        tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: list[int], strip_special: bool = True) -> list[str]:
        """Decode token IDs back to reaction ID strings."""
        result = []
        for tid in token_ids:
            tok = self.itos[tid] if 0 <= tid < len(self.itos) else UNK_TOKEN
            if strip_special and tok in SPECIAL_TOKENS:
                if tok == EOS_TOKEN:
                    break
                continue
            result.append(tok)
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"reaction_ids": self.itos[len(SPECIAL_TOKENS):]}
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> MetabolicVocab:
        data = json.loads(Path(path).read_text())
        return cls(data["reaction_ids"])
