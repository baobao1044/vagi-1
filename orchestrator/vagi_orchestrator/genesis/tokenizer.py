from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


@dataclass(slots=True)
class CharTokenizer:
    tokens: list[str]

    @classmethod
    def build_from_texts(cls, texts: list[str]) -> "CharTokenizer":
        charset = sorted({ch for text in texts for ch in text})
        tokens = SPECIAL_TOKENS + charset
        return cls(tokens=tokens)

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: idx for idx, token in enumerate(self.tokens)}

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        mapping = self.token_to_id
        ids = [mapping.get(ch, self.unk_id) for ch in text]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        specials = set(SPECIAL_TOKENS)
        chars: list[str] = []
        for token_id in ids:
            if token_id < 0 or token_id >= len(self.tokens):
                token = "<unk>"
            else:
                token = self.tokens[token_id]
            if skip_special and token in specials:
                continue
            chars.append(token)
        return "".join(chars)

    def save(self, path: Path) -> None:
        payload = {"tokens": self.tokens}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokens = payload["tokens"]
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("Invalid vocab file: `tokens` must be a non-empty list")
        return cls(tokens=[str(token) for token in tokens])

