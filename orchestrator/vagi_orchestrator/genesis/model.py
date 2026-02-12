from __future__ import annotations

import torch
from torch import nn


class TinyGruLm(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(input_ids)
        output, next_hidden = self.gru(x, hidden)
        logits = self.lm_head(output)
        return logits, next_hidden

