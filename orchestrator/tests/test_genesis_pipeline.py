from __future__ import annotations

import json
from pathlib import Path

from vagi_orchestrator.genesis.pipeline import GenesisConfig, train_and_export


def test_genesis_train_and_export_creates_safetensors_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "genesis-v0"
    result = train_and_export(
        output_dir=out_dir,
        config=GenesisConfig(
            repeats=4,
            embed_dim=32,
            hidden_dim=64,
            num_layers=1,
            seq_len=32,
            batch_size=16,
            epochs=2,
            lr=5e-3,
            max_seq_len=64,
        ),
    )
    assert (out_dir / "model.safetensors").exists()
    assert (out_dir / "vocab.json").exists()
    assert (out_dir / "manifest.json").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["arch"] == "tiny-gru-lm"
    assert manifest["model_sha256"]
    assert manifest["vocab_sha256"]
    assert isinstance(result["smoke_text"], str)

