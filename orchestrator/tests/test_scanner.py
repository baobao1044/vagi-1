from __future__ import annotations

from pathlib import Path

from vagi_orchestrator.scanner import scan_codebase


def test_scanner_finds_high_risk_patterns(tmp_path: Path) -> None:
    sample = tmp_path / "sample.rs"
    sample.write_text(
        """
fn main() {
    // TODO: remove debug path
    let value = maybe().unwrap();
}
""",
        encoding="utf-8",
    )
    scanned_files, issues, _plan = scan_codebase(str(tmp_path))
    assert scanned_files == 1
    assert len(issues) >= 2
    assert any(issue.rule == "panic-risk-unwrap" for issue in issues)

