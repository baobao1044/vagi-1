from __future__ import annotations

from pathlib import Path

from .models import ScanIssue

TEXT_EXTENSIONS = {
    ".rs",
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".go",
    ".java",
    ".kt",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
}

EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "dist",
    "build",
}


def scan_codebase(path: str) -> tuple[int, list[ScanIssue], list[str]]:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"path does not exist: {root}")

    issues: list[ScanIssue] = []
    scanned_files = 0

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in file_path.parts):
            continue
        if file_path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        scanned_files += 1
        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        issues.extend(_scan_lines(file_path, lines))

    remediation_plan = [
        "Ưu tiên xử lý tất cả `high` severity trước.",
        "Bổ sung test cho vùng code có `TODO` hoặc gọi `unwrap()`.",
        "Ràng buộc verifier rules để chặn `eval/exec` ở pipeline CI.",
    ]
    return scanned_files, issues, remediation_plan


def _scan_lines(file_path: Path, lines: list[str]) -> list[ScanIssue]:
    issues: list[ScanIssue] = []
    for idx, line in enumerate(lines, start=1):
        content = line.strip()
        lower = content.lower()
        if "todo" in lower:
            issues.append(
                ScanIssue(
                    path=str(file_path),
                    line=idx,
                    severity="medium",
                    rule="todo-left-in-code",
                    message="TODO tồn tại trong code sản xuất, cần xác minh backlog.",
                )
            )
        if "unwrap()" in content:
            issues.append(
                ScanIssue(
                    path=str(file_path),
                    line=idx,
                    severity="high",
                    rule="panic-risk-unwrap",
                    message="`unwrap()` có thể gây panic runtime, nên thay bằng xử lý lỗi tường minh.",
                )
            )
        if "eval(" in lower or "exec(" in lower:
            issues.append(
                ScanIssue(
                    path=str(file_path),
                    line=idx,
                    severity="high",
                    rule="dynamic-exec-risk",
                    message="Phát hiện thực thi động `eval/exec`, nguy cơ code injection.",
                )
            )
    return issues

