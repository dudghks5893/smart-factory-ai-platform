"""Lightweight contracts for final documentation and repository hygiene."""

from __future__ import annotations

import re
from pathlib import Path

_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
_REQUIRED_README_SECTIONS = (
    "## 1. 문제 정의와 목표",
    "## 2. 주요 기능",
    "## 3. 시스템 아키텍처",
    "## 4. Vision AI와 데이터 계약",
    "## 5. Serving과 inspection data",
    "## 6. MLOps와 운영",
    "## 7. SOP RAG assistant",
    "## 8. Final benchmark 결과",
    "## 9. Repository 구조",
    "## 10. 빠른 시작과 model workflow",
    "## 11. Docker Compose",
    "## 12. 테스트와 CI",
    "## 13. Kubernetes와 GCP foundation",
    "## 14. 문서",
    "## 15. 한계와 미검증 범위",
    "## 16. 향후 작업",
    "## 17. 기술 스택",
    "## 18. 완료 상태",
)


# ADD 2026-08-22: Documentation test에서 repository root를 반환한다.
def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ADD 2026-08-22: Final README의 required section과 ordering을 검증한다.
def test_readme_has_final_entry_document_sections() -> None:
    readme = (_project_root() / "README.md").read_text(encoding="utf-8")

    positions = [readme.index(section) for section in _REQUIRED_README_SECTIONS]

    assert positions == sorted(positions)
    assert "working_tree_dirty" not in readme
    assert "production GKE/Cloud SQL 배포" in readme


# ADD 2026-08-22: README/docs의 local Markdown link target이 실제 file 또는 directory인지 검증한다.
def test_relative_documentation_links_resolve() -> None:
    root = _project_root()
    documents = (root / "README.md", *(root / "docs").rglob("*.md"))
    broken: list[str] = []

    for document in documents:
        text = document.read_text(encoding="utf-8")
        for target in _MARKDOWN_LINK.findall(text):
            path_target = target.split("#", maxsplit=1)[0]
            if not path_target or "://" in path_target or path_target.startswith("mailto:"):
                continue
            resolved = (document.parent / path_target).resolve()
            if not resolved.exists():
                broken.append(f"{document.relative_to(root)} -> {target}")

    assert not broken, "Broken documentation links:\n" + "\n".join(broken)


# ADD 2026-08-22: Generated/artifact paths의 root ignore policy와 stale placeholder 부재를 검증한다.
# MODIFY 2026-08-24: Tracked example source도 repository hygiene 검사 범위에 포함한다.
def test_repository_hygiene_contract_has_no_stale_gitkeep() -> None:
    root = _project_root()
    ignore_rules = set((root / ".gitignore").read_text(encoding="utf-8").splitlines())
    required_rules = {
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        ".DS_Store",
        ".env",
        "*.db",
        "*.sqlite",
        "*.sqlite3",
        "/data/",
        "/artifacts/",
        "/models/",
        "/checkpoints/",
        "/outputs/",
        "/mlruns/",
    }
    source_roots = (
        root / "apps",
        root / "configs",
        root / "docs",
        root / "examples",
        root / "infra",
        root / "manuals",
        root / "ml",
        root / "monitoring",
        root / "pipelines",
        root / "scripts",
        root / "services",
        root / "tests",
    )

    assert required_rules <= ignore_rules
    assert not [path for source_root in source_roots for path in source_root.rglob(".gitkeep")]
