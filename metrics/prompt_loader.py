from __future__ import annotations

from pathlib import Path


def read_text(rel_name: str) -> str:
    base = Path(__file__).resolve().parent
    return (base / "prompts" / rel_name).read_text(encoding="utf-8")
