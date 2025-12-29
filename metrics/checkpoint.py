from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import atomic_write_json, iter_jsonl


@dataclass
class ResumeState:
    processed_record_ids: set[str]
    processed_count: int


def load_resume_state(output_jsonl: Path) -> ResumeState:
    processed: set[str] = set()
    count = 0
    if output_jsonl.exists():
        for obj in iter_jsonl(output_jsonl, skip_bad_lines=True):
            rid = obj.get("record_id")
            if isinstance(rid, str) and rid:
                processed.add(rid)
            count += 1
    return ResumeState(processed_record_ids=processed, processed_count=count)


def write_checkpoint(checkpoint_path: Path, *, processed_count: int, total_seen: int, total_submitted: int) -> None:
    atomic_write_json(
        checkpoint_path,
        {
            "processed_count": processed_count,
            "total_seen": total_seen,
            "total_submitted": total_submitted,
        },
    )
