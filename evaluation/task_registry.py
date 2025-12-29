from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .output_manager import append_jsonl
from .utils import utc_now_iso


@dataclass(frozen=True)
class TaskEvent:
    event: str  # start|resume|complete|fail
    at: str
    command: str
    output_dir: str
    model: str
    task_type: str
    signature: str
    run_id: str
    extra: dict[str, Any]


def append_task_event(outputs_root: Path, event: TaskEvent) -> None:
    path = outputs_root / "_task_records.jsonl"
    append_jsonl(
        path,
        {
            "event": event.event,
            "at": event.at,
            "command": event.command,
            "output_dir": event.output_dir,
            "model": event.model,
            "task_type": event.task_type,
            "signature": event.signature,
            "run_id": event.run_id,
            **(event.extra or {}),
        },
    )


def now_event(
    *,
    event: str,
    command: str,
    output_dir: str,
    model: str,
    task_type: str,
    signature: str,
    run_id: str,
    extra: dict[str, Any] | None = None,
) -> TaskEvent:
    return TaskEvent(
        event=event,
        at=utc_now_iso(),
        command=command,
        output_dir=output_dir,
        model=model,
        task_type=task_type,
        signature=signature,
        run_id=run_id,
        extra=extra or {},
    )
