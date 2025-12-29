from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import atomic_write_json, ensure_dir, safe_name, utc_now_iso


@dataclass(frozen=True)
class RunPaths:
    root: Path
    run_dir: Path
    log_file: Path
    checkpoint_file: Path
    results_file: Path
    meta_file: Path


def _new_run_id() -> str:
    # deterministic enough for filesystem; collisions extremely unlikely
    return "run_" + utc_now_iso().replace(":", "").replace("+00:00", "Z")


def select_or_create_run(
    *,
    outputs_root: Path,
    model: str,
    task_type: str,
    signature: str,
    resume_run_id: str | None = None,
    resume_signature: str | None = None,
) -> tuple[RunPaths, dict[str, Any], bool]:
    """Return (paths, meta, resumed).

    If there is an unfinished run with the same signature, resume it.
    Otherwise create a new run.
    """

    model_dir = ensure_dir(outputs_root / safe_name(model))
    task_dir = ensure_dir(model_dir / safe_name(task_type))

    if resume_run_id and resume_signature:
        raise ValueError("resume_run_id and resume_signature are mutually exclusive")

    # Forced resume by run_id
    if resume_run_id:
        run_dir = task_dir / str(resume_run_id)
        meta_file = run_dir / "run_meta.json"
        if not run_dir.exists() or not run_dir.is_dir() or not meta_file.exists():
            raise FileNotFoundError(f"run_id not found: {run_dir}")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        paths = RunPaths(
            root=outputs_root,
            run_dir=run_dir,
            log_file=run_dir / "run.log",
            checkpoint_file=run_dir / "checkpoint.json",
            results_file=run_dir / "results.jsonl",
            meta_file=run_dir / "run_meta.json",
        )
        return paths, meta, True

    # Forced resume by signature (latest unfinished)
    if resume_signature:
        candidates: list[tuple[str, Path]] = []
        for child in task_dir.iterdir():
            if not child.is_dir():
                continue
            meta_file = child / "run_meta.json"
            if not meta_file.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if meta.get("signature") != resume_signature:
                continue
            if meta.get("status") == "completed":
                continue
            candidates.append((meta.get("run_id", child.name), child))

        if not candidates:
            raise FileNotFoundError(f"no unfinished run found for signature={resume_signature}")

        _, run_dir = sorted(candidates, key=lambda x: x[1].name)[-1]
        meta_file = run_dir / "run_meta.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        paths = RunPaths(
            root=outputs_root,
            run_dir=run_dir,
            log_file=run_dir / "run.log",
            checkpoint_file=run_dir / "checkpoint.json",
            results_file=run_dir / "results.jsonl",
            meta_file=run_dir / "run_meta.json",
        )
        return paths, meta, True

    # Resume: find latest matching signature with status != completed
    candidates: list[tuple[str, Path]] = []
    for child in task_dir.iterdir():
        if not child.is_dir():
            continue
        meta_file = child / "run_meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if meta.get("signature") != signature:
            continue
        if meta.get("status") == "completed":
            continue
        candidates.append((meta.get("run_id", child.name), child))

    resumed = False
    if candidates:
        _, run_dir = sorted(candidates, key=lambda x: x[1].name)[-1]
        meta_file = run_dir / "run_meta.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        resumed = True
    else:
        run_id = _new_run_id()
        run_dir = ensure_dir(task_dir / run_id)
        meta = {
            "run_id": run_id,
            "signature": signature,
            "status": "running",
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
        atomic_write_json(run_dir / "run_meta.json", meta)

    paths = RunPaths(
        root=outputs_root,
        run_dir=run_dir,
        log_file=run_dir / "run.log",
        checkpoint_file=run_dir / "checkpoint.json",
        results_file=run_dir / "results.jsonl",
        meta_file=run_dir / "run_meta.json",
    )
    return paths, meta, resumed


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "done_record_ids": [],
            "updated_at": None,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint = dict(checkpoint)
    checkpoint["updated_at"] = utc_now_iso()
    atomic_write_json(path, checkpoint)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
