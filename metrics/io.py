from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path, *, skip_bad_lines: bool = False) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj: Any = json.loads(line)
                # Some corpora store each line as a JSON-encoded string of JSON.
                # Example line: "{\"pathway_id\": ...}".
                if isinstance(obj, str):
                    try:
                        obj = json.loads(obj)
                    except Exception:  # noqa: BLE001
                        pass
            except Exception as e:  # noqa: BLE001
                if skip_bad_lines:
                    continue
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {e}") from e
            if not isinstance(obj, dict):
                if skip_bad_lines:
                    continue
                raise ValueError(f"Expected JSON object at line {line_no} in {path}")
            yield obj


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")
        f.flush()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
