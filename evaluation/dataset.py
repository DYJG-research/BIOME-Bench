from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .utils import sha1_text


def compute_record_id(*, messages: list[dict[str, Any]], pathway_id: str, pubmed_id: str) -> str:
    user_content = ""
    for m in messages:
        if m.get("role") == "user":
            user_content = str(m.get("content", ""))
            break
    rid_source = f"{pathway_id}|{pubmed_id}|{user_content}"
    return sha1_text(rid_source)


@dataclass(frozen=True)
class DatasetRecord:
    record_id: str
    messages: list[dict[str, Any]]
    pathway_id: str
    pubmed_id: str

    @staticmethod
    def from_obj(obj: dict[str, Any]) -> "DatasetRecord":
        messages = obj.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("record.messages must be a non-empty list")

        pathway_id = str(obj.get("pathway_id", ""))
        pubmed_id = str(obj.get("pubmed_id", ""))

        record_id = compute_record_id(messages=messages, pathway_id=pathway_id, pubmed_id=pubmed_id)
        return DatasetRecord(
            record_id=record_id,
            messages=messages,
            pathway_id=pathway_id,
            pubmed_id=pubmed_id,
        )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e


def load_dataset(path: str | Path) -> list[DatasetRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        objs = list(_iter_jsonl(path))
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(".json dataset must be a JSON array")
        objs = data
    else:
        raise ValueError("dataset must be .jsonl or .json")

    records: list[DatasetRecord] = []
    for obj in objs:
        if not isinstance(obj, dict):
            raise ValueError("each dataset item must be a JSON object")
        records.append(DatasetRecord.from_obj(obj))
    return records


def build_infer_messages(
    messages: list[dict[str, Any]],
    *,
    no_think: bool = False,
    prompt_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Use dataset messages as input for inference.

    Convention:
    - If the last message is an assistant reference answer, exclude it from the prompt.
    - Otherwise, use all messages as-is.
    """
    base = messages[:-1] if (messages and messages[-1].get("role") == "assistant") else messages

    # Copy to avoid mutating source dataset records (and thus saved `messages`).
    copied: list[dict[str, Any]] = [dict(m) for m in base]

    if no_think and prompt_prefix is None:
        prompt_prefix = "\no_think"

    if no_think and prompt_prefix:
        for i, m in enumerate(copied):
            if m.get("role") == "user":
                content = str(m.get("content", ""))
                if not content.startswith(prompt_prefix):
                    copied[i] = {**m, "content": prompt_prefix + content}
                break

    return copied
