from __future__ import annotations

from typing import Any, Optional


def get_user_query(record: dict[str, Any]) -> str:
    msgs = record.get("messages")
    if not isinstance(msgs, list):
        return ""
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            return str(c) if c is not None else ""
    return ""


def get_ground_truth_text(record: dict[str, Any]) -> str:
    msgs = record.get("messages")
    if not isinstance(msgs, list):
        return ""
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content")
            return str(c) if c is not None else ""
    return ""


def get_model_prediction_text(record: dict[str, Any]) -> str:
    er = record.get("eval_result")
    if not isinstance(er, dict):
        return ""
    c = er.get("response")
    return str(c) if c is not None else ""


def get_key(record: dict[str, Any]) -> Optional[tuple[str, str]]:
    pid = record.get("pathway_id")
    pmid = record.get("pubmed_id")
    if isinstance(pid, str) and isinstance(pmid, str):
        return pid, pmid
    return None
