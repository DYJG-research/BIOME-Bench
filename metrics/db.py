from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import iter_jsonl


@dataclass(frozen=True)
class DbRow:
    pathway_id: str
    pubmed_id: str
    raw: dict[str, Any]

    def target_phenotypes_str(self) -> str:
        standardized = self.raw.get("standardized_entities")
        if isinstance(standardized, dict):
            pp = standardized.get("processes_phenotypes")
        else:
            pp = None

        if pp is None:
            return ""
        if isinstance(pp, list):
            lines: list[str] = []
            for x in pp:
                if x is None:
                    continue
                s = str(x).strip()
                if s:
                    lines.append(s)
            return "\n".join(lines)
        if isinstance(pp, dict):
            return json.dumps(pp, ensure_ascii=False)
        return str(pp)

    def relation_list_str(self) -> tuple[str, int]:
        kg = self.raw.get("knowledge_graph")
        if not isinstance(kg, list):
            return "", 0
        lines: list[str] = []
        for i, rel in enumerate(kg):
            if isinstance(rel, dict):
                s = json.dumps(rel, ensure_ascii=False)
            else:
                s = str(rel)
            lines.append(f"{i}. {s}")
        return "\n".join(lines), len(kg)


def collect_keys_needed(results_jsonl: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for obj in iter_jsonl(results_jsonl, skip_bad_lines=True):
        pid = obj.get("pathway_id")
        pmid = obj.get("pubmed_id")
        if pid is None or pmid is None:
            continue
        keys.add((str(pid), str(pmid)))
    return keys


def _read_first_non_ws_char(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return ""
            if not ch.isspace():
                return ch


def _add_pubmed_entries(
    *,
    index: dict[tuple[str, str], DbRow],
    keys_needed: set[tuple[str, str]],
    pathway_id: Any,
    pubmed_list: Any,
) -> None:
    if not keys_needed:
        return
    if pathway_id is None or not isinstance(pubmed_list, list):
        return
    pid = str(pathway_id)
    for entry in pubmed_list:
        if not isinstance(entry, dict):
            continue
        pmid = entry.get("pubmed_id")
        if pmid is None:
            pmid = entry.get("pmid")
        if pmid is None:
            continue
        k = (pid, str(pmid))
        if k not in keys_needed:
            continue
        index[k] = DbRow(pathway_id=pid, pubmed_id=str(pmid), raw=entry)
        if len(index) >= len(keys_needed):
            return


def build_db_index(db_jsonl: Path, keys_needed: set[tuple[str, str]]) -> dict[tuple[str, str], DbRow]:
    index: dict[tuple[str, str], DbRow] = {}
    if not keys_needed:
        return index

    # Support both JSONL (one object per line) and a single JSON array file.
    # The benchmark `hsa_main.jsonl` is actually a JSON array of pathway objects,
    # each containing a `pubmed` list where per-PMID annotations live.
    first = _read_first_non_ws_char(db_jsonl)
    if first == "[":
        with db_jsonl.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for pathway in data:
                if not isinstance(pathway, dict):
                    continue
                # Pathway id can be stored as `id` (benchmark) or `pathway_id`.
                pid = pathway.get("pathway_id")
                if pid is None:
                    pid = pathway.get("id")
                _add_pubmed_entries(
                    index=index,
                    keys_needed=keys_needed,
                    pathway_id=pid,
                    pubmed_list=pathway.get("pubmed"),
                )
                if len(index) >= len(keys_needed):
                    return index
        return index

    # Fallback: JSONL, possibly with either (a) direct rows or (b) pathway rows
    # containing embedded pubmed entries.
    for obj in iter_jsonl(db_jsonl, skip_bad_lines=True):
        if not isinstance(obj, dict):
            continue
        if "pubmed" in obj and isinstance(obj.get("pubmed"), list):
            pid = obj.get("pathway_id")
            if pid is None:
                pid = obj.get("id")
            _add_pubmed_entries(index=index, keys_needed=keys_needed, pathway_id=pid, pubmed_list=obj.get("pubmed"))
            if len(index) >= len(keys_needed):
                break
            continue

        pid = obj.get("pathway_id")
        if pid is None:
            pid = obj.get("id")
        pmid = obj.get("pubmed_id")
        if pmid is None:
            pmid = obj.get("pmid")
        if pid is None or pmid is None:
            continue

        k = (str(pid), str(pmid))
        if k not in keys_needed:
            continue
        index[k] = DbRow(pathway_id=str(pid), pubmed_id=str(pmid), raw=obj)
        if len(index) >= len(keys_needed):
            break
    return index
