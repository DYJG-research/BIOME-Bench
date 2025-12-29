from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import DatasetRecord, load_dataset


@dataclass(frozen=True)
class MissingReport:
    total: int
    done_pairs: int
    missing_pairs: int
    missing_records: list[DatasetRecord]


def _pair_key(pathway_id: str, pubmed_id: str) -> tuple[str, str]:
    return (str(pathway_id), str(pubmed_id))


def _extract_pair(obj: dict[str, Any]) -> tuple[str, str] | None:
    p = obj.get("pathway_id")
    m = obj.get("pubmed_id")
    if p is not None and m is not None:
        ps = str(p)
        ms = str(m)
        if ps and ms:
            return _pair_key(ps, ms)

    er = obj.get("eval_result")
    if isinstance(er, dict):
        p = er.get("pathway_id")
        m = er.get("pubmed_id")
        if p is not None and m is not None:
            ps = str(p)
            ms = str(m)
            if ps and ms:
                return _pair_key(ps, ms)

    return None


def compute_missing_pairs(*, data_path: str | Path, results_path: str | Path) -> MissingReport:
    data_path = Path(data_path)
    results_path = Path(results_path)

    records = load_dataset(data_path)
    all_pairs: dict[tuple[str, str], DatasetRecord] = {}
    for r in records:
        all_pairs[_pair_key(r.pathway_id, r.pubmed_id)] = r

    done: set[tuple[str, str]] = set()
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(obj, dict):
                    pair = _extract_pair(obj)
                    if pair is not None:
                        done.add(pair)

    missing_keys = [k for k in all_pairs.keys() if k not in done]
    missing_records = [all_pairs[k] for k in missing_keys]

    return MissingReport(
        total=len(all_pairs),
        done_pairs=len(done & set(all_pairs.keys())),
        missing_pairs=len(missing_records),
        missing_records=missing_records,
    )


def write_missing_dataset_jsonl(*, report: MissingReport, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for r in report.missing_records:
            f.write(
                json.dumps(
                    {
                        "messages": r.messages,
                        "pathway_id": r.pathway_id,
                        "pubmed_id": r.pubmed_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return out_path
