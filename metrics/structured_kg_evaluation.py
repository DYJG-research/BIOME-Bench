from __future__ import annotations

import argparse
from contextlib import nullcontext
import re
import statistics
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

try:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except Exception:  # pragma: no cover
    Progress = None  # type: ignore[assignment]

# Allow running as a standalone script from src/metrics.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.checkpoint import load_resume_state, write_checkpoint
from metrics.config import load_task_configs
from metrics.db import build_db_index, collect_keys_needed
from metrics.log import get_logger
from metrics.io import append_jsonl, iter_jsonl
from metrics.openai_client import chat_json
from metrics.prompt_loader import read_text
from metrics.record import get_key, get_model_prediction_text


LOG = get_logger("task3_kg")


def _species_from_pathway_id(pathway_id: str) -> str:
    # Convention: pathway IDs are like "hsa04380", "mmu01234", "rnoXXXXX".
    # We route DB by the first 3 characters.
    return pathway_id[:3].lower() if pathway_id else ""


def _infer_species_from_db_path(db_path: Path) -> str:
    # Common naming: hsa_main.jsonl, mmu_main.jsonl, rno_main.jsonl
    name = db_path.name.lower()
    m = re.match(r"^(?P<sp>[a-z]{3})[_\.-]", name)
    if m:
        return m.group("sp")
    stem = db_path.stem.lower()
    return stem[:3] if len(stem) >= 3 and stem[:3].isalpha() else ""


def _resolve_output_paths(output: Path) -> tuple[Path, Path, Path]:
    if output.suffix.lower() == ".jsonl":
        out_jsonl = output
        out_dir = output.parent
    else:
        task_dirname = "task3_kg"
        out_dir = output if output.name == task_dirname else (output / task_dirname)
        out_jsonl = out_dir / "results.jsonl"
    checkpoint = out_dir / "checkpoint_task3_kg.json"
    stats = out_dir / "task3_kg_stats.txt"
    return out_jsonl, checkpoint, stats


def _validate_cfg(name: str, cfg: Any) -> None:
    missing = []
    if not getattr(cfg, "model", ""):
        missing.append("model")
    if not getattr(cfg, "base_url", ""):
        missing.append("base_url")
    if not getattr(cfg, "api_key", ""):
        missing.append("api_key")
    if missing:
        raise SystemExit(f"{name} missing fields in openai_config.json: {', '.join(missing)}")


def _worker(
    record: dict[str, Any],
    *,
    db_index_by_species: dict[str, dict[tuple[str, str], Any]],
    default_db_index: dict[tuple[str, str], Any] | None,
    system_prompt: str,
    user_tmpl: str,
    cfg: Any,
) -> dict[str, Any]:
    rid = record.get("record_id")
    LOG.info("calling model", record_id=rid)
    key = get_key(record)
    if not key:
        raise ValueError("missing (pathway_id, pubmed_id) in record")

    species = _species_from_pathway_id(key[0])
    db_index = db_index_by_species.get(species)
    if db_index is None:
        db_index = default_db_index
    if db_index is None:
        raise ValueError(f"No DB provided for species='{species}' (pathway_id={key[0]!r}).")

    db_row = db_index.get(key) if key else None
    if not db_row:
        raise ValueError(f"DB row not found for key={key}")

    relation_list_str, total_relations = db_row.relation_list_str()
    if total_relations <= 0:
        raise ValueError(f"knowledge_graph empty for key={key}")

    user_prompt = user_tmpl.format(
        text=get_model_prediction_text(record),
        relation_list_str=relation_list_str,
    )

    res = chat_json(cfg=cfg, system_prompt=system_prompt, user_prompt=user_prompt, log_name="task3_kg")

    supported = []
    if isinstance(res, dict) and isinstance(res.get("supported_relation_ids"), list):
        supported = res.get("supported_relation_ids")

    valid_ids: set[int] = set()
    for x in supported:
        if isinstance(x, int) and 0 <= x < total_relations:
            valid_ids.add(x)
        elif isinstance(x, str) and x.isdigit():
            v = int(x)
            if 0 <= v < total_relations:
                valid_ids.add(v)

    coverage = (len(valid_ids) / total_relations) if total_relations else 0.0

    out = dict(record)
    out["task3_kg_entailment"] = {
        "total_relations": total_relations,
        "supported_relation_ids": sorted(valid_ids),
        "coverage": coverage,
        "extract_result": res,
    }
    return out


def _summarize(output_jsonl: Path) -> tuple[int, int, float]:
    n = 0
    ok = 0
    covs: list[float] = []
    if not output_jsonl.exists():
        return 0, 0, 0.0
    for obj in iter_jsonl(output_jsonl, skip_bad_lines=True):
        n += 1
        t = obj.get("task3_kg_entailment")
        if not isinstance(t, dict):
            continue
        v = t.get("coverage")
        if isinstance(v, (int, float)):
            ok += 1
            covs.append(float(v))
    return n, ok, (statistics.mean(covs) if covs else 0.0)


def main() -> int:
    p = argparse.ArgumentParser(description="Task3: Knowledge Graph entailment extraction + coverage")
    p.add_argument("--results", required=True, type=Path, help="Input results.jsonl to evaluate")
    p.add_argument(
        "--db",
        required=True,
        type=Path,
        nargs="+",
        help="Database JSON/JSONL files (e.g. hsa_main.jsonl mmu_main.jsonl rno_main.jsonl)",
    )
    p.add_argument("--output", required=True, type=Path, help="Output JSONL file path or output directory")
    p.add_argument("--threads", type=int, default=1, help="Worker threads (default: 1)")
    p.add_argument(
        "--openai-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "config.json",
        help="Path to config/config.json",
    )
    args = p.parse_args()

    if args.threads <= 0:
        raise SystemExit("--threads must be >= 1")

    LOG.info("loading openai config", path=args.openai_config)
    task_cfg = load_task_configs(args.openai_config)
    _validate_cfg("CommonModel", task_cfg.common)

    output_jsonl, checkpoint_path, stats_path = _resolve_output_paths(args.output)
    LOG.info("output", output_jsonl=output_jsonl)
    resume = load_resume_state(output_jsonl)
    LOG.info("resume", already_processed=resume.processed_count)

    # Progress total (for ETA). We count only records in this input file.
    total_records = 0
    already_done_in_input = 0
    for record in iter_jsonl(args.results, skip_bad_lines=True):
        total_records += 1
        rid = record.get("record_id")
        if isinstance(rid, str) and rid in resume.processed_record_ids:
            already_done_in_input += 1

    todo_records = max(0, total_records - already_done_in_input)
    LOG.info(
        "progress total",
        total_records=total_records,
        already_done_in_input=already_done_in_input,
        todo_records=todo_records,
    )

    t0 = time.time()
    LOG.info("scan results keys", results=args.results)
    keys_needed = collect_keys_needed(args.results)
    LOG.info("scan done", keys_needed=len(keys_needed), elapsed_s=f"{time.time()-t0:.2f}")

    # Partition required keys by species so we can build smaller DB indices.
    keys_by_species: dict[str, set[tuple[str, str]]] = {}
    for pid, pmid in keys_needed:
        keys_by_species.setdefault(_species_from_pathway_id(pid), set()).add((pid, pmid))

    db_paths = list(args.db)
    if not db_paths:
        raise SystemExit("--db must include at least 1 path")

    db_index_by_species: dict[str, dict[tuple[str, str], Any]] = {}
    default_db_index: dict[tuple[str, str], Any] | None = None

    if len(db_paths) == 1:
        t1 = time.time()
        LOG.info("build db index", db=db_paths[0])
        default_db_index = build_db_index(db_paths[0], keys_needed)
        LOG.info("db index ready", db_index_size=len(default_db_index), elapsed_s=f"{time.time()-t1:.2f}")
        if keys_needed and len(default_db_index) == 0:
            sample_key = next(iter(keys_needed))
            LOG.warn(
                "db index empty",
                sample_needed_key=sample_key,
                hint="benchmark db keys = (pathway.id, pubmed[].pmid)",
            )
    else:
        # Infer species mapping from db file names.
        db_path_by_species: dict[str, Path] = {}
        for dbp in db_paths:
            sp = _infer_species_from_db_path(dbp)
            if not sp:
                raise SystemExit(f"Cannot infer species code from db file name: {dbp}")
            if sp in db_path_by_species and db_path_by_species[sp] != dbp:
                raise SystemExit(f"Duplicate DB provided for species '{sp}': {db_path_by_species[sp]} and {dbp}")
            db_path_by_species[sp] = dbp

        needed_species = {sp for sp, ks in keys_by_species.items() if ks}
        missing = sorted(sp for sp in needed_species if sp not in db_path_by_species)
        if missing:
            raise SystemExit(
                "Missing DB(s) for species: "
                + ", ".join(missing)
                + ". Provide matching files via --db (e.g. hsa_main.jsonl mmu_main.jsonl rno_main.jsonl)."
            )

        for sp in sorted(needed_species):
            t1 = time.time()
            dbp = db_path_by_species[sp]
            LOG.info("build db index", species=sp, db=dbp)
            idx = build_db_index(dbp, keys_by_species.get(sp, set()))
            db_index_by_species[sp] = idx
            LOG.info(
                "db index ready",
                species=sp,
                db_index_size=len(idx),
                elapsed_s=f"{time.time()-t1:.2f}",
            )

    # For multi-DB mode, ensure we at least built something for every needed species.
    if default_db_index is None and not db_index_by_species and keys_needed:
        raise SystemExit("DB indices are empty; check --db paths and file contents")

    system_prompt = read_text("Structured_KG_Evaluation_System.txt")
    user_tmpl = read_text("Structured_KG_Evaluation_System_User.txt")

    total_seen = 0
    total_submitted = 0
    processed_count = resume.processed_count

    max_inflight = max(1, args.threads * 2)
    inflight = set()

    def _submit_one(ex: ThreadPoolExecutor, record: dict[str, Any]):
        nonlocal total_submitted
        fut = ex.submit(
            _worker,
            record,
            db_index_by_species=db_index_by_species,
            default_db_index=default_db_index,
            system_prompt=system_prompt,
            user_tmpl=user_tmpl,
            cfg=task_cfg.common,
        )
        inflight.add(fut)
        total_submitted += 1

    t2 = time.time()
    LOG.info("processing results", results=args.results)

    progress = None
    progress_task_id = None
    if Progress is not None:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            transient=False,
            speed_estimate_period=300.0,
        )

    progress_cm = progress if progress is not None else nullcontext()
    with progress_cm:
        if progress is not None:
            progress_task_id = progress.add_task("task3_kg_entailment", total=total_records)
            if already_done_in_input:
                progress.advance(progress_task_id, already_done_in_input)

        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            for record in iter_jsonl(args.results, skip_bad_lines=True):
                total_seen += 1
                rid = record.get("record_id")
                if isinstance(rid, str) and rid in resume.processed_record_ids:
                    continue

                while len(inflight) >= max_inflight:
                    done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        inflight.remove(fut)
                        try:
                            out = fut.result()
                        except Exception as e:  # noqa: BLE001
                            LOG.error("worker error", err=str(e))
                            continue

                        append_jsonl(output_jsonl, out)
                        processed_count += 1
                        if progress is not None and progress_task_id is not None:
                            progress.advance(progress_task_id, 1)
                        out_rid = out.get("record_id")
                        if isinstance(out_rid, str) and out_rid:
                            resume.processed_record_ids.add(out_rid)

                        t = out.get("task3_kg_entailment")
                        cov = None
                        if isinstance(t, dict) and isinstance(t.get("coverage"), (int, float)):
                            cov = float(t["coverage"])
                        LOG.success(
                            "done",
                            record_id=out_rid,
                            coverage=cov,
                            processed=processed_count,
                            submitted=total_submitted,
                            seen=total_seen,
                        )

                        write_checkpoint(
                            checkpoint_path,
                            processed_count=processed_count,
                            total_seen=total_seen,
                            total_submitted=total_submitted,
                        )

                _submit_one(ex, record)
                if total_submitted % 100 == 0:
                    LOG.info(
                        "progress",
                        submitted=total_submitted,
                        inflight=len(inflight),
                        processed=processed_count,
                        elapsed_s=f"{time.time()-t2:.1f}",
                    )

        while inflight:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                inflight.remove(fut)
                try:
                    out = fut.result()
                except Exception as e:  # noqa: BLE001
                    LOG.error("worker error", err=str(e))
                    continue

                append_jsonl(output_jsonl, out)
                processed_count += 1
                if progress is not None and progress_task_id is not None:
                    progress.advance(progress_task_id, 1)
                out_rid = out.get("record_id")
                if isinstance(out_rid, str) and out_rid:
                    resume.processed_record_ids.add(out_rid)

                t = out.get("task3_kg_entailment")
                cov = None
                if isinstance(t, dict) and isinstance(t.get("coverage"), (int, float)):
                    cov = float(t["coverage"])
                LOG.success(
                    "done",
                    record_id=out_rid,
                    coverage=cov,
                    processed=processed_count,
                    submitted=total_submitted,
                    seen=total_seen,
                )

                write_checkpoint(
                    checkpoint_path,
                    processed_count=processed_count,
                    total_seen=total_seen,
                    total_submitted=total_submitted,
                )

    n, ok, mean_cov = _summarize(output_jsonl)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        f"task3_kg_entailment\n"
        f"output_jsonl: {output_jsonl}\n"
        f"records_in_output: {n}\n"
        f"records_with_coverage: {ok}\n"
        f"mean_coverage: {mean_cov}\n",
        encoding="utf-8",
    )
    LOG.success("summary", mean_coverage=mean_cov, ok=f"{ok}/{n}", stats_path=stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
