from __future__ import annotations

import logging
import os
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

from .config import InferenceConfig
from .dataset import DatasetRecord, build_infer_messages, compute_record_id, load_dataset
from .openai_compat import OpenAICompatClient
from .output_manager import append_jsonl, load_checkpoint, save_checkpoint, select_or_create_run
from .task_registry import append_task_event, now_event
from .thinking import ThinkingRules
from .utils import atomic_write_json, file_sha1, sha1_json, utc_now_iso


@dataclass(frozen=True)
class RunArgs:
    data_path: Path
    task_type: str
    base_url: str
    api_key: str
    model: str
    threads: int
    config_path: Path
    outputs_root: Path
    resume_run_id: str | None = None
    resume_signature: str | None = None


def _build_signature(args: RunArgs, cfg: InferenceConfig) -> str:
    dataset_hash = file_sha1(args.data_path)
    think_rules_hash = "builtin"
    try:
        rules_path = Path(cfg.thinking_rules_file)
        if rules_path.exists():
            think_rules_hash = file_sha1(rules_path)
    except Exception:  # noqa: BLE001
        think_rules_hash = "error"
    return sha1_json(
        {
            "task_type": args.task_type,
            "model": args.model,
            "base_url": args.base_url,
            "dataset_sha1": dataset_hash,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "no_think": cfg.no_think,
            "thinking_rules_sha1": think_rules_hash,
        }
    )


def _setup_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("multiomics_evaluation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    # File handler (append for resume)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler (rich)
    try:
        from rich.logging import RichHandler

        ch = RichHandler(rich_tracebacks=True, show_time=False, show_level=True, show_path=False)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
    except Exception:  # noqa: BLE001
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


def _make_output_obj(record: DatasetRecord, model: str, response: str) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "messages": record.messages,
        "eval_result": {"model": model, "response": response},
        "pathway_id": record.pathway_id,
        "pubmed_id": record.pubmed_id,
    }


def _pair_key(pathway_id: str, pubmed_id: str) -> tuple[str, str]:
    return (str(pathway_id), str(pubmed_id))


def _done_ids_from_results(results_file: Path) -> set[str]:
    if not results_file.exists():
        return set()

    done: set[str] = set()
    with results_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                # Skip corrupt/truncated line (e.g., crash while writing)
                continue

            rid = obj.get("record_id")
            if isinstance(rid, str) and rid:
                done.add(rid)
                continue

            try:
                messages = obj.get("messages")
                if not isinstance(messages, list):
                    continue
                pathway_id = str(obj.get("pathway_id", ""))
                pubmed_id = str(obj.get("pubmed_id", ""))
                done.add(compute_record_id(messages=messages, pathway_id=pathway_id, pubmed_id=pubmed_id))
            except Exception:  # noqa: BLE001
                # Be tolerant to older/partial formats
                continue

    return done


def _done_pairs_from_results(results_file: Path) -> set[tuple[str, str]]:
    if not results_file.exists():
        return set()

    done: set[tuple[str, str]] = set()
    with results_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(obj, dict):
                continue

            pathway_id = obj.get("pathway_id")
            pubmed_id = obj.get("pubmed_id")
            if pathway_id is not None and pubmed_id is not None:
                p = str(pathway_id)
                pm = str(pubmed_id)
                if p and pm:
                    done.add(_pair_key(p, pm))
                    continue

            er = obj.get("eval_result")
            if isinstance(er, dict):
                pathway_id = er.get("pathway_id")
                pubmed_id = er.get("pubmed_id")
                if pathway_id is not None and pubmed_id is not None:
                    p = str(pathway_id)
                    pm = str(pubmed_id)
                    if p and pm:
                        done.add(_pair_key(p, pm))

    return done


def run_evaluation(args: RunArgs, command_line: str) -> Path:
    cfg = InferenceConfig.from_json_file(args.config_path)

    thinking_rules = ThinkingRules(rules=[])
    try:
        rules_path = Path(cfg.thinking_rules_file)
        if rules_path.exists():
            thinking_rules = ThinkingRules.from_json_file(rules_path)
    except Exception:  # noqa: BLE001
        # Fall back to no-op rules.
        thinking_rules = ThinkingRules(rules=[])

    think_action = thinking_rules.resolve(model=args.model, no_think=cfg.no_think)
    computed_signature = _build_signature(args, cfg)

    paths, meta, resumed = select_or_create_run(
        outputs_root=args.outputs_root,
        model=args.model,
        task_type=args.task_type,
        signature=computed_signature,
        resume_run_id=args.resume_run_id,
        resume_signature=args.resume_signature,
    )
    effective_signature = str(meta.get("signature") or computed_signature)
    logger = _setup_logging(paths.log_file)

    # Task registry
    append_task_event(
        args.outputs_root,
        now_event(
            event="resume" if resumed else "start",
            command=command_line,
            output_dir=str(paths.run_dir),
            model=args.model,
            task_type=args.task_type,
            signature=effective_signature,
            run_id=meta.get("run_id", paths.run_dir.name),
        ),
    )

    logger.info("Output dir: %s", paths.run_dir)
    logger.info("Dataset: %s", args.data_path)
    logger.info("Task type: %s | Model: %s | Threads: %d", args.task_type, args.model, args.threads)

    records = load_dataset(args.data_path)
    dataset_ids = {r.record_id for r in records}
    checkpoint = load_checkpoint(paths.checkpoint_file)

    done_ids: set[str] = set(checkpoint.get("done_record_ids") or [])

    # Reconcile with results.jsonl: result append can happen before checkpoint save.
    done_from_results = _done_ids_from_results(paths.results_file)
    recovered = 0
    if done_from_results:
        before = len(done_ids)
        done_ids |= done_from_results
        done_ids &= dataset_ids
        recovered += max(0, len(done_ids) - before)
    else:
        done_ids &= dataset_ids

    # Keep checkpoint consistent with derived record_id state.
    if recovered:
        logger.info("Recovered %d done items from results/checkpoint", recovered)
    save_checkpoint(paths.checkpoint_file, {"done_record_ids": sorted(done_ids)})

    pending = [r for r in records if r.record_id not in done_ids]

    total_records = len(records)
    done_records = len(done_ids)
    logger.info("Total records: %d | Done records: %d | Pending records: %d", total_records, done_records, len(pending))

    client = OpenAICompatClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout_s=cfg.request_timeout_s,
        max_retries=cfg.max_retries,
        retry_backoff_s=cfg.retry_backoff_s,
    )

    save_every = cfg.save_every if cfg.save_every > 0 else max(1, args.threads)
    newly_done = 0
    processed = done_records

    # ---- Stall diagnostics / heartbeat ----
    hb_s = int(os.environ.get("MULTIOMICS_EVAL_HEARTBEAT_S", "60"))
    stall_s = int(os.environ.get("MULTIOMICS_EVAL_STALL_S", str(max(180, hb_s * 3))))
    diag_file = paths.run_dir / "stall_diagnostics.log"
    diag_lock = threading.Lock()
    diag_state: dict[str, Any] = {
        "last_activity_ts": time.time(),
        "in_flight": 0,
        "last_record_id": None,
        "last_pathway_id": None,
        "last_pubmed_id": None,
    }

    def _touch_activity(*, rec: DatasetRecord | None, delta_in_flight: int = 0) -> None:
        with diag_lock:
            diag_state["last_activity_ts"] = time.time()
            if delta_in_flight:
                diag_state["in_flight"] = max(0, int(diag_state.get("in_flight", 0)) + delta_in_flight)
            if rec is not None:
                diag_state["last_record_id"] = rec.record_id
                diag_state["last_pathway_id"] = rec.pathway_id
                diag_state["last_pubmed_id"] = rec.pubmed_id

    def _dump_all_thread_traces(reason: str) -> None:
        try:
            import faulthandler

            with diag_file.open("a", encoding="utf-8") as f:
                f.write(f"\n=== {utc_now_iso()} {reason} ===\n")
                faulthandler.dump_traceback(file=f, all_threads=True)
        except Exception:
            # Never let diagnostics crash the run.
            pass

    def _heartbeat() -> None:
        if hb_s <= 0:
            return
        last_dump_ts = 0.0
        while True:
            time.sleep(hb_s)
            with diag_lock:
                last_activity_ts = float(diag_state.get("last_activity_ts", time.time()))
                in_flight = int(diag_state.get("in_flight", 0))
                last_rid = diag_state.get("last_record_id")
                last_pid = diag_state.get("last_pathway_id")
                last_pmid = diag_state.get("last_pubmed_id")

            idle_s = time.time() - last_activity_ts
            logger.info(
                "Heartbeat: Done=%d/%d Pending=%d InFlight=%d Idle=%.1fs Last=%s (%s/%s)",
                len(done_ids),
                total_pairs,
                len(pending),
                in_flight,
                idle_s,
                last_rid,
                last_pid,
                last_pmid,
            )
            if idle_s >= stall_s and (time.time() - last_dump_ts) >= stall_s:
                last_dump_ts = time.time()
                _dump_all_thread_traces(f"STALL_DETECTED idle={idle_s:.1f}s")

    threading.Thread(target=_heartbeat, name="multiomics-heartbeat", daemon=True).start()

    def worker(rec: DatasetRecord) -> tuple[str, str]:
        _touch_activity(rec=rec, delta_in_flight=1)
        messages = build_infer_messages(rec.messages, no_think=cfg.no_think, prompt_prefix=think_action.prompt_prefix)
        try:
            response = client.chat_completions(
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                extra_payload=think_action.request_params,
            )
        finally:
            _touch_activity(rec=rec, delta_in_flight=-1)
        return rec.record_id, response

    try:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            transient=False,
            # LLM requests can take long; use a longer window so ETA doesn't stay blank.
            speed_estimate_period=300.0,
        )
        with progress:
            task_id = progress.add_task("evaluating", total=total_records)
            if processed:
                # Advance once to set initial completed count (supports resume + ETA speed calc)
                progress.advance(task_id, processed)

            # If resuming and everything already done, show 100% and skip executor.
            if not pending:
                if total_records > processed:
                    progress.advance(task_id, total_records - processed)
            else:
                with ThreadPoolExecutor(max_workers=args.threads) as ex:
                    futures = {ex.submit(worker, r): r for r in pending}
                    for fut in as_completed(futures):
                        rec = futures[fut]
                        try:
                            rid, response = fut.result()
                            append_jsonl(paths.results_file, _make_output_obj(rec, args.model, response))
                            done_ids.add(rid)
                            newly_done += 1
                        except Exception as e:  # noqa: BLE001
                            logger.exception(
                                "Failed record_id=%s pathway_id=%s pubmed_id=%s: %s",
                                rec.record_id,
                                rec.pathway_id,
                                rec.pubmed_id,
                                e,
                            )

                        # Progress counts processed pairs (success or failure), starting from checkpointed done.
                        processed += 1
                        progress.advance(task_id, 1)

                        if newly_done > 0 and newly_done % save_every == 0:
                            save_checkpoint(
                                paths.checkpoint_file,
                                {
                                    "done_record_ids": sorted(done_ids),
                                },
                            )
                            meta["updated_at"] = utc_now_iso()
                            atomic_write_json(paths.meta_file, meta)

        # Final checkpoint + meta
        save_checkpoint(
            paths.checkpoint_file,
            {
                "done_record_ids": sorted(done_ids),
            },
        )
        meta["updated_at"] = utc_now_iso()
        if total_records == 0 or len(done_ids) >= total_records:
            meta["status"] = "completed"
            meta["completed_at"] = utc_now_iso()
        else:
            meta["status"] = "partial"
        atomic_write_json(paths.meta_file, meta)

        append_task_event(
            args.outputs_root,
            now_event(
                event="complete" if meta.get("status") == "completed" else "partial",
                command=command_line,
                output_dir=str(paths.run_dir),
                model=args.model,
                task_type=args.task_type,
                signature=effective_signature,
                run_id=meta.get("run_id", paths.run_dir.name),
                extra={"status": meta.get("status"), "done_records": len(done_ids), "total_records": total_records},
            ),
        )

        logger.info(
            "Finished. Status=%s DoneRecords=%d/%d",
            meta.get("status"),
            len(done_ids),
            total_records,
        )
        return paths.run_dir
    except Exception as e:  # noqa: BLE001
        meta["updated_at"] = utc_now_iso()
        meta["status"] = "failed"
        atomic_write_json(paths.meta_file, meta)
        append_task_event(
            args.outputs_root,
            now_event(
                event="fail",
                command=command_line,
                output_dir=str(paths.run_dir),
                model=args.model,
                task_type=args.task_type,
                signature=effective_signature,
                run_id=meta.get("run_id", paths.run_dir.name),
                extra={"error": str(e)},
            ),
        )
        raise
