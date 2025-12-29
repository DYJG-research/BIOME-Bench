from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
from numpy.linalg import norm

# Allow running as a standalone script from src/metrics.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.checkpoint import load_resume_state, write_checkpoint
from metrics.config import load_task_configs
from metrics.log import get_logger
from metrics.io import append_jsonl, iter_jsonl
from metrics.openai_client import embed
from metrics.record import get_model_prediction_text, get_user_query


LOG = get_logger("task2_embed")


def _resolve_output_paths(output: Path) -> tuple[Path, Path, Path]:
    if output.suffix.lower() == ".jsonl":
        out_jsonl = output
        out_dir = output.parent
    else:
        task_dirname = "task2_embed"
        out_dir = output if output.name == task_dirname else (output / task_dirname)
        out_jsonl = out_dir / "results.jsonl"
    checkpoint = out_dir / "checkpoint_task2_embed.json"
    stats = out_dir / "task2_embed_stats.txt"
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


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    A = np.asarray(a, dtype=np.float32)
    B = np.asarray(b, dtype=np.float32)

    na = float(norm(A))
    nb = float(norm(B))
    if na == 0.0 or nb == 0.0:
        return 0.0

    return float(np.dot(A, B) / (na * nb))


def _worker(record: dict[str, Any], *, cfg: Any) -> dict[str, Any]:
    rid = record.get("record_id")
    LOG.info("embedding", record_id=rid)
    # Spec要求：用 messages.user.content（称为 ground truth） vs eval_result.response
    t1 = get_user_query(record)
    t2 = get_model_prediction_text(record)

    e1 = embed(cfg=cfg, text=t1, log_name="task2_embed")
    e2 = embed(cfg=cfg, text=t2, log_name="task2_embed")
    sim = _cosine(e1, e2)

    out = dict(record)
    out["task2_embed_similarity"] = {
        "similarity": sim,
    }
    return out


def _summarize(output_jsonl: Path) -> tuple[int, int, float]:
    n = 0
    ok = 0
    sims: list[float] = []
    if not output_jsonl.exists():
        return 0, 0, 0.0
    for obj in iter_jsonl(output_jsonl, skip_bad_lines=True):
        n += 1
        t = obj.get("task2_embed_similarity")
        if not isinstance(t, dict):
            continue
        v = t.get("similarity")
        if isinstance(v, (int, float)):
            ok += 1
            sims.append(float(v))
    return n, ok, (statistics.mean(sims) if sims else 0.0)


def main() -> int:
    p = argparse.ArgumentParser(description="Task2: Embedding cosine similarity evaluation")
    p.add_argument("--results", required=True, type=Path, help="Input results.jsonl to evaluate")
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
    _validate_cfg("EmbedModel", task_cfg.embed)

    output_jsonl, checkpoint_path, stats_path = _resolve_output_paths(args.output)
    LOG.info("output", output_jsonl=output_jsonl)
    resume = load_resume_state(output_jsonl)
    LOG.info("resume", already_processed=resume.processed_count)

    total_seen = 0
    total_submitted = 0
    processed_count = resume.processed_count

    max_inflight = max(1, args.threads * 2)
    inflight = set()

    def _submit_one(ex: ThreadPoolExecutor, record: dict[str, Any]):
        nonlocal total_submitted
        fut = ex.submit(_worker, record, cfg=task_cfg.embed)
        inflight.add(fut)
        total_submitted += 1

    t0 = time.time()
    LOG.info("processing results", results=args.results)

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
                    out_rid = out.get("record_id")
                    if isinstance(out_rid, str) and out_rid:
                        resume.processed_record_ids.add(out_rid)

                    t = out.get("task2_embed_similarity")
                    sim = None
                    if isinstance(t, dict) and isinstance(t.get("similarity"), (int, float)):
                        sim = float(t["similarity"])
                    LOG.success(
                        "done",
                        record_id=out_rid,
                        similarity=sim,
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
            if total_submitted % 200 == 0:
                LOG.info(
                    "progress",
                    submitted=total_submitted,
                    inflight=len(inflight),
                    processed=processed_count,
                    elapsed_s=f"{time.time()-t0:.1f}",
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
                out_rid = out.get("record_id")
                if isinstance(out_rid, str) and out_rid:
                    resume.processed_record_ids.add(out_rid)

                t = out.get("task2_embed_similarity")
                sim = None
                if isinstance(t, dict) and isinstance(t.get("similarity"), (int, float)):
                    sim = float(t["similarity"])
                LOG.success(
                    "done",
                    record_id=out_rid,
                    similarity=sim,
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

    n, ok, mean_sim = _summarize(output_jsonl)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        f"task2_embed_similarity\n"
        f"output_jsonl: {output_jsonl}\n"
        f"records_in_output: {n}\n"
        f"records_with_similarity: {ok}\n"
        f"mean_similarity: {mean_sim}\n",
        encoding="utf-8",
    )
    LOG.success("summary", mean_similarity=mean_sim, ok=f"{ok}/{n}", stats_path=stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
