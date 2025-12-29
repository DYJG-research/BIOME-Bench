from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from .orchestrator import generate_sh
from .runner import RunArgs, run_evaluation
from .missing import compute_missing_pairs, write_missing_dataset_jsonl


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="multiomics-eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run evaluation")
    run.add_argument("--data", required=True, type=Path, help="dataset path (.jsonl or .json)")
    run.add_argument("--task-type", required=True, help="task type label (used in outputs)")
    run.add_argument("--base-url", default=None, help="OpenAI-compatible base url (overrides config)")
    run.add_argument("--api-key", default=None, help="OpenAI-compatible api key (overrides config)")
    run.add_argument("--model", default=None, help="model name (overrides config)")
    run.add_argument("--threads", type=int, default=1, help="thread count (default: 1)")
    run.add_argument("--config", type=Path, default=Path("config/config.json"), help="config json")
    run.add_argument("--outputs", type=Path, default=Path("outputs"), help="outputs root")
    run.add_argument(
        "--resume-run-id",
        default=None,
        help="force resume a specific run_id (directory name under outputs/<model>/<task-type>/)",
    )
    run.add_argument(
        "--resume-signature",
        default=None,
        help="force resume the latest unfinished run with the given signature",
    )

    orch = sub.add_parser("orchestrate", help="generate a sh script for multiple tasks")
    orch.add_argument("--tasks", required=True, type=Path, help="tasks json file")
    orch.add_argument("--out", required=True, type=Path, help="output sh path")

    miss = sub.add_parser("missing", help="list missing (pathway_id,pubmed_id) by comparing dataset to results.jsonl")
    miss.add_argument("--data", required=True, type=Path, help="dataset path (.jsonl or .json)")
    miss.add_argument("--run-dir", required=True, type=Path, help="run directory containing results.jsonl")
    miss.add_argument(
        "--out",
        default=None,
        type=Path,
        help="optional: write a filtered dataset (.jsonl) containing only missing records",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = _build_parser()
    ns = p.parse_args(argv)

    if ns.cmd == "orchestrate":
        generate_sh(ns.tasks, ns.out)
        ns.out.chmod(0o755)
        print(f"Wrote: {ns.out}")
        return 0

    if ns.cmd == "missing":
        if not ns.data.exists():
            raise SystemExit(f"dataset not found: {ns.data}")
        if not ns.run_dir.exists():
            raise SystemExit(f"run dir not found: {ns.run_dir}")

        results_path = ns.run_dir / "results.jsonl"
        report = compute_missing_pairs(data_path=ns.data, results_path=results_path)
        print(
            "TotalPairs={total} DonePairs={done} MissingPairs={missing}".format(
                total=report.total,
                done=report.done_pairs,
                missing=report.missing_pairs,
            )
        )
        for r in report.missing_records:
            print(f"{r.pathway_id}\t{r.pubmed_id}\t{r.record_id}")

        if ns.out is not None:
            out_path = write_missing_dataset_jsonl(report=report, out_path=ns.out)
            print(f"Wrote: {out_path}")

        return 0

    if ns.threads < 1:
        raise SystemExit("--threads must be >= 1")
    if not ns.data.exists():
        raise SystemExit(f"dataset not found: {ns.data}")
    if not ns.config.exists():
        raise SystemExit(f"config not found: {ns.config}")

    if ns.resume_run_id and ns.resume_signature:
        raise SystemExit("--resume-run-id and --resume-signature are mutually exclusive")

    # Load config early to fill defaults
    from .config import InferenceConfig
    cfg_obj = InferenceConfig.from_json_file(ns.config)
    
    # Resolve final values: CLI > Config
    final_model = ns.model or cfg_obj.model
    final_base_url = ns.base_url or cfg_obj.base_url
    final_api_key = ns.api_key or cfg_obj.api_key

    if not final_model:
        raise SystemExit("model must be provided via --model or config file")
    if not final_base_url:
        raise SystemExit("base-url must be provided via --base-url or config file")
    # api-key can be empty string if user desires, but usually check existence

    args = RunArgs(
        data_path=ns.data,
        task_type=ns.task_type,
        base_url=final_base_url,
        api_key=final_api_key or "",
        model=final_model,
        threads=ns.threads,
        config_path=ns.config,
        outputs_root=ns.outputs,
        resume_run_id=ns.resume_run_id,
        resume_signature=ns.resume_signature,
    )
    cmdline = "multiomics-eval " + " ".join(shlex.quote(x) for x in argv)
    out_dir = run_evaluation(args, cmdline)
    print(f"Output: {out_dir}")
    return 0
