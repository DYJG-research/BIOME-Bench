#!/usr/bin/env python3
import sys
import os
import subprocess
import shutil
import json
import argparse
from pathlib import Path
import time

# --- Defaults ---
DEFAULT_CONFIG = "config/config.json"
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs_demo")
METRICS_DIR = Path("metrics")

TASK_A_DATA = DATA_DIR / "TASK-A-test.jsonl"
TASK_B_DATA = DATA_DIR / "TASK-B-test.jsonl"

DB_FILES = [
    DATA_DIR / "hsa.jsonl",
    DATA_DIR / "mmu.jsonl",
    DATA_DIR / "rno.jsonl"
]

def print_step(msg):
    print(f"\n{'='*60}")
    print(f"[*] {msg}")
    print(f"{ '='*60}")

def run_cmd(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

def find_latest_results(task_dir: Path) -> Path:
    """Find the results.jsonl file in the most recent run directory."""
    if not task_dir.exists():
         raise FileNotFoundError(f"Task directory not found: {task_dir}")
    
    # Find all subdirectories starting with "run_"
    runs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    
    if not runs:
        raise FileNotFoundError(f"No run directories found in {task_dir}")
    
    # Sort by name (which includes timestamp) to get the latest
    runs.sort(key=lambda x: x.name)
    latest_run = runs[-1]
    
    results_file = latest_run / "results.jsonl"
    if not results_file.exists():
         raise FileNotFoundError(f"results.jsonl not found in {latest_run}")
    
    print(f"[*] Found latest results: {results_file}")
    return results_file

def load_model_name(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get("EvalModel", {}).get("api_config", {}).get("model", "UnknownModel")
    except Exception as e:
        print(f"Error reading config {config_path}: {e}")
        return "UnknownModel"

def main():
    parser = argparse.ArgumentParser(description="Run BIOME-Bench Demo Pipeline")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to config.json")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for evaluation")
    parser.add_argument("--outputs", type=str, default=str(OUTPUT_DIR), help="Root directory for outputs")
    parser.add_argument("--data-a", type=str, default=str(TASK_A_DATA), help="Path to Task A data")
    parser.add_argument("--data-b", type=str, default=str(TASK_B_DATA), help="Path to Task B data")
    parser.add_argument("--db", type=str, nargs="+", default=[str(f) for f in DB_FILES], help="Path to DB files")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_root = Path(args.outputs)
    threads = args.threads
    task_a_data = Path(args.data_a)
    task_b_data = Path(args.data_b)
    db_files = [Path(f) for f in args.db]
    
    model_name = load_model_name(config_path)
    print(f"[*] Using Model: {model_name}")
    print(f"[*] Config: {config_path}")
    print(f"[*] Threads: {threads}")
    print(f"[*] Task A Data: {task_a_data}")
    print(f"[*] Task B Data: {task_b_data}")
    print(f"[*] DB Files: {db_files}")

    # 0. Cleanup previous demo outputs
    if output_root.exists():
        print_step(f"Cleaning up previous outputs in {output_root}...")
        shutil.rmtree(output_root)
    
    # 1. Task A: Relation Prediction
    print_step("Step 1: Running Task A (Relation Prediction) Evaluation")
    task_a_type = "task1_relation_prediction"
    cmd_a = [
        sys.executable, "-m", "evaluation", "run",
        "--data", str(task_a_data),
        "--task-type", task_a_type,
        "--threads", str(threads),
        "--config", str(config_path),
        "--outputs", str(output_root)
    ]
    run_cmd(cmd_a)

    # 2. Task B: Mechanism Analysis
    print_step("Step 2: Running Task B (Mechanism Analysis) Evaluation")
    task_b_type = "task2_analysis"
    cmd_b = [
        sys.executable, "-m", "evaluation", "run",
        "--data", str(task_b_data),
        "--task-type", task_b_type,
        "--threads", str(threads),
        "--config", str(config_path),
        "--outputs", str(output_root)
    ]
    run_cmd(cmd_b)

    # 3. Task A Scoring
    print_step("Step 3: Calculating Task A Metrics")
    # Output path convention: outputs/<model>/<task-type>/run_<timestamp>/results.jsonl
    task_a_dir = output_root / model_name / task_a_type
    try:
        task_a_results = find_latest_results(task_a_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    cmd_score_a = [
        sys.executable, str(METRICS_DIR / "biomolecular_interaction_inference_acc.py"),
        "--input", str(task_a_results)
    ]
    run_cmd(cmd_score_a)

    # 4. Task B Scoring (LLM-as-a-Judge)
    print_step("Step 4: Calculating Task B Metrics (LLM-as-a-Judge)")
    task_b_dir = output_root / model_name / task_b_type
    try:
        task_b_results = find_latest_results(task_b_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Judge output can go into the same run directory or a separate one. 
    # Let's put it in the same run directory for clarity.
    task_b_judge_out = task_b_results.parent / "judge_results.jsonl"

    # Check for DB files
    existing_dbs = []
    for db in db_files:
        if db.exists():
            existing_dbs.append(str(db))
        else:
            print(f"Warning: DB file not found: {db}")
    
    if not existing_dbs:
         print(f"Error: No DB files found (hsa.jsonl, mmu.jsonl, rno.jsonl). Scoring requires at least one.")
         sys.exit(1)

    cmd_score_b = [
        sys.executable, str(METRICS_DIR / "LLM-as-a-Judge.py"),
        "--results", str(task_b_results),
        "--db"
    ] + existing_dbs + [
        "--output", str(task_b_judge_out),
        "--threads", str(threads),
        "--openai-config", str(config_path)
    ]
    run_cmd(cmd_score_b)

    print_step("Demo Pipeline Completed Successfully!")
    print(f"Check outputs in: {output_root}")

if __name__ == "__main__":
    main()
