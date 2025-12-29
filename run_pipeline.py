#!/usr/bin/env python3
import sys
import subprocess
import json
import argparse
import shutil
from pathlib import Path

# --- Defaults ---
DEFAULT_CONFIG = "config/config.json"
DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("outputs")
METRICS_DIR = Path("metrics")

TASK_A_DATA = DATA_DIR / "TASK-A.jsonl"
TASK_B_DATA = DATA_DIR / "TASK-B.jsonl"

DB_FILES = [
    DATA_DIR / "hsa.jsonl",
    DATA_DIR / "mmu.jsonl",
    DATA_DIR / "rno.jsonl"
]

def check_dependencies():
    print("[*] Checking dependencies...")
    try:
        import rich
        import requests
        import openai
        print("[+] Dependencies detected.")
    except ImportError:
        print("[-] Missing dependencies. Installing from requirements.txt...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("[+] Installation complete.")
        except subprocess.CalledProcessError:
            print("[!] Failed to install dependencies.")
            sys.exit(1)

def load_model_name(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get("EvalModel", {}).get("api_config", {}).get("model", "UnknownModel")
    except Exception as e:
        print(f"Error reading config {config_path}: {e}")
        return "UnknownModel"

def find_latest_results(task_dir: Path) -> Path:
    if not task_dir.exists():
         raise FileNotFoundError(f"Task directory not found: {task_dir}")
    runs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not runs:
        raise FileNotFoundError(f"No run directories found in {task_dir}")
    runs.sort(key=lambda x: x.name)
    latest_run = runs[-1]
    results_file = latest_run / "results.jsonl"
    if not results_file.exists():
         raise FileNotFoundError(f"results.jsonl not found in {latest_run}")
    return results_file

def print_step(msg):
    print(f"\n{'='*60}")
    print(f"[*] {msg}")
    print(f"{ '='*60}")

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser(description="BIOME-Bench Full Pipeline Runner")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to config.json")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads")
    parser.add_argument("--outputs", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--data-a", type=str, default=str(TASK_A_DATA), help="Path to Task A data")
    parser.add_argument("--data-b", type=str, default=str(TASK_B_DATA), help="Path to Task B data")
    parser.add_argument("--db", type=str, nargs="+", default=[str(f) for f in DB_FILES], help="Path to DB files")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation and only run scoring")
    
    args = parser.parse_args()
    
    check_dependencies()
    
    config_path = Path(args.config)
    output_root = Path(args.outputs)
    threads = args.threads
    task_a_data = Path(args.data_a)
    task_b_data = Path(args.data_b)
    db_files = [Path(f) for f in args.db]
    model_name = load_model_name(config_path)

    print_step(f"Pipeline Config:\nModel: {model_name}\nConfig: {config_path}\nThreads: {threads}\nOutputs: {output_root}\nTask A Data: {task_a_data}\nTask B Data: {task_b_data}\nDB Files: {db_files}")

    # ... (rest of the code) ...

    # 4. Scoring Task B
    print_step("Scoring Task B (LLM-as-a-Judge)")
    try:
        task_b_results = find_latest_results(output_root / model_name / "task2_analysis")
        judge_out = task_b_results.parent / "judge_results.jsonl"
        
        existing_dbs = [str(db) for db in db_files if db.exists()]
        if not existing_dbs:
            print("Error: No DB files found for scoring.")
            sys.exit(1)

        run_cmd([
            sys.executable, str(METRICS_DIR / "LLM-as-a-Judge.py"),
            "--results", str(task_b_results),
            "--db"
        ] + existing_dbs + [
            "--output", str(judge_out),
            "--threads", str(threads),
            "--openai-config", str(config_path)
        ])
    except Exception as e:
        print(f"Error scoring Task B: {e}")

    print_step("Pipeline Completed!")

if __name__ == "__main__":
    main()