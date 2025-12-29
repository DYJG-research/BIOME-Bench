from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_sh(tasks_path: Path, out_path: Path) -> None:
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("tasks file must be a JSON array")

    lines: list[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    
    # Import locally to avoid circular import issues if any
    from .config import InferenceConfig

    for t in tasks:
        if not isinstance(t, dict):
            raise ValueError("each task must be an object")

        data = t["data"]
        task_type = t["task_type"]
        
        # Optional overrides in tasks.json
        base_url = t.get("base_url")
        api_key = t.get("api_key")
        model = t.get("model")
        
        threads = int(t.get("threads", 1))
        config_path = t.get("config", "config/config.json")
        outputs = t.get("outputs", "outputs")
        
        # Validate that if not in task, it must be in config
        if not (base_url and model):
            try:
                cfg_obj = InferenceConfig.from_json_file(config_path)
                if not base_url:
                    base_url = cfg_obj.base_url
                if not model:
                    model = cfg_obj.model
                # api_key can be empty/None from config
            except Exception:
                # Config might not exist yet or be invalid; orchestrator generates script anyway.
                # But we should warn or let runtime fail.
                pass
        
        cmd_parts = [
            "multiomics-eval run",
            f"--data {data}",
            f"--task-type {task_type}",
            f"--threads {threads}",
            f"--config {config_path}",
            f"--outputs {outputs}",
        ]
        
        # Only add flags if they are explicitly overriding config, 
        # OR if we want to "bake in" the values into the script.
        # However, since the CLI now supports loading from config, 
        # we can omit them if they match config defaults. 
        # But for robustness/reproducibility of the script, it is safer 
        # to include them if they were specified in tasks.json.
        # If they were NOT in tasks.json, we let the CLI load from config at runtime.
        
        if t.get("base_url"):
            cmd_parts.append(f"--base-url {base_url}")
        if t.get("api_key"):
            cmd_parts.append(f"--api-key {api_key}")
        if t.get("model"):
            cmd_parts.append(f"--model {model}")

        lines.append(" ".join(cmd_parts))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
