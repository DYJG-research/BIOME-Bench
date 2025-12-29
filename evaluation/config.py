from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceConfig:
    # API Config
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    request_timeout_s: int = 120
    max_retries: int = 5
    retry_backoff_s: float = 1.0

    # Generation Config
    temperature: float = 0.0
    max_tokens: int = 8192
    no_think: bool = False
    thinking_rules_file: str = "thinking_rules.json"

    # Runtime
    save_every: int = 0

    @staticmethod
    def from_json_file(path: str | Path) -> "InferenceConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        # Support root level EvalModel or direct root for backward compat logic if needed,
        # but here we strictly follow the new structure as requested.
        data = raw.get("EvalModel", raw)
        
        api = data.get("api_config", {})
        gen = data.get("generation_config", {})
        
        # Flatten into one config object
        return InferenceConfig(
            # API
            model=str(api.get("model", "")),
            base_url=str(api.get("base_url", "")),
            api_key=str(api.get("api_key", "")),
            request_timeout_s=int(api.get("timeout", 120)),
            max_retries=int(api.get("max_retries", 5)),
            retry_backoff_s=float(api.get("retry_delay", 1.0)),
            
            # Gen
            temperature=float(gen.get("temperature", 0.0)),
            max_tokens=int(gen.get("max_tokens", 8192)),
            no_think=bool(gen.get("no_think", False)),
            thinking_rules_file=str(gen.get("thinking_rules_file", "thinking_rules.json")),
            
            # Runtime
            save_every=int(data.get("save_every", 0)),
        )
