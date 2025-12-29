from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .openai_client import OpenAIChatConfig, OpenAIEmbedConfig


@dataclass(frozen=True)
class TaskConfigs:
    judge: OpenAIChatConfig
    common: OpenAIChatConfig
    embed: OpenAIEmbedConfig


def load_task_configs(path: Path) -> TaskConfigs:
    raw = json.loads(path.read_text(encoding="utf-8"))

    def _get(d: dict[str, Any], key: str, default: Any = None) -> Any:
        v = d.get(key, default)
        return v

    def _chat(section: str) -> OpenAIChatConfig:
        sec = raw.get(section) or {}
        api = sec.get("api_config") or {}
        gen = sec.get("generation_config") or {}
        return OpenAIChatConfig(
            model=str(_get(api, "model", "")),
            base_url=str(_get(api, "base_url", "")),
            api_key=str(_get(api, "api_key", "")),
            timeout_s=int(_get(api, "timeout", 60)),
            max_retries=int(_get(api, "max_retries", 10)),
            retry_delay_s=float(_get(api, "retry_delay", 5)),
            max_tokens=int(_get(gen, "max_tokens", 10240)),
            temperature=float(_get(gen, "temperature", 0.0)),
        )

    def _embed(section: str) -> OpenAIEmbedConfig:
        sec = raw.get(section) or {}
        api = sec.get("api_config") or {}
        return OpenAIEmbedConfig(
            model=str(_get(api, "model", "")),
            base_url=str(_get(api, "base_url", "")),
            api_key=str(_get(api, "api_key", "")),
            timeout_s=int(_get(api, "timeout", 60)),
            max_retries=int(_get(api, "max_retries", 10)),
            retry_delay_s=float(_get(api, "retry_delay", 5)),
        )

    return TaskConfigs(judge=_chat("JudgeModel"), common=_chat("JudgeModel"), embed=_embed("EmbedModel"))
