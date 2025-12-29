from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9._-]", "_", text)
    return text[:200] or "unnamed"


def sha1_text(text: str) -> str:
    h = hashlib.sha1()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def sha1_json(obj: Any) -> str:
    return sha1_text(json.dumps(obj, ensure_ascii=False, sort_keys=True))


def file_sha1(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _extract_json(text: str) -> Dict[str, Any]:
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
        return {"_parsed": obj}

    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output is not JSON")
    obj = json.loads(t[start : end + 1])
    if isinstance(obj, dict):
        return obj
    return {"_parsed": obj}


def clean_response_to_json(text: str) -> Dict[str, Any]:
    """清洗 LLM 返回文本并解析为 JSON (增强版)。

    - 移除 <think>...</think>
    - 移除 Markdown 代码块标记 (```json ... ```)
    - 优先使用 json_repair 容错解析（若安装）
    - 兜底：提取第一个 '{' 到最后一个 '}' 再 json.loads
    """

    if not text:
        return {}

    t = text
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)
    t = re.sub(r"```json\s*", "", t, flags=re.IGNORECASE)
    t = t.replace("```", "")
    t = t.strip()

    # Prefer json_repair for tolerance.
    try:
        import json_repair  # type: ignore

        obj = json_repair.loads(t)
        if isinstance(obj, dict):
            return obj
        if obj is None:
            return {}
        return {"_parsed": obj}
    except Exception as e:
        # Fallback: best-effort brace extraction.
        try:
            return _extract_json(t)
        except Exception:
            print(f"❌ JSON Repair/parse failed. Raw text snippet: {t[:100]}...")
            raise e
