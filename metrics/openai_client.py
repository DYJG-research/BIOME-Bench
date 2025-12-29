from __future__ import annotations

import atexit
import json
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


_thread_local = threading.local()
_clients_lock = threading.Lock()
_clients: list[Any] = []
_cleanup_registered = False


def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            # Best-effort cleanup; never fail during interpreter shutdown.
            pass


def _close_all_clients() -> None:
    with _clients_lock:
        clients = list(_clients)
        _clients.clear()
    for c in clients:
        _close_client(c)


def _register_cleanup_once() -> None:
    global _cleanup_registered
    if _cleanup_registered:
        return
    _cleanup_registered = True
    atexit.register(_close_all_clients)


def _get_thread_openai_client(OpenAI_cls: Any, *, base_url: str, api_key: str, timeout_s: int) -> Any:
    """Return a per-thread OpenAI client.

    The OpenAI Python client holds an underlying HTTP connection pool.
    Creating a new client per request (especially with keep-alive) can leak
    sockets and exhaust server/client resources under multithreading.
    """

    key = (base_url, api_key, int(timeout_s))
    client = getattr(_thread_local, "openai_client", None)
    client_key = getattr(_thread_local, "openai_client_key", None)
    if client is not None and client_key == key:
        return client

    if client is not None:
        _close_client(client)

    new_client = OpenAI_cls(base_url=base_url, api_key=api_key, timeout=timeout_s)
    _thread_local.openai_client = new_client
    _thread_local.openai_client_key = key

    with _clients_lock:
        _clients.append(new_client)
    _register_cleanup_once()
    return new_client


@dataclass(frozen=True)
class OpenAIChatConfig:
    model: str
    base_url: str
    api_key: str
    timeout_s: int
    max_retries: int
    retry_delay_s: float
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class OpenAIEmbedConfig:
    model: str
    base_url: str
    api_key: str
    timeout_s: int
    max_retries: int
    retry_delay_s: float


def _normalize_base_url(base_url: str) -> str:
    u = base_url.strip().rstrip("/")
    if not u:
        return u
    # If caller already supplies a versioned base (e.g., /v1, /v4), keep as-is.
    if re.search(r"/v\d+$", u):
        return u
    return u + "/v1"


def _extract_json(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output is not JSON")
    return json.loads(t[start : end + 1])


def clean_response_to_json(text: str) -> dict[str, Any]:
    """Clean LLM output and parse as JSON.

    Handles common noise:
    - <think>...</think>
    - Markdown code fences (```json ... ```)
    - Slightly broken JSON via json_repair (when installed)
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
        # Some models might return a list; keep only dict to match our contract.
        if obj is None:
            return {}
        return {"_parsed": obj}
    except Exception:
        # Fallback: best-effort brace extraction.
        return _extract_json(t)


def _sleep_backoff(base: float, attempt: int) -> None:
    # exponential backoff + jitter
    delay = base * (2**attempt)
    delay *= 0.75 + random.random() * 0.5
    # Avoid extremely long sleeps that look like a hang.
    delay = min(delay, 30.0)
    time.sleep(delay)


def chat_json(*, cfg: OpenAIChatConfig, system_prompt: str, user_prompt: str, log_name: str = "openai") -> dict[str, Any]:
    # Force disable proxy for localhost connections to avoid 502 errors from intermediate proxies
    import os
    os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1,0.0.0.0"

    from openai import OpenAI  # local import to avoid hard dependency for unrelated tests

    from .log import get_logger

    log = get_logger(log_name)

    base_url = _normalize_base_url(cfg.base_url)
    # Workaround for httpx/localhost issues: prefer 127.0.0.1
    if "localhost" in base_url:
        base_url = base_url.replace("localhost", "127.0.0.1")
    
    client = _get_thread_openai_client(OpenAI, base_url=base_url, api_key=cfg.api_key, timeout_s=cfg.timeout_s)

    # Some OpenAI-compatible servers reject newer fields (e.g., response_format)
    # or large max_tokens; progressively fall back when we see 400.
    allow_response_format = False
    max_tokens_override: Optional[int] = None

    last_err: Optional[BaseException] = None
    for attempt in range(cfg.max_retries + 1):
        t_attempt = time.time()
        log.info(
            "chat request",
            attempt=f"{attempt+1}/{cfg.max_retries+1}",
            model=cfg.model,
            base_url=base_url,
            timeout_s=cfg.timeout_s,
        )
        try:
            mt = max_tokens_override if max_tokens_override is not None else cfg.max_tokens
            kwargs: dict[str, Any] = {
                "model": cfg.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": cfg.temperature,
                "max_tokens": mt,
            }
            # If supported by backend/model, force JSON output.
            if allow_response_format:
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            log.success("chat ok", attempt=attempt + 1, elapsed_s=f"{time.time()-t_attempt:.2f}")
            return clean_response_to_json(content)
        except Exception as e:  # noqa: BLE001
            last_err = e
            
            # Extract detailed error info if available
            err_msg = f"{type(e).__name__}: {e}"
            if hasattr(e, "body") and e.body:
                err_msg += f" | Body: {e.body}"
            
            log.error(
                "chat error",
                attempt=attempt + 1,
                elapsed_s=f"{time.time()-t_attempt:.2f}",
                err=err_msg,
            )

            # Heuristic fallback for OpenAI-compatible servers returning 400.
            # We avoid importing OpenAI exception types at module import time.
            is_400 = " 400" in str(e) or "400" in str(e)
            if is_400:
                if allow_response_format:
                    allow_response_format = False
                    log.warn("fallback", reason="disable response_format")
                elif max_tokens_override is None and cfg.max_tokens > 2048:
                    max_tokens_override = 2048
                    log.warn("fallback", reason="max_tokens -> 2048")
                elif max_tokens_override is not None and max_tokens_override > 1024:
                    max_tokens_override = 1024
                    log.warn("fallback", reason="max_tokens -> 1024")

            if attempt >= cfg.max_retries:
                break
            log.info("retrying", backoff_base_s=cfg.retry_delay_s, backoff_cap_s=30)
            _sleep_backoff(cfg.retry_delay_s, attempt)
    raise RuntimeError(f"OpenAI chat_json failed after retries: {last_err}")


def embed(*, cfg: OpenAIEmbedConfig, text: str, log_name: str = "openai") -> list[float]:
    # Force disable proxy for localhost connections
    import os
    os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1,0.0.0.0"

    from openai import OpenAI  # local import

    from .log import get_logger

    log = get_logger(log_name)

    base_url = _normalize_base_url(cfg.base_url)
    client = _get_thread_openai_client(OpenAI, base_url=base_url, api_key=cfg.api_key, timeout_s=cfg.timeout_s)

    last_err: Optional[BaseException] = None
    for attempt in range(cfg.max_retries + 1):
        t_attempt = time.time()
        log.info(
            "embed request",
            attempt=f"{attempt+1}/{cfg.max_retries+1}",
            model=cfg.model,
            base_url=_normalize_base_url(cfg.base_url),
            timeout_s=cfg.timeout_s,
        )
        try:
            resp = client.embeddings.create(model=cfg.model, input=text)
            log.success("embed ok", attempt=attempt + 1, elapsed_s=f"{time.time()-t_attempt:.2f}")
            return list(resp.data[0].embedding)
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.error(
                "embed error",
                attempt=attempt + 1,
                elapsed_s=f"{time.time()-t_attempt:.2f}",
                err=f"{type(e).__name__}: {e}",
            )
            if attempt >= cfg.max_retries:
                break
            log.info("retrying", backoff_base_s=cfg.retry_delay_s, backoff_cap_s=30)
            _sleep_backoff(cfg.retry_delay_s, attempt)
    raise RuntimeError(f"OpenAI embed failed after retries: {last_err}")
