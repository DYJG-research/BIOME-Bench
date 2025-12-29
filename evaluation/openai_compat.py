from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from urllib.parse import urlsplit, urlunsplit

import requests


class OpenAICompatError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAICompatClient:
    base_url: str
    api_key: str
    model: str
    timeout_s: int
    max_retries: int
    retry_backoff_s: float

    def _chat_completions_url(self) -> str:
        base = self.base_url.strip().rstrip("/")
        if not base:
            return "/v1/chat/completions"

        u = urlsplit(base)
        path = u.path.rstrip("/")

        # If user already provided a versioned base like /v4 or /v1, do NOT force /v1.
        if re.search(r"/v\d+$", path):
            new_path = path + "/chat/completions"
        else:
            new_path = path + "/v1/chat/completions"

        return urlunsplit((u.scheme, u.netloc, new_path, u.query, u.fragment))

    def chat_completions(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        extra_payload: dict[str, Any] | None = None,
    ) -> str:
        logger = logging.getLogger("multiomics_evaluation")
        url = self._chat_completions_url()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if extra_payload:
            for k, v in extra_payload.items():
                if k in {"model", "messages", "temperature", "max_tokens"}:
                    continue
                payload[k] = v

        # Split timeout into (connect, read) to avoid very long hangs on unreachable hosts.
        connect_timeout_s = max(1, min(10, int(self.timeout_s)))
        timeout = (connect_timeout_s, int(self.timeout_s))

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt == 0:
                    logger.info("POST %s (timeout=%ss, retries=%d)", url, self.timeout_s, self.max_retries)
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code >= 400:
                    raise OpenAICompatError(f"HTTP {resp.status_code}: {resp.text[:1000]}")

                data = resp.json()
                choices = data.get("choices")
                if not choices:
                    raise OpenAICompatError(f"No choices in response: {data}")
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if content is None:
                    raise OpenAICompatError(f"No message.content in response: {data}")
                return str(content)
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt >= self.max_retries:
                    break
                sleep_s = self.retry_backoff_s * (2**attempt)
                logger.warning(
                    "Request failed (attempt %d/%d): %s; retry in %.2fs",
                    attempt + 1,
                    self.max_retries + 1,
                    repr(e),
                    sleep_s,
                )
                time.sleep(sleep_s)

        raise OpenAICompatError(f"Request failed after retries: {last_err}")
