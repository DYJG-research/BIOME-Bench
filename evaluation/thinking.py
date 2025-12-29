from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_RESERVED_PAYLOAD_KEYS = {"model", "messages", "temperature", "max_tokens"}


@dataclass(frozen=True)
class ThinkingAction:
    prompt_prefix: str | None = None
    request_params: dict[str, Any] | None = None


@dataclass(frozen=True)
class ThinkingRule:
    match_contains: list[str]
    when_no_think: ThinkingAction
    when_think: ThinkingAction


@dataclass(frozen=True)
class ThinkingRules:
    rules: list[ThinkingRule]

    @staticmethod
    def from_json_file(path: str | Path) -> "ThinkingRules":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rules_data = data.get("rules")
        if not isinstance(rules_data, list):
            raise ValueError("thinking_rules.json must contain a top-level list field 'rules'")

        rules: list[ThinkingRule] = []
        for item in rules_data:
            if not isinstance(item, dict):
                continue
            match_contains = item.get("match_contains")
            if isinstance(match_contains, str):
                match_contains = [match_contains]
            if not isinstance(match_contains, list) or not match_contains:
                continue
            match_contains = [str(x).lower() for x in match_contains if str(x).strip()]
            if not match_contains:
                continue

            when_no_think = _parse_action(item.get("when_no_think"))
            when_think = _parse_action(item.get("when_think"))
            rules.append(ThinkingRule(match_contains=match_contains, when_no_think=when_no_think, when_think=when_think))

        return ThinkingRules(rules=rules)

    def resolve(self, *, model: str, no_think: bool) -> ThinkingAction:
        name = (model or "").lower()
        for rule in self.rules:
            if any(s in name for s in rule.match_contains):
                action = rule.when_no_think if no_think else rule.when_think
                return _sanitize_action(action)
        return ThinkingAction()


def _parse_action(obj: Any) -> ThinkingAction:
    if not isinstance(obj, dict):
        return ThinkingAction()
    prompt_prefix = obj.get("prompt_prefix")
    if prompt_prefix is not None:
        prompt_prefix = str(prompt_prefix)
    request_params = obj.get("request_params")
    if request_params is not None and not isinstance(request_params, dict):
        request_params = None
    return ThinkingAction(prompt_prefix=prompt_prefix, request_params=request_params)


def _sanitize_action(action: ThinkingAction) -> ThinkingAction:
    params = dict(action.request_params or {})
    for k in list(params.keys()):
        if k in _RESERVED_PAYLOAD_KEYS:
            params.pop(k, None)
    return ThinkingAction(prompt_prefix=action.prompt_prefix, request_params=params or None)
