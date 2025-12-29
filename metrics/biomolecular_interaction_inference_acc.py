import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Dict, Tuple


REL_VOCAB = [
    "activates",
    "inhibits",
    "upregulates_expression",
    "downregulates_expression",
    "regulates",
    "binds",
    "dissociates_from",
    "phosphorylates",
    "dephosphorylates",
    "ubiquitinates",
    "glycosylates",
    "methylates",
    "produces",
    "consumes",
    "converts_to",
    "leads_to",
    "increases_level",
    "decreases_level",
]
REL_SET = set(REL_VOCAB)


def _normalize_text(text: str) -> str:
    t = text.strip().lower()
    # strip common wrappers
    t = t.replace("**", " ")
    t = t.replace("`", " ")
    t = t.replace("\n", "\n")
    t = re.sub(r"<\s*/?\s*think\s*>", " ", t)
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    return t.strip()


# word-boundary-like match for snake_case labels
_REL_PATTERN = re.compile(
    "|".join(
        rf"(?<![a-z0-9_]){re.escape(r)}(?![a-z0-9_])"
        for r in sorted(REL_VOCAB, key=len, reverse=True)
    )
)


def _strip_punct(s: str) -> str:
    # remove common punctuation around labels, e.g. "relation: activates", '"activates"', etc.
    s = s.strip()
    s = s.strip(" \t\r\n\"'“”‘’`.,;:()[]{}<>")
    return s.strip()


def extract_relation(text: Optional[str]) -> Optional[str]:
    """
    Robustly extract relation label from model output.
    Strategy:
      1) Check from tail: last non-empty line exact-match (after light punctuation stripping).
      2) Otherwise, search the entire normalized text and return the *last* occurrence of any label.
    """
    if not text:
        return None

    raw = text
    # 1) tail-first exact match on lines
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    for ln in reversed(lines):
        cand = _strip_punct(_normalize_text(ln))
        if cand in REL_SET:
            return cand
        # also allow patterns like "relation: activates"
        if ":" in cand:
            tail = _strip_punct(cand.split(":")[-1])
            if tail in REL_SET:
                return tail

    # 2) tail-first by taking last regex match in whole text
    t = _normalize_text(raw)
    t_flat = t.replace("\n", " ")
    if t_flat in REL_SET:
        return t_flat

    matches = list(_REL_PATTERN.finditer(t_flat))
    if not matches:
        return None
    return matches[-1].group(0)


def iter_jsonl(path: Path, *, limit: int | None = None, skip_bad_lines: bool = True) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        yielded = 0
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:  # noqa: BLE001
                if skip_bad_lines:
                    continue
                raise ValueError(f"Invalid JSON at line {line_no}")
            yield obj
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def get_gold(record: dict) -> Optional[str]:
    msgs = record.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    last = msgs[-1]
    if not isinstance(last, dict):
        return None
    return extract_relation(last.get("content"))


def get_pred(record: dict) -> Optional[str]:
    eval_result = record.get("eval_result")
    if not isinstance(eval_result, dict):
        return None
    return extract_relation(eval_result.get("response"))


@dataclass(frozen=True)
class Metrics:
    total: int
    correct: int
    valid_pred: int
    acc: float
    micro_recall: float
    micro_precision: float
    micro_f1: float
    macro_f1: float
    macro_f1_all: float


def _safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def compute_metrics(records: Iterable[dict]) -> Metrics:
    total = 0
    correct = 0
    valid_pred = 0

    # per-class counts for true macro-F1
    tp: Dict[str, int] = {r: 0 for r in REL_VOCAB}
    fp: Dict[str, int] = {r: 0 for r in REL_VOCAB}
    fn: Dict[str, int] = {r: 0 for r in REL_VOCAB}
    support: Dict[str, int] = {r: 0 for r in REL_VOCAB}  # gold frequency

    for rec in records:
        gold = get_gold(rec)
        pred = get_pred(rec)

        if gold is None:
            continue
        if gold not in REL_SET:
            # if gold label is malformed/outside vocab, skip to avoid corrupting metrics
            continue

        total += 1
        support[gold] += 1

        if pred is not None:
            if pred in REL_SET:
                valid_pred += 1
            else:
                pred = None  # treat as invalid
        # update correct / per-class
        if pred == gold:
            correct += 1
            tp[gold] += 1
        else:
            fn[gold] += 1
            if pred is not None:
                # wrong but valid label => FP for predicted label
                fp[pred] += 1

    acc = _safe_div(correct, total)

    # micro (single-label): micro-recall == accuracy when counting invalid as wrong
    micro_recall = acc
    micro_precision = _safe_div(correct, valid_pred)
    micro_f1 = (
        _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)
        if (micro_precision + micro_recall) else 0.0
    )

    # true macro-F1: per-class F1 then average
    per_class_f1: Dict[str, float] = {}
    for r in REL_VOCAB:
        prec_r = _safe_div(tp[r], tp[r] + fp[r])
        rec_r = _safe_div(tp[r], tp[r] + fn[r])
        f1_r = _safe_div(2 * prec_r * rec_r, prec_r + rec_r) if (prec_r + rec_r) else 0.0
        per_class_f1[r] = f1_r

    # macro over classes that appear in gold (common in papers to avoid averaging many empty classes)
    present = [r for r in REL_VOCAB if support[r] > 0]
    macro_f1 = sum(per_class_f1[r] for r in present) / len(present) if present else 0.0

    # macro over all classes (including those with 0 support, will be smaller if many unused labels)
    macro_f1_all = sum(per_class_f1.values()) / len(REL_VOCAB) if REL_VOCAB else 0.0

    return Metrics(
        total=total,
        correct=correct,
        valid_pred=valid_pred,
        acc=acc,
        micro_recall=micro_recall,
        micro_precision=micro_precision,
        micro_f1=micro_f1,
        macro_f1=macro_f1,
        macro_f1_all=macro_f1_all,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Compute acc/micro/macro metrics for relation classification.")
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to results.jsonl (each line contains messages with gold + eval_result.response as prediction).",
    )
    p.add_argument("--limit", type=int, default=None, help="Only evaluate first N JSONL records")
    p.add_argument(
        "--no-skip-bad-lines",
        action="store_true",
        help="Fail fast if a JSONL line is invalid (default: skip bad/truncated lines)",
    )
    args = p.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")

    m = compute_metrics(iter_jsonl(args.input, limit=args.limit, skip_bad_lines=not args.no_skip_bad_lines))
    print(
        json.dumps(
            {
                "total": m.total,
                "correct": m.correct,
                "valid_pred": m.valid_pred,
                "acc": round(m.acc, 6),
                "micro_precision": round(m.micro_precision, 6),
                "micro_recall": round(m.micro_recall, 6),
                "micro_f1": round(m.micro_f1, 6),
                "macro_f1": round(m.macro_f1, 6),
                "macro_f1_all": round(m.macro_f1_all, 6),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
