"""Evaluate the natural-language pipeline against ``eval/nl_cases.yaml``.

Scores each case on the outcome after resolution (graph ids), so a parser is
judged on whether the right query would run, not on surface wording:

  status     the pipeline outcome (answered / needs_clarification / ...)
  intent     the query type chosen
  slots      every specified slot matches (role ids, skill-id sets, economies)
  exact      status, intent and all slots correct together

With an LLM configured, the summary also counts the cases where a failed call
fell back to the rule parser, so LLM accuracy is never silently mixed with
rule-parser accuracy, and how many written answers the grounding check rejected.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from skillgraph.nl.pipeline import Assistant

DEFAULT_CASES = Path(__file__).resolve().parents[3] / "eval" / "nl_cases.yaml"
SLOTS = ("target_role", "current_role", "known_skills", "skills", "source_economy", "target_economy", "bidirectional")


@dataclass
class CaseResult:
    id: str
    group: str
    question: str
    ok_status: bool
    ok_intent: bool
    ok_slots: bool
    exact: bool
    got: dict
    expected: dict
    ms: float
    misses: list[str] = field(default_factory=list)
    parser: str = ""
    answer_source: str = ""
    grounding_problems: list[str] = field(default_factory=list)


def _observed(result) -> dict:
    r = result.resolved
    return {
        "status": result.status,
        "intent": result.intent.value if result.intent else None,
        "target_role": (r.get("target_role") or {}).get("id"),
        "current_role": (r.get("current_role") or {}).get("id"),
        "known_skills": sorted(s["id"] for s in r.get("known_skills", [])
                               if not r.get("current_role")),  # skills inferred from a role are not "said"
        "skills": sorted(s["id"] for s in r.get("skills", [])),
        "source_economy": r.get("source_economy"),
        "target_economy": r.get("target_economy"),
        "bidirectional": r.get("bidirectional"),
    }


def evaluate(assistant: Assistant, cases_path: Path = DEFAULT_CASES) -> tuple[dict, list[CaseResult]]:
    cases = yaml.safe_load(cases_path.read_text(encoding="utf-8"))
    rows: list[CaseResult] = []
    for case in cases:
        t0 = time.perf_counter()
        result = assistant.ask(case["q"])
        ms = (time.perf_counter() - t0) * 1000
        got = _observed(result)
        misses = []
        ok_status = got["status"] == case["status"]
        if not ok_status:
            misses.append(f"status {got['status']} != {case['status']}")
        ok_intent = "intent" not in case or got["intent"] == case["intent"]
        if not ok_intent:
            misses.append(f"intent {got['intent']} != {case['intent']}")
        ok_slots = True
        for slot in SLOTS:
            if slot not in case:
                continue
            want = sorted(case[slot]) if isinstance(case[slot], list) else case[slot]
            if got[slot] != want:
                ok_slots = False
                misses.append(f"{slot} {got[slot]} != {want}")
        rows.append(CaseResult(case["id"], case["group"], case["q"], ok_status, ok_intent, ok_slots,
                               ok_status and ok_intent and ok_slots,
                               {k: v for k, v in got.items() if k in case or k == "status"},
                               {k: v for k, v in case.items() if k not in ("id", "group", "q")},
                               round(ms, 1), misses, result.parser, result.answer_source,
                               result.grounding_problems))
    n = len(rows)
    summary = {
        "parser": "llm" if assistant.llm is not None else "rules",
        "model": getattr(assistant.llm, "model", None),
        "cases": n,
        "status_accuracy": round(sum(r.ok_status for r in rows) / n, 3),
        "intent_accuracy": round(sum(r.ok_intent for r in rows) / n, 3),
        "slot_accuracy": round(sum(r.ok_slots for r in rows) / n, 3),
        "exact_match": round(sum(r.exact for r in rows) / n, 3),
        "by_group": {g: f"{sum(r.exact for r in rows if r.group == g)}/{sum(r.group == g for r in rows)}"
                     for g in dict.fromkeys(r.group for r in rows)},
        "median_ms": sorted(r.ms for r in rows)[n // 2],
        "parser_fallbacks": sum(r.parser == "rules" for r in rows) if assistant.llm is not None else 0,
        "llm_answers": sum(r.answer_source == "llm" for r in rows),
        "grounding_rejections": sum(bool(r.grounding_problems) for r in rows),
    }
    return summary, rows


def save(summary: dict, rows: list[CaseResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "cases": [asdict(r) for r in rows]}, indent=2,
                               ensure_ascii=False), encoding="utf-8")
