"""Cheap hallucination check for LLM-written answers.

Before an LLM answer is shown it must pass two checks against the query
result it was given:

1. Every catalogue entity it names (skill, role, course, provider) appears in
   the result. Naming a course that is not in the result is exactly the
   failure the user would act on.
2. Every hour figure and percentage it quotes can be found in the result.

If either fails, the pipeline falls back to the deterministic template answer
and records what was caught.
"""

from __future__ import annotations

import json
import re


def _numbers(obj) -> set[float]:
    found: set[float] = set()
    for token in re.findall(r"-?\d+(?:\.\d+)?", json.dumps(obj)):
        x = float(token)
        found |= {round(x, 1), float(round(x))}
        if 0 < x <= 1:  # shares stored as 0-1 may be quoted as percentages
            found |= {round(x * 100, 1), float(round(x * 100))}
    return found


def check_answer(answer: str, payload, catalogue_names: list[str]) -> list[str]:
    payload_text = json.dumps(payload, ensure_ascii=False).casefold()
    answer_cf = answer.casefold()
    problems: list[str] = []
    # longest names first, so "Deep Learning & Neural Networks" is judged before "Deep Learning"
    for name in sorted(set(catalogue_names), key=len, reverse=True):
        key = name.casefold()
        if len(key) < 4 or key not in answer_cf:
            continue
        if key not in payload_text:
            problems.append(f"names '{name}', which is not in the query result")
        answer_cf = answer_cf.replace(key, " ")
    allowed = _numbers(payload)
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(%|h\b|hrs?\b|hours?\b)", answer, re.I):
        v = round(float(value), 1)
        if v not in allowed and round(v, 0) not in allowed:
            problems.append(f"quotes {value}{unit}, which is not in the query result")
    return problems
