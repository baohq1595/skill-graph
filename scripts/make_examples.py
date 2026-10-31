"""Regenerate docs/examples from the running system.

    python scripts/make_examples.py            # offline (rule parser + templates)
    GEMINI_API_KEY=... python scripts/make_examples.py --llm

Writes docs/examples/<mode>/: one file per query type and a natural-language
transcript. Nothing in these files is hand-written.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from skillgraph import render
from skillgraph.context import get_context
from skillgraph.nl.llm import llm_from_env
from skillgraph.nl.pipeline import Assistant

ROOT = Path(__file__).resolve().parents[1]
NL_QUESTIONS = [
    # the three required templates
    "What skills do I need to become a Data Scientist?",
    "I know Python and SQL. What's the fastest path to AI Engineer?",
    "Which Digital skills are most transferable to the Green economy?",
    # the five sample queries shipped with the dataset
    "What skills does a Data Scientist need?",
    "I know Python and SQL. What is the fastest path to becoming an AI Engineer?",
    "Which skills are most transferable between the Digital and Green economies?",
    "I am a healthcare data analyst. What skill gaps do I have for a Sustainability Consultant role?",
    "What SkillsFuture courses should I take to learn Machine Learning?",
    # how the system handles ambiguity and things it cannot answer
    "What skills do I need to become a data analyst?",
    "What's the weather in Singapore?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="Use Gemini (needs GEMINI_API_KEY).")
    args = ap.parse_args()
    ctx = get_context()
    llm = llm_from_env() if args.llm else None
    if args.llm and llm is None:
        raise SystemExit("--llm needs GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.")
    mode = "llm" if llm else "offline"
    out = ROOT / "docs" / "examples" / mode
    out.mkdir(parents=True, exist_ok=True)
    stamp = f"_Generated {date.today().isoformat()} by `scripts/make_examples.py`" + \
            (f" with `{llm.model}`._" if llm else " in offline mode._")

    if not llm:
        for kind, run in (("learning_path", ctx.service.learning_path), ("gap_analysis", ctx.service.gap_analysis)):
            parts = [f"# {kind.replace('_', ' ').title()} — sample learners", "", stamp, ""]
            for lp in ctx.dataset.learner_profiles:
                result = run(lp.current_skills, lp.target_role)
                parts += [f"---\n\n**{lp.id} · {lp.name}** — {lp.description}  ",
                          "Current skills: " + ", ".join(ctx.store.names(lp.current_skills)), "",
                          render.render(result)]
            (out / f"{kind}.md").write_text("\n".join(parts), encoding="utf-8")
        t = ctx.service.transferability("Digital", "Green")
        (out / "transferability.md").write_text(f"{stamp}\n\n{render.render(t)}", encoding="utf-8")
        (out / "learning_path_LP-02.json").write_text(
            ctx.service.learning_path(ctx.learner("LP-02").current_skills, "ROL-02").model_dump_json(indent=2),
            encoding="utf-8")

    assistant = Assistant(ctx, llm=llm)
    parts = [f"# Natural-language queries ({mode})", "", stamp, ""]
    for q in NL_QUESTIONS:
        r = assistant.ask(q)
        usage = [s.detail.get("usage") for s in r.trace if s.detail.get("usage")]
        parts += ["---", "", f"### Q: {q}", "",
                  f"`status={r.status}` · `parser={r.parser}` · `answer={r.answer_source}`"
                  + (f" · tokens {usage}" if usage else ""), "",
                  "<details><summary>Parsed intent</summary>", "", "```json",
                  json.dumps(r.parsed, indent=2), "```", "</details>", "",
                  r.answer, ""]
        if r.grounding_problems:
            parts += ["> LLM answer rejected by the grounding check: " + "; ".join(r.grounding_problems), ""]
    (out / "nl_queries.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
