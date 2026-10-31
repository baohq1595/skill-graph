"""Natural language -> structured graph query -> grounded answer.

    question
      │ 1. parse    LLM structured output (or the rule parser offline / on failure)
      ▼
    ParsedQuery  (the user's words, no ids)
      │ 2. resolve  words -> graph ids via the resolver; ambiguity becomes a question back
      ▼
    resolved ids
      │ 3. execute  deterministic graph query (services.queries)
      ▼
    typed result
      │ 4. answer   LLM writes from the result only, then a grounding check;
      ▼             the template renderer is used offline or if the check fails
    AskResult (answer + result + full trace)

The LLM never touches the graph and never produces ids, and every number in an
answer comes from step 3.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from skillgraph import render
from skillgraph.context import AppContext
from skillgraph.graph.schema import Label
from skillgraph.nl.grounding import check_answer
from skillgraph.nl.llm import LLMClient, LLMError, llm_from_env
from skillgraph.nl.prompts import ANSWER_SYSTEM, PARSE_SYSTEM, answer_user_message, catalogue_block
from skillgraph.nl.rules import RuleParser
from skillgraph.nl.schema import Intent, ParsedQuery, llm_json_schema
from skillgraph.services.resolver import Kind, MatchType, Resolution

Status = Literal["answered", "needs_clarification", "not_found", "unsupported", "error"]

SUPPORTED = [
    "What skills do I need to become a Data Scientist?",
    "I know Python and SQL. What's the fastest path to AI Engineer?",
    "Which Digital skills are most transferable to the Green economy?",
    "I'm a health data analyst. What skill gaps do I have for Sustainability Consultant?",
    "What courses teach Machine Learning?",
]


class TraceStep(BaseModel):
    step: str
    ms: float
    detail: dict[str, Any] = Field(default_factory=dict)


class AskResult(BaseModel):
    question: str
    status: Status
    intent: Intent | None = None
    parser: str
    parsed: dict[str, Any] | None = None
    resolved: dict[str, Any] = Field(default_factory=dict)
    answer: str
    answer_source: Literal["llm", "template"]
    results: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    grounding_problems: list[str] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)


class _Stop(Exception):
    def __init__(self, status: Status, message: str) -> None:
        self.status, self.message = status, message


class Assistant:
    def __init__(self, ctx: AppContext, llm: LLMClient | None | Literal["env"] = "env") -> None:
        self.ctx = ctx
        self.llm = llm_from_env() if llm == "env" else llm
        self.rules = RuleParser(ctx.resolver)
        store = ctx.store
        self._catalogue = catalogue_block(
            [store.node(i)["name"] for i in store.ids(Label.ROLE)],
            [store.node(i)["name"] for i in store.ids(Label.SKILL)])
        self._entity_names = sorted(
            {store.node(i)["name"] for lab in (Label.SKILL, Label.ROLE, Label.COURSE) for i in store.ids(lab)}
            | {store.node(i)["provider"] for i in store.ids(Label.COURSE)})

    # ------------------------------------------------------------------ public
    def ask(self, question: str) -> AskResult:
        trace: list[TraceStep] = []
        question = (question or "").strip()
        if not question:
            return self._finish(question, "unsupported", None, "rules", None, {}, [], [],
                                "Ask a question about skills, roles or courses.", trace)
        parsed, parser = self._parse(question, trace)
        try:
            resolved, notes = self._resolve(parsed, trace)
            results = self._execute(parsed, resolved, trace)
        except _Stop as stop:
            return self._finish(question, stop.status, parsed, parser, parsed.model_dump(mode="json"),
                                {}, [], [], stop.message, trace)
        answer, source, problems = self._answer(question, parsed, results, notes, trace)
        out = self._finish(question, "answered", parsed, parser, parsed.model_dump(mode="json"), resolved,
                           [r.model_dump(mode="json") for r in results], notes, answer, trace,
                           source=source)
        out.grounding_problems = problems
        return out

    # ------------------------------------------------------------------- steps
    def _parse(self, question: str, trace: list[TraceStep]) -> tuple[ParsedQuery, str]:
        if self.llm is not None:
            t0 = time.perf_counter()
            try:
                data, reply = self.llm.json(PARSE_SYSTEM + "\n\n" + self._catalogue, question,
                                            llm_json_schema(), effort="low")
                parsed = ParsedQuery.model_validate(data)
                trace.append(TraceStep(step="parse", ms=_ms(t0), detail={
                    "parser": "llm", "model": reply.model, "cached": reply.cached, "usage": reply.usage,
                    "output": parsed.model_dump(mode="json")}))
                return parsed, "llm"
            except (LLMError, ValueError) as exc:
                trace.append(TraceStep(step="parse", ms=_ms(t0), detail={
                    "parser": "llm", "error": str(exc), "fallback": "rules"}))
        t0 = time.perf_counter()
        parsed = self.rules.parse(question)
        trace.append(TraceStep(step="parse", ms=_ms(t0), detail={
            "parser": "rules", "output": parsed.model_dump(mode="json")}))
        return parsed, "rules"

    def _resolve_one(self, text: str, kind: Kind, label: str) -> Resolution:
        res = self.ctx.resolver.resolve(text, kind)
        if res.match is MatchType.AMBIGUOUS:
            options = "\n".join(f"- {c.name}" for c in res.candidates)
            raise _Stop("needs_clarification", f"Which {label} do you mean by '{text}'?\n{options}")
        if not res.ok:
            hint = ("\nClosest matches:\n" + "\n".join(f"- {c.name}" for c in res.candidates)
                    if res.candidates else "")
            raise _Stop("not_found", f"I couldn't find a {label} called '{text}' in the skills framework.{hint}")
        return res

    def _resolve(self, p: ParsedQuery, trace: list[TraceStep]) -> tuple[dict[str, Any], list[str]]:
        t0 = time.perf_counter()
        resolved: dict[str, Any] = {}
        notes: list[str] = []
        if p.intent is Intent.OUT_OF_SCOPE:
            raise _Stop("unsupported", "I can answer questions about skills, job roles, learning paths and "
                        "courses in the SkillsFuture framework. For example:\n"
                        + "\n".join(f"- {q}" for q in SUPPORTED))

        if p.intent in (Intent.SKILLS_FOR_ROLE, Intent.LEARNING_PATH, Intent.GAP_ANALYSIS):
            if not p.target_role:
                raise _Stop("needs_clarification", "Which role are you aiming for?")
            target = self._resolve_one(p.target_role, Kind.ROLE, "role")
            resolved["target_role"] = {"id": target.id, "name": target.name, "match": target.match.value}

        if p.intent in (Intent.LEARNING_PATH, Intent.GAP_ANALYSIS):
            known: list[str] = []
            for text in p.known_skills:
                res = self.ctx.resolver.resolve(text, Kind.SKILL)
                if res.match is MatchType.AMBIGUOUS:
                    options = "\n".join(f"- {c.name}" for c in res.candidates)
                    raise _Stop("needs_clarification", f"Which skill do you mean by '{text}'?\n{options}")
                if res.ok:
                    known.append(res.id)  # type: ignore[arg-type]
                else:
                    notes.append(f"'{text}' is not a skill in the framework, so it was not counted.")
            if p.current_role:
                current = self._resolve_one(p.current_role, Kind.ROLE, "role")
                resolved["current_role"] = {"id": current.id, "name": current.name}
                from_role = self.ctx.store.required_skills(current.id)  # type: ignore[arg-type]
                known += [s for s in from_role if s not in known]
                notes.append(f"Assumed you have the skills the framework lists for {current.name}.")
            resolved["known_skills"] = [{"id": s, "name": self.ctx.store.node(s)["name"]} for s in known]

        if p.intent is Intent.TRANSFERABLE_SKILLS:
            econ = {k: self.ctx.resolver.economy(v or "") for k, v in
                    (("source", p.source_economy), ("target", p.target_economy))}
            if not econ["source"] or not econ["target"]:
                raise _Stop("needs_clarification", "Which two economies should I compare? Choose from "
                            "Digital, Care and Green, e.g. 'Which Digital skills are most transferable to Green?'")
            if econ["source"] == econ["target"]:
                raise _Stop("needs_clarification", "Please name two different economies (Digital, Care, Green).")
            resolved.update(source_economy=econ["source"], target_economy=econ["target"],
                            bidirectional=p.bidirectional)

        if p.intent is Intent.COURSES_FOR_SKILL:
            if not p.skills:
                raise _Stop("needs_clarification", "Which skill would you like courses for?")
            resolved["skills"] = []
            for text in p.skills:
                res = self._resolve_one(text, Kind.SKILL, "skill")
                resolved["skills"].append({"id": res.id, "name": res.name})
        trace.append(TraceStep(step="resolve", ms=_ms(t0), detail=resolved | {"notes": notes}))
        return resolved, notes

    def _execute(self, p: ParsedQuery, r: dict[str, Any], trace: list[TraceStep]) -> list:
        t0 = time.perf_counter()
        svc = self.ctx.service
        known = [s["id"] for s in r.get("known_skills", [])]
        if p.intent is Intent.SKILLS_FOR_ROLE:
            results = [svc.role_requirements(r["target_role"]["id"])]
        elif p.intent is Intent.LEARNING_PATH:
            results = [svc.learning_path(known, r["target_role"]["id"],
                                         mode="fastest" if p.fastest else "thorough")]
        elif p.intent is Intent.GAP_ANALYSIS:
            results = [svc.gap_analysis(known, r["target_role"]["id"])]
        elif p.intent is Intent.TRANSFERABLE_SKILLS:
            results = [svc.transferability(r["source_economy"], r["target_economy"])]
            if r["bidirectional"]:
                results.append(svc.transferability(r["target_economy"], r["source_economy"]))
        else:
            results = [svc.courses_for_skill(s["id"]) for s in r["skills"]]
        trace.append(TraceStep(step="execute", ms=_ms(t0), detail={
            "query": p.intent.value, "results": [type(x).__name__ for x in results]}))
        return results

    def _answer(self, question: str, p: ParsedQuery, results: list, notes: list[str],
                trace: list[TraceStep]) -> tuple[str, Literal["llm", "template"], list[str]]:
        template = "\n".join(render.render(x) for x in results)
        if notes:
            template += "\n" + "\n".join(f"- {n}" for n in notes)
        if self.llm is None:
            trace.append(TraceStep(step="answer", ms=0.0, detail={"source": "template", "reason": "offline"}))
            return template, "template", []
        payload = {"notes": notes, "results": [_compact(x.model_dump(mode="json")) for x in results]}
        t0 = time.perf_counter()
        try:
            reply = self.llm.text(ANSWER_SYSTEM, answer_user_message(question, p.intent.value, payload),
                                  effort="medium")
        except LLMError as exc:
            trace.append(TraceStep(step="answer", ms=_ms(t0), detail={"source": "template", "error": str(exc)}))
            return template, "template", []
        problems = check_answer(reply.text, payload, self._entity_names)
        trace.append(TraceStep(step="answer", ms=_ms(t0), detail={
            "source": "template" if problems else "llm", "model": reply.model, "cached": reply.cached,
            "usage": reply.usage, "grounding_problems": problems}))
        if problems:
            return template, "template", problems
        return reply.text, "llm", []

    def _finish(self, question, status, parsed, parser, parsed_dump, resolved, results, notes, answer,
                trace, source: Literal["llm", "template"] = "template") -> AskResult:
        return AskResult(question=question, status=status, intent=parsed.intent if parsed else None,
                         parser=parser, parsed=parsed_dump, resolved=resolved, answer=answer,
                         answer_source=source, results=results, notes=notes, trace=trace)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


_DROP = {"id", "tracks", "teaches", "economy"}


def _compact(obj: Any) -> Any:
    """Shrink a result for the answer prompt: drop ids and repeated metadata."""
    if isinstance(obj, dict):
        return {k: _compact(v) for k, v in obj.items() if k not in _DROP and v not in (None, [], {})}
    if isinstance(obj, list):
        return [_compact(v) for v in obj]
    return obj
