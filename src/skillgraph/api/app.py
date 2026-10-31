"""REST API over the same service layer as the CLI.

Interactive documentation is served at /docs. Every endpoint that takes a skill
or role accepts a name, an alias, an id or a misspelling; ambiguous or unknown
names return 422 with the candidates, so a client can ask its user to choose.
"""

from __future__ import annotations

from importlib import resources
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator

from skillgraph import __version__
from skillgraph.config import load_dotenv
from skillgraph.context import get_context
from skillgraph.graph.schema import Label
from skillgraph.services.queries import UnknownEntity
from skillgraph.services.resolver import Kind, ResolutionError
from skillgraph.services.results import (
    GapAnalysis, LearningPath, RoleRequirements, SkillCourses, Transferability,
)

load_dotenv()

app = FastAPI(
    title="Skill & Competency Graph API",
    version=__version__,
    description="Learning paths, gap analysis and natural-language queries over the SkillsFuture framework.",
)


@app.exception_handler(ResolutionError)
def _resolution_error(_: Request, exc: ResolutionError) -> JSONResponse:
    res = exc.resolution
    return JSONResponse(status_code=422, content={
        "error": str(exc), "query": res.query, "kind": res.kind.value, "match": res.match.value,
        "candidates": [{"id": c.id, "name": c.name, "score": c.score} for c in res.candidates]})


@app.exception_handler(UnknownEntity)
def _unknown(_: Request, exc: UnknownEntity) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": exc.args[0]})


# ---------------------------------------------------------------- request bodies
class LearnerQuery(BaseModel):
    """Either a sample learner, or a target role plus current skills."""

    learner_id: str | None = Field(None, examples=["LP-01"])
    target_role: str | None = Field(None, examples=["AI Engineer"])
    current_skills: list[str] = Field(default_factory=list, examples=[["Python", "SQL"]])

    @model_validator(mode="after")
    def _needs_target(self):
        if not self.learner_id and not self.target_role:
            raise ValueError("Give learner_id, or target_role (with optional current_skills).")
        return self

    def resolve(self) -> tuple[list[str], str]:
        ctx = get_context()
        if self.learner_id:
            try:
                lp = ctx.learner(self.learner_id)
            except KeyError as exc:
                raise HTTPException(404, exc.args[0]) from exc
            return lp.current_skills, lp.target_role
        return (ctx.resolver.require_many(self.current_skills, Kind.SKILL),
                ctx.resolver.require(self.target_role, Kind.ROLE))  # type: ignore[arg-type]


class PathQuery(LearnerQuery):
    mode: Literal["fastest", "thorough"] = "fastest"


class AskQuery(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000,
                          examples=["I know Python and SQL. What's the fastest path to AI Engineer?"])


# ---------------------------------------------------------------- endpoints
@app.get("/health", tags=["meta"])
def health() -> dict:
    ctx = get_context()
    from skillgraph.nl.llm import llm_from_env
    llm = llm_from_env()
    return {"status": "ok", "nodes": ctx.store.g.number_of_nodes(), "edges": ctx.store.g.number_of_edges(),
            "llm": f"{llm.model} ({getattr(getattr(llm, 'inner', llm), 'backend', 'llm')})" if llm
                   else "offline (rule parser)"}


@app.get("/data-quality", tags=["meta"])
def data_quality() -> dict:
    report = get_context().report
    return {"stats": report.stats, "issues": [
        {"severity": i.severity.value, "code": i.code, "message": i.message, "refs": list(i.refs)}
        for i in report.issues]}


@app.get("/catalog/{kind}", tags=["catalog"])
def catalog(kind: Literal["roles", "skills", "courses", "sectors", "tracks"]) -> list[dict]:
    label = {"roles": Label.ROLE, "skills": Label.SKILL, "courses": Label.COURSE,
             "sectors": Label.SECTOR, "tracks": Label.TRACK}[kind]
    store = get_context().store
    return [{k: v for k, v in store.node(i).items() if k != "label"} for i in store.ids(label)]


@app.get("/learners", tags=["catalog"])
def learners() -> list[dict]:
    return [lp.model_dump() for lp in get_context().dataset.learner_profiles]


@app.get("/roles/{role}/skills", response_model=RoleRequirements, tags=["queries"])
def role_skills(role: str) -> RoleRequirements:
    ctx = get_context()
    return ctx.service.role_requirements(ctx.resolver.require(role, Kind.ROLE))


@app.post("/gap-analysis", response_model=GapAnalysis, tags=["queries"])
def gap_analysis(body: LearnerQuery) -> GapAnalysis:
    known, role = body.resolve()
    return get_context().service.gap_analysis(known, role)


@app.post("/learning-path", response_model=LearningPath, tags=["queries"])
def learning_path(body: PathQuery) -> LearningPath:
    known, role = body.resolve()
    return get_context().service.learning_path(known, role, mode=body.mode)


@app.get("/transferability", response_model=Transferability, tags=["queries"])
def transferability(source: str = Query(..., examples=["Digital"]), target: str = Query(..., examples=["Green"]),
                    limit: int = Query(8, ge=1, le=52)) -> Transferability:
    ctx = get_context()
    a, b = ctx.resolver.economy(source), ctx.resolver.economy(target)
    if not a or not b:
        raise HTTPException(422, f"Unknown economy '{source if not a else target}'. Use Digital, Care or Green.")
    return ctx.service.transferability(a, b, limit)


@app.get("/skills/{skill}/courses", response_model=SkillCourses, tags=["queries"])
def skill_courses(skill: str) -> SkillCourses:
    ctx = get_context()
    return ctx.service.courses_for_skill(ctx.resolver.require(skill, Kind.SKILL))


@app.post("/ask", tags=["natural language"])
def ask(body: AskQuery) -> dict:
    """Plain-English question -> structured graph query -> grounded answer, with the full trace."""
    from skillgraph.nl.pipeline import Assistant

    return Assistant(get_context()).ask(body.question).model_dump(mode="json")


@app.get("/graph", tags=["visualisation"])
def graph() -> dict:
    """All nodes and relationships, for the visualisation."""
    return get_context().store.export()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return resources.files("skillgraph.viz.static").joinpath("index.html").read_text(encoding="utf-8")
