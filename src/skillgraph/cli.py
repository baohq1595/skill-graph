"""Command-line interface.

    skillgraph validate                         data-quality report for the dataset
    skillgraph role "Data Scientist"            skills a role requires
    skillgraph gap --learner LP-01              gap analysis for a sample learner
    skillgraph gap --skills "Python, SQL" --role "AI Engineer"
    skillgraph path --learner LP-02 [--mode thorough]
    skillgraph transfer Digital Green
    skillgraph courses "Machine Learning"
    skillgraph ask "I know Python and SQL. What's the fastest path to AI Engineer?"
    skillgraph eval                              score the natural-language layer
    skillgraph serve                             REST API + visualisation on :8000

Skills and roles can be given by name, alias, id or with typos ("pyton").
Add --json to any query command for machine-readable output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from skillgraph import render
from skillgraph.config import load_dotenv
from skillgraph.context import get_context
from skillgraph.ingest.loader import DatasetError
from skillgraph.services.queries import UnknownEntity
from skillgraph.services.resolver import Kind, ResolutionError

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Mini Skill & Competency Graph over the SkillsFuture framework.")
console = Console()
load_dotenv()
JsonOpt = typer.Option(False, "--json", help="Print the raw result as JSON.")


def _ctx():
    try:
        return get_context()
    except DatasetError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)


def _emit(result, as_json: bool) -> None:
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        console.print(Markdown(render.render(result)))


def _guard(fn):
    """Turn lookup errors into one readable line and a non-zero exit."""
    try:
        return fn()
    except (ResolutionError, UnknownEntity, KeyError) as exc:
        console.print(f"[red]{exc.args[0] if exc.args else exc}[/red]")
        raise typer.Exit(1)


def _learner_inputs(ctx, learner: Optional[str], skills: Optional[str], role: Optional[str]):
    if learner:
        lp = ctx.learner(learner)
        console.print(f"[dim]{lp.id} · {lp.name} — {lp.description}[/dim]")
        return lp.current_skills, lp.target_role
    if not role:
        console.print("[red]Give --learner, or --role (and optionally --skills).[/red]")
        raise typer.Exit(1)
    names = [s.strip() for s in (skills or "").split(",") if s.strip()]
    return ctx.resolver.require_many(names, Kind.SKILL), ctx.resolver.require(role, Kind.ROLE)


@app.command()
def validate(out: Optional[Path] = typer.Option(None, help="Also write the report as markdown.")):
    """Ingest the dataset and print its data-quality report."""
    ctx = _ctx()
    report = ctx.report
    console.print(Markdown(report.to_markdown()))
    console.print(f"Graph: {ctx.store.g.number_of_nodes()} nodes, {ctx.store.g.number_of_edges()} relationships.")
    if out:
        out.write_text(report.to_markdown(), encoding="utf-8")
        console.print(f"Wrote {out}")


@app.command()
def learners():
    """List the sample learner profiles."""
    ctx = _ctx()
    table = Table("ID", "Name", "Current skills", "Target role")
    for lp in ctx.dataset.learner_profiles:
        table.add_row(lp.id, lp.name, ", ".join(ctx.store.names(lp.current_skills)),
                      ctx.store.node(lp.target_role)["name"])
    console.print(table)


@app.command()
def role(name: str = typer.Argument(..., help="Role name, alias or id"), as_json: bool = JsonOpt):
    """Skills a role requires, in learning order, with courses."""
    ctx = _ctx()
    _guard(lambda: _emit(ctx.service.role_requirements(ctx.resolver.require(name, Kind.ROLE)), as_json))


@app.command()
def gap(learner: Optional[str] = typer.Option(None, help="Sample learner id, e.g. LP-01"),
        skills: Optional[str] = typer.Option(None, help="Comma-separated current skills"),
        role: Optional[str] = typer.Option(None, help="Target role"),
        as_json: bool = JsonOpt):
    """Gap analysis: missing skills ranked by priority, with courses."""
    ctx = _ctx()

    def run():
        known, target = _learner_inputs(ctx, learner, skills, role)
        _emit(ctx.service.gap_analysis(known, target), as_json)
    _guard(run)


@app.command()
def path(learner: Optional[str] = typer.Option(None, help="Sample learner id, e.g. LP-02"),
         skills: Optional[str] = typer.Option(None, help="Comma-separated current skills"),
         role: Optional[str] = typer.Option(None, help="Target role"),
         mode: str = typer.Option("fastest", help="fastest | thorough (also adds recommended prerequisites)"),
         as_json: bool = JsonOpt):
    """Learning path: skills in prerequisite order, staged, with a minimum-hours course plan."""
    ctx = _ctx()
    if mode not in ("fastest", "thorough"):
        console.print("[red]--mode must be 'fastest' or 'thorough'.[/red]")
        raise typer.Exit(1)

    def run():
        known, target = _learner_inputs(ctx, learner, skills, role)
        _emit(ctx.service.learning_path(known, target, mode=mode), as_json)  # type: ignore[arg-type]
    _guard(run)


@app.command()
def transfer(source: str, target: str, limit: int = typer.Option(8, min=1, max=52), as_json: bool = JsonOpt):
    """Skills of one economy most transferable to another (Digital, Care, Green)."""
    ctx = _ctx()

    def run():
        a, b = ctx.resolver.economy(source), ctx.resolver.economy(target)
        if not a or not b:
            raise UnknownEntity(f"Unknown economy '{source if not a else target}'. Use Digital, Care or Green.")
        _emit(ctx.service.transferability(a, b, limit), as_json)
    _guard(run)


@app.command()
def courses(skill: str, as_json: bool = JsonOpt):
    """Courses that teach a skill."""
    ctx = _ctx()
    _guard(lambda: _emit(ctx.service.courses_for_skill(ctx.resolver.require(skill, Kind.SKILL)), as_json))


@app.command()
def ask(question: str,
        trace: bool = typer.Option(False, "--trace", help="Show the parse/resolve/execute/answer trace."),
        as_json: bool = JsonOpt):
    """Ask in plain English. Uses Gemini when GEMINI_API_KEY is set, the offline parser otherwise."""
    from skillgraph.nl.pipeline import Assistant

    result = Assistant(_ctx()).ask(question)
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    mode = f"parser={result.parser} · answer={result.answer_source} · status={result.status}"
    console.print(f"[dim]{mode}[/dim]")
    console.print(Markdown(result.answer))
    if result.grounding_problems:
        console.print("[yellow]LLM answer rejected by the grounding check: "
                      + "; ".join(result.grounding_problems) + "[/yellow]")
    if trace:
        table = Table("Step", "ms", "Detail")
        for step in result.trace:
            table.add_row(step.step, f"{step.ms:g}", json.dumps(step.detail, ensure_ascii=False)[:300])
        console.print(table)


@app.command("eval")
def evaluate_cmd(cases: Optional[Path] = typer.Option(None, help="Cases file (default: development + holdout)"),
                 offline: bool = typer.Option(False, help="Force the rule parser even if an API key is set."),
                 out_dir: Path = typer.Option(Path("eval/results"), help="Where to write per-case results.")):
    """Score the natural-language layer on the evaluation sets."""
    from skillgraph.nl.evaluation import DEFAULT_CASES, evaluate, save
    from skillgraph.nl.pipeline import Assistant

    assistant = Assistant(_ctx(), llm=None) if offline else Assistant(_ctx())
    files = [cases] if cases else [DEFAULT_CASES, DEFAULT_CASES.with_name("nl_holdout.yaml")]
    table = Table("Set", "Parser", "Cases", "Status", "Intent", "Slots", "Exact", "By group", "Fallbacks")
    for f in files:
        summary, rows = evaluate(assistant, f)
        save(summary, rows, out_dir / f"{f.stem}.{summary['parser']}.json")
        table.add_row(f.stem, summary["parser"], str(summary["cases"]),
                      *(f"{summary[k]:.0%}" for k in ("status_accuracy", "intent_accuracy", "slot_accuracy",
                                                      "exact_match")),
                      ", ".join(f"{g} {v}" for g, v in summary["by_group"].items()),
                      str(summary["parser_fallbacks"]))
        for r in rows:
            if not r.exact:
                console.print(f"[yellow]{f.stem} {r.id}[/yellow] {r.question}\n    " + "; ".join(r.misses))
    console.print(table)


@app.command("export-cypher")
def export_cypher(out: Path = typer.Argument(Path("skillgraph.cypher"))):
    """Write a Cypher script that loads this graph into Neo4j."""
    from skillgraph.graph.neo4j_export import to_cypher

    out.write_text(to_cypher(_ctx().store), encoding="utf-8")
    console.print(f"Wrote {out}. Load it with: cypher-shell -f {out}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Run the REST API and the graph visualisation."""
    import uvicorn

    console.print(f"API docs: http://{host}:{port}/docs   Visualisation: http://{host}:{port}/")
    uvicorn.run("skillgraph.api.app:app", host=host, port=port)


if __name__ == "__main__":
    app()
