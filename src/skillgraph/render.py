"""Markdown rendering of query results.

One renderer per result type, used by the CLI (printed through rich) and by
the natural-language layer as its deterministic fallback, so a reply looks
the same with or without an LLM.
"""

from __future__ import annotations

from skillgraph.services.results import (
    GapAnalysis, LearningPath, RoleRequirements, SkillCourses, Transferability,
)


def _hours(h: float) -> str:
    return f"{h:g} h"


def role_requirements(r: RoleRequirements) -> str:
    lines = [f"## Skills for {r.role.name}",
             f"{r.role.sector} · {r.role.economy} economy · {r.role.seniority or 'any'} level", "",
             "| # | Skill | Level | Needs first | Courses |", "|--:|---|---|---|---|"]
    for i, item in enumerate(r.required, 1):
        courses = ", ".join(f"{c.name} ({_hours(c.duration_hours)})" for c in item.courses[:2]) or "none in catalogue"
        lines.append(f"| {i} | {item.skill.name} | {item.skill.difficulty} | "
                     f"{', '.join(item.prerequisites) or '—'} | {courses} |")
    lines += [""] + [f"- {n}" for n in r.notes]
    return "\n".join(lines).rstrip() + "\n"


def gap_analysis(g: GapAnalysis) -> str:
    lines = [f"## Gap analysis: {g.target_role.name}",
             f"You have **{len(g.have)} of {len(g.have) + len(g.gaps)}** required skills "
             f"({g.coverage:.0%}).", ""]
    if g.gaps:
        lines += ["| Rank | Skill gap | Priority | Why | Suggested course |", "|--:|---|---|---|---|"]
        for item in g.gaps:
            course = item.courses[0].course if item.courses else None
            course_txt = f"{course.name} ({course.provider}, {_hours(course.duration_hours)})" if course \
                else "no catalogue course"
            lines.append(f"| {item.rank} | {item.skill.name} | {item.priority} ({item.priority_score:g}) | "
                         f"{'; '.join(item.reasons[:2])} | {course_txt} |")
    if g.enabling_prerequisites:
        lines += ["", "**Learn first (not listed for the role, but a gap needs it):** "
                  + ", ".join(e.skill.name for e in g.enabling_prerequisites)]
    lines += [""] + [f"- {n}" for n in g.notes]
    return "\n".join(lines).rstrip() + "\n"


def learning_path(p: LearningPath) -> str:
    lines = [f"## Learning path to {p.target_role.name} ({p.mode})"]
    if p.already_have:
        lines.append("Already have: " + ", ".join(p.already_have))
    lines.append("")
    for stage in range(1, p.stages + 1):
        steps = [s for s in p.steps if s.stage == stage]
        lines.append(f"**Stage {stage}**" + (" (learn in parallel)" if len(steps) > 1 else ""))
        lines.append("")
        for s in steps:
            after = f" — after {', '.join(s.after)}" if s.after else ""
            lines.append(f"{s.order}. {s.skill.name} ({s.skill.difficulty}){after}. _{s.why}_")
        lines.append("")
    if p.course_plan:
        lines += ["**Course plan**", "", "| Stage | Course | Provider | Hours | Covers |", "|--:|---|---|--:|---|"]
        for c in p.course_plan:
            lines.append(f"| {c.stage} | {c.course.name} | {c.course.provider} | "
                         f"{c.course.duration_hours:g} | {', '.join(c.covers)} |")
        lines += ["", f"Total **{_hours(p.total_course_hours)}** of training; "
                      f"**{_hours(p.critical_path_hours)}** if same-stage courses run in parallel"
                      + (" (minimum-hours plan)." if p.plan_is_optimal else " (approximate plan).")]
    if p.self_study:
        lines.append("No catalogue course for: " + ", ".join(p.self_study) + " — self-study or on the job.")
    if p.recommended_extras:
        lines.append("Optional, makes the path easier: " + ", ".join(p.recommended_extras) + ".")
    lines += [f"- {n}" for n in p.notes]
    return "\n".join(lines).rstrip() + "\n"


def transferability(t: Transferability) -> str:
    lines = [f"## {t.source_economy} skills most transferable to the {t.target_economy} economy", "",
             "| Rank | Skill | Score | Evidence |", "|--:|---|--:|---|"]
    for item in t.items:
        lines.append(f"| {item.rank} | {item.skill.name} | {item.score:g} | {'; '.join(item.evidence)} |")
    lines += [""] + [f"- {n}" for n in t.notes]
    return "\n".join(lines).rstrip() + "\n"


def skill_courses(s: SkillCourses) -> str:
    lines = [f"## Courses for {s.skill.name}", ""]
    for i, c in enumerate(s.courses, 1):
        lines.append(f"{i}. **{c.course.name}** — {c.course.provider}, {_hours(c.course.duration_hours)}"
                     + (", SkillsFuture Credit eligible" if c.course.sfc_funded else "")
                     + (f". Also teaches {', '.join(c.also_covers)}" if c.also_covers else ""))
    lines += [""] + [f"- {n}" for n in s.notes]
    return "\n".join(lines).rstrip() + "\n"


RENDERERS = {
    RoleRequirements: role_requirements, GapAnalysis: gap_analysis, LearningPath: learning_path,
    Transferability: transferability, SkillCourses: skill_courses,
}


def render(result) -> str:
    return RENDERERS[type(result)](result)
