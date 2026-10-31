"""Course recommendation and minimum-hours course planning.

Choosing courses to cover a set of skills in the fewest hours is weighted set
cover, which is NP-hard in general. Here only the courses that teach at least
one needed skill are candidates (typically under 10), so the exact optimum is
found by exhaustive search. Above ``EXACT_LIMIT`` candidates the planner falls
back to the classic greedy approximation (cost per newly covered skill) and
says so in the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx

from skillgraph.graph.store import GraphStore
from skillgraph.models import PrereqStrength
from skillgraph.services.refs import course_ref
from skillgraph.services.results import CourseSuggestion

EXACT_LIMIT = 18  # 2^18 subsets, well under a second


def _difficulty_rank(value: str | None) -> int:
    return {"Foundation": 1, "Intermediate": 2, "Advanced": 3}.get(value or "", 2)


def suggest_courses(store: GraphStore, skill_id: str, needed: set[str], limit: int = 3) -> list[CourseSuggestion]:
    """Rank the courses that teach ``skill_id`` for this learner.

    Order: covers more of the learner's other needs, then level matches the
    skill, then SkillsFuture-funded, then shorter.
    """
    skill = store.node(skill_id)
    ranked = []
    for cid in store.courses_teaching(skill_id):
        course = store.node(cid)
        also = [s for s in store.skills_taught(cid) if s in needed and s != skill_id]
        level_gap = abs(_difficulty_rank(course.get("difficulty")) - _difficulty_rank(skill["difficulty"]))
        ranked.append(((-len(also), level_gap, not course.get("sfc_funded"), course["duration_hours"]), cid, also))
    ranked.sort()
    out = []
    for (_, level_gap, unfunded, hours), cid, also in ranked[:limit]:
        why = []
        if also:
            why.append(f"also covers {len(also)} other skill(s) you need")
        if level_gap == 0:
            why.append("level matches the skill")
        if not unfunded:
            why.append("SkillsFuture Credit eligible")
        why.append(f"{hours:g} hours")
        out.append(CourseSuggestion(course=course_ref(store, cid),
                                    also_covers=[store.node(s)["name"] for s in also], why=why))
    return out


@dataclass(frozen=True)
class Plan:
    courses: list[str]               # chosen course ids
    assignment: dict[str, str]       # skill id -> course id that covers it
    uncovered: set[str]              # skills no catalogue course teaches
    hours: float
    optimal: bool


def followable(store: GraphStore, course_id: str, available: set[str]) -> bool:
    """Can a learner follow this course? Every skill it teaches must have its
    hard prerequisites either available (known or on the path) or taught by the
    course itself. An MLOps course is no way to learn DevOps for someone with
    no Machine Learning and none planned."""
    taught = set(store.skills_taught(course_id))
    return all(p in available or p in taught
               for s in taught for p in store.prerequisites(s, PrereqStrength.HARD))


def plan_courses(store: GraphStore, skills: set[str], known: set[str] | None = None) -> Plan:
    available = skills | (known or set())
    teachable = {s for s in skills if store.courses_teaching(s)}
    uncovered = skills - teachable
    candidates = []
    for s in teachable:
        options = store.courses_teaching(s)
        ok = [c for c in options if followable(store, c, available)]
        candidates += ok or options  # if nothing is followable, keep the options rather than drop the skill
    candidates = sorted(set(candidates))
    covers = {c: set(store.skills_taught(c)) & teachable for c in candidates}
    hours = {c: float(store.node(c)["duration_hours"]) for c in candidates}

    if not teachable:
        return Plan([], {}, uncovered, 0.0, True)
    if len(candidates) <= EXACT_LIMIT:
        chosen, optimal = _exact_cover(teachable, candidates, covers, hours), True
    else:
        chosen, optimal = _greedy_cover(teachable, candidates, covers, hours), False

    assignment: dict[str, str] = {}
    for s in sorted(teachable):  # each skill goes to the shortest chosen course teaching it
        assignment[s] = min((c for c in chosen if s in covers[c]), key=lambda c: (hours[c], c))
    return Plan(chosen, assignment, uncovered, sum(hours[c] for c in chosen), optimal)


def _exact_cover(target: set[str], candidates: list[str], covers: dict[str, set[str]],
                 hours: dict[str, float]) -> list[str]:
    best: tuple[float, int, tuple[str, ...]] | None = None
    for size in range(1, len(candidates) + 1):
        for combo in combinations(candidates, size):
            if set().union(*(covers[c] for c in combo)) >= target:
                key = (sum(hours[c] for c in combo), size, combo)
                if best is None or key < best:
                    best = key
        # a smaller set that already covers the target can still be beaten on
        # hours by a larger one, so the search is not stopped early
    return list(best[2]) if best else []


def _greedy_cover(target: set[str], candidates: list[str], covers: dict[str, set[str]],
                  hours: dict[str, float]) -> list[str]:
    remaining, chosen = set(target), []
    while remaining:
        c = min((c for c in candidates if covers[c] & remaining),
                key=lambda c: (hours[c] / len(covers[c] & remaining), c))
        chosen.append(c)
        remaining -= covers[c]
    return chosen


def schedule_courses(store: GraphStore, plan: Plan, path_skills: set[str]) -> tuple[dict[str, int], float]:
    """Order the chosen courses so prerequisites are taught first.

    Course B comes after course A when B covers a skill whose prerequisite
    (on the path) is covered by A. Returns each course's stage and the
    critical-path hours: total time if same-stage courses run in parallel.
    """
    prereq = store.prerequisite_graph()
    dag = nx.DiGraph()
    dag.add_nodes_from(plan.courses)
    for skill, course in plan.assignment.items():
        for pre in prereq.predecessors(skill):
            if pre in path_skills and pre in plan.assignment and plan.assignment[pre] != course:
                dag.add_edge(plan.assignment[pre], course)
    while not nx.is_directed_acyclic_graph(dag):  # two courses teaching each other's prerequisites
        dag.remove_edge(*nx.find_cycle(dag)[0])
    stage = {c: depth for depth, gen in enumerate(nx.topological_generations(dag), start=1) for c in gen}
    hours = {c: float(store.node(c)["duration_hours"]) for c in plan.courses}
    longest: dict[str, float] = {}
    for c in nx.topological_sort(dag):
        longest[c] = hours[c] + max((longest[p] for p in dag.predecessors(c)), default=0.0)
    return stage, max(longest.values(), default=0.0)
