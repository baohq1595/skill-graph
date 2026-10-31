"""The query service: role requirements, gap analysis, learning paths,
cross-economy transferability and course lookup.

All methods take graph ids. Name resolution happens one layer up
(``services.resolver``) so the algorithms stay deterministic and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from skillgraph.graph.schema import Label
from skillgraph.graph.store import GraphStore
from skillgraph.models import PrereqStrength, Transferability as TransferRating
from skillgraph.services import course_plan as cp
from skillgraph.services.prerequisites import order_skills, required_closure, unlock_counts
from skillgraph.services.refs import course_ref, name, role_ref, skill_ref
from skillgraph.services.results import (
    CoursePlanItem, GapAnalysis, GapItem, LearningPath, PathStep, RoleRequirements, RoleSkill,
    SkillCourses, TransferItem, Transferability,
)


class UnknownEntity(KeyError):
    """An id that is not in the graph (or not of the expected kind)."""


@dataclass(frozen=True)
class PriorityWeights:
    """How a gap's priority score (0-100) is built. Weights sum to 1.

    The dataset carries no Core/Supporting importance, so priority comes from
    the graph itself:
      unlocks   - share of the learner's remaining path that depends on it
      national  - flagged as an SDFE 2025 priority skill
      demand    - share of all roles that require it (market breadth)
      readiness - share of its hard prerequisites the learner already has
    """

    unlocks: float = 0.35
    national: float = 0.20
    demand: float = 0.25
    readiness: float = 0.20
    high: float = 60.0
    medium: float = 40.0


@dataclass(frozen=True)
class TransferWeights:
    """How a cross-economy transferability score (0-100) is built.

      direct     - roles in the target economy that require the skill
      rating     - the dataset's own transferability rating (High/Medium/Low)
      breadth    - number of distinct sectors whose roles require it
      structural - sits in a target-economy track, or links by prerequisite to a target-economy skill
    """

    direct: float = 0.40
    rating: float = 0.30
    breadth: float = 0.20
    structural: float = 0.10


class SkillGraphService:
    def __init__(self, store: GraphStore, priority: PriorityWeights | None = None,
                 transfer: TransferWeights | None = None) -> None:
        self.store = store
        self.pw = priority or PriorityWeights()
        self.tw = transfer or TransferWeights()
        all_skills = store.ids(Label.SKILL)
        self._max_demand = max((len(store.roles_requiring(s)) for s in all_skills), default=1) or 1
        self._n_roles = len(store.ids(Label.ROLE))

    # ----------------------------------------------------------- helpers
    def _check(self, node_id: str, label: Label) -> None:
        if not self.store.has(node_id, label):
            raise UnknownEntity(f"{node_id} is not a {label.value} in the graph.")

    def _check_skills(self, skill_ids: list[str]) -> set[str]:
        for s in skill_ids:
            self._check(s, Label.SKILL)
        return set(skill_ids)

    def _priorities(self, to_learn: set[str], known: set[str]) -> dict[str, tuple[float, list[str]]]:
        unlocks = unlock_counts(self.store, to_learn)
        max_unlock = max(unlocks.values(), default=0)
        out: dict[str, tuple[float, list[str]]] = {}
        for s in to_learn:
            node = self.store.node(s)
            dependents = [d for d in self.store.dependents(s) if d in to_learn]
            hard_pre = self.store.prerequisites(s, PrereqStrength.HARD)
            demand = len(self.store.roles_requiring(s))
            readiness = (sum(p in known for p in hard_pre) / len(hard_pre)) if hard_pre else 1.0
            parts = {
                "unlocks": (unlocks[s] / max_unlock) if max_unlock else 0.0,
                "national": 1.0 if node.get("sdfe_priority") else 0.0,
                "demand": demand / self._max_demand,
                "readiness": readiness,
            }
            score = 100 * sum(getattr(self.pw, k) * v for k, v in parts.items())
            reasons = []
            if unlocks[s]:
                reasons.append(f"unlocks {unlocks[s]} later skill(s) on your path, e.g. "
                               + ", ".join(name(self.store, d) for d in dependents[:2]))
            if parts["national"]:
                reasons.append("SkillsFuture SDFE 2025 priority skill")
            reasons.append(f"required by {demand} of {self._n_roles} job roles")
            missing = [p for p in hard_pre if p not in known]
            soft_missing = [p for p in self.store.prerequisites(s, PrereqStrength.SOFT) if p not in known]
            if missing:
                reasons.append("learn first: " + ", ".join(name(self.store, p) for p in missing))
            else:
                reasons.append("ready to start: prerequisites met")
            if soft_missing:
                reasons.append("easier after: " + ", ".join(name(self.store, p) for p in soft_missing))
            out[s] = (round(score, 1), reasons)
        return out

    def _tier(self, score: float) -> Literal["High", "Medium", "Low"]:
        return "High" if score >= self.pw.high else "Medium" if score >= self.pw.medium else "Low"

    def _gap_item(self, rank: int, s: str, prio: tuple[float, list[str]], needed: set[str],
                  known: set[str]) -> GapItem:
        courses = cp.suggest_courses(self.store, s, needed)
        missing = [name(self.store, p) for p in self.store.prerequisites(s, PrereqStrength.HARD)
                   if p not in known]
        return GapItem(rank=rank, skill=skill_ref(self.store, s), priority_score=prio[0],
                       priority=self._tier(prio[0]), reasons=prio[1], missing_prerequisites=missing,
                       courses=courses, no_catalogue_course=not courses)

    # ----------------------------------------------------------- queries
    def role_requirements(self, role_id: str) -> RoleRequirements:
        self._check(role_id, Label.ROLE)
        required = set(self.store.required_skills(role_id))
        to_learn, foundations = required_closure(self.store, required, known=set())
        ordering = order_skills(self.store, required)
        items = [RoleSkill(skill=skill_ref(self.store, s),
                           roles_requiring=len(self.store.roles_requiring(s)),
                           prerequisites=[name(self.store, p) for p in self.store.prerequisites(s)],
                           courses=[course_ref(self.store, c) for c in self.store.courses_teaching(s)])
                 for s in ordering.order]
        notes = []
        if foundations:
            notes.append("The role does not list these, but its skills need them first: "
                         + ", ".join(name(self.store, f) for f in sorted(foundations)) + ".")
        no_course = [i.skill.name for i in items if not i.courses]
        if no_course:
            notes.append("No catalogue course teaches: " + ", ".join(no_course) + ".")
        return RoleRequirements(role=role_ref(self.store, role_id), required=items,
                                foundations=[skill_ref(self.store, f) for f in sorted(foundations)],
                                notes=notes)

    def gap_analysis(self, known_skills: list[str], role_id: str) -> GapAnalysis:
        self._check(role_id, Label.ROLE)
        known = self._check_skills(known_skills)
        required = set(self.store.required_skills(role_id))
        gaps = required - known
        to_learn, enabling = required_closure(self.store, gaps, known)
        prio = self._priorities(to_learn, known)

        ranked_gaps = sorted(gaps, key=lambda s: (-prio[s][0], name(self.store, s)))
        ranked_enabling = sorted(enabling, key=lambda s: (-prio[s][0], name(self.store, s)))
        role = role_ref(self.store, role_id)
        notes = []
        if not gaps:
            notes.append(f"You already have all {len(required)} skills required for {role.name}.")
        if enabling:
            notes.append(f"{len(enabling)} skill(s) not listed for the role are needed first because "
                         "a gap depends on them: " + ", ".join(name(self.store, s) for s in ranked_enabling) + ".")
        extra = sorted(known - required)
        if extra:
            notes.append("Skills you have that this role does not list: "
                         + ", ".join(name(self.store, s) for s in extra) + ".")
        return GapAnalysis(
            target_role=role,
            coverage=round(len(required & known) / len(required), 3) if required else 1.0,
            have=[skill_ref(self.store, s) for s in sorted(required & known)],
            gaps=[self._gap_item(i, s, prio[s], to_learn, known) for i, s in enumerate(ranked_gaps, 1)],
            enabling_prerequisites=[self._gap_item(i, s, prio[s], to_learn, known)
                                    for i, s in enumerate(ranked_enabling, 1)],
            notes=notes,
        )

    def learning_path(self, known_skills: list[str], role_id: str,
                      mode: Literal["fastest", "thorough"] = "fastest") -> LearningPath:
        self._check(role_id, Label.ROLE)
        known = self._check_skills(known_skills)
        required = set(self.store.required_skills(role_id))
        targets = required - known
        to_learn, pulled = required_closure(self.store, targets, known, include_soft=(mode == "thorough"))
        extras = sorted({p for s in to_learn for p in self.store.prerequisites(s, PrereqStrength.SOFT)
                         if p not in known and p not in to_learn})
        prio = self._priorities(to_learn, known)
        ordering = order_skills(self.store, to_learn, {s: v[0] for s, v in prio.items()})

        steps = []
        for i, s in enumerate(ordering.order, 1):
            after = [p for p in self.store.prerequisites(s) if p in to_learn]
            if s in targets:
                why = f"required by {name(self.store, role_id)}"
            else:
                needed_by = [d for d in self.store.dependents(s) if d in to_learn]
                why = "needed before " + ", ".join(name(self.store, d) for d in needed_by)
            steps.append(PathStep(order=i, stage=ordering.stage[s], skill=skill_ref(self.store, s),
                                  kind="required" if s in targets else "prerequisite",
                                  after=[name(self.store, p) for p in after], why=why))

        plan = cp.plan_courses(self.store, to_learn, known)
        course_stage, critical = cp.schedule_courses(self.store, plan, to_learn)
        covers: dict[str, list[str]] = {}
        for skill, course in plan.assignment.items():
            covers.setdefault(course, []).append(name(self.store, skill))
        course_items = sorted(
            (CoursePlanItem(stage=course_stage[c], course=course_ref(self.store, c), covers=sorted(covers.get(c, [])))
             for c in plan.courses),
            key=lambda item: (item.stage, item.course.name))

        notes = []
        if not targets:
            notes.append(f"You already have every skill required for {name(self.store, role_id)}.")
        if pulled:
            notes.append("Added because a required skill depends on them: "
                         + ", ".join(name(self.store, s) for s in sorted(pulled)) + ".")
        if ordering.dropped_soft_edges:
            notes.append("Ignored recommended orderings that formed a cycle: "
                         + ", ".join(f"{a}->{b}" for a, b in ordering.dropped_soft_edges) + ".")
        if not plan.optimal:
            notes.append("Course plan uses the greedy approximation (too many candidate courses for an exact search).")
        return LearningPath(
            target_role=role_ref(self.store, role_id), mode=mode,
            already_have=[name(self.store, s) for s in sorted(required & known)],
            steps=steps, stages=max(ordering.stage.values(), default=0),
            course_plan=course_items, total_course_hours=plan.hours, critical_path_hours=critical,
            plan_is_optimal=plan.optimal,
            self_study=[name(self.store, s) for s in sorted(plan.uncovered)],
            recommended_extras=[name(self.store, s) for s in extras] if mode == "fastest" else [],
            notes=notes,
        )

    def transferability(self, source: str, target: str, limit: int = 8) -> Transferability:
        economies = {self.store.node(e)["name"] for e in self.store.ids(Label.ECONOMY)}
        for e in (source, target):
            if e not in economies:
                raise UnknownEntity(f"{e} is not an economy. Choose from {sorted(economies)}.")
        target_roles = set(self.store.roles_in_economy(target))
        target_skills = set(self.store.skills_in_economy(target))
        target_tracks = {t for t in self.store.ids(Label.TRACK)
                         if self.store.node(self.store.node(t)["sector_id"])["economy"] == target}

        rows = []
        for s in self.store.skills_in_economy(source):
            node = self.store.node(s)
            requiring = self.store.roles_requiring(s)
            direct = [r for r in requiring if r in target_roles]
            sectors = {self.store.node(r)["sector_id"] for r in requiring}
            in_target_track = [t for t in node.get("track_ids", []) if t in target_tracks]
            linked = [x for x in self.store.dependents(s) + self.store.prerequisites(s) if x in target_skills]
            rows.append((s, node, direct, sectors, in_target_track, linked))

        max_direct = max((len(r[2]) for r in rows), default=0)
        max_sectors = max((len(r[3]) for r in rows), default=0)
        items = []
        for s, node, direct, sectors, in_track, linked in rows:
            parts = {
                "direct": len(direct) / max_direct if max_direct else 0.0,
                "rating": TransferRating(node["transferability"]).weight,
                "breadth": len(sectors) / max_sectors if max_sectors else 0.0,
                "structural": 1.0 if (in_track or linked) else 0.0,
            }
            score = round(100 * sum(getattr(self.tw, k) * v for k, v in parts.items()), 1)
            evidence = [f"rated {node['transferability']} transferability in the framework"]
            if direct:
                evidence.append(f"required by {len(direct)} {target} role(s): "
                                + ", ".join(name(self.store, r) for r in direct))
            evidence.append(f"required by roles in {len(sectors)} sector(s)" if sectors
                            else "not required by any role in the dataset")
            if in_track:
                evidence.append("part of the " + ", ".join(name(self.store, t) for t in in_track)
                                + f" track in the {target} economy")
            if linked:
                evidence.append("prerequisite link to " + ", ".join(name(self.store, x) for x in linked))
            items.append((score, node["name"], s, evidence, direct))

        items.sort(key=lambda x: (-x[0], x[1]))
        direct_overlap = sum(1 for *_, d in items if d)
        notes = []
        if direct_overlap == 0:
            notes.append(f"No {target} role in this dataset requires a {source} skill directly, so the "
                         "ranking rests on each skill's transferability rating, how widely it is used "
                         f"across sectors, and any structural link into the {target} economy.")
        return Transferability(
            source_economy=source, target_economy=target, direct_overlap=direct_overlap,
            items=[TransferItem(rank=i, skill=skill_ref(self.store, s), score=score, evidence=ev,
                                target_roles=[name(self.store, r) for r in direct])
                   for i, (score, _, s, ev, direct) in enumerate(items[:limit], 1)],
            notes=notes,
        )

    def courses_for_skill(self, skill_id: str) -> SkillCourses:
        self._check(skill_id, Label.SKILL)
        courses = cp.suggest_courses(self.store, skill_id, {skill_id}, limit=10)
        prereqs = [name(self.store, p) for p in self.store.prerequisites(skill_id)]
        notes = [] if courses else [f"No catalogue course teaches {name(self.store, skill_id)}."]
        if prereqs:
            notes.append("Before starting, you should know: " + ", ".join(prereqs) + ".")
        return SkillCourses(skill=skill_ref(self.store, skill_id), courses=courses,
                            prerequisites=prereqs, notes=notes)
