"""Graph-level data-quality checks, run on every ingestion.

Errors make the graph unusable (a dangling reference, a hard prerequisite
cycle) and stop ingestion. Warnings are real-world data problems the system
works around, and each one says how. Info records documented README-vs-data
differences so a reviewer sees them without reading the code.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

import networkx as nx

from skillgraph.models import Dataset, PrereqStrength


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    message: str
    refs: tuple[str, ...] = ()


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    def add(self, severity: Severity, code: str, message: str, *refs: str) -> None:
        self.issues.append(Issue(severity, code, message, tuple(refs)))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def ok(self) -> bool:
        return not self.errors

    def by_severity(self, severity: Severity) -> list[Issue]:
        return [i for i in self.issues if i.severity is severity]

    def to_markdown(self) -> str:
        lines = ["# Data quality report", "", "| Entity | Count |", "|---|--:|"]
        lines += [f"| {k} | {v} |" for k, v in self.stats.items()]
        for severity in Severity:
            found = self.by_severity(severity)
            lines += ["", f"## {severity.value.title()}s ({len(found)})", ""]
            lines += [f"- **{i.code}** — {i.message}" for i in found] or ["- none"]
        return "\n".join(lines) + "\n"


def validate(ds: Dataset) -> ValidationReport:
    report = ValidationReport(stats={
        "sectors": len(ds.sectors), "skill_tracks": len(ds.skill_tracks),
        "skills": len(ds.skills), "job_roles": len(ds.job_roles),
        "courses": len(ds.courses), "sector_has_track": len(ds.sector_has_track),
        "skill_required_by_role": len(ds.skill_required_by_role),
        "skill_prerequisite_of": len(ds.skill_prerequisite_of),
        "learner_profiles": len(ds.learner_profiles),
    })
    _unique_ids(ds, report)
    _references(ds, report)
    _track_sector_consistency(ds, report)
    _prerequisites(ds, report)
    _coverage(ds, report)
    _readme_drift(ds, report)
    return report


def _unique_ids(ds: Dataset, report: ValidationReport) -> None:
    all_ids = [n.id for group in (ds.sectors, ds.skill_tracks, ds.skills, ds.job_roles, ds.courses)
               for n in group]
    for node_id, n in Counter(all_ids).items():
        if n > 1:
            report.add(Severity.ERROR, "duplicate_id", f"{node_id} is used by {n} nodes.", node_id)
    for label, group in (("skill", ds.skills), ("role", ds.job_roles), ("course", ds.courses)):
        for name, n in Counter(x.name.casefold() for x in group).items():
            if n > 1:
                report.add(Severity.WARNING, "duplicate_name",
                           f"{n} {label}s share the name '{name}'; name lookups will ask the user to choose.")


def _references(ds: Dataset, report: ValidationReport) -> None:
    sectors = {s.id for s in ds.sectors}
    tracks = {t.id for t in ds.skill_tracks}
    skills = {s.id for s in ds.skills}
    roles = {r.id for r in ds.job_roles}

    def check(ref: str, pool: set[str], where: str) -> None:
        if ref not in pool:
            report.add(Severity.ERROR, "dangling_reference", f"{where} points to unknown id {ref}.", ref)

    for t in ds.skill_tracks:
        check(t.sector_id, sectors, f"track {t.id}.sector_id")
    for s in ds.skills:
        if not s.track_ids:
            report.add(Severity.WARNING, "skill_without_track", f"{s.id} belongs to no track.", s.id)
        for t in s.track_ids:
            check(t, tracks, f"skill {s.id}.track_ids")
    for r in ds.job_roles:
        check(r.sector_id, sectors, f"role {r.id}.sector_id")
    for c in ds.courses:
        for s in c.skill_ids:
            check(s, skills, f"course {c.id}.skill_ids")
    for e in ds.sector_has_track:
        check(e.source, sectors, "sector_has_track.source")
        check(e.target, tracks, "sector_has_track.target")
    for e in ds.skill_required_by_role:
        check(e.source, skills, "skill_required_by_role.source")
        check(e.target, roles, "skill_required_by_role.target")
    for e in ds.skill_prerequisite_of:
        check(e.source, skills, "skill_prerequisite_of.source")
        check(e.target, skills, "skill_prerequisite_of.target")
    for lp in ds.learner_profiles:
        for s in lp.current_skills:
            check(s, skills, f"learner {lp.id}.current_skills")
        check(lp.target_role, roles, f"learner {lp.id}.target_role")


def _track_sector_consistency(ds: Dataset, report: ValidationReport) -> None:
    declared = {(t.sector_id, t.id) for t in ds.skill_tracks}
    edged = {(e.source, e.target) for e in ds.sector_has_track}
    for sector, track in sorted(declared - edged):
        report.add(Severity.WARNING, "track_sector_missing_edge",
                   f"Track {track} declares sector {sector} but has no sector_has_track edge; "
                   "the edge is added from the track's sector_id.", track)
    for sector, track in sorted(edged - declared):
        report.add(Severity.WARNING, "track_sector_conflict",
                   f"Edge {sector} HAS {track} disagrees with the track's sector_id; both are kept.", track)


def _prerequisites(ds: Dataset, report: ValidationReport) -> None:
    skills = {s.id: s for s in ds.skills}
    hard = nx.DiGraph()
    full = nx.DiGraph()
    for e in ds.skill_prerequisite_of:
        full.add_edge(e.source, e.target)
        if e.strength is PrereqStrength.HARD:
            hard.add_edge(e.source, e.target)
        if e.source == e.target:
            report.add(Severity.ERROR, "self_prerequisite", f"{e.source} is its own prerequisite.", e.source)

    for cycle in nx.simple_cycles(hard):
        report.add(Severity.ERROR, "hard_prerequisite_cycle",
                   "Hard prerequisite cycle " + " -> ".join(cycle + cycle[:1])
                   + "; no valid learning order exists.", *cycle)
    for cycle in nx.simple_cycles(full):
        steps = list(zip(cycle, cycle[1:] + cycle[:1]))
        if all(hard.has_edge(a, b) for a, b in steps):
            continue  # already reported as a hard cycle
        report.add(Severity.WARNING, "soft_prerequisite_cycle",
                   "Cycle through recommended prerequisites " + " -> ".join(cycle + cycle[:1])
                   + "; soft edges in it are ignored when ordering.", *cycle)

    soft = [e for e in ds.skill_prerequisite_of if e.strength is PrereqStrength.SOFT]
    if soft:
        report.add(Severity.INFO, "soft_prerequisites",
                   f"{len(soft)} of {len(ds.skill_prerequisite_of)} prerequisite notes describe a "
                   "recommendation ('helps', 'supports') rather than a requirement. They guide ordering "
                   "but do not block: " + "; ".join(f"{e.source}->{e.target}" for e in soft) + ".")

    for e in ds.skill_prerequisite_of:
        src, dst = skills.get(e.source), skills.get(e.target)
        if src and dst and src.difficulty.rank > dst.difficulty.rank:
            kind = "hard" if e.strength is PrereqStrength.HARD else "soft"
            report.add(Severity.WARNING, "difficulty_inversion",
                       f"{src.name} ({src.difficulty.value}) is a {kind} prerequisite of "
                       f"{dst.name} ({dst.difficulty.value}).", e.source, e.target)


def _coverage(ds: Dataset, report: ValidationReport) -> None:
    taught = {s for c in ds.courses for s in c.skill_ids}
    required = {e.source for e in ds.skill_required_by_role}
    no_course = [s for s in ds.skills if s.id not in taught]
    if no_course:
        report.add(Severity.WARNING, "skill_without_course",
                   f"{len(no_course)} skills have no course in the catalogue; gap analysis marks them "
                   "'no catalogue course': " + ", ".join(s.name for s in no_course) + ".",
                   *(s.id for s in no_course))
    unused = [s for s in ds.skills if s.id not in required]
    if unused:
        report.add(Severity.INFO, "skill_not_required_by_any_role",
                   f"{len(unused)} skills are required by no role: " + ", ".join(s.name for s in unused) + ".",
                   *(s.id for s in unused))
    roles_with_skills = {e.target for e in ds.skill_required_by_role}
    for r in ds.job_roles:
        if r.id not in roles_with_skills:
            report.add(Severity.WARNING, "role_without_skills", f"{r.name} requires no skills.", r.id)
    sector_economy = {s.id: s.economy for s in ds.sectors}
    track_sector = {t.id: t.sector_id for t in ds.skill_tracks}
    for s in ds.skills:
        track_economies = {sector_economy.get(track_sector.get(t)) for t in s.track_ids} - {None}
        if track_economies and s.economy not in track_economies:
            report.add(Severity.INFO, "skill_economy_differs_from_tracks",
                       f"{s.name} is tagged {s.economy.value} but sits only in "
                       f"{'/'.join(sorted(e.value for e in track_economies))} tracks; "
                       "the skill's own tag is used for economy queries.", s.id)


def _readme_drift(ds: Dataset, report: ValidationReport) -> None:
    if ds.skill_required_by_role and all(e.importance is None for e in ds.skill_required_by_role):
        report.add(Severity.INFO, "missing_importance",
                   "The dataset README describes an importance (Core/Supporting) on skill->role edges, "
                   "but no edge has it. Gap priority is derived from the graph instead "
                   "(see PriorityWeights in services/queries.py).")
    if ds.job_roles and all(r.description is None for r in ds.job_roles):
        report.add(Severity.INFO, "missing_role_description",
                   "The README lists a description on job roles; the data has none.")
    if ds.courses and all(c.url is None and c.mode is None for c in ds.courses):
        report.add(Severity.INFO, "missing_course_fields",
                   "The README lists mode, certification and url on courses; the data has "
                   "difficulty and sfc_funded instead. Both sets are supported.")
    seniorities = sorted({r.seniority for r in ds.job_roles if r.seniority})
    if any(s not in {"Junior", "Mid", "Senior"} for s in seniorities):
        report.add(Severity.INFO, "seniority_values",
                   f"Seniority values in the data are {seniorities}; the README lists Junior/Mid/Senior. "
                   "Stored as given.")
