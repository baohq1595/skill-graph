"""Turn graph nodes into the reference objects used in results."""

from __future__ import annotations

from skillgraph.graph.schema import Rel
from skillgraph.graph.store import GraphStore
from skillgraph.services.results import CourseRef, RoleRef, SkillRef


def skill_ref(store: GraphStore, skill_id: str) -> SkillRef:
    n = store.node(skill_id)
    return SkillRef(
        id=skill_id, name=n["name"], difficulty=n["difficulty"], economy=n["economy"],
        transferability=n["transferability"], sdfe_priority=bool(n.get("sdfe_priority")),
        tracks=[store.node(t)["name"] for t in n.get("track_ids", []) if store.has(t)],
    )


def role_ref(store: GraphStore, role_id: str) -> RoleRef:
    n = store.node(role_id)
    sector = store.node(n["sector_id"])["name"] if store.has(n["sector_id"]) else n["sector_id"]
    return RoleRef(id=role_id, name=n["name"], sector=sector, economy=n["economy"],
                   seniority=n.get("seniority"))


def course_ref(store: GraphStore, course_id: str) -> CourseRef:
    n = store.node(course_id)
    return CourseRef(
        id=course_id, name=n["name"], provider=n["provider"], duration_hours=n["duration_hours"],
        difficulty=n.get("difficulty"), sfc_funded=n.get("sfc_funded"),
        teaches=[store.node(s)["name"] for s in store.skills_taught(course_id)],
    )


def name(store: GraphStore, node_id: str) -> str:
    return store.node(node_id)["name"]


__all__ = ["skill_ref", "role_ref", "course_ref", "name", "Rel"]
