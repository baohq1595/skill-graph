"""The graph store: the only module that knows the graph is NetworkX.

Query services depend on ``GraphStore`` (a Protocol), not on NetworkX, so a
Neo4j-backed store can implement the same methods with Cypher and every
service, the CLI, the API and the tests keep working unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Protocol

import networkx as nx

from skillgraph.graph.schema import Label, Rel, economy_id
from skillgraph.models import Dataset, PrereqStrength


class GraphStore(Protocol):
    def node(self, node_id: str) -> dict[str, Any]: ...
    def has(self, node_id: str, label: Label | None = None) -> bool: ...
    def ids(self, label: Label) -> list[str]: ...
    def required_skills(self, role_id: str) -> list[str]: ...
    def roles_requiring(self, skill_id: str) -> list[str]: ...
    def prerequisites(self, skill_id: str, strength: PrereqStrength | None = None) -> list[str]: ...
    def dependents(self, skill_id: str, strength: PrereqStrength | None = None) -> list[str]: ...
    def prerequisite_strength(self, source: str, target: str) -> PrereqStrength | None: ...
    def prerequisite_graph(self, strength: PrereqStrength | None = None) -> nx.DiGraph: ...
    def courses_teaching(self, skill_id: str) -> list[str]: ...
    def skills_taught(self, course_id: str) -> list[str]: ...
    def roles_in_economy(self, economy: str) -> list[str]: ...
    def skills_in_economy(self, economy: str) -> list[str]: ...
    def export(self) -> dict[str, list[dict[str, Any]]]: ...


class NetworkXStore:
    """In-memory property graph built from a validated ``Dataset``."""

    def __init__(self, ds: Dataset) -> None:
        self.g = nx.MultiDiGraph()
        self._build(ds)
        self._index()

    # ------------------------------------------------------------ build
    def _add_node(self, node_id: str, label: Label, **props: Any) -> None:
        props.pop("id", None)
        self.g.add_node(node_id, id=node_id, label=label.value, **props)

    def _add_edge(self, source: str, target: str, rel: Rel, **props: Any) -> None:
        self.g.add_edge(source, target, key=rel.value, type=rel.value, **props)

    def _build(self, ds: Dataset) -> None:
        for name in sorted({s.economy.value for s in ds.sectors}
                           | {s.economy.value for s in ds.skills}
                           | {r.economy.value for r in ds.job_roles}):
            self._add_node(economy_id(name), Label.ECONOMY, name=name)
        for s in ds.sectors:
            self._add_node(s.id, Label.SECTOR, **s.model_dump(mode="json"))
            self._add_edge(s.id, economy_id(s.economy.value), Rel.IN_ECONOMY)
        for t in ds.skill_tracks:
            self._add_node(t.id, Label.TRACK, **t.model_dump(mode="json"))
        for sk in ds.skills:
            self._add_node(sk.id, Label.SKILL, **sk.model_dump(mode="json"))
        for r in ds.job_roles:
            self._add_node(r.id, Label.ROLE, **r.model_dump(mode="json"))
            self._add_edge(r.id, r.sector_id, Rel.IN_SECTOR)
        for c in ds.courses:
            self._add_node(c.id, Label.COURSE, **c.model_dump(mode="json"))

        # HAS: union of explicit edges and each track's own sector_id, so a
        # missing edge in either source does not orphan a track.
        for source, target in {(e.source, e.target) for e in ds.sector_has_track} | {
                (t.sector_id, t.id) for t in ds.skill_tracks}:
            self._add_edge(source, target, Rel.HAS)
        # CONTAINS and TEACHES are embedded in node data, not in the edge lists.
        for sk in ds.skills:
            for t in sk.track_ids:
                self._add_edge(t, sk.id, Rel.CONTAINS)
        for c in ds.courses:
            for s in c.skill_ids:
                self._add_edge(c.id, s, Rel.TEACHES)
        for e in ds.skill_required_by_role:
            self._add_edge(e.source, e.target, Rel.REQUIRED_BY, importance=e.importance)
        for e in ds.skill_prerequisite_of:
            self._add_edge(e.source, e.target, Rel.PREREQUISITE_OF,
                           strength=e.strength.value, note=e.note)

    def _index(self) -> None:
        """Typed adjacency indexes, so lookups never scan every edge."""
        self._by_label: dict[str, list[str]] = defaultdict(list)
        for node_id, data in self.g.nodes(data=True):
            self._by_label[data["label"]].append(node_id)
        self._out: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._in: dict[tuple[str, str], list[str]] = defaultdict(list)
        for u, v, data in self.g.edges(data=True):
            self._out[(u, data["type"])].append(v)
            self._in[(v, data["type"])].append(u)
        self._prereq: dict[str | None, nx.DiGraph] = {}
        for strength in (None, PrereqStrength.HARD, PrereqStrength.SOFT):
            dg = nx.DiGraph()
            dg.add_nodes_from(self._by_label[Label.SKILL.value])
            for u, v, data in self.g.edges(data=True):
                if data["type"] == Rel.PREREQUISITE_OF.value and (
                        strength is None or data["strength"] == strength.value):
                    dg.add_edge(u, v, strength=data["strength"], note=data.get("note"))
            self._prereq[strength.value if strength else None] = dg

    # ------------------------------------------------------------ reads
    def node(self, node_id: str) -> dict[str, Any]:
        if node_id not in self.g:
            raise KeyError(node_id)
        return dict(self.g.nodes[node_id])

    def has(self, node_id: str, label: Label | None = None) -> bool:
        return node_id in self.g and (label is None or self.g.nodes[node_id]["label"] == label.value)

    def ids(self, label: Label) -> list[str]:
        return sorted(self._by_label[label.value])

    def required_skills(self, role_id: str) -> list[str]:
        return sorted(self._in[(role_id, Rel.REQUIRED_BY.value)])

    def roles_requiring(self, skill_id: str) -> list[str]:
        return sorted(self._out[(skill_id, Rel.REQUIRED_BY.value)])

    def _prereq_view(self, strength: PrereqStrength | None) -> nx.DiGraph:
        return self._prereq[strength.value if strength else None]

    def prerequisites(self, skill_id: str, strength: PrereqStrength | None = None) -> list[str]:
        g = self._prereq_view(strength)
        return sorted(g.predecessors(skill_id)) if skill_id in g else []

    def dependents(self, skill_id: str, strength: PrereqStrength | None = None) -> list[str]:
        g = self._prereq_view(strength)
        return sorted(g.successors(skill_id)) if skill_id in g else []

    def prerequisite_strength(self, source: str, target: str) -> PrereqStrength | None:
        data = self._prereq[None].get_edge_data(source, target)
        return PrereqStrength(data["strength"]) if data else None

    def prerequisite_graph(self, strength: PrereqStrength | None = None) -> nx.DiGraph:
        return self._prereq_view(strength)

    def courses_teaching(self, skill_id: str) -> list[str]:
        return sorted(self._in[(skill_id, Rel.TEACHES.value)])

    def skills_taught(self, course_id: str) -> list[str]:
        return sorted(self._out[(course_id, Rel.TEACHES.value)])

    def roles_in_economy(self, economy: str) -> list[str]:
        return [r for r in self.ids(Label.ROLE) if self.g.nodes[r]["economy"] == economy]

    def skills_in_economy(self, economy: str) -> list[str]:
        return [s for s in self.ids(Label.SKILL) if self.g.nodes[s]["economy"] == economy]

    def names(self, ids: Iterable[str]) -> list[str]:
        return [self.g.nodes[i]["name"] for i in ids]

    def export(self) -> dict[str, list[dict[str, Any]]]:
        nodes = [dict(data) for _, data in self.g.nodes(data=True)]
        edges = [{"source": u, "target": v, **{k: val for k, val in data.items() if val is not None}}
                 for u, v, data in self.g.edges(data=True)]
        return {"nodes": nodes, "edges": edges}
