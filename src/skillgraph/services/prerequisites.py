"""Prerequisite reasoning: what must be learned, and in what order.

Two edge strengths matter here (see ``ingest.loader.classify_prerequisite``):

* HARD prerequisites are pulled into a path even if the target role does not
  list them: you cannot learn Deep Learning without Machine Learning.
* SOFT prerequisites never add skills in "fastest" mode, but when both ends
  are already on the path they still order it (GenAI before Prompt Engineering).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import networkx as nx

from skillgraph.graph.store import GraphStore
from skillgraph.models import PrereqStrength


def required_closure(store: GraphStore, targets: set[str], known: set[str],
                     include_soft: bool = False) -> tuple[set[str], set[str]]:
    """Skills to learn = targets plus any unmet prerequisites, transitively.

    Returns ``(to_learn, pulled_in)`` where ``pulled_in`` are the skills added
    only because something on the path needs them. Search stops at known skills:
    if the learner already has Machine Learning, its own prerequisites are moot.
    """
    strength = None if include_soft else PrereqStrength.HARD
    to_learn = {t for t in targets if t not in known}
    queue = deque(to_learn)
    while queue:
        skill = queue.popleft()
        for pre in store.prerequisites(skill, strength):
            if pre not in known and pre not in to_learn:
                to_learn.add(pre)
                queue.append(pre)
    return to_learn, to_learn - targets


@dataclass(frozen=True)
class Ordering:
    order: list[str]                 # a valid learning sequence
    stage: dict[str, int]            # 1-based; same stage = can be learned in parallel
    dropped_soft_edges: list[tuple[str, str]]


def order_skills(store: GraphStore, skills: set[str], priority: dict[str, float] | None = None) -> Ordering:
    """Topologically order ``skills`` using every prerequisite edge between them.

    Stage = length of the longest prerequisite chain ending at the skill, so a
    stage holds everything that can be studied in parallel once earlier stages
    are done. Within a stage, higher priority first, then easier first.
    If soft edges ever create a cycle, they are dropped (hard cycles are
    rejected at ingestion), so an order always exists.
    """
    priority = priority or {}
    full = store.prerequisite_graph()
    dag = nx.DiGraph()
    dag.add_nodes_from(skills)
    dag.add_edges_from((u, v, d) for u, v, d in full.subgraph(skills).edges(data=True))
    dropped: list[tuple[str, str]] = []
    while not nx.is_directed_acyclic_graph(dag):
        cycle = nx.find_cycle(dag)
        soft = [(u, v) for u, v in cycle if dag.edges[u, v].get("strength") == PrereqStrength.SOFT.value]
        edge = soft[0] if soft else cycle[0]
        dag.remove_edge(*edge)
        dropped.append(edge)

    stage: dict[str, int] = {}
    for depth, generation in enumerate(nx.topological_generations(dag), start=1):
        for s in generation:
            stage[s] = depth

    def key(s: str) -> tuple:
        n = store.node(s)
        difficulty = {"Foundation": 1, "Intermediate": 2, "Advanced": 3}.get(n["difficulty"], 2)
        return (stage[s], -priority.get(s, 0.0), difficulty, n["name"])

    return Ordering(sorted(skills, key=key), stage, dropped)


def unlock_counts(store: GraphStore, skills: set[str]) -> dict[str, int]:
    """For each skill, how many other skills in the set depend on it (transitively)."""
    full = store.prerequisite_graph().subgraph(skills)
    return {s: len(nx.descendants(full, s)) for s in skills}
