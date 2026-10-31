"""Entity resolution: free-text names to graph ids.

Every entry point (CLI, API, natural language) goes through here, so "Python",
"python3", "SKL-01" and "Pyton" all mean the same node, and an ambiguous name
("data analyst") becomes a clarification instead of a guess.

Matching order, stopping at the first that succeeds:
    1. exact id ("SKL-01")
    2. exact or alias match on a normalised name
    3. fuzzy match, accepted only when clearly better than the runner-up
The LLM never produces ids itself; it only proposes the surface text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from importlib import resources
from typing import Iterable

import yaml
from rapidfuzz import fuzz, process

from skillgraph.graph.schema import Label
from skillgraph.graph.store import GraphStore
from skillgraph.models import Economy

ACCEPT = 88       # fuzzy score accepted outright
CONSIDER = 72     # fuzzy score worth considering when it beats the runner-up
MARGIN = 8        # how clearly the best candidate must beat the second


class Kind(str, Enum):
    SKILL = "skill"
    ROLE = "role"
    COURSE = "course"


_LABEL = {Kind.SKILL: Label.SKILL, Kind.ROLE: Label.ROLE, Kind.COURSE: Label.COURSE}


class MatchType(str, Enum):
    ID = "id"
    EXACT = "exact"
    ALIAS = "alias"
    FUZZY = "fuzzy"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class Candidate:
    id: str
    name: str
    score: float


@dataclass(frozen=True)
class Resolution:
    query: str
    kind: Kind
    match: MatchType
    id: str | None = None
    name: str | None = None
    candidates: tuple[Candidate, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.id is not None


class ResolutionError(ValueError):
    def __init__(self, resolution: Resolution) -> None:
        self.resolution = resolution
        options = ", ".join(f"{c.name} ({c.id})" for c in resolution.candidates)
        if resolution.match is MatchType.AMBIGUOUS:
            msg = f"'{resolution.query}' could mean several {resolution.kind.value}s: {options}."
        else:
            msg = f"No {resolution.kind.value} called '{resolution.query}'."
            if options:
                msg += f" Did you mean: {options}?"
        super().__init__(msg)


def normalise(text: str) -> str:
    text = text.casefold().replace("&", " and ").replace("’", "'")
    text = re.sub(r"[^\w/+#]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # light plural folding on the last word: "data scientists" -> "data scientist"
    if len(text) > 4 and text.endswith("s") and not text.endswith(("ss", "ics", "ops", "sis", "ies")):
        text = text[:-1]
    return text


def _derived_aliases(name: str) -> set[str]:
    """Aliases that follow from the name itself, so most need no curation."""
    out: set[str] = set()
    for acronym in re.findall(r"\(([^)]+)\)", name):           # "Regulatory Technology (RegTech)"
        out.add(acronym)
    base = re.sub(r"\s*\([^)]*\)", "", name).strip()
    out.add(base)
    if " / " in base:                                            # "AI / Machine Learning Engineer"
        left, right = base.split(" / ", 1)
        tail = " ".join(right.split()[1:])
        out.add(f"{left} {tail}".strip())
        out.add(right)
    return out


class Resolver:
    def __init__(self, store: GraphStore, aliases: dict | None = None) -> None:
        self.store = store
        aliases = aliases if aliases is not None else _load_aliases()
        self._index: dict[Kind, dict[str, str]] = {k: {} for k in Kind}
        self._names: dict[Kind, dict[str, str]] = {k: {} for k in Kind}
        self._collisions: dict[Kind, dict[str, set[str]]] = {k: {} for k in Kind}
        for kind in Kind:
            for node_id in store.ids(_LABEL[kind]):
                name = store.node(node_id)["name"]
                self._names[kind][node_id] = name
                for alias in {name} | _derived_aliases(name):
                    self._add(kind, alias, node_id)
        for kind, key in ((Kind.SKILL, "skills"), (Kind.ROLE, "roles")):
            for node_id, forms in (aliases.get(key) or {}).items():
                if store.has(node_id, _LABEL[kind]):
                    for form in forms:
                        self._add(kind, form, node_id)
        self._economy_aliases: dict[str, str] = {}
        for econ, forms in (aliases.get("economies") or {}).items():
            for form in [econ, *forms]:
                self._economy_aliases[normalise(form)] = econ

    def _add(self, kind: Kind, text: str, node_id: str) -> None:
        key = normalise(text)
        if not key:
            return
        existing = self._index[kind].get(key)
        if existing and existing != node_id:
            self._collisions[kind].setdefault(key, {existing}).add(node_id)
        else:
            self._index[kind][key] = node_id

    # ------------------------------------------------------------- public
    def name(self, kind: Kind, node_id: str) -> str:
        return self._names[kind][node_id]

    def resolve(self, text: str, kind: Kind) -> Resolution:
        query = (text or "").strip()
        if not query:
            return Resolution(query, kind, MatchType.NOT_FOUND)
        upper = query.upper()
        if self.store.has(upper, _LABEL[kind]):
            return Resolution(query, kind, MatchType.ID, upper, self._names[kind][upper])
        key = normalise(query)
        if key in self._collisions[kind]:
            ids = sorted(self._collisions[kind][key])
            return Resolution(query, kind, MatchType.AMBIGUOUS,
                              candidates=tuple(Candidate(i, self._names[kind][i], 100.0) for i in ids))
        if key in self._index[kind]:
            node_id = self._index[kind][key]
            exact = normalise(self._names[kind][node_id]) == key
            return Resolution(query, kind, MatchType.EXACT if exact else MatchType.ALIAS,
                              node_id, self._names[kind][node_id])
        return self._fuzzy(query, key, kind)

    def _fuzzy(self, query: str, key: str, kind: Kind) -> Resolution:
        keys = list(self._index[kind])
        hits = process.extract(key, keys, scorer=fuzz.WRatio, limit=8)
        best: dict[str, float] = {}
        for alias, score, _ in hits:            # several aliases can point at one node
            node_id = self._index[kind][alias]
            best[node_id] = max(best.get(node_id, 0.0), float(score))
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        candidates = tuple(Candidate(i, self._names[kind][i], round(s, 1)) for i, s in ranked[:3])
        if not ranked or ranked[0][1] < CONSIDER:
            return Resolution(query, kind, MatchType.NOT_FOUND,
                              candidates=tuple(c for c in candidates if c.score >= 55))
        top_id, top = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if top - runner_up < MARGIN and (top < ACCEPT or runner_up >= ACCEPT):
            return Resolution(query, kind, MatchType.AMBIGUOUS,
                              candidates=tuple(c for c in candidates if top - c.score < MARGIN))
        if top >= ACCEPT or top - runner_up >= MARGIN:
            return Resolution(query, kind, MatchType.FUZZY, top_id, self._names[kind][top_id], candidates)
        return Resolution(query, kind, MatchType.NOT_FOUND, candidates=candidates)

    def require(self, text: str, kind: Kind) -> str:
        """Resolve or raise ``ResolutionError`` with suggestions."""
        res = self.resolve(text, kind)
        if not res.ok:
            raise ResolutionError(res)
        return res.id  # type: ignore[return-value]

    def require_many(self, texts: Iterable[str], kind: Kind) -> list[str]:
        out: list[str] = []
        for t in texts:
            node_id = self.require(t, kind)
            if node_id not in out:
                out.append(node_id)
        return out

    def economy(self, text: str) -> str | None:
        """Economy name from free text; sector names map to their economy."""
        key = normalise(text or "")
        if key in self._economy_aliases:
            return self._economy_aliases[key]
        for e in Economy:
            if e.value.casefold() == key:
                return e.value
        for sector_id in self.store.ids(Label.SECTOR):
            sector = self.store.node(sector_id)
            if normalise(sector["name"]) == key:
                return sector["economy"]
        return None

    def scan(self, text: str, kind: Kind) -> list[str]:
        """Find every entity of ``kind`` named verbatim (or by alias) in free text.

        Longest match first, non-overlapping. Used by the offline parser and to
        check that LLM-extracted mentions really occur in the question.
        """
        words = normalise(text).split()
        found: list[str] = []
        i = 0
        while i < len(words):
            for n in range(min(6, len(words) - i), 0, -1):
                key = " ".join(words[i:i + n])
                node_id = self._index[kind].get(key)
                if node_id and key not in self._collisions[kind]:
                    if node_id not in found:
                        found.append(node_id)
                    i += n
                    break
            else:
                i += 1
        return found


def _load_aliases() -> dict:
    text = resources.files("skillgraph.resources").joinpath("aliases.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}
