"""Load the SkillsFuture JSON into validated domain models.

Ingestion is two steps so problems are reported, not swallowed:

1. ``load_dataset`` parses the file into ``Dataset`` (schema-level checks via
   pydantic) and normalises the known README-vs-data differences.
2. ``ingest.validate.validate`` runs the graph-level checks (dangling references,
   prerequisite cycles, contradictions) and returns a report.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from skillgraph.models import Dataset, PrereqStrength

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "skillsfuture_dataset.json"

# A prerequisite note that only *recommends* an ordering. Everything else is
# treated as a hard requirement, the conservative default.
_SOFT_PATTERN = re.compile(r"\b(helps?|helpful|supports?|useful|recommended|beneficial)\b", re.I)
_HARD_PATTERN = re.compile(r"\b(is\s+)?prerequisite\b|\brequired\b|\bmust\b", re.I)


class DatasetError(ValueError):
    """The input file cannot be turned into a usable dataset."""


def data_path() -> Path:
    """Dataset location: ``SKILLGRAPH_DATA`` if set, else the bundled copy."""
    return Path(os.environ.get("SKILLGRAPH_DATA", DEFAULT_DATA_PATH))


def classify_prerequisite(note: str | None) -> PrereqStrength:
    """Decide whether a prerequisite blocks progress or only guides ordering.

    The dataset has no explicit strength field, but its notes are consistent:
    "X is prerequisite for Y" versus "X helps with Y" / "X supports Y".
    A note that says "prerequisite" wins over a softening word.
    """
    if not note:
        return PrereqStrength.HARD
    if _HARD_PATTERN.search(note) and not _SOFT_PATTERN.search(note):
        return PrereqStrength.HARD
    if _SOFT_PATTERN.search(note):
        return PrereqStrength.SOFT
    return PrereqStrength.HARD


def _normalise(raw: dict[str, Any]) -> dict[str, Any]:
    edges = raw.get("edges") or {}
    prerequisites = []
    for edge in edges.get("skill_prerequisite_of", []):
        edge = dict(edge)
        # An explicit strength in a future export wins over the inferred one.
        edge.setdefault("strength", classify_prerequisite(edge.get("note")).value)
        prerequisites.append(edge)

    return {
        "sectors": raw.get("sectors", []),
        "skill_tracks": raw.get("skill_tracks", []),
        "skills": raw.get("skills", []),
        "job_roles": raw.get("job_roles", []),
        "courses": raw.get("courses", []),
        "sector_has_track": edges.get("sector_has_track", []),
        "skill_required_by_role": edges.get("skill_required_by_role", []),
        "skill_prerequisite_of": prerequisites,
        "learner_profiles": raw.get("sample_learner_profiles", []),
        # README calls it sample_nl_queries; the file calls it sample_queries.
        "sample_queries": raw.get("sample_queries", raw.get("sample_nl_queries", [])),
    }


def load_dataset(path: str | Path | None = None) -> Dataset:
    path = Path(path) if path else data_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetError(f"Dataset not found at {path}. Set SKILLGRAPH_DATA to its location.") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise DatasetError(f"{path}: expected a JSON object at the top level.")
    try:
        return Dataset.model_validate(_normalise(raw))
    except Exception as exc:  # pydantic.ValidationError, surfaced with the file name
        raise DatasetError(f"{path} does not match the expected schema:\n{exc}") from exc
