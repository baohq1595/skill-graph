"""Domain models for the SkillsFuture dataset.

The models are written against the data as delivered, not the dataset README.
Where the two disagree (see ``ingest.validate``), fields the README promises but
the data omits are optional, and unknown extra fields are kept rather than
rejected, so a richer future export loads without code changes.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Economy(str, Enum):
    DIGITAL = "Digital"
    CARE = "Care"
    GREEN = "Green"


class Difficulty(str, Enum):
    FOUNDATION = "Foundation"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"

    @property
    def rank(self) -> int:
        return {"Foundation": 1, "Intermediate": 2, "Advanced": 3}[self.value]


class Transferability(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

    @property
    def weight(self) -> float:
        return {"High": 1.0, "Medium": 0.5, "Low": 0.0}[self.value]


class PrereqStrength(str, Enum):
    """How strictly a prerequisite edge should be enforced.

    HARD blocks a learning path ("ML is prerequisite for Deep Learning").
    SOFT only guides ordering ("GenAI understanding helps with Prompt Engineering").
    """

    HARD = "hard"
    SOFT = "soft"


class _Node(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    id: str
    name: str


class Sector(_Node):
    economy: Economy
    description: str | None = None


class SkillTrack(_Node):
    sector_id: str
    description: str | None = None


class Skill(_Node):
    track_ids: list[str] = Field(default_factory=list)
    difficulty: Difficulty
    economy: Economy
    transferability: Transferability
    sdfe_priority: bool = False


class JobRole(_Node):
    sector_id: str
    seniority: str | None = None  # data uses "Mid-Senior"; README lists Junior/Mid/Senior
    economy: Economy
    description: str | None = None  # promised by the README, absent in the data


class Course(_Node):
    provider: str
    duration_hours: float = Field(ge=0)
    skill_ids: list[str] = Field(default_factory=list)
    difficulty: Difficulty | None = None
    sfc_funded: bool | None = None
    # promised by the README, absent in the data
    mode: str | None = None
    certification: str | None = None
    url: str | None = None


class Edge(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    source: str
    target: str


class RequiredByEdge(Edge):
    importance: str | None = None  # README: Core/Supporting; absent in every delivered edge


class PrerequisiteEdge(Edge):
    note: str | None = None
    strength: PrereqStrength = PrereqStrength.HARD


class LearnerProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    description: str | None = None
    current_skills: list[str]
    target_role: str
    note: str | None = None


class Dataset(BaseModel):
    """The whole input file, validated."""

    sectors: list[Sector]
    skill_tracks: list[SkillTrack]
    skills: list[Skill]
    job_roles: list[JobRole]
    courses: list[Course]
    sector_has_track: list[Edge]
    skill_required_by_role: list[RequiredByEdge]
    skill_prerequisite_of: list[PrerequisiteEdge]
    learner_profiles: list[LearnerProfile] = Field(default_factory=list)
    sample_queries: list[str] = Field(default_factory=list)
