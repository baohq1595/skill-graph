"""Output structures shared by the CLI, the API and the LLM layer.

Every ranking carries the reasons behind it, so a learner (or the LLM writing
the reply) can say *why* a skill is high priority, not just that it is.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SkillRef(BaseModel):
    id: str
    name: str
    difficulty: str
    economy: str
    transferability: str
    sdfe_priority: bool
    tracks: list[str] = Field(default_factory=list)


class RoleRef(BaseModel):
    id: str
    name: str
    sector: str
    economy: str
    seniority: str | None = None


class CourseRef(BaseModel):
    id: str
    name: str
    provider: str
    duration_hours: float
    difficulty: str | None = None
    sfc_funded: bool | None = None
    teaches: list[str] = Field(default_factory=list, description="Skill names taught")


class CourseSuggestion(BaseModel):
    course: CourseRef
    also_covers: list[str] = Field(default_factory=list,
                                   description="Other skills the learner still needs that this course teaches")
    why: list[str] = Field(default_factory=list)


class GapItem(BaseModel):
    rank: int
    skill: SkillRef
    priority_score: float = Field(ge=0, le=100)
    priority: Literal["High", "Medium", "Low"]
    reasons: list[str]
    missing_prerequisites: list[str] = Field(default_factory=list, description="Names")
    courses: list[CourseSuggestion] = Field(default_factory=list)
    no_catalogue_course: bool = False


class GapAnalysis(BaseModel):
    target_role: RoleRef
    coverage: float = Field(description="Share of the role's required skills already held, 0-1")
    have: list[SkillRef]
    gaps: list[GapItem]
    enabling_prerequisites: list[GapItem] = Field(
        default_factory=list,
        description="Skills the role does not list but that a gap needs first")
    notes: list[str] = Field(default_factory=list)


class PathStep(BaseModel):
    order: int
    stage: int = Field(description="Steps in the same stage can be learned in parallel")
    skill: SkillRef
    kind: Literal["required", "prerequisite"]
    after: list[str] = Field(default_factory=list, description="Names of skills that must come first")
    why: str


class CoursePlanItem(BaseModel):
    stage: int
    course: CourseRef
    covers: list[str] = Field(description="Names of the path skills this course covers")


class LearningPath(BaseModel):
    target_role: RoleRef
    mode: Literal["fastest", "thorough"]
    already_have: list[str]
    steps: list[PathStep]
    stages: int
    course_plan: list[CoursePlanItem]
    total_course_hours: float
    critical_path_hours: float = Field(description="Hours if courses in the same stage are taken in parallel")
    plan_is_optimal: bool = Field(description="True when the course plan is the exact minimum-hours cover")
    self_study: list[str] = Field(default_factory=list, description="Path skills no catalogue course teaches")
    recommended_extras: list[str] = Field(default_factory=list,
                                          description="Soft prerequisites left out in fastest mode")
    notes: list[str] = Field(default_factory=list)


class RoleSkill(BaseModel):
    skill: SkillRef
    roles_requiring: int
    prerequisites: list[str] = Field(default_factory=list)
    courses: list[CourseRef] = Field(default_factory=list)


class RoleRequirements(BaseModel):
    role: RoleRef
    required: list[RoleSkill]
    foundations: list[SkillRef] = Field(default_factory=list,
                                        description="Prerequisites of the required skills that the role does not list")
    notes: list[str] = Field(default_factory=list)


class TransferItem(BaseModel):
    rank: int
    skill: SkillRef
    score: float = Field(ge=0, le=100)
    evidence: list[str]
    target_roles: list[str] = Field(default_factory=list,
                                    description="Roles in the target economy that require the skill")


class Transferability(BaseModel):
    source_economy: str
    target_economy: str
    direct_overlap: int = Field(description="Source-economy skills required by at least one target-economy role")
    items: list[TransferItem]
    notes: list[str] = Field(default_factory=list)


class SkillCourses(BaseModel):
    skill: SkillRef
    courses: list[CourseSuggestion]
    prerequisites: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
