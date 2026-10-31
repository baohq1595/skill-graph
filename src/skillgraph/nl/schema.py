"""The contract between natural language and the graph.

The parser (LLM or rule-based) fills ``ParsedQuery`` with the *words* the user
used. It never produces graph ids: ``nl.pipeline`` resolves those words through
``services.resolver``, so a hallucinated skill fails resolution visibly instead
of becoming a wrong answer.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    SKILLS_FOR_ROLE = "skills_for_role"            # template 1
    LEARNING_PATH = "learning_path"                # template 2
    TRANSFERABLE_SKILLS = "transferable_skills"    # template 3
    GAP_ANALYSIS = "gap_analysis"                  # sample query 4
    COURSES_FOR_SKILL = "courses_for_skill"        # sample query 5
    OUT_OF_SCOPE = "out_of_scope"


class ParsedQuery(BaseModel):
    intent: Intent
    target_role: str | None = Field(None, description="Role the user wants to move into")
    current_role: str | None = Field(None, description="Role the user says they hold now")
    known_skills: list[str] = Field(default_factory=list, description="Skills the user says they have")
    skills: list[str] = Field(default_factory=list, description="Skills the user asks about (courses intent)")
    source_economy: str | None = None
    target_economy: str | None = None
    bidirectional: bool = Field(False, description="True for 'between A and B' rather than 'A to B'")
    fastest: bool = Field(True, description="False when the user asks for a thorough or complete path")


def llm_json_schema() -> dict:
    """JSON schema for structured outputs: every key required, no extras."""
    nullable = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    strings = {"type": "array", "items": {"type": "string"}}
    props = {
        "intent": {"type": "string", "enum": [i.value for i in Intent]},
        "target_role": nullable,
        "current_role": nullable,
        "known_skills": strings,
        "skills": strings,
        "source_economy": nullable,
        "target_economy": nullable,
        "bidirectional": {"type": "boolean"},
        "fastest": {"type": "boolean"},
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
