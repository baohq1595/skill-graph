"""Graph schema: node labels and relationship types.

    (Economy)<-[:IN_ECONOMY]-(Sector)-[:HAS]->(SkillTrack)-[:CONTAINS]->(Skill)
    (JobRole)-[:IN_SECTOR]->(Sector)
    (Skill)-[:REQUIRED_BY {importance}]->(JobRole)
    (Skill)-[:PREREQUISITE_OF {strength, note}]->(Skill)
    (Course)-[:TEACHES]->(Skill)

Relationship directions follow the brief (Skill REQUIRED_BY Role, Course
TEACHES Skill), so the same names work unchanged as Neo4j relationship types.
"""

from enum import Enum


class Label(str, Enum):
    ECONOMY = "Economy"
    SECTOR = "Sector"
    TRACK = "SkillTrack"
    SKILL = "Skill"
    ROLE = "JobRole"
    COURSE = "Course"


class Rel(str, Enum):
    IN_ECONOMY = "IN_ECONOMY"
    HAS = "HAS"
    CONTAINS = "CONTAINS"
    REQUIRED_BY = "REQUIRED_BY"
    PREREQUISITE_OF = "PREREQUISITE_OF"
    TEACHES = "TEACHES"
    IN_SECTOR = "IN_SECTOR"


def economy_id(name: str) -> str:
    return f"ECO-{name}"
