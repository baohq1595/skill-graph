# SkillsFuture Skills Framework — Curated Dataset

## Overview

This dataset is a curated subset of Singapore's SkillsFuture Skills Framework, prepared for the Senior AI Engineer take-home assignment. It provides a ready-to-use graph data source so you can focus on modelling, querying, and LLM integration rather than data wrangling.

## File

**`skillsfuture_dataset.json`** — single JSON file (~30 KB)

## Schema

### Nodes

| Entity | Count | Key Fields |
|--------|-------|------------|
| **Sectors** | 8 | `id`, `name`, `economy` (Digital/Care/Green), `description` |
| **Skill Tracks** | 12 | `id`, `name`, `sector_id`, `description` |
| **Skills** | 52 | `id`, `name`, `track_ids[]`, `difficulty` (Foundation/Intermediate/Advanced), `economy`, `transferability` (High/Medium/Low), `sdfe_priority` (bool) |
| **Job Roles** | 18 | `id`, `name`, `sector_id`, `seniority` (Junior/Mid/Senior), `economy`, `description` |
| **Courses** | 30 | `id`, `name`, `provider`, `skill_ids[]`, `duration_hours`, `mode` (Online/Classroom/Hybrid), `certification`, `url` |

### Edges (Relationships)

| Relationship | Count | Description |
|-------------|-------|-------------|
| `sector_has_track` | 12 | Links sectors to their skill tracks |
| `skill_required_by_role` | 74 | Links skills to the job roles that require them, with `importance` (Core/Supporting) |
| `skill_prerequisite_of` | 30 | Defines prerequisite chains between skills |

Note: `track_contains_skill` and `course_teaches_skill` relationships are embedded in the node data (via `track_ids` on skills and `skill_ids` on courses).

### Economies

The dataset spans Singapore's 3 growth economies from the SDFE 2025 report:

- **Digital** — ICT, Financial Services, Manufacturing (4 sectors)
- **Care** — Healthcare, Education & Training, Social Services (3 sectors)
- **Green** — Built Environment, Energy & Chemicals (2 sectors)

Note: Some sectors contribute to multiple economies; the primary mapping is used.

### Sample Data

The dataset includes test fixtures to help you validate your implementation:

- **3 Learner Profiles** — career switcher, upskilling developer, cross-sector transition
- **5 Sample NL Queries** — covering the 3 required LLM query templates, with expected output hints

## ID Convention

All IDs follow a prefix pattern for easy identification:

| Entity | Prefix | Example |
|--------|--------|---------|
| Sector | `SEC-` | `SEC-01` |
| Skill Track | `TRK-` | `TRK-01` |
| Skill | `SKL-` | `SKL-01` |
| Job Role | `ROL-` | `ROL-01` |
| Course | `CRS-` | `CRS-01` |

## Data Sources

This dataset is derived from publicly available information:

- [SkillsFuture Skills Framework](https://www.skillsfuture.gov.sg/skills-framework) — sector and job role structures
- [SDFE 2025 Report](https://jobsandskills.skillsfuture.gov.sg/sdfe-2025) — priority skills and growth economies
- [MySkillsFuture Course Directory](https://data.gov.sg) — training provider and course information

The data has been simplified and restructured for this assignment. Real-world SkillsFuture data is significantly larger (38 sectors, 119+ job roles, 80+ skills).

## Quick Start

```python
import json

with open("skillsfuture_dataset.json") as f:
    data = json.load(f)

# Access nodes
sectors = data["sectors"]           # 8 sectors
skills = data["skills"]             # 52 skills
job_roles = data["job_roles"]       # 18 job roles
courses = data["courses"]           # 30 courses

# Access edges
edges = data["edges"]
sector_track = edges["sector_has_track"]        # 12 edges
skill_role = edges["skill_required_by_role"]    # 74 edges
prerequisites = edges["skill_prerequisite_of"]  # 30 edges

# Access test fixtures
learners = data["sample_learner_profiles"]      # 3 profiles
queries = data["sample_nl_queries"]             # 5 NL queries
```
