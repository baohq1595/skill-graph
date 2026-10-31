# Data quality report

| Entity | Count |
|---|--:|
| sectors | 8 |
| skill_tracks | 12 |
| skills | 52 |
| job_roles | 18 |
| courses | 30 |
| sector_has_track | 12 |
| skill_required_by_role | 74 |
| skill_prerequisite_of | 30 |
| learner_profiles | 3 |

## Errors (0)

- none

## Warnings (2)

- **difficulty_inversion** — Generative AI (Intermediate) is a soft prerequisite of Prompt Engineering (Foundation).
- **skill_without_course** — 9 skills have no course in the catalogue; gap analysis marks them 'no catalogue course': Security Architecture, Regulatory Technology (RegTech), Patient Data Analytics, Clinical Decision Support Systems, EdTech Platform Development, Care Coordination Technology, Client Communications, Building Energy Modelling, Sustainable Engineering.

## Infos (6)

- **soft_prerequisites** — 6 of 30 prerequisite notes describe a recommendation ('helps', 'supports') rather than a requirement. They guide ordering but do not block: SKL-04->SKL-08; SKL-08->SKL-09; SKL-03->SKL-02; SKL-17->SKL-18; SKL-02->SKL-51; SKL-01->SKL-51.
- **skill_not_required_by_any_role** — 6 skills are required by no role: Natural Language Processing, Computer Vision, Blockchain & DLT, Care Coordination Technology, Client Communications, Energy Management & Optimisation.
- **missing_importance** — The dataset README describes an importance (Core/Supporting) on skill->role edges, but no edge has it. Gap priority is derived from the graph instead (see PriorityWeights in services/queries.py).
- **missing_role_description** — The README lists a description on job roles; the data has none.
- **missing_course_fields** — The README lists mode, certification and url on courses; the data has difficulty and sfc_funded instead. Both sets are supported.
- **seniority_values** — Seniority values in the data are ['Mid', 'Mid-Senior', 'Senior']; the README lists Junior/Mid/Senior. Stored as given.
