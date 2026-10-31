from itertools import combinations

import pytest

from skillgraph.graph.schema import Label
from skillgraph.services import course_plan
from skillgraph.services.queries import UnknownEntity


def _all_role_scenarios(ctx):
    """Every role, from a blank start and from each sample learner's skills."""
    starts = [[]] + [lp.current_skills for lp in ctx.dataset.learner_profiles]
    return [(known, role) for role in ctx.store.ids(Label.ROLE) for known in starts]


# --------------------------------------------------------------- learning path
def test_every_path_respects_every_prerequisite(ctx):
    graph = ctx.store.prerequisite_graph()
    for known, role in _all_role_scenarios(ctx):
        path = ctx.service.learning_path(known, role)
        position = {step.skill.id: step.order for step in path.steps}
        stage = {step.skill.id: step.stage for step in path.steps}
        for u, v in graph.edges():
            if u in position and v in position:
                assert position[u] < position[v], (role, u, v)
                assert stage[u] < stage[v], (role, u, v)


def test_every_path_closes_the_gap(ctx):
    for known, role in _all_role_scenarios(ctx):
        path = ctx.service.learning_path(known, role)
        learned = {s.skill.id for s in path.steps}
        assert set(ctx.store.required_skills(role)) <= learned | set(known)
        # every hard prerequisite of a path skill is either known or earlier on the path
        for step in path.steps:
            for pre in ctx.store.prerequisites(step.skill.id, strength=None):
                strength = ctx.store.prerequisite_strength(pre, step.skill.id).value
                if strength == "hard":
                    assert pre in known or pre in learned


def test_unlisted_prerequisite_is_pulled_in(ctx):
    # Priya: the AI/ML Engineer role does not list Statistics, but ML needs it.
    path = ctx.service.learning_path(["SKL-01", "SKL-03", "SKL-17", "SKL-18", "SKL-19"], "ROL-02")
    stats = next(s for s in path.steps if s.skill.id == "SKL-52")
    assert stats.kind == "prerequisite" and stats.stage == 1


def test_known_skill_prerequisites_are_not_relearned(ctx):
    # Knowing ML makes its own prerequisites (Python, Statistics) irrelevant.
    path = ctx.service.learning_path(["SKL-04"], "ROL-02")
    assert "SKL-52" not in {s.skill.id for s in path.steps}


def test_soft_prerequisite_orders_but_does_not_add(ctx):
    fastest = ctx.service.learning_path(["SKL-01"], "ROL-14")      # needs Prompt Engineering
    ids = [s.skill.id for s in fastest.steps]
    assert "SKL-08" not in ids                                   # GenAI only "helps" Prompt Engineering
    assert "Generative AI" in fastest.recommended_extras
    thorough = ctx.service.learning_path(["SKL-01"], "ROL-14", mode="thorough")
    ids = [s.skill.id for s in thorough.steps]
    assert ids.index("SKL-08") < ids.index("SKL-09")


def test_already_qualified_learner_gets_an_empty_path(ctx):
    required = ctx.store.required_skills("ROL-09")
    path = ctx.service.learning_path(required, "ROL-09")
    assert path.steps == [] and path.total_course_hours == 0
    assert any("already have" in n for n in path.notes)


# ---------------------------------------------------------------- course plan
def _brute_force_hours(ctx, skills, known):
    teachable = {s for s in skills if ctx.store.courses_teaching(s)}
    available = set(skills) | set(known)
    cands = set()
    for s in teachable:
        options = ctx.store.courses_teaching(s)
        ok = [c for c in options if course_plan.followable(ctx.store, c, available)]
        cands |= set(ok or options)
    cands = sorted(cands)
    best = None
    for k in range(1, len(cands) + 1):
        for combo in combinations(cands, k):
            if set().union(*(set(ctx.store.skills_taught(c)) for c in combo)) >= teachable:
                hours = sum(ctx.store.node(c)["duration_hours"] for c in combo)
                best = hours if best is None else min(best, hours)
    return best or 0.0


def test_course_plan_is_the_true_minimum(ctx):
    for known, role in _all_role_scenarios(ctx):
        path = ctx.service.learning_path(known, role)
        skills = {s.skill.id for s in path.steps}
        assert path.plan_is_optimal
        assert path.total_course_hours == pytest.approx(_brute_force_hours(ctx, skills, known))


def test_alex_plan_beats_the_greedy_choice(ctx):
    path = ctx.service.learning_path(["SKL-02", "SKL-51", "SKL-20"], "ROL-01")
    assert path.total_course_hours == 108          # greedy would pick 124 hours
    assert path.critical_path_hours <= path.total_course_hours


def test_plan_never_uses_a_course_the_learner_cannot_follow(ctx):
    # MLOps & Model Deployment is the shortest course teaching DevOps, but it also
    # teaches MLOps, which needs Machine Learning that this learner will not have.
    path = ctx.service.learning_path(["SKL-01"], "ROL-03")
    assert "CRS-09" not in {c.course.id for c in path.course_plan}
    assert "CRS-14" in {c.course.id for c in path.course_plan}


def test_greedy_fallback_still_covers_everything(ctx, monkeypatch):
    monkeypatch.setattr(course_plan, "EXACT_LIMIT", 0)
    path = ctx.service.learning_path([], "ROL-02")
    covered = {name for item in path.course_plan for name in item.covers}
    assert {s.skill.name for s in path.steps} - set(path.self_study) <= covered
    assert not path.plan_is_optimal
    assert any("greedy" in n for n in path.notes)


def test_course_order_respects_prerequisites(ctx):
    path = ctx.service.learning_path(["SKL-01", "SKL-03"], "ROL-02")
    stage = {c.course.id: c.stage for c in path.course_plan}
    assert stage["CRS-02"] < stage["CRS-03"]        # ML course before Deep Learning course
    assert stage["CRS-02"] < stage["CRS-09"]        # and before MLOps


def test_skill_without_course_is_self_study(ctx):
    path = ctx.service.learning_path([], "ROL-16")
    assert "Sustainable Engineering" in path.self_study


# --------------------------------------------------------------- gap analysis
@pytest.mark.parametrize("learner", ["LP-01", "LP-02", "LP-03"])
def test_gaps_are_exactly_the_missing_required_skills(ctx, learner):
    lp = ctx.learner(learner)
    gap = ctx.service.gap_analysis(lp.current_skills, lp.target_role)
    required = set(ctx.store.required_skills(lp.target_role))
    assert {g.skill.id for g in gap.gaps} == required - set(lp.current_skills)
    scores = [g.priority_score for g in gap.gaps]
    assert scores == sorted(scores, reverse=True)
    assert [g.rank for g in gap.gaps] == list(range(1, len(gap.gaps) + 1))
    assert gap.coverage == pytest.approx(len(required & set(lp.current_skills)) / len(required), abs=1e-3)


def test_gap_priority_puts_unlocking_skills_first(ctx):
    lp = ctx.learner("LP-03")   # Sustainability Management unlocks Carbon Accounting and ESG Reporting
    gap = ctx.service.gap_analysis(lp.current_skills, lp.target_role)
    assert gap.gaps[0].skill.name == "Sustainability Management"
    assert gap.gaps[0].priority == "High"


def test_every_gap_has_courses_or_is_flagged(ctx):
    for known, role in _all_role_scenarios(ctx):
        for item in ctx.service.gap_analysis(known, role).gaps:
            assert bool(item.courses) != item.no_catalogue_course
            for suggestion in item.courses:
                assert item.skill.name in suggestion.course.teaches


def test_unknown_ids_are_rejected(ctx):
    with pytest.raises(UnknownEntity):
        ctx.service.gap_analysis(["SKL-01"], "ROL-99")
    with pytest.raises(UnknownEntity):
        ctx.service.learning_path(["SKL-999"], "ROL-01")


# ------------------------------------------------------------- other queries
def test_transferability_uses_direct_demand_when_it_exists(ctx):
    result = ctx.service.transferability("Digital", "Care")
    assert result.direct_overlap > 0
    assert result.items[0].target_roles      # top skill is required by a Care role


def test_transferability_explains_when_no_direct_overlap(ctx):
    result = ctx.service.transferability("Digital", "Green")
    assert result.direct_overlap == 0 and result.notes
    assert all(not item.target_roles for item in result.items)
    assert [i.score for i in result.items] == sorted((i.score for i in result.items), reverse=True)


def test_transferability_rejects_unknown_economy(ctx):
    with pytest.raises(UnknownEntity):
        ctx.service.transferability("Digital", "Blue")


def test_role_requirements_lists_unlisted_foundations(ctx):
    req = ctx.service.role_requirements("ROL-02")
    assert {f.name for f in req.foundations} == {"Statistical Analysis", "DevOps & CI/CD"}


def test_courses_for_skill(ctx):
    assert [c.course.id for c in ctx.service.courses_for_skill("SKL-04").courses] == ["CRS-02"]
    assert ctx.service.courses_for_skill("SKL-15").notes[0].startswith("No catalogue course")
