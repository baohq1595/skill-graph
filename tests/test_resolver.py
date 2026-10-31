import pytest

from skillgraph.services.resolver import Kind, MatchType, ResolutionError, normalise


@pytest.mark.parametrize("text, kind, expected_id, match", [
    ("SKL-04", Kind.SKILL, "SKL-04", MatchType.ID),
    ("skl-04", Kind.SKILL, "SKL-04", MatchType.ID),
    ("Machine Learning", Kind.SKILL, "SKL-04", MatchType.EXACT),
    ("Python", Kind.SKILL, "SKL-01", MatchType.ALIAS),
    ("SQL", Kind.SKILL, "SKL-03", MatchType.ALIAS),
    ("RegTech", Kind.SKILL, "SKL-26", MatchType.ALIAS),            # derived from "(RegTech)"
    ("machine lerning", Kind.SKILL, "SKL-04", MatchType.FUZZY),
    ("Pyton", Kind.SKILL, "SKL-01", MatchType.FUZZY),
    ("AI Engineer", Kind.ROLE, "ROL-02", MatchType.ALIAS),
    ("Machine Learning Engineer", Kind.ROLE, "ROL-02", MatchType.ALIAS),  # derived from the slash name
    ("Data Scientists", Kind.ROLE, "ROL-01", MatchType.EXACT),            # plural folded
    ("healthcare data analyst", Kind.ROLE, "ROL-13", MatchType.ALIAS),
])
def test_resolves(ctx, text, kind, expected_id, match):
    res = ctx.resolver.resolve(text, kind)
    assert (res.id, res.match) == (expected_id, match)


@pytest.mark.parametrize("text, kind", [("data analyst", Kind.ROLE), ("security", Kind.SKILL)])
def test_ambiguous_names_are_not_guessed(ctx, text, kind):
    res = ctx.resolver.resolve(text, kind)
    assert res.match is MatchType.AMBIGUOUS and res.id is None and len(res.candidates) >= 2


def test_unknown_name_raises_with_message(ctx):
    with pytest.raises(ResolutionError, match="No role called 'astronaut'"):
        ctx.resolver.require("astronaut", Kind.ROLE)


@pytest.mark.parametrize("text, expected", [
    ("Digital", "Digital"), ("green", "Green"), ("tech", "Digital"),
    ("Healthcare", "Care"), ("Built Environment", "Green"), ("finance", "Digital"), ("moon", None),
])
def test_economy(ctx, text, expected):
    assert ctx.resolver.economy(text) == expected


def test_scan_finds_mentions_in_free_text(ctx):
    text = "I know Python and SQL. What is the fastest path to becoming an AI Engineer?"
    assert ctx.resolver.scan(text, Kind.SKILL) == ["SKL-01", "SKL-03"]
    assert ctx.resolver.scan(text, Kind.ROLE) == ["ROL-02"]


def test_normalise():
    assert normalise("  SQL & Database   Management ") == "sql and database management"
    assert normalise("Data Scientists") == "data scientist"
    assert normalise("Analytics") == "analytics"
