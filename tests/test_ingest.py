import pytest

from skillgraph.context import build_context
from skillgraph.ingest.loader import DatasetError, classify_prerequisite, load_dataset
from skillgraph.ingest.validate import Severity, validate
from skillgraph.models import PrereqStrength


def test_loads_every_entity_from_the_delivered_file():
    ds = load_dataset()
    assert (len(ds.sectors), len(ds.skill_tracks), len(ds.skills), len(ds.job_roles), len(ds.courses)) \
        == (8, 12, 52, 18, 30)
    assert (len(ds.sector_has_track), len(ds.skill_required_by_role), len(ds.skill_prerequisite_of)) \
        == (12, 74, 30)
    assert len(ds.learner_profiles) == 3
    assert len(ds.sample_queries) == 5


@pytest.mark.parametrize("note, expected", [
    ("Python is prerequisite for ML", PrereqStrength.HARD),
    ("Containers prerequisite for Cloud Architecture", PrereqStrength.HARD),
    ("ML foundations helpful for GenAI understanding", PrereqStrength.SOFT),
    ("GenAI understanding helps with Prompt Engineering", PrereqStrength.SOFT),
    ("SQL supports Data Analytics", PrereqStrength.SOFT),
    (None, PrereqStrength.HARD),
    ("", PrereqStrength.HARD),
])
def test_prerequisite_strength_is_read_from_the_note(note, expected):
    assert classify_prerequisite(note) is expected


def test_explicit_strength_in_the_data_wins(write_dataset):
    def mutate(d):
        d["edges"]["skill_prerequisite_of"][0]["strength"] = "soft"
    ds = load_dataset(write_dataset(mutate))
    assert ds.skill_prerequisite_of[0].strength is PrereqStrength.SOFT


def test_delivered_dataset_has_no_errors_and_reports_its_known_problems():
    report = validate(load_dataset())
    codes = {i.code for i in report.issues}
    assert report.ok
    assert {"missing_importance", "soft_prerequisites", "skill_without_course",
            "difficulty_inversion", "seniority_values"} <= codes


def test_dangling_reference_stops_ingestion(write_dataset):
    path = write_dataset(lambda d: d["courses"][0]["skill_ids"].append("SKL-999"))
    report = validate(load_dataset(path))
    assert any(i.code == "dangling_reference" and "SKL-999" in i.refs for i in report.errors)
    with pytest.raises(DatasetError, match="SKL-999"):
        build_context(path)


def test_hard_prerequisite_cycle_stops_ingestion(write_dataset):
    # ML -> Deep Learning already exists; adding Deep Learning -> ML closes a hard loop
    path = write_dataset(lambda d: d["edges"]["skill_prerequisite_of"].append(
        {"source": "SKL-05", "target": "SKL-04", "note": "DL is prerequisite for ML"}))
    report = validate(load_dataset(path))
    assert any(i.code == "hard_prerequisite_cycle" for i in report.errors)


def test_soft_cycle_is_only_a_warning(write_dataset):
    path = write_dataset(lambda d: d["edges"]["skill_prerequisite_of"].append(
        {"source": "SKL-09", "target": "SKL-08", "note": "Prompting helps with GenAI"}))
    report = validate(load_dataset(path))
    assert report.ok
    assert any(i.code == "soft_prerequisite_cycle" for i in report.by_severity(Severity.WARNING))


def test_track_edge_missing_is_repaired_from_sector_id(write_dataset):
    path = write_dataset(lambda d: d["edges"]["sector_has_track"].pop())
    ctx = build_context(path)
    assert any(i.code == "track_sector_missing_edge" for i in ctx.report.issues)
    assert ctx.store.g.has_edge("SEC-01", "TRK-12")  # still linked via the track's sector_id


def test_missing_file_and_bad_json_give_clear_errors(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        load_dataset(bad)


def test_schema_violation_names_the_file(write_dataset):
    path = write_dataset(lambda d: d["skills"][0].update(difficulty="Expert"))
    with pytest.raises(DatasetError, match="does not match the expected schema"):
        load_dataset(path)
