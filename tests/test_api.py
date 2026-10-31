import pytest
from fastapi.testclient import TestClient

from skillgraph.api.app import app


@pytest.fixture(scope="module")
def client(monkeypatch_module=None):
    return TestClient(app)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("SKILLGRAPH_LLM", "off")


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["nodes"] == 123 and body["llm"].startswith("offline")


def test_gap_analysis_by_learner(client):
    body = client.post("/gap-analysis", json={"learner_id": "LP-01"}).json()
    assert body["target_role"]["name"] == "Data Scientist"
    assert [g["skill"]["name"] for g in body["gaps"]][0] == "Python Programming"


def test_learning_path_accepts_names_and_typos(client):
    body = client.post("/learning-path", json={"target_role": "ai engineer", "current_skills": ["pyton", "SQL"]}).json()
    assert body["target_role"]["id"] == "ROL-02"
    assert body["already_have"] == ["Python Programming"]
    assert body["total_course_hours"] > 0


def test_ambiguous_name_returns_422_with_candidates(client):
    res = client.post("/gap-analysis", json={"target_role": "data analyst"})
    assert res.status_code == 422
    assert {c["id"] for c in res.json()["candidates"]} >= {"ROL-10", "ROL-13"}


def test_missing_target_is_a_validation_error(client):
    assert client.post("/learning-path", json={"current_skills": ["Python"]}).status_code == 422


def test_unknown_learner_is_404(client):
    assert client.post("/gap-analysis", json={"learner_id": "LP-99"}).status_code == 404


def test_transferability_and_courses(client):
    t = client.get("/transferability", params={"source": "tech", "target": "Care"}).json()
    assert t["source_economy"] == "Digital" and t["direct_overlap"] > 0
    assert client.get("/transferability", params={"source": "Digital", "target": "Moon"}).status_code == 422
    c = client.get("/skills/machine learning/courses").json()
    assert c["courses"][0]["course"]["id"] == "CRS-02"


def test_role_skills_and_catalog(client):
    assert len(client.get("/roles/Data Scientist/skills").json()["required"]) == 6
    assert len(client.get("/catalog/skills").json()) == 52
    assert len(client.get("/learners").json()) == 3


def test_ask_offline(client):
    body = client.post("/ask", json={"question": "What skills does a Data Scientist need?"}).json()
    assert body["status"] == "answered" and body["parser"] == "rules" and body["results"]


def test_graph_and_page(client):
    g = client.get("/graph").json()
    assert len(g["nodes"]) == 123 and len(g["edges"]) == 263
    assert "Skill Graph Explorer" in client.get("/").text


def test_data_quality(client):
    codes = {i["code"] for i in client.get("/data-quality").json()["issues"]}
    assert "missing_importance" in codes


def test_dotenv_never_overrides_the_environment(tmp_path, monkeypatch):
    from skillgraph.config import load_dotenv

    env = tmp_path / ".env"
    env.write_text("# comment\nSKILLGRAPH_T1=from_file\nexport SKILLGRAPH_T2='quoted'\nSKILLGRAPH_T3=file\n")
    monkeypatch.setenv("SKILLGRAPH_T3", "from_env")
    monkeypatch.delenv("SKILLGRAPH_T1", raising=False)
    monkeypatch.delenv("SKILLGRAPH_T2", raising=False)
    assert sorted(load_dotenv(env)) == ["SKILLGRAPH_T1", "SKILLGRAPH_T2"]
    import os
    assert (os.environ["SKILLGRAPH_T1"], os.environ["SKILLGRAPH_T2"], os.environ["SKILLGRAPH_T3"]) \
        == ("from_file", "quoted", "from_env")
