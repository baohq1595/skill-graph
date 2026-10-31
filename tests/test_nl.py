import json

import pytest

from skillgraph.nl.grounding import check_answer
from skillgraph.nl.llm import CachedClient, GeminiClient, LLMError, LLMRefusal, LLMReply
from skillgraph.nl.pipeline import Assistant
from skillgraph.nl.rules import RuleParser
from skillgraph.nl.schema import Intent, llm_json_schema


# ---------------------------------------------------------------- test doubles
class FakeLLM:
    """Returns scripted outputs and records what it was asked."""

    model = "fake"

    def __init__(self, parsed=None, answer="", fail=None):
        self.parsed, self.answer, self.fail, self.calls = parsed, answer, fail, []

    def json(self, system, user, schema, *, effort="low"):
        self.calls.append(("json", user))
        if self.fail == "parse":
            raise LLMError("boom")
        return self.parsed, LLMReply(text=json.dumps(self.parsed), model="fake", latency_ms=1.0)

    def text(self, system, user, *, effort="low"):
        self.calls.append(("text", user))
        if self.fail == "answer":
            raise LLMRefusal("declined")
        return LLMReply(text=self.answer, model="fake", latency_ms=1.0)


def parsed(**kw):
    base = {"intent": "skills_for_role", "target_role": None, "current_role": None, "known_skills": [],
            "skills": [], "source_economy": None, "target_economy": None, "bidirectional": False,
            "fastest": True}
    return base | kw


# ------------------------------------------------------------ rule parser
SAMPLE_EXPECTATIONS = [
    ("What skills does a Data Scientist need?", Intent.SKILLS_FOR_ROLE, {"target_role": "Data Scientist"}),
    ("I know Python and SQL. What is the fastest path to becoming an AI Engineer?", Intent.LEARNING_PATH,
     {"target_role": "AI / Machine Learning Engineer",
      "known_skills": ["Python Programming", "SQL & Database Management"]}),
    ("Which skills are most transferable between the Digital and Green economies?", Intent.TRANSFERABLE_SKILLS,
     {"source_economy": "Digital", "target_economy": "Green", "bidirectional": True}),
    ("I am a healthcare data analyst. What skill gaps do I have for a Sustainability Consultant role?",
     Intent.GAP_ANALYSIS, {"target_role": "Sustainability Consultant", "current_role": "Health Data Analyst",
                           "known_skills": []}),
    ("What SkillsFuture courses should I take to learn Machine Learning?", Intent.COURSES_FOR_SKILL,
     {"skills": ["Machine Learning"]}),
]


@pytest.mark.parametrize("question, intent, fields", SAMPLE_EXPECTATIONS)
def test_rule_parser_handles_every_sample_query(ctx, question, intent, fields):
    p = RuleParser(ctx.resolver).parse(question)
    assert p.intent is intent
    for key, value in fields.items():
        assert getattr(p, key) == value


# ------------------------------------------------------------ pipeline (offline)
@pytest.fixture
def offline(ctx):
    return Assistant(ctx, llm=None)


def test_every_sample_query_is_answered_offline(ctx, offline):
    for q in ctx.dataset.sample_queries:
        r = offline.ask(q)
        assert r.status == "answered", q
        assert r.parser == "rules" and r.answer_source == "template"
        assert r.results and r.answer


def test_between_economies_answers_both_directions(offline):
    r = offline.ask("Which skills are most transferable between the Digital and Green economies?")
    assert [(x["source_economy"], x["target_economy"]) for x in r.results] == [("Digital", "Green"),
                                                                              ("Green", "Digital")]


def test_current_role_is_turned_into_skills(offline):
    r = offline.ask("I am a healthcare data analyst. What skill gaps do I have for a Sustainability Consultant role?")
    assert {s["name"] for s in r.resolved["known_skills"]} >= {"Data Analytics", "Health Informatics"}
    assert any("Assumed you have" in n for n in r.notes)


def test_ambiguous_role_asks_instead_of_guessing(offline):
    r = offline.ask("What skills do I need to become a data analyst?")
    assert r.status == "needs_clarification"
    assert "FinTech Data Analyst" in r.answer and "Health Data Analyst" in r.answer
    assert r.results == []


def test_unknown_known_skill_is_reported_not_fatal(offline):
    r = offline.ask("I know Python and basket weaving. What is the fastest path to Data Engineer?")
    assert r.status == "answered"
    assert any("basket weaving" in n for n in r.notes)


def test_out_of_scope_gets_examples(offline):
    r = offline.ask("What's the weather tomorrow?")
    assert r.status == "unsupported" and "Data Scientist" in r.answer


def test_empty_question(offline):
    assert offline.ask("   ").status == "unsupported"


# ------------------------------------------------------------ pipeline with an LLM
def test_llm_parse_and_grounded_answer(ctx):
    llm = FakeLLM(parsed(target_role="Data Scientist"),
                  answer="A Data Scientist needs Python Programming, SQL & Database Management and "
                         "Machine Learning. Machine Learning Fundamentals (60 h) covers ML.")
    r = Assistant(ctx, llm=llm).ask("what does a data scientist need")
    assert (r.status, r.parser, r.answer_source) == ("answered", "llm", "llm")
    assert r.grounding_problems == []
    # the answer prompt carries the graph result, not the catalogue
    assert "Machine Learning Fundamentals" in llm.calls[1][1]


def test_hallucinated_course_falls_back_to_template(ctx):
    llm = FakeLLM(parsed(target_role="Data Scientist"),
                  answer="Take Deep Learning & Neural Networks (40 h) first.")   # not in this result
    r = Assistant(ctx, llm=llm).ask("what does a data scientist need")
    assert r.answer_source == "template"
    assert any("Deep Learning & Neural Networks" in p for p in r.grounding_problems)
    assert r.answer.startswith("## Skills for Data Scientist")


def test_llm_parse_failure_falls_back_to_rules(ctx):
    llm = FakeLLM(fail="parse", answer="ok")
    r = Assistant(ctx, llm=llm).ask("What skills does a Data Scientist need?")
    assert r.parser == "rules" and r.status == "answered"
    assert r.trace[0].detail["error"] == "boom"


def test_llm_refusal_on_answer_uses_template(ctx):
    llm = FakeLLM(parsed(target_role="Data Scientist"), fail="answer")
    r = Assistant(ctx, llm=llm).ask("x")
    assert r.status == "answered" and r.answer_source == "template"


def test_llm_invented_role_is_not_trusted(ctx):
    llm = FakeLLM(parsed(target_role="Chief Astronaut Officer"))
    r = Assistant(ctx, llm=llm).ask("how do I become a chief astronaut officer")
    assert r.status == "not_found" and r.results == []


# ------------------------------------------------------------ grounding
def test_grounding_check():
    payload = {"results": [{"name": "Machine Learning", "hours": 60.0, "coverage": 0.333}]}
    names = ["Machine Learning", "Deep Learning", "Machine Learning Fundamentals"]
    assert check_answer("Learn Machine Learning (60 hours). You have 33% of it.", payload, names) == []
    problems = check_answer("Then Deep Learning, about 45 h.", payload, names)
    assert len(problems) == 2


# --------------------------------------------------------------- Gemini client
def _mock_gemini(handler):
    import httpx
    from google import genai
    from google.genai import types

    return genai.Client(api_key="test-key", vertexai=False, http_options=types.HttpOptions(
        base_url="https://api.test", httpx_client=httpx.Client(transport=httpx.MockTransport(handler))))


def _response(text, finish="STOP", block=None):
    body = {"candidates": [{"content": {"role": "model", "parts": [{"text": "thinking...", "thought": True},
                                                                     {"text": text}] if text else []},
                            "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 7, "thoughtsTokenCount": 3},
            "modelVersion": "gemini-3.8-flash"}
    if block:
        body = {"promptFeedback": {"blockReason": block}}
    return body


def test_gemini_request_shape():
    import httpx

    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["key"] = request.headers.get("x-goog-api-key")
        return httpx.Response(200, json=_response(json.dumps(parsed(target_role="Data Scientist"))))

    client = GeminiClient(client=_mock_gemini(handler))
    data, reply = client.json("SYSTEM", "question", llm_json_schema())
    body, config = seen["body"], seen["body"]["generationConfig"]
    assert data["target_role"] == "Data Scientist"
    assert reply.usage["input_tokens"] == 12 and reply.usage["thinking_tokens"] == 3
    assert "models/gemini-3.8-flash:generateContent" in seen["url"] and seen["key"] == "test-key"
    assert body["systemInstruction"]["parts"][0]["text"] == "SYSTEM"
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"]["required"] == llm_json_schema()["required"]
    thinking = config["thinkingConfig"]  # the SDK sends snake_case; the API accepts both
    assert thinking.get("thinkingLevel", thinking.get("thinking_level")) == "LOW"
    assert "temperature" not in config


def test_gemini_thought_parts_are_not_returned_as_text():
    import httpx

    client = GeminiClient(client=_mock_gemini(lambda r: httpx.Response(200, json=_response("the answer"))))
    assert client.text("s", "u").text == "the answer"


@pytest.mark.parametrize("payload", [_response("", "SAFETY"), _response("", block="PROHIBITED_CONTENT")])
def test_gemini_block_is_a_refusal(payload):
    import httpx

    client = GeminiClient(client=_mock_gemini(lambda r: httpx.Response(200, json=payload)))
    with pytest.raises(LLMRefusal):
        client.text("s", "u")


def test_gemini_http_error_is_wrapped():
    import httpx

    error = {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}}
    client = GeminiClient(client=_mock_gemini(lambda r: httpx.Response(400, json=error)))
    with pytest.raises(LLMError, match="400"):
        client.text("s", "u")


def test_gemini_timeout_is_retried_then_wrapped():
    import httpx

    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    client = GeminiClient(client=_mock_gemini(handler))
    with pytest.raises(LLMError, match="ReadTimeout"):
        client.text("s", "u")
    assert len(calls) == 2


def test_gemini_timeout_recovers_on_retry():
    import httpx

    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json=_response("ok"))

    assert GeminiClient(client=_mock_gemini(handler)).text("s", "u").text == "ok"


def test_gemini_ignores_vertex_environment(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert GeminiClient().client.vertexai is False


def _fake_service_account(path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
    path.write_text(json.dumps({"type": "service_account", "project_id": "demo-project", "private_key_id": "k1",
                                "private_key": key, "client_email": "sa@demo-project.iam.gserviceaccount.com",
                                "client_id": "1", "token_uri": "https://oauth2.googleapis.com/token"}))
    return path


def test_vertex_uses_only_the_named_service_account(tmp_path, monkeypatch):
    from skillgraph.nl.llm import llm_from_env

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/ambient.json")
    monkeypatch.setenv("SKILLGRAPH_VERTEX_CREDENTIALS", str(_fake_service_account(tmp_path / "sa.json")))
    monkeypatch.setenv("SKILLGRAPH_LLM_CACHE", "off")
    client = llm_from_env()
    api = client.client._api_client
    assert client.backend == "vertex" and api.vertexai
    assert (api.project, api.location) == ("demo-project", "global")
    assert api._credentials.service_account_email == "sa@demo-project.iam.gserviceaccount.com"


def test_vertex_missing_file_is_a_clear_error(monkeypatch):
    from skillgraph.nl.llm import llm_from_env

    monkeypatch.setenv("SKILLGRAPH_LLM", "vertex")
    monkeypatch.setenv("SKILLGRAPH_VERTEX_CREDENTIALS", "/nonexistent/sa.json")
    with pytest.raises(LLMError, match="not found"):
        llm_from_env()


def test_cache_returns_identical_reply_without_calling_again(tmp_path):
    inner = FakeLLM(parsed(), answer="hello")
    cached = CachedClient(inner, tmp_path)
    first = cached.text("s", "u")
    second = cached.text("s", "u")
    assert (first.text, second.text, second.cached) == ("hello", "hello", True)
    assert len(inner.calls) == 1
    cached.text("s", "different question")
    assert len(inner.calls) == 2
