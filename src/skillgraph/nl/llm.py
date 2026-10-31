"""LLM client: one small interface, a Gemini implementation, and a disk cache.

Configuration (environment variables):
    SKILLGRAPH_LLM          auto (default) | gemini | vertex | off
                            auto uses the Gemini API when GEMINI_API_KEY (or
                            GOOGLE_API_KEY) is set, Vertex AI when
                            SKILLGRAPH_VERTEX_CREDENTIALS is set, otherwise
                            runs offline.
    SKILLGRAPH_LLM_MODEL    default gemini-3.8-flash
    SKILLGRAPH_VERTEX_CREDENTIALS
                            path to a service-account JSON file (Vertex AI)
    GOOGLE_CLOUD_PROJECT    Vertex project (default: the file's project_id)
    GOOGLE_CLOUD_LOCATION   Vertex location (default: global)
    SKILLGRAPH_LLM_CACHE    cache directory (default .cache/llm), or "off"

The cache keys on the full request, so a re-run of the same question with the
same prompts and model is free and reproducible (useful for demos and evals);
changing a prompt changes the key.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

DEFAULT_MODEL = "gemini-3.8-flash"
CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
TIMEOUT_MS = 120_000
NETWORK_ATTEMPTS = 2  # a dropped or timed-out connection is retried once; HTTP 429/5xx retries are the SDK's
# Responses blocked for content reasons are reported as refusals, not errors.
BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}


class LLMError(RuntimeError):
    """The model could not be reached or returned something unusable."""


class LLMRefusal(LLMError):
    """The model declined the request (stop_reason == "refusal")."""


@dataclass
class LLMReply:
    text: str
    model: str
    latency_ms: float
    cached: bool = False
    usage: dict[str, int] = field(default_factory=dict)


class LLMClient(Protocol):
    model: str

    def json(self, system: str, user: str, schema: dict, *, effort: str = "low") -> tuple[dict, LLMReply]: ...
    def text(self, system: str, user: str, *, effort: str = "low") -> LLMReply: ...


class GeminiClient:
    """Gemini through the official ``google-genai`` SDK.

    Two backends, same model family and request: the Gemini Developer API
    (API key, the default) or Vertex AI (service account, ``on_vertex``).

    * Structured output (``response_json_schema``) returns schema-valid JSON
      for intent parsing.
    * ``effort`` maps to Gemini's ``thinking_level``: parsing runs at low,
      answer writing at medium.
    * The system prompt is identical across calls (static rules plus the
      catalogue), so Gemini's implicit prompt caching applies to it.
    * Credentials are always explicit: an API key, or the one service-account
      file named in configuration. A stray ``GOOGLE_GENAI_USE_VERTEXAI`` or
      application-default credential on the machine is never picked up.
    """

    def __init__(self, model: str = DEFAULT_MODEL, client: Any = None, api_key: str | None = None) -> None:
        import httpx  # installed with google-genai
        from google import genai  # optional dependency, imported only when used
        from google.genai import errors, types

        self._errors, self._types = errors, types
        self._network_errors = (OSError, httpx.HTTPError)
        self.model = model
        if client is None:
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            client = genai.Client(api_key=key, vertexai=False,
                                  http_options=_http_options(types))
        self.client = client
        self.backend = "vertex" if getattr(client, "vertexai", False) else "gemini-api"

    @classmethod
    def on_vertex(cls, credentials_file: str | Path, model: str = DEFAULT_MODEL, *,
                  project: str | None = None, location: str = "global") -> "GeminiClient":
        """Vertex AI, authenticated with the given service-account file only."""
        from google import genai
        from google.genai import types
        from google.oauth2 import service_account

        path = Path(credentials_file).expanduser()
        if not path.is_file():
            raise LLMError(f"Vertex credentials file not found: {path}")
        creds = service_account.Credentials.from_service_account_file(str(path), scopes=[CLOUD_SCOPE])
        client = genai.Client(vertexai=True, credentials=creds, project=project or creds.project_id,
                              location=location, http_options=_http_options(types))
        return cls(model, client=client)

    def _call(self, system: str, user: str, *, effort: str, max_tokens: int,
              schema: dict | None = None) -> LLMReply:
        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            thinking_config=t.ThinkingConfig(thinking_level=effort.upper()),
            response_mime_type="application/json" if schema is not None else None,
            response_json_schema=schema,
        )
        started = time.perf_counter()
        try:
            response = self._generate(user, config)
        except self._errors.APIError as exc:
            if exc.code in (401, 403):
                raise LLMError(f"Gemini credentials were rejected ({self.backend}): {exc.message}") from exc
            if exc.code == 429:
                raise LLMError("Gemini API rate limit or quota reached.") from exc
            raise LLMError(f"Gemini API error {exc.code}: {exc.message}") from exc
        except self._network_errors as exc:  # connection failures, timeouts
            raise LLMError(f"Could not reach the Gemini API: {type(exc).__name__}: {exc}") from exc
        latency = (time.perf_counter() - started) * 1000

        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and feedback.block_reason:
            raise LLMRefusal(f"The request was blocked ({_name(feedback.block_reason)}).")
        if not response.candidates:
            raise LLMError("The model returned no candidates.")
        candidate = response.candidates[0]
        reason = _name(candidate.finish_reason)
        if reason in BLOCKED:
            raise LLMRefusal(f"The model declined this request ({reason}).")
        if reason == "MAX_TOKENS":
            raise LLMError("The model ran out of output tokens before finishing.")
        parts = candidate.content.parts if candidate.content and candidate.content.parts else []
        text = "".join(p.text for p in parts if p.text and not p.thought).strip()
        if not text:
            raise LLMError("The model returned no text.")
        u = response.usage_metadata
        usage = {"input_tokens": (u and u.prompt_token_count) or 0,
                 "output_tokens": (u and u.candidates_token_count) or 0,
                 "thinking_tokens": (u and u.thoughts_token_count) or 0,
                 "cache_read_input_tokens": (u and u.cached_content_token_count) or 0}
        return LLMReply(text=text, model=response.model_version or self.model, latency_ms=latency, usage=usage)

    def _generate(self, user: str, config: Any) -> Any:
        for attempt in range(1, NETWORK_ATTEMPTS + 1):
            try:
                return self.client.models.generate_content(model=self.model, contents=user, config=config)
            except self._network_errors:
                if attempt == NETWORK_ATTEMPTS:
                    raise

    def json(self, system: str, user: str, schema: dict, *, effort: str = "low") -> tuple[dict, LLMReply]:
        reply = self._call(system, user, effort=effort, max_tokens=4000, schema=schema)
        try:
            return json.loads(reply.text), reply
        except json.JSONDecodeError as exc:
            raise LLMError(f"Structured output was not valid JSON: {exc}") from exc

    def text(self, system: str, user: str, *, effort: str = "low") -> LLMReply:
        return self._call(system, user, effort=effort, max_tokens=8000)


def _http_options(types: Any) -> Any:
    retry = types.HttpRetryOptions(attempts=3, http_status_codes=[429, 500, 502, 503, 504])
    return types.HttpOptions(timeout=TIMEOUT_MS, retry_options=retry)


def _name(value: Any) -> str:
    """Enum or plain string -> its upper-case name ("" for None)."""
    if value is None:
        return ""
    return str(getattr(value, "name", value)).upper()


class CachedClient:
    """Wraps any LLMClient with a content-addressed disk cache."""

    def __init__(self, inner: LLMClient, directory: str | Path) -> None:
        self.inner = inner
        self.model = inner.model
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, kind: str, *parts: Any) -> Path:
        blob = json.dumps([kind, self.model, *parts], sort_keys=True, ensure_ascii=False)
        return self.dir / f"{hashlib.sha256(blob.encode()).hexdigest()[:32]}.json"

    def _load(self, path: Path) -> LLMReply | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return LLMReply(text=data["text"], model=data["model"], latency_ms=0.0, cached=True,
                        usage=data.get("usage", {}))

    def _save(self, path: Path, reply: LLMReply) -> None:
        path.write_text(json.dumps({"text": reply.text, "model": reply.model, "usage": reply.usage},
                                   ensure_ascii=False), encoding="utf-8")

    def json(self, system: str, user: str, schema: dict, *, effort: str = "low") -> tuple[dict, LLMReply]:
        path = self._key("json", system, user, schema, effort)
        hit = self._load(path)
        if hit:
            return json.loads(hit.text), hit
        data, reply = self.inner.json(system, user, schema, effort=effort)
        self._save(path, reply)
        return data, reply

    def text(self, system: str, user: str, *, effort: str = "low") -> LLMReply:
        path = self._key("text", system, user, effort)
        hit = self._load(path)
        if hit:
            return hit
        reply = self.inner.text(system, user, effort=effort)
        self._save(path, reply)
        return reply


def llm_from_env() -> LLMClient | None:
    """The configured client, or None for offline mode."""
    mode = os.environ.get("SKILLGRAPH_LLM", "auto").strip().lower()
    if mode == "off":
        return None
    has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    vertex_file = os.environ.get("SKILLGRAPH_VERTEX_CREDENTIALS", "").strip()
    if mode == "auto":
        if not (has_key or vertex_file):
            return None
        mode = "gemini" if has_key else "vertex"
    model = os.environ.get("SKILLGRAPH_LLM_MODEL", DEFAULT_MODEL)
    try:
        if mode == "vertex":
            if not vertex_file:
                raise LLMError("SKILLGRAPH_LLM=vertex needs SKILLGRAPH_VERTEX_CREDENTIALS (a service-account file).")
            client: LLMClient = GeminiClient.on_vertex(
                vertex_file, model, project=os.environ.get("GOOGLE_CLOUD_PROJECT") or None,
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
        else:
            client = GeminiClient(model)
    except ImportError as exc:
        raise LLMError("The LLM layer needs the google-genai package: pip install -e '.[llm]'") from exc
    cache = os.environ.get("SKILLGRAPH_LLM_CACHE", ".cache/llm")
    return client if cache.lower() == "off" else CachedClient(client, cache)
