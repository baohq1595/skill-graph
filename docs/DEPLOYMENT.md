# Deployment requirements

This document covers what you need to run the system, the settings you can change, how to run it locally or in Docker, and what I would add before putting it in production.

## What you need

| Item | Details |
|---|---|
| Python | 3.11 or newer. I tested it on 3.12 and 3.13. |
| Memory | Less than 150 MB. The graph is small (123 nodes and 263 links) and lives in memory. |
| Database | None. The graph is built from the JSON file when the app starts, which takes less than a second. |
| Internet | None for the graph queries. The app only calls Google when Gemini is turned on: `generativelanguage.googleapis.com` with an API key, or `aiplatform.googleapis.com` with Vertex AI. |
| Secrets | A Gemini API key, or a Vertex AI service-account file. Both are optional. Without them, the app runs offline with the rule-based parser and template answers. |
| Web page | The API serves the explorer page itself, at `/`. The page loads two libraries (`vis-network` and `marked`) from cdn.jsdelivr.net. |

## Settings

You can set these as environment variables, or put them in a `.env` file in the project folder. Copy `.env.example` to start. The `.env` file is never committed.

| Setting | Default | What it does |
|---|---|---|
| `GEMINI_API_KEY` | (none) | Turns on Gemini through the Gemini API. You can get a free key from Google AI Studio. |
| `SKILLGRAPH_VERTEX_CREDENTIALS` | (none) | Path to a service-account file. Turns on Gemini through Vertex AI instead of an API key. |
| `GOOGLE_CLOUD_PROJECT` | the project in the service-account file | The Vertex AI project to use. |
| `GOOGLE_CLOUD_LOCATION` | `global` | The Vertex AI location to use. |
| `SKILLGRAPH_LLM` | `auto` | `auto` uses the API key if there is one, then Vertex if there is a service-account file, and otherwise runs offline. You can also force `gemini`, `vertex` or `off`. |
| `SKILLGRAPH_LLM_MODEL` | `gemini-3.8-flash` | Which Gemini model to use. |
| `SKILLGRAPH_LLM_CACHE` | `.cache/llm` | Where Gemini replies are cached. Set it to `off` to turn the cache off. |
| `SKILLGRAPH_DATA` | `data/skillsfuture_dataset.json` | Where the dataset file is. |

## Running it

### On your machine

Install it in editable mode, so it can find the dataset and the evaluation files in the project folder:

```bash
pip install -e ".[llm]"
skillgraph serve --host 0.0.0.0 --port 8000
```

### With Docker

```bash
docker build -t skillgraph .
docker run -p 8000:8000 --env-file .env skillgraph
```

You can also pass the key directly with `-e GEMINI_API_KEY=...`, or leave out `--env-file` to run offline.

For Vertex AI, don't copy the service-account file into the image. Mount it when you run the container instead, read-only:

```bash
docker run -p 8000:8000 -v /path/to/sa.json:/secrets/sa.json:ro \
  -e SKILLGRAPH_VERTEX_CREDENTIALS=/secrets/sa.json skillgraph
```

Then open http://127.0.0.1:8000 for the explorer and http://127.0.0.1:8000/docs for the API.

About the image:

- It is based on `python:3.12-slim` and is about 380 MB.
- It runs as a normal user called `app`, not as root.
- It has a health check that calls `/health`.
- Credential files and `.env` are listed in `.dockerignore`, so they can't end up in the image by accident.

I built and tested it with Docker 29 on an Apple Silicon Mac. The build takes under 30 seconds and the container reports healthy. Every endpoint works, including `/ask`, the explorer page, and the 422 error for an unclear role name. I also ran it with a wrong API key on purpose: the Gemini call failed cleanly, the rule parser answered instead, and the error showed up in the trace.

If you are on a Mac without Docker Desktop, you can use colima instead: `brew install colima docker docker-buildx && colima start`.

## Checking that it is healthy

- **`GET /health`** returns the number of nodes and links, and which model is in use (or "offline").
- **`GET /data-quality`** returns the data-quality report.
- **`POST /ask`** returns a trace with every answer: the parser used, the model, cache hits, token counts, the time for each step, and any fact-check problems. In production, you would send these traces to your logging system.

## Before production

These are out of scope for the assignment, but this is what I would add before real users:

- **Login and rate limits on `/ask`.** Every question costs Gemini tokens, so it needs protecting.
- **A shared graph for many servers.** Each server process builds its own copy of the graph. That is fine at this size. For the full framework, move to one shared Neo4j database (`skillgraph export-cypher` writes the loading script).
- **A shared reply cache.** The cache is on the local disk today. With several servers, use Redis or cloud storage instead.
- **CORS settings,** if the explorer is served from a different domain than the API.
- **Local copies of the two web libraries,** if the servers can't reach the internet.
