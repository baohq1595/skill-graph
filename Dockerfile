FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY eval ./eval
RUN pip install --no-cache-dir -e ".[llm]" \
 && useradd --create-home app && chown -R app /app
USER app

ENV SKILLGRAPH_DATA=/app/data/skillsfuture_dataset.json \
    SKILLGRAPH_LLM_CACHE=/app/.cache/llm \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["skillgraph", "serve", "--host", "0.0.0.0", "--port", "8000"]
