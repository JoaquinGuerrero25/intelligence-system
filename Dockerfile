FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KNOWLEDGE_BASE_PATH=/srv/data/knowledge_base.json

WORKDIR /srv
COPY pyproject.toml README.md ./
COPY intelligence ./intelligence
RUN pip install .
COPY data ./data

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --retries=12 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "intelligence.api:app", "--host", "0.0.0.0", "--port", "8000"]
