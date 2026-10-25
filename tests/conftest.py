"""Fixtures compartidas. Los tests no usan red, ni API keys, ni Redis real, ni Phoenix."""

import pytest

from intelligence.config import Settings
from intelligence.rag import build_rag_service


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, llm_provider="mock", phoenix_enabled=False, worker_concurrency=3)


@pytest.fixture
async def rag(settings):
    servicio = build_rag_service(settings)
    await servicio.ensure_indexed()
    yield servicio
    await servicio.close()
