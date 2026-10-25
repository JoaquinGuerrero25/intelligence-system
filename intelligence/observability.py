"""Observabilidad con Arize Phoenix (OpenInference).

`init_observability()` registra un exportador OTLP hacia Phoenix e instrumenta LangChain/LangGraph:
cada nodo del grafo, cada llamada al LLM (tokens y modelo, de donde Phoenix calcula el costo) y
cada herramienta queda como un span dentro de la traza del job. Los spans propios (`job`,
`rag.search`) se crean con `tracer()`.
"""

import logging

from opentelemetry import trace

from intelligence.config import Settings

log = logging.getLogger(__name__)
_iniciado = False


def init_observability(settings: Settings) -> bool:
    global _iniciado
    if _iniciado or not settings.phoenix_enabled:
        return _iniciado
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    endpoint = settings.phoenix_collector_endpoint.rstrip("/")
    provider = register(
        project_name=settings.phoenix_project_name,
        endpoint=f"{endpoint}/v1/traces",
        batch=True,
        set_global_tracer_provider=True,
    )
    LangChainInstrumentor().instrument(tracer_provider=provider)
    _iniciado = True
    log.info("Phoenix activo -> %s (proyecto %s)", endpoint, settings.phoenix_project_name)
    return True


def tracer() -> trace.Tracer:
    return trace.get_tracer("intelligence")
