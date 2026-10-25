"""Fábrica del LLM. Proveedor, modelo y claves salen de la configuración (variables de entorno)."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from intelligence.config import Settings

DEFAULT_MODELS = {"openai": "gpt-4o-mini", "google": "gemini-3.5-flash"}


def create_llm(settings: Settings, role: str = "generic") -> BaseChatModel:
    """Crea el chat model. Los modelos de LangChain se usan solo vía `ainvoke` (async de punta a punta).

    `role` solo lo usa el proveedor `mock` (modelo heurístico sin red por rol del agente).
    """
    if settings.llm_provider == "mock":
        from intelligence.llm.mock import HeuristicChat

        return HeuristicChat(role=role)

    modelo = settings.llm_model or DEFAULT_MODELS[settings.llm_provider]
    rpm = settings.llm_requests_per_minute
    limitador = InMemoryRateLimiter(requests_per_second=rpm / 60, max_bucket_size=1) if rpm else None

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=modelo, temperature=0, api_key=settings.openai_api_key, rate_limiter=limitador
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=modelo,
        temperature=0,
        max_retries=6,
        google_api_key=settings.google_api_key,
        rate_limiter=limitador,
    )
