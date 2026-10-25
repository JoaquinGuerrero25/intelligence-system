"""Embeddings asíncronos. `local` no usa red (hashing de n-gramas): sirve para correr sin API keys."""

import hashlib
import math
import re
import unicodedata
from typing import Protocol

from intelligence.config import Settings


class Embedder(Protocol):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


_WORD = re.compile(r"[a-z0-9]+")


class LocalHashEmbedder:
    """Feature hashing de palabras + trigramas de caracteres, normalizado en L2.

    No es semántico como un modelo neuronal, pero captura similitud morfológica
    ("reseñas" ~ "reseña") y es determinista, rápido y sin dependencias externas.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

    def _features(self, text: str) -> list[str]:
        palabras = _WORD.findall(strip_accents(text.lower()))
        feats = [f"w:{p}" for p in palabras]
        for p in palabras:
            padded = f"^{p}$"
            feats += [f"c:{padded[i : i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for f in self._features(text):
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            vec[h % self.dimension] += 1.0 if (h >> 63) & 1 else -1.0
        norma = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norma for v in vec]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class LangChainEmbedder:
    """Adaptador sobre los embeddings de LangChain usando SIEMPRE sus métodos async (`aembed_*`)."""

    def __init__(self, inner):
        self.inner = inner

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.inner.aembed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        return await self.inner.aembed_query(text)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "local":
        return LocalHashEmbedder(settings.embedding_dimension)
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return LangChainEmbedder(
            OpenAIEmbeddings(
                model=settings.embedding_model or "text-embedding-3-small",
                dimensions=settings.embedding_dimension,
                api_key=settings.openai_api_key,
            )
        )
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return LangChainEmbedder(
        GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model or "models/gemini-embedding-001",
            output_dimensionality=settings.embedding_dimension,
            google_api_key=settings.google_api_key,
        )
    )
