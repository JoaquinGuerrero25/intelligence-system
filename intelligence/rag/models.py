"""Contratos de datos de la capa RAG."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Categoria = Literal["resenas", "ventas", "politicas", "arquitectura"]


class KBDocument(BaseModel):
    id: str = Field(pattern=r"^doc-\d{3}$")
    titulo: str
    fuente: str
    categoria: Categoria
    calificacion: float | None = Field(None, ge=0, le=5)
    texto: str = Field(min_length=1)

    @property
    def indexable(self) -> str:
        """Texto que se embebe y se indexa en BM25: el título aporta contexto al contenido."""
        return f"{self.titulo}. {self.texto}"


class VectorRecord(BaseModel):
    id: str
    values: list[float]
    metadata: dict[str, str | float | None]


class VectorHit(BaseModel):
    id: str
    score: float


class RetrievedChunk(BaseModel):
    """Resultado de una búsqueda híbrida: el documento + cómo se llegó a él."""

    id: str
    titulo: str
    fuente: str
    categoria: Categoria
    calificacion: float | None = None
    texto: str
    score: float = Field(description="Puntaje RRF ponderado (BM25 + vectorial).")
    encontrado_por: list[Literal["bm25", "vectorial"]]


def load_knowledge_base(path: Path) -> list[KBDocument]:
    return [KBDocument.model_validate(d) for d in json.loads(path.read_text(encoding="utf-8"))]
