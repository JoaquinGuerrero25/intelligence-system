"""Recuperación híbrida: BM25 (léxico) + vectorial (semántico) fusionados con Reciprocal Rank Fusion.

Las dos ramas corren en paralelo (`asyncio.gather`): BM25 es CPU y va a un thread, la rama
vectorial es I/O (embedding de la consulta + query al vector store) y es nativamente async.
"""

import asyncio
import logging
import re

from rank_bm25 import BM25Okapi

from intelligence.config import Settings
from intelligence.observability import tracer
from intelligence.rag.embeddings import Embedder, build_embedder, strip_accents
from intelligence.rag.models import KBDocument, RetrievedChunk, VectorRecord, load_knowledge_base
from intelligence.rag.stores import VectorStore, build_vector_store

log = logging.getLogger(__name__)

RRF_K = 60
STOPWORDS = frozenset(
    """a al algo como con cual cuales cuando de del donde el ella en entre es esa ese esta este esto hay la las
    lo los mas me mi muy no o para pero por porque que se si sin sobre su sus te tiene un una uno unos y ya yo
    son ser hace hacer puedo puede cuanto cuantos calcula investiga busca analiza""".split()
)
_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(strip_accents(text.lower())) if len(t) > 1 and t not in STOPWORDS]


class RAGService:
    def __init__(
        self,
        documents: list[KBDocument],
        embedder: Embedder,
        store: VectorStore,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ):
        if not documents:
            raise ValueError("La base de conocimiento está vacía.")
        self.documents = {d.id: d for d in documents}
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.weights = {"bm25": bm25_weight, "vectorial": vector_weight}
        self._bm25 = BM25Okapi([tokenize(d.indexable) for d in documents])
        self._orden = [d.id for d in documents]

    async def ensure_indexed(self) -> int:
        """Indexa la base si el store está incompleto. Devuelve cuántos documentos hay indexados."""
        total = len(self.documents)
        if await self.store.count() >= total:
            return total
        docs = list(self.documents.values())
        vectores = await self.embedder.embed_documents([d.indexable for d in docs])
        await self.store.upsert(
            [
                VectorRecord(id=d.id, values=v, metadata={"categoria": d.categoria, "titulo": d.titulo})
                for d, v in zip(docs, vectores, strict=True)
            ]
        )
        log.info("RAG: %d documentos indexados", total)
        return total

    def _buscar_bm25(self, query: str, k: int, categoria: str | None) -> list[str]:
        tokens = tokenize(query)
        if not tokens:
            return []
        puntajes = self._bm25.get_scores(tokens)
        ranking = sorted(zip(self._orden, puntajes, strict=True), key=lambda x: -x[1])
        return [i for i, p in ranking if p > 0 and categoria in (None, self.documents[i].categoria)][:k]

    async def _buscar_vectorial(self, query: str, k: int, categoria: str | None) -> list[str]:
        vector = await self.embedder.embed_query(query)
        hits = await self.store.query(vector, k, categoria)
        return [h.id for h in hits if h.id in self.documents]

    async def search(self, query: str, k: int | None = None, categoria: str | None = None) -> list[RetrievedChunk]:
        k = k or self.top_k
        candidatos = max(k * 2, 10)
        with tracer().start_as_current_span("rag.search") as span:
            span.set_attribute("openinference.span.kind", "RETRIEVER")
            span.set_attribute("input.value", query)
            lexico, semantico = await asyncio.gather(
                asyncio.to_thread(self._buscar_bm25, query, candidatos, categoria),
                self._buscar_vectorial(query, candidatos, categoria),
            )
            resultado = self._fusionar({"bm25": lexico, "vectorial": semantico})[:k]
            for i, r in enumerate(resultado):
                span.set_attribute(f"retrieval.documents.{i}.document.id", r.id)
                span.set_attribute(f"retrieval.documents.{i}.document.content", r.texto)
                span.set_attribute(f"retrieval.documents.{i}.document.score", r.score)
            return resultado

    def _fusionar(self, rankings: dict[str, list[str]]) -> list[RetrievedChunk]:
        puntaje: dict[str, float] = {}
        origen: dict[str, list[str]] = {}
        for rama, ids in rankings.items():
            for rango, doc_id in enumerate(ids, start=1):
                puntaje[doc_id] = puntaje.get(doc_id, 0.0) + self.weights[rama] / (RRF_K + rango)
                origen.setdefault(doc_id, []).append(rama)
        return [
            RetrievedChunk(**self.documents[i].model_dump(), score=round(p, 5), encontrado_por=origen[i])
            for i, p in sorted(puntaje.items(), key=lambda x: -x[1])
        ]

    def get_document(self, doc_id: str) -> KBDocument | None:
        return self.documents.get(doc_id)

    async def close(self) -> None:
        await self.store.close()


def build_rag_service(settings: Settings) -> RAGService:
    return RAGService(
        documents=load_knowledge_base(settings.knowledge_base_path),
        embedder=build_embedder(settings),
        store=build_vector_store(settings),
        top_k=settings.rag_top_k,
        bm25_weight=settings.rag_bm25_weight,
        vector_weight=settings.rag_vector_weight,
    )
