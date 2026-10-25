"""Vector stores asíncronos: en memoria (sin dependencias) y Pinecone (SDK async oficial)."""

import math
from typing import Protocol

from intelligence.config import Settings
from intelligence.rag.models import VectorHit, VectorRecord


class VectorStore(Protocol):
    async def count(self) -> int: ...

    async def upsert(self, records: list[VectorRecord]) -> None: ...

    async def query(
        self, vector: list[float], top_k: int, categoria: str | None = None
    ) -> list[VectorHit]: ...

    async def close(self) -> None: ...


def _coseno(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return num / den if den else 0.0


class MemoryVectorStore:
    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    async def count(self) -> int:
        return len(self._records)

    async def upsert(self, records: list[VectorRecord]) -> None:
        for r in records:
            self._records[r.id] = r

    async def query(self, vector: list[float], top_k: int, categoria: str | None = None) -> list[VectorHit]:
        candidatos = [r for r in self._records.values() if categoria in (None, r.metadata.get("categoria"))]
        hits = [VectorHit(id=r.id, score=_coseno(vector, r.values)) for r in candidatos]
        return sorted(hits, key=lambda h: -h.score)[:top_k]

    async def close(self) -> None:
        self._records.clear()


class PineconeVectorStore:
    """Pinecone con `PineconeAsyncio`: nombre de índice, namespace y región vienen del entorno."""

    def __init__(self, settings: Settings):
        self.s = settings
        self._pc = None
        self._index = None

    async def _get_index(self):
        if self._index is None:
            from pinecone import PineconeAsyncio, ServerlessSpec

            self._pc = PineconeAsyncio(api_key=self.s.pinecone_api_key)
            if not await self._pc.has_index(self.s.pinecone_index_name):
                await self._pc.create_index(
                    name=self.s.pinecone_index_name,
                    dimension=self.s.embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=self.s.pinecone_cloud, region=self.s.pinecone_region),
                )
            desc = await self._pc.describe_index(self.s.pinecone_index_name)
            self._index = self._pc.IndexAsyncio(host=desc.host)
        return self._index

    async def count(self) -> int:
        stats = await (await self._get_index()).describe_index_stats()
        ns = (stats.namespaces or {}).get(self.s.pinecone_namespace)
        return int(ns.vector_count) if ns else 0

    async def upsert(self, records: list[VectorRecord]) -> None:
        index = await self._get_index()
        vectors = [
            {
                "id": r.id,
                "values": r.values,
                "metadata": {k: v for k, v in r.metadata.items() if v is not None},
            }
            for r in records
        ]
        await index.upsert(vectors=vectors, namespace=self.s.pinecone_namespace)

    async def query(self, vector: list[float], top_k: int, categoria: str | None = None) -> list[VectorHit]:
        index = await self._get_index()
        resp = await index.query(
            vector=vector,
            top_k=top_k,
            namespace=self.s.pinecone_namespace,
            filter={"categoria": {"$eq": categoria}} if categoria else None,
            include_metadata=False,
        )
        return [VectorHit(id=m.id, score=float(m.score)) for m in resp.matches]

    async def close(self) -> None:
        if self._index is not None:
            await self._index.close()
        if self._pc is not None:
            await self._pc.close()


def build_vector_store(settings: Settings) -> VectorStore:
    return PineconeVectorStore(settings) if settings.vector_backend == "pinecone" else MemoryVectorStore()
