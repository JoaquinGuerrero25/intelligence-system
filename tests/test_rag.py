import pytest

from intelligence.rag.embeddings import LocalHashEmbedder
from intelligence.rag.models import VectorRecord
from intelligence.rag.service import RAGService
from intelligence.rag.stores import MemoryVectorStore


async def test_busqueda_hibrida_encuentra_resenas(rag):
    ids = {c.id for c in await rag.search("reseñas AuroraBook X14", k=8, categoria="resenas")}
    assert {"doc-001", "doc-002", "doc-003", "doc-004", "doc-005"} <= ids


async def test_filtro_por_categoria(rag):
    chunks = await rag.search("devoluciones y envíos", k=5, categoria="politicas")
    assert chunks and all(c.categoria == "politicas" for c in chunks)
    assert {"doc-012", "doc-013"} <= {c.id for c in chunks}


async def test_cada_resultado_indica_que_rama_lo_encontro(rag):
    chunks = await rag.search("garantía de la AuroraBook", k=3)
    assert chunks[0].id == "doc-011"
    assert chunks[0].encontrado_por and set(chunks[0].encontrado_por) <= {"bm25", "vectorial"}


async def test_consulta_sin_terminos_utiles_no_rompe(rag):
    assert isinstance(await rag.search("de la el", k=3), list)


async def test_ensure_indexed_es_idempotente(rag):
    assert await rag.ensure_indexed() == len(rag.documents)
    assert await rag.store.count() == len(rag.documents)


async def test_embedder_local_es_determinista_y_normalizado():
    e = LocalHashEmbedder(64)
    a, b = await e.embed_query("reseñas de la notebook"), await e.embed_query("reseñas de la notebook")
    assert a == b
    assert abs(sum(x * x for x in a) - 1) < 1e-9

    def dot(x, y):
        return sum(p * q for p, q in zip(x, y, strict=True))

    # similitud morfológica: "reseña" está más cerca de "reseñas" que de "garantía"
    r, rs, g = [await e.embed_query(t) for t in ("reseña", "reseñas", "garantía")]
    assert dot(r, rs) > dot(r, g)


async def test_memory_store_respeta_filtro():
    s = MemoryVectorStore()
    await s.upsert(
        [
            VectorRecord(id="a", values=[1, 0], metadata={"categoria": "x"}),
            VectorRecord(id="b", values=[1, 0], metadata={"categoria": "y"}),
        ]
    )
    assert [h.id for h in await s.query([1, 0], 5, "y")] == ["b"]


def test_base_vacia_es_error():
    with pytest.raises(ValueError):
        RAGService([], LocalHashEmbedder(8), MemoryVectorStore())
