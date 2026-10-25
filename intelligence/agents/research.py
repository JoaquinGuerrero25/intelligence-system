"""Agente Investigador: su única fuente de verdad es el RAG híbrido (BM25 + vectorial).

Las herramientas validan entradas y salidas con Pydantic y son `async`: la búsqueda no bloquea el loop.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from intelligence.rag import KBDocument, RAGService, RetrievedChunk
from intelligence.rag.models import Categoria

SYSTEM_PROMPT = """Eres el Agente de Investigación. Tu único trabajo es RECUPERAR información.
- Usa `buscar_en_base_conocimiento` (puedes llamarla varias veces con consultas distintas) y
  `leer_documento` si necesitas el texto completo. Filtra por `categoria` cuando sepas qué buscas.
- No analices ni calcules: eso lo hace otro agente. Entrega los datos crudos y relevantes.
- Responde con una lista de hallazgos; cada dato debe citar su documento como [doc-XXX].
- Incluye los textos completos de reseñas y las cifras exactas (calificaciones, unidades, precios).
- Si no encuentras nada, dilo explícitamente. No inventes datos."""


class BuscarInput(BaseModel):
    consulta: str = Field(min_length=2, max_length=300, description="Consulta en lenguaje natural.")
    categoria: Categoria | None = Field(None, description="Restringe la búsqueda a una categoría.")
    max_resultados: int = Field(5, ge=1, le=10)


class BuscarOutput(BaseModel):
    resultados: list[RetrievedChunk]
    aviso: str = ""


class LeerInput(BaseModel):
    doc_id: str = Field(pattern=r"^doc-\d{3}$", description="Id del documento, ej. 'doc-008'.")


class LeerOutput(BaseModel):
    documento: KBDocument | None = None
    error: str = ""


def build_research_tools(rag: RAGService) -> list[BaseTool]:
    @tool("buscar_en_base_conocimiento", args_schema=BuscarInput)
    async def buscar(consulta: str, categoria: Categoria | None = None, max_resultados: int = 5) -> str:
        """Búsqueda híbrida (BM25 + vectorial) en la base de conocimiento: reseñas, ventas, políticas, docs."""
        hallados = await rag.search(consulta, k=max_resultados, categoria=categoria)
        return BuscarOutput(resultados=hallados, aviso="" if hallados else "Sin coincidencias.").model_dump_json()

    @tool("leer_documento", args_schema=LeerInput)
    async def leer(doc_id: str) -> str:
        """Devuelve un documento completo de la base de conocimiento por su id."""
        doc = rag.get_document(doc_id)
        if doc is None:
            return LeerOutput(error=f"Documento {doc_id!r} no existe.").model_dump_json()
        return LeerOutput(documento=doc).model_dump_json()

    return [buscar, leer]


def build_research_agent(llm: BaseChatModel, rag: RAGService):
    return create_react_agent(llm, build_research_tools(rag), prompt=SYSTEM_PROMPT, name="investigador")
