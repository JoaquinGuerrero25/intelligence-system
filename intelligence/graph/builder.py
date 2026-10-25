"""Ensamblado del grafo: Supervisor -> {Investigador, Analista} -> Validación -> [Aprobación humana] -> Síntesis."""

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from intelligence.agents import build_analyst_agent, build_research_agent
from intelligence.config import Settings
from intelligence.graph.common import make_specialist_node
from intelligence.graph.hitl import aprobacion_humana, ejecutar_accion, route_after_approval
from intelligence.graph.state import (
    ANALISTA,
    APROBACION,
    EJECUTAR_ACCION,
    INVESTIGADOR,
    SINTETIZADOR,
    SUPERVISOR,
    VALIDACION,
    OrchestratorState,
)
from intelligence.graph.supervisor import make_supervisor_node, route_after_supervisor
from intelligence.graph.validation import make_synthesizer_node, route_after_validation, validacion
from intelligence.llm import create_llm
from intelligence.rag import RAGService


def _contexto(state: OrchestratorState) -> str:
    # Los especialistas solo reciben las notas del investigador, no el historial del sistema.
    return state.get("research_notes", "")


def build_graph(
    llm: BaseChatModel,
    rag: RAGService,
    research_llm: BaseChatModel | None = None,
    analyst_llm: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Compila el grafo. `checkpointer` (AsyncRedisSaver en producción) persiste cada paso por thread_id."""
    g = StateGraph(OrchestratorState)

    g.add_node(SUPERVISOR, make_supervisor_node(llm))
    g.add_node(
        INVESTIGADOR,
        make_specialist_node(
            INVESTIGADOR, build_research_agent(research_llm or llm, rag), "research_notes", _contexto
        ),
    )
    g.add_node(
        ANALISTA,
        make_specialist_node(ANALISTA, build_analyst_agent(analyst_llm or llm), "analysis_result", _contexto),
    )
    g.add_node(VALIDACION, validacion)
    g.add_node(APROBACION, aprobacion_humana)
    g.add_node(EJECUTAR_ACCION, ejecutar_accion)
    g.add_node(SINTETIZADOR, make_synthesizer_node(llm))

    g.add_edge(START, SUPERVISOR)
    g.add_conditional_edges(SUPERVISOR, route_after_supervisor)
    g.add_edge(INVESTIGADOR, SUPERVISOR)
    g.add_edge(ANALISTA, SUPERVISOR)
    # Validación aprobada (o sin presupuesto) -> pasa por la compuerta HITL antes de sintetizar.
    g.add_conditional_edges(
        VALIDACION, route_after_validation, {SUPERVISOR: SUPERVISOR, SINTETIZADOR: APROBACION}
    )
    g.add_conditional_edges(
        APROBACION,
        route_after_approval,
        {EJECUTAR_ACCION: EJECUTAR_ACCION, SINTETIZADOR: SINTETIZADOR, END: END},
    )
    g.add_edge(EJECUTAR_ACCION, SINTETIZADOR)
    g.add_edge(SINTETIZADOR, END)

    return g.compile(checkpointer=checkpointer)


def create_graph(settings: Settings, rag: RAGService, checkpointer: BaseCheckpointSaver | None = None):
    """Grafo listo para ejecutar con los LLMs configurados (un modelo por rol; en mock, heurísticos sin red)."""
    return build_graph(
        llm=create_llm(settings, "supervisor"),
        rag=rag,
        research_llm=create_llm(settings, "investigador"),
        analyst_llm=create_llm(settings, "analista"),
        checkpointer=checkpointer,
    )
