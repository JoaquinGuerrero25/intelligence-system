"""Utilidades compartidas por los nodos: normalización de texto y nodo especialista async."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from intelligence.graph.state import OrchestratorState


def texto(content: Any) -> str:
    """Normaliza el contenido de un mensaje (str o lista de bloques) a texto plano."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b if isinstance(b, str) else b.get("text", "") for b in content if b)
    return str(content)


def ultima_pregunta(state: OrchestratorState) -> str:
    """Último mensaje del usuario: en un hilo multi-turno es la pregunta del turno actual."""
    return next((texto(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")


def historial_previo(state: OrchestratorState, max_turnos: int = 3) -> str:
    """Resumen de los turnos anteriores del hilo (preguntas y respuestas finales), recortado."""
    previos = state["messages"][:-1] if state["messages"] else []
    pares = [
        f"{'Usuario' if isinstance(m, HumanMessage) else 'Respuesta previa'}: {texto(m.content)[:400]}"
        for m in previos
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and m.name == "sintetizador")
    ]
    return "\n".join(pares[-2 * max_turnos :])


def make_specialist_node(
    name: str,
    agent: Any,
    result_field: str,
    context: Callable[[OrchestratorState], str],
) -> Callable[[OrchestratorState], Awaitable[dict]]:
    """Envuelve un agente ReAct como nodo async del grafo.

    Evita la contaminación de contexto: el especialista NO ve `messages` del sistema; recibe solo la
    instrucción del supervisor + el contexto mínimo que `context(state)` decida compartirle.
    """

    async def node(state: OrchestratorState) -> dict:
        instruccion = state["instruction"]
        extra = context(state)
        prompt = f"Tarea: {instruccion}"
        if extra:
            prompt += f"\n\nContexto disponible:\n{extra}"

        resultado = await agent.ainvoke({"messages": [HumanMessage(prompt)]}, {"recursion_limit": 14})
        mensajes = resultado["messages"]
        salida = next(
            (texto(m.content) for m in reversed(mensajes) if isinstance(m, AIMessage) and m.content), ""
        )
        herramientas = [m.name for m in mensajes if isinstance(m, ToolMessage) and m.name]

        return {
            "messages": [AIMessage(content=salida, name=name)],
            result_field: salida,
            "contributions": [
                {
                    "agent": name,
                    "step": state.get("step_count", 0),
                    "instruction": instruccion,
                    "output": salida,
                    "tools_used": herramientas,
                }
            ],
        }

    return node
