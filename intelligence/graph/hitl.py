"""Human-in-the-loop: nodo de aprobación (interrupt) + nodo que ejecuta la acción tras aprobarse.

Una tarea es "crítica" si pide una acción con efectos secundarios (enviar, reembolsar, eliminar...) o si
el cliente fuerza la revisión. El grafo se PAUSA con `interrupt()` —el estado queda en el checkpointer—
y no avanza hasta que llegue `Command(resume=...)` desde el endpoint de aprobación.
"""

import re
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.types import interrupt

from intelligence.graph.common import ultima_pregunta
from intelligence.graph.state import EJECUTAR_ACCION, SINTETIZADOR, OrchestratorState
from intelligence.rag.embeddings import strip_accents

# Raíces de verbos con efectos secundarios (comparadas sin tildes, en minúsculas).
ACCIONES_CRITICAS = (
    "envia", "enviar", "mand", "emit", "reembols", "elimin", "borr", "cancel",
    "pagar", "transfer", "publica", "ejecut", "despleg", "comprar", "notifica",
)  # fmt: skip


def clasificar_criticidad(consulta: str, forzada: bool = False) -> str:
    """Devuelve el motivo por el que la tarea es crítica, o "" si no lo es."""
    if forzada:
        return "El cliente solicitó revisión humana explícita."
    q = strip_accents(consulta.lower())
    hallados = sorted({a for a in ACCIONES_CRITICAS if re.search(rf"\b{re.escape(a)}", q)})
    if hallados:
        return f"La tarea implica una acción con efectos secundarios (detectado: {', '.join(hallados)})."
    return ""


async def aprobacion_humana(state: OrchestratorState) -> dict:
    """Nodo HITL. Si la tarea es crítica, pausa el grafo hasta recibir la decisión humana.

    Al reanudar, el nodo se re-ejecuta desde el inicio: por eso antes de `interrupt()` no hay efectos
    secundarios (la acción real vive en `ejecutar_accion`, posterior a la aprobación).
    """
    pregunta = ultima_pregunta(state)
    motivo = clasificar_criticidad(pregunta, state.get("force_approval", False))
    if not motivo:
        return {"requires_approval": False}

    decision = interrupt(
        {
            "tipo": "aprobacion_requerida",
            "motivo": motivo,
            "pregunta": pregunta,
            "resumen_analisis": state.get("analysis_result", "")[:1200],
        }
    )
    aprobada = bool(decision.get("approved"))
    registro = {
        "approved": aprobada,
        "reviewer": decision.get("reviewer", "anonimo"),
        "comment": decision.get("comment", ""),
    }
    if aprobada:
        msg = f"Aprobada por {registro['reviewer']}."
        return {
            "requires_approval": True,
            "approval": registro,
            "messages": [AIMessage(content=msg, name="aprobacion")],
        }
    msg = (
        f"Tarea RECHAZADA por {registro['reviewer']}. Motivo: {registro['comment'] or 'sin comentario'}. "
        "No se ejecutó ninguna acción."
    )
    return {
        "requires_approval": True,
        "approval": registro,
        "final_answer": msg,
        "task_completed": True,
        "messages": [AIMessage(content=msg, name="aprobacion")],
    }


def route_after_approval(state: OrchestratorState) -> Literal["ejecutar_accion", "sintetizador", "__end__"]:
    if not state.get("requires_approval"):
        return SINTETIZADOR
    return EJECUTAR_ACCION if state["approval"]["approved"] else END


async def ejecutar_accion(state: OrchestratorState) -> dict:
    """Efecto secundario (simulado) que solo ocurre tras la aprobación humana."""
    entrada = (
        f"Acción ejecutada tras aprobación de {state['approval']['reviewer']}: {ultima_pregunta(state)[:200]}"
    )
    return {"action_log": [entrada]}
