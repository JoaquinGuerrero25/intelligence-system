"""Estado compartido del grafo.

Hereda de MessagesState (historial de la conversación, persistido por el checkpointer bajo un
`thread_id`) y agrega campos estructurados por turno. Cada nuevo mensaje del usuario en el mismo
hilo reinicia los campos de control con `nuevo_turno()`, pero conserva `messages`.
"""

from typing import Annotated, Any, TypedDict

from langgraph.graph import MessagesState

# Nombres de los nodos (única fuente de verdad para el routing).
SUPERVISOR = "supervisor"
INVESTIGADOR = "investigador"
ANALISTA = "analista"
VALIDACION = "validacion"
SINTETIZADOR = "sintetizador"
APROBACION = "aprobacion_humana"
EJECUTAR_ACCION = "ejecutar_accion"
FINISH = "FINISH"

MAX_STEPS = 6  # tope de decisiones del supervisor: corta el "Supervisor Infinito"
MAX_REVISIONS = 1  # cuántas veces se puede re-delegar a un mismo especialista


def acumular(actual: list | None, nuevo: list | None) -> list:
    """Reducer append-only. Recibir `None` reinicia la lista (inicio de un turno nuevo)."""
    if nuevo is None:
        return []
    return (actual or []) + nuevo


class Contribution(TypedDict):
    """Registro de lo que aportó un agente: quién, en qué paso, con qué herramientas."""

    agent: str
    step: int
    instruction: str
    output: str
    tools_used: list[str]


class OrchestratorState(MessagesState):
    # --- control de flujo ---
    next_agent: str
    instruction: str  # tarea concreta para el especialista elegido (lo único que recibe)
    step_count: int
    revisions: dict[str, int]
    task_completed: bool

    # --- resultados por agente (trazabilidad) ---
    research_notes: str
    analysis_result: str
    contributions: Annotated[list[Contribution], acumular]

    # --- validación y cierre ---
    validation_feedback: str
    validation_attempts: int
    final_answer: str

    # --- human-in-the-loop ---
    force_approval: bool
    requires_approval: bool
    approval: dict
    action_log: Annotated[list[str], acumular]


def nuevo_turno(query: str, force_approval: bool = False) -> dict[str, Any]:
    """Entrada del grafo para un turno: mensaje del usuario + reinicio de los campos de control."""
    return {
        "messages": [("user", query)],
        "next_agent": "",
        "instruction": "",
        "step_count": 0,
        "revisions": {},
        "task_completed": False,
        "research_notes": "",
        "analysis_result": "",
        "contributions": None,
        "validation_feedback": "",
        "validation_attempts": 0,
        "final_answer": "",
        "force_approval": force_approval,
        "requires_approval": False,
        "approval": {},
        "action_log": None,
    }
