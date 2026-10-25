"""Contratos de la API: todo lo que entra y sale por HTTP pasa por estos modelos Pydantic."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DONE = "DONE"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


TERMINAL_STATES = {JobStatus.DONE, JobStatus.REJECTED, JobStatus.FAILED}


class TaskIn(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"query": "Investiga las reseñas de la AuroraBook X14 y calcula la calificación promedio."},
                {
                    "query": "Reseñas de la AuroraBook X14 y envía el resumen por correo",
                    "thread_id": "ana-001",
                },
            ]
        }
    )

    query: str = Field(min_length=3, max_length=2000, description="Tarea o pregunta del usuario.")
    thread_id: str | None = Field(
        None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.-]+$",
        description="Hilo de conversación. Mismo thread_id = misma memoria (persistida en Redis). "
        "Si se omite, se crea un hilo nuevo.",
    )
    require_approval: bool = Field(False, description="Fuerza la pausa HITL aunque la tarea no sea crítica.")


class TaskAccepted(BaseModel):
    job_id: str
    thread_id: str
    status: JobStatus = JobStatus.PENDING


class ApprovalIn(BaseModel):
    approved: bool
    reviewer: str = Field("anonimo", min_length=1, max_length=64)
    comment: str = Field("", max_length=500)


class ApprovalRequest(BaseModel):
    tipo: str
    motivo: str
    pregunta: str
    resumen_analisis: str = ""


class JobOut(BaseModel):
    job_id: str
    thread_id: str
    status: JobStatus
    query: str
    result: str | None = None
    error: str | None = None
    approval_request: ApprovalRequest | None = None
    action_log: list[str] = Field(default_factory=list)
    created_at: float
    updated_at: float


class HistoryMessage(BaseModel):
    role: str = Field(description="user | assistant | supervisor | investigador | analista | ...")
    content: str


class ThreadHistory(BaseModel):
    thread_id: str
    messages: list[HistoryMessage]


class HealthOut(BaseModel):
    status: str
    redis: bool
    rag_documents: int
    llm_provider: str
    tracing: bool


def to_jsonable(value: Any) -> Any:
    """Convierte valores de LangGraph (p. ej. Interrupt) a algo serializable en JSON."""
    return value.model_dump() if isinstance(value, BaseModel) else value
