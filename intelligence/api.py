"""API FastAPI: POST /tasks encola y responde al instante; GET /tasks/{id} consulta el estado.

Uso:  uvicorn intelligence.api:app --port 8000
"""

import asyncio
import inspect
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from redis.asyncio import Redis

from intelligence.config import Settings, get_settings
from intelligence.graph import create_graph
from intelligence.logging_setup import setup_logging
from intelligence.observability import init_observability
from intelligence.rag import build_rag_service
from intelligence.schemas import (
    ApprovalIn,
    HealthOut,
    HistoryMessage,
    JobOut,
    JobStatus,
    TaskAccepted,
    TaskIn,
    ThreadHistory,
)
from intelligence.store import JobStore
from intelligence.worker import worker_loop

log = logging.getLogger(__name__)

DESCRIPTION = """
Sistema de IA con **RAG híbrido** + **agentes con Supervisor** (LangGraph) + trazabilidad en **Arize Phoenix**.

* `POST /tasks` encola una tarea y devuelve un `job_id` al instante.
* `GET /tasks/{job_id}` consulta el estado (`PENDING → RUNNING → DONE`).
* Las tareas con efectos secundarios se pausan en `WAITING_APPROVAL` hasta `POST /tasks/{job_id}/approve`.
* `GET /threads/{thread_id}/messages` devuelve la conversación persistida por el checkpointer (Redis).
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = settings
    setup_logging(settings.log_level)
    app.state.tracing = init_observability(settings)

    redis: Redis = getattr(app.state, "redis", None) or Redis.from_url(
        settings.redis_url, decode_responses=True
    )
    app.state.redis = redis
    app.state.jobs = JobStore(redis)

    rag = build_rag_service(settings)
    app.state.rag_documents = await rag.ensure_indexed()

    # Checkpointer persistente (Redis Stack: RedisJSON + RediSearch). Los tests inyectan uno en memoria.
    checkpointer = getattr(app.state, "checkpointer", None)
    cm = None
    if checkpointer is None:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        cm = AsyncRedisSaver.from_conn_string(settings.redis_url)
        checkpointer = await cm.__aenter__()
        await checkpointer.asetup()

    grafo = create_graph(settings, rag, checkpointer)
    app.state.graph = grafo
    workers = [
        asyncio.create_task(worker_loop(f"w{i}", app.state.jobs, grafo, settings.job_timeout_s))
        for i in range(settings.worker_concurrency)
    ]
    log.info(
        "API lista: llm=%s vector=%s workers=%d", settings.llm_provider, settings.vector_backend, len(workers)
    )
    try:
        yield
    finally:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        if cm:
            await cm.__aexit__(None, None, None)
        await rag.close()
        await redis.aclose()


# Las versiones recientes de FastAPI emiten spans HTTP propios; los apagamos para que en Phoenix solo
# queden las trazas del sistema (una por job) y no el ruido del polling de GET /tasks/{id}.
_SIN_SPANS_HTTP = (
    {"telemetry": {"tracing": False, "metrics": False}}
    if "telemetry" in inspect.signature(FastAPI.__init__).parameters
    else {}
)

app = FastAPI(
    title="The Intelligence System",
    version="1.0.0",
    description=DESCRIPTION,
    lifespan=lifespan,
    **_SIN_SPANS_HTTP,
)


def _jobs(request: Request) -> JobStore:
    return request.app.state.jobs


@app.post("/tasks", status_code=202, response_model=TaskAccepted, tags=["tareas"])
async def crear_tarea(body: TaskIn, request: Request) -> TaskAccepted:
    """Encola la tarea y devuelve el job_id sin esperar al agente."""
    jobs = _jobs(request)
    thread_id = body.thread_id or uuid.uuid4().hex[:16]
    job_id = await jobs.create(body.query, thread_id, body.require_approval)
    if not await jobs.acquire_thread(thread_id, job_id):
        await jobs.update(job_id, status=JobStatus.FAILED, error="Hilo ocupado")
        raise HTTPException(409, f"El hilo {thread_id!r} ya tiene un job activo; esperá a que termine.")
    await jobs.enqueue({"job_id": job_id})
    return TaskAccepted(job_id=job_id, thread_id=thread_id)


@app.get("/tasks/{job_id}", response_model=JobOut, tags=["tareas"])
async def estado_tarea(job_id: str, request: Request) -> JobOut:
    job = await _jobs(request).get(job_id)
    if job is None:
        raise HTTPException(404, "Job inexistente")
    return job


@app.post("/tasks/{job_id}/approve", status_code=202, response_model=TaskAccepted, tags=["tareas"])
async def aprobar(job_id: str, body: ApprovalIn, request: Request) -> TaskAccepted:
    """Señal externa que reanuda un job pausado en WAITING_APPROVAL (aprobándolo o rechazándolo)."""
    jobs = _jobs(request)
    job = await jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job inexistente")
    if job.status != JobStatus.WAITING_APPROVAL:
        raise HTTPException(409, f"El job está en estado {job.status}; solo se aprueba WAITING_APPROVAL.")
    if not await jobs.claim_approval(job_id):  # evita doble aprobación concurrente
        raise HTTPException(409, "El job ya fue resuelto.")
    await jobs.update(job_id, status=JobStatus.PENDING)
    await jobs.enqueue({"job_id": job_id, "resume": body.model_dump()})
    return TaskAccepted(job_id=job_id, thread_id=job.thread_id)


@app.get("/threads/{thread_id}/messages", response_model=ThreadHistory, tags=["conversaciones"])
async def historial(thread_id: str, request: Request) -> ThreadHistory:
    """Conversación del hilo, leída del checkpointer (sobrevive a reinicios del servidor)."""
    snapshot = await request.app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.values:
        raise HTTPException(404, "Hilo inexistente")
    mensajes = [
        HistoryMessage(
            role=m.name or ("user" if m.type == "human" else "assistant"),
            content=m.content if isinstance(m.content, str) else str(m.content),
        )
        for m in snapshot.values.get("messages", [])
    ]
    return ThreadHistory(thread_id=thread_id, messages=mensajes)


@app.get("/health", response_model=HealthOut, tags=["operación"])
async def health(request: Request) -> HealthOut:
    redis_ok = bool(await request.app.state.redis.ping())
    s: Settings = request.app.state.settings
    return HealthOut(
        status="ok" if redis_ok else "degraded",
        redis=redis_ok,
        rag_documents=request.app.state.rag_documents,
        llm_provider=s.llm_provider,
        tracing=request.app.state.tracing,
    )
