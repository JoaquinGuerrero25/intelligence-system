"""Worker: consume la cola de Redis y ejecuta el grafo en segundo plano.

Ciclo de vida del job: PENDING -> RUNNING -> (WAITING_APPROVAL -> PENDING -> RUNNING)* -> DONE | REJECTED | FAILED.
Cualquier excepción se captura y deja el job en FAILED con el error, para que el cliente nunca quede
esperando en un polling infinito.
"""

import asyncio
import logging
import traceback

from langgraph.types import Command

from intelligence.graph import nuevo_turno
from intelligence.observability import tracer
from intelligence.schemas import ApprovalRequest, JobStatus, to_jsonable
from intelligence.store import JobStore

log = logging.getLogger(__name__)


async def procesar(job_id: str, resume: dict | None, jobs: JobStore, grafo, timeout_s: float) -> None:
    raw = await jobs.raw(job_id)
    if raw is None:
        return
    thread_id = raw["thread_id"]
    await jobs.update(job_id, status=JobStatus.RUNNING)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}

    # Span raíz: agrupa la traza del job y la asocia a una "sesión" de Phoenix (= el hilo de conversación).
    with tracer().start_as_current_span("job") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("session.id", thread_id)
        span.set_attribute("job.id", job_id)
        span.set_attribute("job.resume", resume is not None)
        span.set_attribute("input.value", raw["query"])
        try:
            if resume is not None:
                entrada = Command(resume=resume)
            else:
                entrada = nuevo_turno(raw["query"], force_approval=raw.get("force_approval") == "true")
            resultado = await asyncio.wait_for(grafo.ainvoke(entrada, config), timeout=timeout_s)

            if interrupciones := resultado.get("__interrupt__"):
                solicitud = ApprovalRequest.model_validate(to_jsonable(interrupciones[0].value))
                await jobs.update(
                    job_id, status=JobStatus.WAITING_APPROVAL, approval_request=solicitud.model_dump()
                )
                span.set_attribute("job.status", "WAITING_APPROVAL")
                return

            aprobado = resultado.get("approval", {}).get("approved", True)
            estado = JobStatus.DONE if aprobado else JobStatus.REJECTED
            span.set_attribute("output.value", resultado.get("final_answer", ""))
            span.set_attribute("job.status", estado.value)
            await jobs.finish(
                job_id, thread_id, estado,
                result=resultado.get("final_answer", ""), action_log=resultado.get("action_log", []),
            )  # fmt: skip
        except Exception as e:  # noqa: BLE001 - toda falla debe reflejarse en Redis
            log.error("Job %s falló: %s", job_id, traceback.format_exc())
            span.record_exception(e)
            span.set_attribute("job.status", "FAILED")
            msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            try:
                await jobs.finish(job_id, thread_id, JobStatus.FAILED, error=msg)
            except Exception:  # noqa: BLE001
                log.exception("No se pudo marcar FAILED el job %s", job_id)


async def worker_loop(nombre: str, jobs: JobStore, grafo, timeout_s: float) -> None:
    log.info("worker %s listo", nombre)
    while True:
        try:
            item = await jobs.dequeue(timeout=1)
            if item:
                await procesar(item["job_id"], item.get("resume"), jobs, grafo, timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - un error de Redis no mata al worker
            log.exception("error en el loop del worker %s", nombre)
            await asyncio.sleep(1)
