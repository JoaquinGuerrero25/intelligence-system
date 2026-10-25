"""Estado de los jobs y cola de trabajo en Redis (cliente asíncrono: nunca bloquea el event loop)."""

import json
import time
import uuid
from typing import Any

from redis.asyncio import Redis

from intelligence.schemas import TERMINAL_STATES, JobOut, JobStatus

QUEUE_KEY = "intelligence:queue"
JOB_TTL_S = 60 * 60 * 24  # los jobs expiran a las 24 h
THREAD_LOCK_TTL_S = 60 * 60

_JSON_FIELDS = ("approval_request", "action_log")


def _key(job_id: str) -> str:
    return f"intelligence:job:{job_id}"


class JobStore:
    def __init__(self, redis: Redis):
        self.r = redis

    async def create(self, query: str, thread_id: str, force_approval: bool = False) -> str:
        job_id = uuid.uuid4().hex
        now = time.time()
        await self.r.hset(
            _key(job_id),
            mapping={
                "job_id": job_id,
                "thread_id": thread_id,
                "status": JobStatus.PENDING.value,
                "query": query,
                "force_approval": json.dumps(force_approval),
                "created_at": now,
                "updated_at": now,
            },
        )
        await self.r.expire(_key(job_id), JOB_TTL_S)
        return job_id

    async def update(self, job_id: str, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        mapping = {
            k: json.dumps(v, default=str)
            if k in _JSON_FIELDS
            else str(v.value if isinstance(v, JobStatus) else v)
            for k, v in fields.items()
        }
        await self.r.hset(_key(job_id), mapping=mapping)

    async def raw(self, job_id: str) -> dict[str, str] | None:
        return (await self.r.hgetall(_key(job_id))) or None

    async def get(self, job_id: str) -> JobOut | None:
        raw = await self.raw(job_id)
        if raw is None:
            return None
        data: dict[str, Any] = {k: v for k, v in raw.items() if k != "force_approval"}
        for campo in _JSON_FIELDS:
            if campo in data:
                data[campo] = json.loads(data[campo])
        return JobOut.model_validate(data)

    # --- exclusión por hilo: un solo job activo por thread_id -----------------------------
    async def acquire_thread(self, thread_id: str, job_id: str) -> bool:
        return bool(
            await self.r.set(f"intelligence:thread:{thread_id}", job_id, nx=True, ex=THREAD_LOCK_TTL_S)
        )

    async def release_thread(self, thread_id: str, job_id: str) -> None:
        clave = f"intelligence:thread:{thread_id}"
        if await self.r.get(clave) == job_id:
            await self.r.delete(clave)

    async def claim_approval(self, job_id: str) -> bool:
        """SET NX atómico: solo una aprobación concurrente gana."""
        return bool(await self.r.set(f"intelligence:approve-lock:{job_id}", "1", nx=True, ex=JOB_TTL_S))

    # --- cola ----------------------------------------------------------------------------
    async def enqueue(self, item: dict) -> None:
        await self.r.rpush(QUEUE_KEY, json.dumps(item))

    async def dequeue(self, timeout: int = 1) -> dict | None:
        res = await self.r.blpop([QUEUE_KEY], timeout=timeout)
        return json.loads(res[1]) if res else None

    async def finish(self, job_id: str, thread_id: str, status: JobStatus, **fields: Any) -> None:
        await self.update(job_id, status=status, **fields)
        if status in TERMINAL_STATES:
            await self.release_thread(thread_id, job_id)
