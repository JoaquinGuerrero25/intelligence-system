"""Demuestra que las conversaciones sobreviven al reinicio del servidor (checkpointer en Redis).

python scripts/persistence_check.py create      # crea un hilo con una pregunta
docker compose restart api                      # reinicia SOLO la API
python scripts/persistence_check.py verify      # el hilo sigue ahí y se puede continuar
"""

import argparse
import asyncio
import sys

import httpx

HILO = "demo-persistencia"


async def esperar(cli: httpx.AsyncClient, job_id: str) -> dict:
    while True:
        job = (await cli.get(f"/tasks/{job_id}")).json()
        if job["status"] in {"DONE", "FAILED", "REJECTED"}:
            return job
        await asyncio.sleep(0.4)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=["create", "verify"])
    ap.add_argument("--url", default="http://localhost:8000")
    a = ap.parse_args()
    async with httpx.AsyncClient(base_url=a.url, timeout=60) as cli:
        if a.fase == "create":
            q = "Investiga las reseñas de la AuroraBook X14 y calcula la calificación promedio."
            r = await cli.post("/tasks", json={"query": q, "thread_id": HILO})
            if r.status_code == 409:
                print(f"El hilo {HILO!r} ya existe; usá 'verify'.")
                return 0
            job = await esperar(cli, r.json()["job_id"])
            print(f"Turno 1: {job['status']} -> {job['result'][:150]}")
            print("Ahora reiniciá la API:  docker compose restart api   y corré:  verify")
            return 0

        h = await cli.get(f"/threads/{HILO}/messages")
        if h.status_code != 200:
            print("FAIL: el hilo no existe tras el reinicio.")
            return 1
        print(f"Tras el reinicio el hilo conserva {len(h.json()['messages'])} mensajes.")
        r = await cli.post(
            "/tasks", json={"query": "Ahora lo mismo para las reseñas de la NovaPad Pro.", "thread_id": HILO}
        )
        job = await esperar(cli, r.json()["job_id"])
        n = sum(
            1 for m in (await cli.get(f"/threads/{HILO}/messages")).json()["messages"] if m["role"] == "user"
        )
        print(f"Turno 2: {job['status']} -> {job['result'][:150]}")
        print(
            f"OK: el hilo ahora tiene {n} preguntas de usuario."
            if n == 2
            else "FAIL: se perdió el historial."
        )
        return 0 if n == 2 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
