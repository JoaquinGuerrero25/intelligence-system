"""Corre las pruebas de punta a punta contra la API en ejecución y deja una traza por cada una en Phoenix.

    python scripts/run_scenarios.py                       # API en http://localhost:8000
    python scripts/run_scenarios.py --url http://host:8000

Escenarios: 1) reseñas -> sentimiento + promedio, 2) ventas -> cálculo de ingresos, 3) producto distinto,
4) tarea crítica con HITL aprobada, 5) tarea crítica RECHAZADA, 6) conversación multi-turno en un mismo hilo.
Todas las peticiones corren en paralelo (httpx async). Sale con código 1 si algún escenario no da lo esperado.
"""

import argparse
import asyncio
import sys
import time
import uuid

import httpx

TERMINALES = {"DONE", "FAILED", "REJECTED"}

ESCENARIOS = [
    {
        "nombre": "1. reseñas: sentimiento + promedio",
        "query": "Investiga las reseñas de la AuroraBook X14, analiza el sentimiento de los clientes y calcula la calificación promedio.",
        "esperado": "DONE",
    },
    {
        "nombre": "2. ventas: unidades e ingreso",
        "query": "Busca las ventas de la AuroraBook X14 y calcula el total de unidades vendidas y el ingreso estimado.",
        "esperado": "DONE",
    },
    {
        "nombre": "3. otro producto: NovaPad Pro",
        "query": "Investiga las reseñas de la NovaPad Pro y calcula su calificación media y el sentimiento.",
        "esperado": "DONE",
    },
    {
        "nombre": "4. tarea crítica (HITL aprobada)",
        "query": "Investiga las reseñas de la AuroraBook X14, calcula la calificación promedio y envía un resumen por correo al equipo de producto.",
        "aprobar": True,
        "esperado": "DONE",
    },
    {
        "nombre": "5. tarea crítica (HITL rechazada)",
        "query": "Busca las ventas de la NovaPad Pro, calcula las unidades totales y publica el resultado en el canal de ventas.",
        "aprobar": False,
        "esperado": "REJECTED",
    },
]


async def esperar(cli: httpx.AsyncClient, job_id: str, aprobar: bool | None) -> dict:
    resuelto = False
    while True:
        job = (await cli.get(f"/tasks/{job_id}")).json()
        if job["status"] == "WAITING_APPROVAL" and aprobar is not None and not resuelto:
            body = {"approved": aprobar, "reviewer": "scenarios", "comment": "" if aprobar else "prueba de rechazo"}
            (await cli.post(f"/tasks/{job_id}/approve", json=body)).raise_for_status()
            resuelto = True
        if job["status"] in TERMINALES:
            return job
        await asyncio.sleep(0.4)


async def escenario(cli: httpx.AsyncClient, e: dict) -> dict:
    t0 = time.perf_counter()
    r = await cli.post("/tasks", json={"query": e["query"]})
    r.raise_for_status()
    job = await esperar(cli, r.json()["job_id"], e.get("aprobar"))
    return {**e, "job": job, "segundos": time.perf_counter() - t0, "ok": job["status"] == e["esperado"]}


async def multi_turno(cli: httpx.AsyncClient) -> dict:
    """Dos preguntas en el mismo hilo: la segunda se resuelve con la memoria de la primera."""
    hilo = f"demo-{uuid.uuid4().hex[:6]}"
    t0 = time.perf_counter()
    pasos = ["Investiga las reseñas de la AuroraBook X14 y calcula la calificación promedio.", "Ahora lo mismo para las reseñas de la NovaPad Pro."]
    job = {}
    for q in pasos:
        r = await cli.post("/tasks", json={"query": q, "thread_id": hilo})
        r.raise_for_status()
        job = await esperar(cli, r.json()["job_id"], None)
    hist = (await cli.get(f"/threads/{hilo}/messages")).json()
    usuarios = sum(1 for m in hist["messages"] if m["role"] == "user")
    return {
        "nombre": "6. multi-turno en un hilo",
        "query": " | ".join(pasos),
        "job": job,
        "segundos": time.perf_counter() - t0,
        "ok": job.get("status") == "DONE" and usuarios == 2,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    a = ap.parse_args()
    async with httpx.AsyncClient(base_url=a.url, timeout=60) as cli:
        salud = (await cli.get("/health")).json()
        print(f"API: {salud}\n")
        t0 = time.perf_counter()
        resultados = await asyncio.gather(*(escenario(cli, e) for e in ESCENARIOS), multi_turno(cli))
        total = time.perf_counter() - t0

    for r in resultados:
        marca = "OK  " if r["ok"] else "FAIL"
        print(f"[{marca}] {r['nombre']:<38} estado={r['job']['status']:<9} {r['segundos']:5.1f}s")
        print(f"        respuesta: {(r['job'].get('result') or r['job'].get('error') or '')[:170]}")
    ok = sum(r["ok"] for r in resultados)
    print(f"\n{ok}/{len(resultados)} escenarios OK en {total:.1f}s (en paralelo). Trazas: Phoenix -> http://localhost:6006")
    return 0 if ok == len(resultados) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
