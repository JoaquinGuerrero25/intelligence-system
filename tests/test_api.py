"""API: Redis falso + checkpointer en memoria + LLM mock (sin red). Incluye 'reinicio' del servidor."""

import time

import fakeredis
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from intelligence.api import app

RESENAS = "Reseñas de la AuroraBook X14, sentimiento y promedio"


@pytest.fixture
def infra(settings):
    """Redis + checkpointer compartidos: sobreviven a 'reiniciar' la app dentro de un test."""
    infra = {
        "redis": fakeredis.FakeAsyncRedis(decode_responses=True),
        "checkpointer": MemorySaver(),
        "settings": settings,
    }
    yield infra
    for k in infra:
        if hasattr(app.state, k):
            delattr(app.state, k)


def arrancar(infra) -> TestClient:
    for k, v in infra.items():
        setattr(app.state, k, v)
    return TestClient(app)


@pytest.fixture
def client(infra):
    with arrancar(infra) as c:
        yield c


def esperar(client, job_id, estados, timeout=20):
    fin = time.time() + timeout
    while time.time() < fin:
        job = client.get(f"/tasks/{job_id}").json()
        if job["status"] in estados:
            return job
        time.sleep(0.05)
    raise AssertionError(f"timeout; último estado: {job['status']}")


def test_health_y_docs(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["redis"] and h["rag_documents"] == 15 and h["llm_provider"] == "mock"
    assert client.get("/openapi.json").json()["info"]["title"] == "The Intelligence System"


def test_validacion_de_entrada(client):
    assert client.post("/tasks", json={"query": "x"}).status_code == 422
    assert client.post("/tasks", json={}).status_code == 422
    assert client.post("/tasks", json={"query": "valida", "thread_id": "../mal"}).status_code == 422
    assert client.post("/tasks/zzz/approve", json={"approved": "tal vez"}).status_code == 422


def test_post_encola_y_devuelve_job_id(client):
    r = client.post("/tasks", json={"query": RESENAS})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "PENDING" and body["job_id"] and body["thread_id"]
    assert client.get("/tasks/inexistente").status_code == 404


def test_flujo_normal_termina_en_done(client):
    jid = client.post("/tasks", json={"query": RESENAS}).json()["job_id"]
    job = esperar(client, jid, {"DONE", "FAILED", "WAITING_APPROVAL"})
    assert job["status"] == "DONE" and "AuroraBook" in job["result"] and job["action_log"] == []


def test_tarea_critica_pausa_hasta_la_aprobacion(client):
    jid = client.post("/tasks", json={"query": RESENAS + " y envía el resumen por correo"}).json()["job_id"]
    job = esperar(client, jid, {"WAITING_APPROVAL", "DONE", "FAILED"})
    assert job["status"] == "WAITING_APPROVAL" and job["approval_request"]["tipo"] == "aprobacion_requerida"
    time.sleep(0.3)
    assert client.get(f"/tasks/{jid}").json()["status"] == "WAITING_APPROVAL"

    assert client.post(f"/tasks/{jid}/approve", json={"approved": True, "reviewer": "ana"}).status_code == 202
    job = esperar(client, jid, {"DONE", "FAILED", "REJECTED"})
    assert job["status"] == "DONE" and "ana" in job["action_log"][0]
    assert client.post(f"/tasks/{jid}/approve", json={"approved": True}).status_code == 409


def test_rechazo_no_ejecuta_la_accion(client):
    jid = client.post("/tasks", json={"query": RESENAS, "require_approval": True}).json()["job_id"]
    esperar(client, jid, {"WAITING_APPROVAL"})
    client.post(f"/tasks/{jid}/approve", json={"approved": False, "reviewer": "ana", "comment": "no"})
    job = esperar(client, jid, {"REJECTED", "DONE", "FAILED"})
    assert job["status"] == "REJECTED" and job["action_log"] == []


def test_aprobar_job_no_pausado_o_inexistente(client):
    jid = client.post("/tasks", json={"query": RESENAS}).json()["job_id"]
    esperar(client, jid, {"DONE"})
    assert client.post(f"/tasks/{jid}/approve", json={"approved": True}).status_code == 409
    assert client.post("/tasks/nope/approve", json={"approved": True}).status_code == 404


def test_un_solo_job_activo_por_hilo(client):
    r1 = client.post("/tasks", json={"query": RESENAS, "thread_id": "h1", "require_approval": True})
    assert r1.status_code == 202
    esperar(client, r1.json()["job_id"], {"WAITING_APPROVAL"})
    assert client.post("/tasks", json={"query": RESENAS, "thread_id": "h1"}).status_code == 409


def test_excepcion_en_el_agente_deja_el_job_en_failed(infra, monkeypatch):
    import intelligence.api as api

    class Roto:
        async def ainvoke(self, *a, **k):
            raise RuntimeError("el LLM se cayó")

    monkeypatch.setattr(api, "create_graph", lambda *a, **k: Roto())
    with arrancar(infra) as c:
        jid = c.post("/tasks", json={"query": "cualquier tarea"}).json()["job_id"]
        job = esperar(c, jid, {"FAILED", "DONE"})
    assert job["status"] == "FAILED" and "el LLM se cayó" in job["error"]


def test_conversacion_persiste_tras_reiniciar_el_servidor(infra):
    with arrancar(infra) as c1:
        r = c1.post("/tasks", json={"query": RESENAS, "thread_id": "persist-1"}).json()
        esperar(c1, r["job_id"], {"DONE"})
    # --- "reinicio": se destruye la app y se levanta otra con el mismo Redis/checkpointer ---
    with arrancar(infra) as c2:
        roles = [m["role"] for m in c2.get("/threads/persist-1/messages").json()["messages"]]
        assert roles[0] == "user" and "sintetizador" in roles
        pregunta = "Y ahora la NovaPad Pro: calificación media"
        r2 = c2.post("/tasks", json={"query": pregunta, "thread_id": "persist-1"}).json()
        esperar(c2, r2["job_id"], {"DONE"})
        assert [m["role"] for m in c2.get("/threads/persist-1/messages").json()["messages"]].count(
            "user"
        ) == 2
        assert c2.get("/threads/no-existe/messages").status_code == 404
