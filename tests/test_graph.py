"""Flujos del grafo con checkpointer en memoria y LLM mock (herramientas y RAG reales)."""

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from intelligence.graph import create_graph, nuevo_turno
from intelligence.graph.hitl import clasificar_criticidad
from intelligence.graph.state import MAX_STEPS
from intelligence.graph.supervisor import Decision, aplicar_guardas
from intelligence.graph.validation import revisar_rubrica, route_after_validation

RESENAS = (
    "Investiga las reseñas de la AuroraBook X14, analiza el sentimiento y calcula la calificación promedio."
)


def cfg(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}, "recursion_limit": 40}


@pytest.fixture
def grafo(settings, rag):
    return create_graph(settings, rag, MemorySaver())


async def test_flujo_completo_supervisor_investigador_analista(grafo):
    r = await grafo.ainvoke(nuevo_turno(RESENAS), cfg("t1"))
    assert "3.4" in r["final_answer"] and "[doc-001]" in r["final_answer"]
    assert [c["agent"] for c in r["contributions"]] == ["investigador", "analista"]
    assert r["contributions"][0]["tools_used"] == ["buscar_en_base_conocimiento"]
    assert set(r["contributions"][1]["tools_used"]) == {"analizar_sentimiento", "calcular_estadisticas"}
    assert r["validation_feedback"] == "" and r["task_completed"]


async def test_calculo_de_ventas(grafo):
    q = "Busca las ventas de la AuroraBook X14 y calcula unidades totales e ingreso."
    r = await grafo.ainvoke(nuevo_turno(q), cfg("t2"))
    assert "4,050" in r["final_answer"] and "3,847,500" in r["final_answer"]


async def test_memoria_multi_turno_en_el_mismo_hilo(grafo):
    await grafo.ainvoke(nuevo_turno(RESENAS), cfg("t3"))
    r2 = await grafo.ainvoke(nuevo_turno("Ahora lo mismo para las reseñas de la NovaPad Pro."), cfg("t3"))
    # el segundo turno reinicia el control pero conserva el historial de mensajes
    assert [c["agent"] for c in r2["contributions"]] == ["investigador", "analista"]
    assert len([m for m in r2["messages"] if m.type == "human"]) == 2


async def test_tarea_critica_pausa_y_reanuda(grafo):
    q = "Reseñas de la AuroraBook X14, calificación promedio y envía el resumen por correo al equipo."
    r = await grafo.ainvoke(nuevo_turno(q), cfg("t4"))
    assert r["__interrupt__"][0].value["tipo"] == "aprobacion_requerida"
    assert (await grafo.aget_state(cfg("t4"))).next == ("aprobacion_humana",)  # pausado en el checkpoint
    r = await grafo.ainvoke(Command(resume={"approved": True, "reviewer": "ana"}), cfg("t4"))
    assert r["final_answer"] and "ana" in r["action_log"][0]


async def test_rechazo_no_ejecuta_accion(grafo):
    await grafo.ainvoke(nuevo_turno(RESENAS, force_approval=True), cfg("t5"))
    r = await grafo.ainvoke(
        Command(resume={"approved": False, "reviewer": "ana", "comment": "no"}), cfg("t5")
    )
    assert "RECHAZADA" in r["final_answer"] and r["action_log"] == []


async def test_el_estado_sobrevive_a_reconstruir_el_grafo(settings, rag):
    """Misma persistencia que tras reiniciar el servidor: nuevo grafo + mismo checkpointer."""
    saver = MemorySaver()
    g1 = create_graph(settings, rag, saver)
    await g1.ainvoke(nuevo_turno(RESENAS, force_approval=True), cfg("t6"))
    g2 = create_graph(settings, rag, saver)
    r = await g2.ainvoke(Command(resume={"approved": True, "reviewer": "luis"}), cfg("t6"))
    assert "luis" in r["action_log"][0]


# ---------- guardas, rúbrica y criticidad (unitarios) ----------
def test_analista_sin_datos_se_redirige_al_investigador():
    sig, _, nota = aplicar_guardas(
        Decision(next_agent="analista", instruction="x", reason="r"), {"step_count": 0}
    )
    assert sig == "investigador" and nota


def test_tope_de_pasos_fuerza_finish():
    decision = Decision(next_agent="investigador", instruction="x", reason="r")
    assert aplicar_guardas(decision, {"step_count": MAX_STEPS})[0] == "FINISH"


def test_revisiones_agotadas_cierra():
    estado = {
        "step_count": 3,
        "research_notes": "n",
        "revisions": {"investigador": 1},
        "contributions": [{"agent": "investigador"}, {"agent": "analista"}],
    }
    decision = Decision(next_agent="investigador", instruction="x", reason="r")
    assert aplicar_guardas(decision, estado)[0] == "FINISH"


def test_rubrica_detecta_faltantes_y_enruta():
    assert len(revisar_rubrica({})) == 2
    ok = {
        "research_notes": "[doc-001] dato",
        "analysis_result": "media 3.4",
        "contributions": [{"agent": "analista", "tools_used": ["calcular_estadisticas"]}],
    }
    assert revisar_rubrica(ok) == []
    assert (
        route_after_validation({"validation_feedback": "x", "validation_attempts": 1, "step_count": 2})
        == "supervisor"
    )
    assert (
        route_after_validation({"validation_feedback": "x", "validation_attempts": 2, "step_count": 2})
        == "sintetizador"
    )


def test_clasificador_de_criticidad():
    assert clasificar_criticidad("Investiga las reseñas y calcula el promedio") == ""
    assert "efectos secundarios" in clasificar_criticidad(
        "Envía un correo con el reporte a todos los clientes"
    )
    assert "efectos secundarios" in clasificar_criticidad("Emití un reembolso")
    assert clasificar_criticidad("hola", forzada=True)
