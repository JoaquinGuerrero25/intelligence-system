"""Las herramientas validan entrada y salida con Pydantic."""

import json

import pytest
from pydantic import ValidationError

from intelligence.agents.analyst import (
    CalculoOutput,
    EstadisticasOutput,
    SentimientoOutput,
    analizar_sentimiento,
    calcular_estadisticas,
    calcular_expresion,
)
from intelligence.agents.research import BuscarOutput, LeerOutput, build_research_tools


async def test_buscar_devuelve_contrato_valido(rag):
    buscar, _ = build_research_tools(rag)
    out = BuscarOutput.model_validate_json(await buscar.ainvoke({"consulta": "ventas NovaPad Pro"}))
    assert out.resultados and out.resultados[0].id in {"doc-009", "doc-015"}


async def test_buscar_rechaza_entradas_invalidas(rag):
    buscar, _ = build_research_tools(rag)
    for malo in (
        {"consulta": "x"},
        {"consulta": "ok ok", "max_resultados": 99},
        {"consulta": "ok ok", "categoria": "otra"},
    ):
        with pytest.raises(ValidationError):
            await buscar.ainvoke(malo)


async def test_leer_documento(rag):
    _, leer = build_research_tools(rag)
    assert LeerOutput.model_validate_json(await leer.ainvoke({"doc_id": "doc-008"})).documento.id == "doc-008"
    assert "no existe" in LeerOutput.model_validate_json(await leer.ainvoke({"doc_id": "doc-999"})).error
    with pytest.raises(ValidationError):
        await leer.ainvoke({"doc_id": "../../etc/passwd"})


async def test_sentimiento_con_negacion():
    textos = ["Excelente y rápido", "Terrible, pésimo servicio", "no recomiendo"]
    r = SentimientoOutput.model_validate_json(await analizar_sentimiento.ainvoke({"textos": textos}))
    assert [d.etiqueta for d in r.detalle] == ["positivo", "negativo", "negativo"]


async def test_estadisticas():
    r = EstadisticasOutput.model_validate_json(
        await calcular_estadisticas.ainvoke({"valores": [5, 4, 2, 5, 1]})
    )
    assert r.media == 3.4 and r.mediana == 4 and r.n == 5
    for malo in ({"valores": []}, {"valores": [1, float("inf")]}):
        with pytest.raises(ValidationError):
            await calcular_estadisticas.ainvoke(malo)


async def test_calculo_seguro():
    ok = CalculoOutput.model_validate_json(
        await calcular_expresion.ainvoke({"expresion": "1200*950 + 1500*950"})
    )
    assert ok.resultado == 2565000
    for peligrosa in ("__import__('os').system('x')", "9**99", "1/0"):
        r = json.loads(await calcular_expresion.ainvoke({"expresion": peligrosa}))
        assert r["error"] and r["resultado"] is None
    with pytest.raises(ValidationError):
        await calcular_expresion.ainvoke({"expresion": "1+" * 200})
