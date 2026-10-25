"""LLM heurístico sin red ni API keys (`LLM_PROVIDER=mock`).

No es un script fijo: decide según el contenido de la conversación (qué aportaron ya los agentes, qué
devolvieron las herramientas, qué pide la tarea). Las herramientas y el RAG se ejecutan de verdad; solo el
"razonamiento" del LLM está simulado. Sirve para correr el sistema completo, los tests y las trazas en una
máquina limpia. Con un LLM real (openai/google) este módulo no se usa.
"""

import json
import re
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from intelligence.graph.supervisor import Decision
from intelligence.rag.embeddings import strip_accents

Role = Literal["supervisor", "investigador", "analista", "sintetizador", "generic"]
_PRODUCTOS = ("aurorabook", "novapad")


def _ultimo_humano(messages: list[BaseMessage]) -> str:
    return next((str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), "")


def _tool_call(name: str, args: dict, id_: str) -> dict:
    return {"name": name, "args": args, "id": id_}


class HeuristicChat(BaseChatModel):
    role: Role = "generic"

    @property
    def _llm_type(self) -> str:
        return "heuristic-mock"

    # --- interfaz de LangChain -------------------------------------------------
    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kw: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._responder(messages))])

    async def _agenerate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kw: Any
    ) -> ChatResult:
        return self._generate(messages)

    def bind_tools(self, tools: Sequence[Any], **kw: Any):
        return self

    def with_structured_output(self, schema: Any, **kw: Any):
        async def _adecidir(msgs: Any) -> Decision:
            return self._decidir(msgs)

        return RunnableLambda(self._decidir, afunc=_adecidir)

    # --- supervisor ---------------------------------------------------------------
    def _decidir(self, messages: Any) -> Decision:
        resumen = _ultimo_humano(list(messages))
        pregunta = (re.search(r"Pregunta del usuario: (.*)", resumen) or [None, ""])[1].strip()
        if "Aporte de analista" in resumen:
            return Decision(
                next_agent="FINISH", reason="Hay datos citados y resultados numéricos del analista."
            )
        if "Aporte de investigador" in resumen:
            return Decision(
                next_agent="analista",
                instruction=f"Analiza con tus herramientas los datos recuperados para responder: {pregunta}",
                reason="Hay datos citados; falta el análisis cuantitativo.",
            )
        return Decision(
            next_agent="investigador",
            instruction=f"Recupera de la base de conocimiento todo lo necesario para: {pregunta}",
            reason="Faltan datos: nada fue investigado todavía.",
        )

    # --- agentes y sintetizador -----------------------------------------------------
    def _responder(self, messages: list[BaseMessage]) -> AIMessage:
        if self.role == "investigador":
            return self._investigar(messages)
        if self.role == "analista":
            return self._analizar(messages)
        return self._sintetizar(messages)

    def _investigar(self, messages: list[BaseMessage]) -> AIMessage:
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        tarea = _ultimo_humano(messages).removeprefix("Tarea: ").strip()
        q = strip_accents(tarea.lower())
        if not tool_msgs:
            familias = {
                "resenas": bool(re.search(r"resena|opinion|sentimiento|calificacion", q)),
                "ventas": bool(re.search(r"venta|ingreso|unidades|facturacion", q)),
                "politicas": bool(re.search(r"devolucion|envio|garantia|soporte|politica", q)),
            }
            activas = [f for f, on in familias.items() if on]
            args: dict[str, Any] = {"consulta": tarea[:300], "max_resultados": 8}
            if len(activas) == 1:
                args["categoria"] = activas[0]
            return AIMessage(
                content="", tool_calls=[_tool_call("buscar_en_base_conocimiento", args, "inv-1")]
            )

        resultados = json.loads(tool_msgs[-1].content).get("resultados", [])
        producto = next((p for p in _PRODUCTOS if p in q), None)
        if producto:
            resultados = [
                r for r in resultados
                if producto in strip_accents(f"{r['titulo']} {r['texto']}".lower()) or r["categoria"] == "politicas"
            ]  # fmt: skip
        if not resultados:
            return AIMessage(content="No se encontraron documentos relevantes en la base de conocimiento.")
        lineas = []
        for r in sorted(resultados, key=lambda r: r["id"]):
            cal = f" ({r['calificacion']:g}/5)" if r["calificacion"] is not None else ""
            lineas.append(f"- [{r['id']}]{cal} {r['titulo']}: {r['texto']}")
        return AIMessage(content="Hallazgos de la base de conocimiento:\n" + "\n".join(lineas))

    def _analizar(self, messages: list[BaseMessage]) -> AIMessage:
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        humano = _ultimo_humano(messages)
        contexto = humano.split("Contexto disponible:", 1)[-1]
        if not tool_msgs:
            return self._planificar_calculo(contexto)

        ids = sorted(set(re.findall(r"doc-\d{3}", contexto)))
        frases = []
        for m in tool_msgs:
            d = json.loads(m.content)
            if m.name == "calcular_estadisticas":
                frases.append(
                    f"calificación/valor medio {d['media']} (mediana {d['mediana']}, mín {d['min']:g}, "
                    f"máx {d['max']:g}, desvío {d['desvio']}, n={d['n']})"
                )
            elif m.name == "analizar_sentimiento":
                c = d["conteo"]
                frases.append(
                    f"sentimiento global {d['etiqueta_global']} (puntaje medio {d['puntaje_medio']}): "
                    f"{c['positivo']} positivas, {c['neutral']} neutrales, {c['negativo']} negativas"
                )
            elif m.name == "calcular_expresion":
                frases.append(
                    f"{d['expresion']} = {d['resultado']:,.0f}" if not d["error"] else f"error: {d['error']}"
                )
        citas = ", ".join(f"[{i}]" for i in ids)
        return AIMessage(content=f"Análisis de {citas}: " + "; ".join(frases) + ".")

    @staticmethod
    def _planificar_calculo(contexto: str) -> AIMessage:
        calls = []
        reseñas = re.findall(r"\[(doc-\d{3})\] \((\d(?:\.\d)?)/5\) [^:]*: (.*)", contexto)
        if reseñas:
            calls.append(_tool_call("analizar_sentimiento", {"textos": [t for _, _, t in reseñas]}, "ana-1"))
            calls.append(
                _tool_call("calcular_estadisticas", {"valores": [float(c) for _, c, _ in reseñas]}, "ana-2")
            )
        else:
            unidades = [int(u) for u in re.findall(r"= (\d+) unidades", contexto)]
            precios = [int(p) for p in re.findall(r"(\d+) USD por unidad", contexto)]
            if unidades:
                expr = "+".join(map(str, unidades))
                calls.append(_tool_call("calcular_expresion", {"expresion": expr}, "ana-1"))
                if precios:
                    calls.append(
                        _tool_call("calcular_expresion", {"expresion": f"({expr})*{precios[0]}"}, "ana-2")
                    )
            else:
                numeros = [float(n) for n in re.findall(r"(?<![\w-])(\d+(?:\.\d+)?)(?![\w-])", contexto)]
                if numeros:
                    calls.append(_tool_call("calcular_estadisticas", {"valores": numeros[:200]}, "ana-1"))
        if not calls:
            return AIMessage(content="El contexto recibido no contiene datos numéricos para analizar.")
        return AIMessage(content="", tool_calls=calls)

    @staticmethod
    def _sintetizar(messages: list[BaseMessage]) -> AIMessage:
        entrada = _ultimo_humano(messages)
        analisis = (
            re.search(r"Resultados del analista:\n(.*?)(?:\n\nAdvertencias|\Z)", entrada, re.S) or [None, ""]
        )[1]
        pregunta = (re.search(r"Pregunta: (.*)", entrada) or [None, ""])[1]
        advertencia = (re.search(r"Advertencias[^:]*: (.*)", entrada) or [None, ""])[1]
        texto = f"Respuesta a «{pregunta.strip()}»: {analisis.strip()}"
        if advertencia:
            texto += f"\nLimitaciones: {advertencia.strip()}"
        return AIMessage(content=texto)
