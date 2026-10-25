"""Agente Analista: sentimiento, estadística y cálculo seguro sobre datos YA recuperados.

Cada herramienta declara un `args_schema` Pydantic (entrada validada antes de ejecutar) y devuelve un
modelo Pydantic serializado (salida con contrato fijo). Son `async` por uniformidad con el resto del sistema.
"""

import ast
import math
import operator
import statistics

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from intelligence.rag.embeddings import strip_accents

SYSTEM_PROMPT = """Eres el Agente de Análisis. Procesas datos que YA fueron recuperados por el investigador.
- NO tienes acceso a fuentes externas: trabaja solo con el contexto recibido.
- Nunca calcules de cabeza: usa SIEMPRE las herramientas (`analizar_sentimiento`,
  `calcular_estadisticas`, `calcular_expresion`).
- Reporta resultados numéricos exactos tal como los devuelven las herramientas y cita los [doc-XXX] analizados.
- Si el contexto no alcanza para el análisis pedido, dilo explícitamente."""


# ------------------------------------------------------------------ sentimiento
_POS = {
    "excelente", "increible", "comodo", "rapido", "recomiendo", "recomendable", "fantastica", "fantastico",
    "buena", "bueno", "buen", "mejor", "encanta", "nitida", "liviana", "genial", "perfecta", "perfecto",
    "correcta", "responde", "totalmente",
}  # fmt: skip
_NEG = {
    "decepcionante", "terrible", "pesimo", "pesima", "lento", "lenta", "calienta", "danada", "danado",
    "tardo", "traba", "regular", "ruido", "problema", "malo", "mala", "peor", "falla",
}  # fmt: skip
_NEGADORES = {"no", "nunca", "tampoco"}


def _puntaje_texto(texto: str) -> float:
    tokens = [w.strip(".,;:!¡?¿()\"'") for w in strip_accents(texto.lower()).split()]
    pos = neg = 0
    for i, w in enumerate(tokens):
        negado = i > 0 and tokens[i - 1] in _NEGADORES
        if w in _POS:
            pos, neg = (pos, neg + 1) if negado else (pos + 1, neg)
        elif w in _NEG:
            pos, neg = (pos + 1, neg) if negado else (pos, neg + 1)
    total = pos + neg
    return 0.0 if total == 0 else round((pos - neg) / total, 3)


def _etiqueta(p: float) -> str:
    return "positivo" if p > 0.2 else "negativo" if p < -0.2 else "neutral"


class SentimientoInput(BaseModel):
    textos: list[str] = Field(min_length=1, max_length=50, description="Textos a clasificar.")


class SentimientoItem(BaseModel):
    n: int
    puntaje: float = Field(ge=-1, le=1)
    etiqueta: str
    inicio: str


class SentimientoOutput(BaseModel):
    detalle: list[SentimientoItem]
    conteo: dict[str, int]
    puntaje_medio: float
    etiqueta_global: str


@tool("analizar_sentimiento", args_schema=SentimientoInput)
async def analizar_sentimiento(textos: list[str]) -> str:
    """Clasifica el sentimiento de cada texto (puntaje -1 a 1) y devuelve el agregado."""
    detalle = [
        SentimientoItem(n=i, puntaje=(p := _puntaje_texto(t)), etiqueta=_etiqueta(p), inicio=t[:50])
        for i, t in enumerate(textos, 1)
    ]
    conteo = {e: sum(1 for d in detalle if d.etiqueta == e) for e in ("positivo", "neutral", "negativo")}
    media = round(statistics.fmean(d.puntaje for d in detalle), 3)
    return SentimientoOutput(
        detalle=detalle, conteo=conteo, puntaje_medio=media, etiqueta_global=_etiqueta(media)
    ).model_dump_json()


# ------------------------------------------------------------------ estadística
class EstadisticasInput(BaseModel):
    valores: list[float] = Field(min_length=1, max_length=1000)

    @field_validator("valores")
    @classmethod
    def _finitos(cls, v: list[float]) -> list[float]:
        if not all(math.isfinite(x) for x in v):
            raise ValueError("Los valores deben ser números finitos.")
        return v


class EstadisticasOutput(BaseModel):
    n: int
    media: float
    mediana: float
    min: float
    max: float
    desvio: float


@tool("calcular_estadisticas", args_schema=EstadisticasInput)
async def calcular_estadisticas(valores: list[float]) -> str:
    """Calcula n, media, mediana, mínimo, máximo y desvío estándar de una lista de números."""
    return EstadisticasOutput(
        n=len(valores),
        media=round(statistics.fmean(valores), 3),
        mediana=statistics.median(valores),
        min=min(valores),
        max=max(valores),
        desvio=round(statistics.stdev(valores), 3) if len(valores) > 1 else 0.0,
    ).model_dump_json()


# ------------------------------------------------------------------ cálculo seguro
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}  # fmt: skip


def _eval(n: ast.AST) -> float:
    if isinstance(n, ast.Constant) and isinstance(n.value, int | float) and not isinstance(n.value, bool):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        a, b = _eval(n.left), _eval(n.right)
        if isinstance(n.op, ast.Pow) and abs(b) > 10:
            raise ValueError("Exponente demasiado grande.")
        return _OPS[type(n.op)](a, b)
    if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_eval(n.operand))
    raise ValueError("Expresión no permitida (solo números y + - * / ** %).")


class CalculoInput(BaseModel):
    expresion: str = Field(min_length=1, max_length=200, description="Ej.: '1200*950 + 1500*950'.")


class CalculoOutput(BaseModel):
    expresion: str
    resultado: float | None = None
    error: str = ""


@tool("calcular_expresion", args_schema=CalculoInput)
async def calcular_expresion(expresion: str) -> str:
    """Evalúa una expresión aritmética segura (sin eval), ej. '1200*950 + 1500*950'."""
    try:
        resultado = _eval(ast.parse(expresion, mode="eval").body)
        return CalculoOutput(expresion=expresion, resultado=round(resultado, 4)).model_dump_json()
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return CalculoOutput(expresion=expresion, error=str(e) or type(e).__name__).model_dump_json()


ANALYST_TOOLS = [analizar_sentimiento, calcular_estadisticas, calcular_expresion]


def build_analyst_agent(llm: BaseChatModel):
    return create_agent(llm, ANALYST_TOOLS, system_prompt=SYSTEM_PROMPT, name="analista")
