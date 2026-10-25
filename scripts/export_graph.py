"""Genera el diagrama Mermaid del grafo REAL compilado (docs/graph.mmd), para pegarlo en el README.

python scripts/export_graph.py
"""

from pathlib import Path

from intelligence.config import Settings
from intelligence.graph import create_graph
from intelligence.rag import build_rag_service


def main() -> None:
    settings = Settings(_env_file=None, llm_provider="mock", phoenix_enabled=False)
    rag = build_rag_service(settings)
    grafo = create_graph(settings, rag)
    destino = Path(__file__).resolve().parent.parent / "docs" / "graph.mmd"
    destino.write_text(grafo.get_graph().draw_mermaid(), encoding="utf-8")
    print(f"Escrito {destino}")


if __name__ == "__main__":
    main()
