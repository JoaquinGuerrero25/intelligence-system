"""Logging estructurado en una línea por evento (fácil de filtrar por job_id)."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for ruidoso in ("httpx", "httpcore", "urllib3", "opentelemetry"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
