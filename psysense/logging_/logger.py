"""
logging_/logger.py

Structured JSON logging for PsySense. Every service calls
get_logger(component_name) instead of logging.getLogger() directly, so
log records carry a consistent schema (timestamp, level, component,
message, plus arbitrary structured context via `extra`) that's easy to
grep, ship to a log aggregator, or insert into the system_logs table.

Save as: logging_/logger.py
(named logging_ deliberately -- a folder literally named `logging` on
the Python path shadows the stdlib module and breaks every other import
in the project)
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        # Anything passed via logger.info(..., extra={"context": {...}})
        # ends up here under "context" rather than polluting top-level keys.
        if hasattr(record, "context"):
            payload["context"] = record.context
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_root_logging(level: int = logging.INFO) -> None:
    """Call once at process startup (main.py, each FastAPI service's
    module-level init). Idempotent -- safe to call more than once."""
    root = logging.getLogger("psysense")
    if root.handlers:
        return  # already configured
    root.setLevel(level)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.propagate = False


def get_logger(component: str) -> logging.Logger:
    """component should be dotted, e.g. 'identity.manager', 'tracking.bytetrack'
    -- it's appended under the 'psysense' root logger namespace."""
    configure_root_logging()
    return logging.getLogger(f"psysense.{component}")