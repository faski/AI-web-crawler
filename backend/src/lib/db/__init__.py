"""MariaDB persistence layer for Creeping Crawler.

Modules:
  connection   pool lifecycle and per-request connections
  models       Pydantic models returned by read queries
  queries      SQL queries (read/write) over web_resources and gold_standard
  init_loader  one-shot bootstrap from gs_data/*_gs.json files
  schema       tables added after init.sql, created at startup
  llm_queries  storage of the LLM-parser evaluation runs
"""

from .connection import close_pool, get_connection, init_pool, ping
from .init_loader import populate_if_empty
from .models import GoldStandardEntry, WebResource
from .schema import apply_schema

__all__ = [
    "close_pool",
    "get_connection",
    "init_pool",
    "ping",
    "populate_if_empty",
    "apply_schema",
    "GoldStandardEntry",
    "WebResource",
]
