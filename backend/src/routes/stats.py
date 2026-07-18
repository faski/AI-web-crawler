"""Route handlers for /db_stats, /db_schema and /status."""

from fastapi import APIRouter

from ..lib.db import ping as ping_database
from ..lib.db import queries
from ..lib.llm import ping as ping_ollama
from ..schemas import DbStatsResponse, HealthResponse

router = APIRouter()


SCHEMA_DESCRIPTION: dict[str, dict[str, str]] = {
    "web_resources": {
        "url": "varchar(2048), PK",
        "domain": "varchar(255)",
        "title": "varchar(2048)",
        "html_text": "longtext",
        "created_at": "datetime",
    },
    "gold_standard": {
        "url": "varchar(2048), PK, FK(web_resources.url)",
        "gold_text": "longtext",
        "created_at": "datetime",
    },
    "evaluations": {
        "url": "varchar(2048), PK, FK(web_resources.url)",
        "precision_val": "double",
        "recall_val": "double",
        "f1": "double",
        "cosine": "double",
        "jaccard": "double",
        "excess_ratio": "double",
        "judge_score": "int",
        "updated_at": "datetime",
    },
}


@router.get("/db_stats", response_model=DbStatsResponse)
def db_stats():
    """Per-domain counts and average evaluations read from the DB cache."""
    return DbStatsResponse(
        web_resources=queries.count_resources_per_domain(),
        gold_standard=queries.count_gold_per_domain(),
        avg_eval=queries.get_avg_eval_per_domain(),
        avg_eval_judge=queries.get_avg_judge_per_domain(),
    )


@router.get("/db_schema")
def db_schema() -> dict:
    """Return the DB schema as a plain JSON dict (table → column → type/keys)."""
    return SCHEMA_DESCRIPTION


@router.get("/status", response_model=HealthResponse)
def status():
    """Per-component health (``ok`` or ``error``). Always returns HTTP 200."""
    return HealthResponse(
        backend="ok",
        database="ok" if ping_database() else "error",
        ollama="ok" if ping_ollama() else "error",
    )
