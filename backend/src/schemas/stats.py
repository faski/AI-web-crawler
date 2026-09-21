"""Pydantic schemas for /db_stats, /db_schema and /status."""

from pydantic import BaseModel


class DbStatsResponse(BaseModel):
    """Response for GET /db_stats: per-domain counts and average evaluations."""

    web_resources: dict[str, int]
    gold_standard: dict[str, int]
    avg_eval: dict[str, dict]
    avg_eval_judge: dict[str, dict]


class HealthResponse(BaseModel):
    """Response for GET /status: ``ok`` or ``error`` for each component."""

    backend: str
    database: str
    ollama: str
    # Not health checks but settings: whether the page may parse with the
    # model, and who would answer. The UI needs all three to say what a click
    # does, since local and remote differ in cost and in how long they take.
    llm_parser: bool = False
    llm_provider: str = "ollama"
    llm_model: str = ""
