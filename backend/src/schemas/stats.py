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
