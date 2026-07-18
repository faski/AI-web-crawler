"""Pydantic model returned by the judge module."""

from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    model_name: str
    judge_score: int = Field(ge=1, le=5)
    judge_feedback: str
