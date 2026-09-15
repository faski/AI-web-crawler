"""Pydantic models returned by the LLM modules."""

from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    model_name: str
    judge_score: int = Field(ge=1, le=5)
    judge_feedback: str


class SelfCheckResult(BaseModel):
    """The model's own verdict on an extraction, with no gold standard."""

    model_name: str
    complete: bool
    coherent: bool
    notes: str
