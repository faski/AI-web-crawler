"""Pydantic schemas for evaluation endpoints."""

from pydantic import BaseModel


class EvaluateRequest(BaseModel):
    """Request body for POST /evaluate and POST /evaluate_judge."""

    parsed_text: str
    gold_text: str


class TokenLevelEval(BaseModel):
    precision: float
    recall: float
    f1: float


class SimilarityEval(BaseModel):
    cosine: float
    jaccard: float
    excess_ratio: float


class EvaluateResponse(BaseModel):
    """Response for POST /evaluate."""

    token_level_eval: TokenLevelEval
    similarity_eval: SimilarityEval


class JudgeEvalResponse(BaseModel):
    """Response for POST /evaluate_judge."""

    model_name: str
    judge_score: int
    judge_feedback: str
    parsed_preview: str  # markdown-stripped parsed text, cut to text_cap
    gold_preview: str    # markdown-stripped gold text, cut to text_cap
    text_cap: int


class FullGsEvalResponse(BaseModel):
    """Response for GET /full_gs_eval: averaged metrics over a domain's gold standard."""

    token_level_eval: TokenLevelEval
    similarity_eval: SimilarityEval
    judge_score: float
