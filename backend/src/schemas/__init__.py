"""Pydantic request/response schemas for the Creeping Crawler API."""

from .domains import DomainsResponse
from .evaluate import (
    EvaluateRequest,
    EvaluateResponse,
    FullGsEvalResponse,
    JudgeEvalResponse,
    SimilarityEval,
    TokenLevelEval,
)
from .gold import (
    AddGoldStandardRequest,
    AddWebResourceRequest,
    DeleteUrlRequest,
    GoldStandardResponse,
    GoldStandardUrlsResponse,
    StatusResponse,
)
from .parse import ParseRequest, ParseResponse
from .stats import DbStatsResponse, HealthResponse

__all__ = [
    "DomainsResponse",
    "EvaluateRequest",
    "EvaluateResponse",
    "FullGsEvalResponse",
    "JudgeEvalResponse",
    "TokenLevelEval",
    "SimilarityEval",
    "GoldStandardResponse",
    "GoldStandardUrlsResponse",
    "AddWebResourceRequest",
    "AddGoldStandardRequest",
    "DeleteUrlRequest",
    "StatusResponse",
    "ParseRequest",
    "ParseResponse",
    "DbStatsResponse",
    "HealthResponse",
]
