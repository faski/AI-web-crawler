from .domains import router as domains_router
from .evaluate import router as evaluate_router
from .gold import router as gold_router
from .parse import router as parse_router
from .stats import router as stats_router

__all__ = [
    "domains_router",
    "evaluate_router",
    "gold_router",
    "parse_router",
    "stats_router",
]
