"""FastAPI application entry point.

Run from the backend/ directory:
    uvicorn src.server:app --host 0.0.0.0 --port 8003

Startup sequence (lifespan):
    1. wait for MariaDB and open the connection pool
    2. seed the gold standard tables from gs_data/ on first boot
    3. precompute and cache quantitative metrics for every GS URL
    4. start the shared crawler browser so the first parse is fast
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .lib import close_crawler, get_crawler
from .lib.db import close_pool, init_pool, populate_if_empty
from .lib.precompute import precompute_quantitative_evaluations
from .routes import (
    domains_router,
    evaluate_router,
    gold_router,
    parse_router,
    stats_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_pool()
    inserted_count = populate_if_empty()
    if inserted_count:
        print(f"[startup] seeded {inserted_count} gold standard entries", flush=True)
    precomputed = await precompute_quantitative_evaluations()
    if precomputed:
        print(f"[startup] precomputed {precomputed} quantitative evaluations", flush=True)
    await get_crawler()
    yield
    await close_crawler()
    close_pool()


app = FastAPI(title="Creeping Crawler API", lifespan=lifespan)

app.include_router(domains_router)
app.include_router(parse_router)
app.include_router(gold_router)
app.include_router(evaluate_router)
app.include_router(stats_router)
