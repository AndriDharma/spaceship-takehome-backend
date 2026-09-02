"""Application entry point."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import chart, chat, dashboard, forecast, health
from core import config, db
from services import schema_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    The schema is read once, here, rather than per request.

    It is the validator's column whitelist, the router's prompt, and the date
    anchor - three things every turn needs and none of which change while the
    process is alive. Reading them at startup also means a misconfigured
    database fails at boot with a clear error, instead of on the first user's
    first question.
    """
    db.init_engine()

    try:
        await schema_service.load()
        meta = schema_service.cached()
        print(
            f"SCHEMA LOADED | {len(meta['columns'])} columns "
            f"| {meta['row_count']} orders "
            f"| {meta['min_date']} to {meta['max_date']}"
        )
    except Exception as exc:
        # Not fatal. The health endpoint reports schema_loaded=false and the
        # container still starts, which is easier to diagnose from Cloud Run
        # logs than a crash loop.
        print(f"SCHEMA LOAD FAILED | {type(exc).__name__}: {exc}")

    yield

    db.close_engine()


app = FastAPI(
    title="Logistics Analytics API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # The frontend is a separate Cloud Run service, so its origin has to be
    # listed explicitly - credentials are not used, but SSE still needs the
    # origin allowed.
    allow_origins=config.CORS_ORIGINS or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(dashboard.router)
app.include_router(chat.router)
app.include_router(chart.router)
app.include_router(forecast.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=True,
    )
