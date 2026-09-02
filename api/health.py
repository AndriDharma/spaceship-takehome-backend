"""Liveness, and a warm-up target.

Cloud Run with min-instances=0 cold-starts on the first request, and this
application imports LangGraph and the Vertex client before it can answer
anything. Hitting this endpoint first pays that cost somewhere the user is not
watching an empty chat panel.
"""

from typing import Any, Dict

from fastapi import APIRouter

from core import config
from services import schema_service

router = APIRouter(tags=["health"])


@router.get("/api/health")
async def health() -> Dict[str, Any]:
    try:
        meta = schema_service.cached()
        schema_loaded = True
        window = f"{meta['min_date']} to {meta['max_date']}"
        rows = meta["row_count"]

    except RuntimeError:
        schema_loaded = False
        window = None
        rows = 0

    return {
        "status": "ok" if schema_loaded else "starting",
        "model": config.GEMINI_MODEL,
        "region": config.VERTEX_REGION,
        "schema_loaded": schema_loaded,
        "data_window": window,
        "row_count": rows,
    }
