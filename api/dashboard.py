"""The descriptive dashboard. No AI in this path at all.

One endpoint returning KPIs and all three charts together. At 400 rows the
whole payload is smaller than the overhead of three round trips, and it means
the frontend renders a complete dashboard from a single fetch.
"""

from typing import Any, Dict

from fastapi import APIRouter

from services import dashboard

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("")
async def get_dashboard() -> Dict[str, Any]:
    return await dashboard.build()
