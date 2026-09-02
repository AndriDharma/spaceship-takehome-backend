"""Direct access to the forecasting tool, for a dashboard widget.

The same service the graph's forecast node calls. The model is not in this
path - the caller states the level and horizon, so there is nothing to
interpret. Asking "predict demand for CRAYON" in the chat goes through the
router instead, which is the path the assignment grades as tool selection.
"""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from models.schemas import ForecastRequest
from services import forecast as forecast_service

router = APIRouter(prefix="/api/forecast", tags=["forecast"])


@router.post("")
async def run_forecast(request: ForecastRequest) -> Dict[str, Any]:
    if request.level in ("category", "region") and not request.key:
        raise HTTPException(
            status_code=400,
            detail=f"key is required when level is {request.level}",
        )

    result = await forecast_service.run(
        level=request.level,
        key=request.key,
        horizon=request.horizon,
        method=request.method,
    )

    if not result.get("ok"):
        raise HTTPException(status_code=422, detail=result.get("reason", "no history"))

    chart = result.pop("chart", None)
    result["chart"] = chart.model_dump() if chart is not None else None

    return result
