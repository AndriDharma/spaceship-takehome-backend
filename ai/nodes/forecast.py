"""The forecasting tool, as a graph node.

It is reached by the router, not by a button, which is what makes it a tool the
AI selects rather than a feature the AI is bypassed for. The same computation
is also exposed at POST /api/forecast for the dashboard, but that endpoint
calls this service directly - the model is not in that path.

Nothing is predicted here. services/forecast.py does the arithmetic and builds
the chart; this node translates the router's parameters into a call and puts
the result on the state for the answer node to narrate.
"""

from typing import Any, Dict

from ai import streaming
from ai.state import GraphState
from services import forecast as forecast_service


async def forecast_node(state: GraphState) -> Dict[str, Any]:
    params = state.get("forecast_params") or {}

    streaming.progress("forecast", "Building the demand forecast")

    try:
        result = await forecast_service.run(
            level=params.get("level", "overall"),
            key=params.get("key"),
            horizon=int(params.get("horizon", 3)),
            method=params.get("method", "moving_average"),
            requested_sku=params.get("requested_sku"),
        )

    except Exception as exc:
        print(f"FORECAST FAILED | {type(exc).__name__}: {exc}")

        streaming.progress(
            "forecast",
            "The forecast could not be computed",
            status=streaming.FAILED,
        )

        return {"forecast": {"ok": False, "reason": str(exc)}}

    if not result.get("ok"):
        streaming.progress(
            "forecast",
            result.get("reason", "Not enough history to forecast"),
            status=streaming.FAILED,
        )

        return {"forecast": result}

    chart = result.pop("chart", None)

    streaming.progress(
        "forecast",
        f"Forecast ready: {result['horizon']} month(s), {result['method_label']}",
        status=streaming.COMPLETED,
        detail={
            "level": result["level"],
            "key": result["key"],
            "method": result["method"],
            "notes": result["notes"],
        },
    )

    if chart is not None:
        # Emitted from here rather than from chart_node: the shape was known
        # before the data was read, so it was built in Python and there is no
        # model call to run in parallel with the answer.
        streaming.emit("chart", chart.model_dump())

    return {
        "forecast": result,
        "chart_config": chart.model_dump() if chart is not None else None,
        # The forecast series is the chart. The row table shows the same
        # numbers so the "Table" tab is not empty.
        "rows": result["history"] + result["forecast"],
        "headers": ["month", "demand", "orders"],
        "row_count": len(result["history"]) + len(result["forecast"]),
    }
