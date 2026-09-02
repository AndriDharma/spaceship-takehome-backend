"""Selecting a chart, concurrently with the answer being written.

This node and answer_node are both scheduled the moment execute_sql finishes,
so the chart model call overlaps the answer stream instead of following it. By
the time the last token arrives the chart is usually already on the client.
That is the whole reason the graph fans out here rather than running the two in
sequence.

The pair writes disjoint state keys - this node owns chart_config and
chart_skipped, answer_node owns answer - because LangGraph rejects two branches
writing the same key without a reducer.

Every failure in here is contained. A chart is a presentation detail; the
answer is the product. Nothing this node does can prevent the answer arriving.
"""

from typing import Any, Dict

from ai import streaming
from ai.llm import get_llm
from ai.state import GraphState
from services import chart_config


async def chart_node(state: GraphState) -> Dict[str, Any]:
    rows = state.get("rows") or []
    headers = state.get("headers") or []

    if not rows:
        return {"chart_config": None, "chart_skipped": "No rows to chart."}

    # Checked before the progress row is emitted, so a single-value answer
    # does not show the user a chart step that was never going to produce one.
    chartable, reason = chart_config.is_chartable(rows, headers)

    if not chartable:
        streaming.emit("chart_skipped", {"reason": reason})

        return {"chart_config": None, "chart_skipped": reason}

    streaming.progress("chart", "Selecting a chart")

    try:
        chart, skip_reason = await chart_config.build(
            rows=rows,
            headers=headers,
            question=state.get("question", ""),
            llm=get_llm(),
        )

    except Exception as exc:
        print(f"CHART NODE FAILED | {type(exc).__name__}: {exc}")

        streaming.progress("chart", "Chart could not be built", status=streaming.SKIPPED)
        streaming.emit("chart_skipped", {"reason": str(exc)})

        return {"chart_config": None, "chart_skipped": str(exc)}

    if chart is None:
        streaming.progress("chart", skip_reason, status=streaming.SKIPPED)
        streaming.emit("chart_skipped", {"reason": skip_reason})

        return {"chart_config": None, "chart_skipped": skip_reason}

    payload = chart.model_dump()

    streaming.progress(
        "chart",
        f"{chart.chartType} chart ready",
        status=streaming.COMPLETED,
        detail={"chartType": chart.chartType, "xKey": chart.xKey, "yKeys": chart.yKeys},
    )

    streaming.emit("chart", payload)

    return {"chart_config": payload, "chart_skipped": ""}
