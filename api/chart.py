"""Chart regeneration for a turn that already ran.

The chart normally arrives on the chat stream, built concurrently with the
answer. This endpoint exists for the case where it did not - the model returned
an unusable config, or the user wants another attempt at the same result.

It re-executes the stored SQL rather than re-reading the stored rows. The
statement was validated when it was written and is re-validated on the way
back out, so the round trip costs one cheap query and guarantees the chart
reflects the data rather than a snapshot of it.
"""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ai.llm import get_llm
from core import db
from models.schemas import ChartRequest
from services import chart_config, schema_service, sql_validator, turns

router = APIRouter(prefix="/api/chart", tags=["chart"])


@router.post("")
async def regenerate(request: ChartRequest) -> Dict[str, Any]:
    turn = await turns.get(request.turn_id)

    if turn is None:
        raise HTTPException(status_code=404, detail="turn not found")

    sql = turn.get("sql_text")

    if not sql:
        raise HTTPException(
            status_code=400,
            detail="This turn has no query behind it, so there is nothing to chart.",
        )

    # Re-validated rather than trusted. The statement is coming back out of a
    # database column, and the executor's contract is that nothing reaches it
    # unvalidated - regardless of where it has been in between.
    validation = sql_validator.validate(sql, schema_service.cached()["column_names"])

    if not validation["ok"]:
        raise HTTPException(status_code=400, detail=validation["reason"])

    try:
        rows, _ = await db.run_generated_sql(validation["sql"])

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"The query failed: {exc}")

    headers = list(rows[0].keys()) if rows else []

    chart, skip_reason = await chart_config.build(
        rows=rows,
        headers=headers,
        question=turn.get("question", ""),
        llm=get_llm(),
    )

    payload = chart.model_dump() if chart else None

    await turns.update_chart(request.turn_id, payload)

    return {
        "turn_id": str(request.turn_id),
        "chart": payload,
        "chart_skipped": skip_reason,
        "row_count": len(rows),
    }
