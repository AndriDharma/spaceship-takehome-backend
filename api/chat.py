"""The streaming chat endpoint.

One request, one connection, one graph run. The reference this was adapted from
splits the same work across three hops - an enqueue endpoint, a Service Bus
worker, and a Redis stream the client subscribes to - because it serves many
tenants concurrently and has to survive a worker dying mid-turn. None of that
applies here, and collapsing it removes Redis, a message bus, and a queue table
from the deployment.

POST rather than GET, so the question travels in a body. That means the browser
cannot use EventSource, which is GET-only; the frontend reads the response with
fetch and a ReadableStream.
"""

from typing import Any, AsyncGenerator, Dict
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ai import streaming
from ai.graph import compiled
from models.schemas import ChatRequest
from services import schema_service, turns

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Tokens are flushed in blocks rather than one at a time. A per-token frame
# spends more time in framing overhead than in content, and the visible result
# is identical.
_FLUSH_AT = 30


def _unpack(part: Any) -> tuple[str, Any]:
    """
    Normalise one item from graph.astream().

    LangGraph yields (mode, payload) tuples when stream_mode is a list, but has
    shipped a {"type": ..., "data": ...} dict in other versions. Handling both
    means a library upgrade cannot silently stop the stream.
    """
    if isinstance(part, tuple) and len(part) == 2:
        return str(part[0]), part[1]

    if isinstance(part, dict) and "type" in part:
        return str(part["type"]), part.get("data")

    return "unknown", part


def _explain(state: Dict[str, Any]) -> Dict[str, Any]:
    validation = state.get("validation") or {}
    forecast = state.get("forecast") or {}

    return {
        "mode": state.get("mode", "direct"),
        "reason": state.get("route_reason", ""),
        "sql": state.get("sql") or None,
        "tables": validation.get("tables", []),
        "columns": validation.get("columns", []),
        "filters": validation.get("filters", ""),
        "group_by": validation.get("group_by", []),
        "row_count": state.get("row_count", 0),
        "truncated": bool(state.get("truncated")),
        "limit_injected": bool(validation.get("limit_injected")),
        "anchor_date": state.get("anchor_date"),
        "retries": state.get("retries", 0),
        "method": forecast.get("method_label"),
    }


async def _run(question: str, session_id: UUID) -> AsyncGenerator[str, None]:
    turn_id = uuid4()

    meta = schema_service.cached()
    window = f"{meta['min_date']} to {meta['max_date']}"

    yield streaming.sse(
        "start",
        {"turn_id": str(turn_id), "session_id": str(session_id), "data_window": window},
    )

    state = {
        "question": question,
        "session_id": str(session_id),
        "history": await turns.recent(session_id),
        "schema_text": schema_service.schema_block(),
        "anchor_date": str(schema_service.anchor_date()),
        "data_window": window,
        "retries": 0,
    }

    buffer = ""
    final: Dict[str, Any] = {}

    try:
        async for part in compiled().astream(state, stream_mode=["custom", "values"]):
            mode, payload = _unpack(part)

            if mode == "values" and isinstance(payload, dict):
                final = payload
                continue

            if mode != "custom" or not isinstance(payload, dict):
                continue

            event = payload.get("event")
            data = payload.get("data") or {}

            if event == "output":
                buffer += data.get("content", "")

                if len(buffer) >= _FLUSH_AT:
                    yield streaming.sse("output", {"content": buffer})
                    buffer = ""

                continue

            # Anything that is not an answer token is a structural event and
            # goes out immediately - but the token buffer has to drain first,
            # or a progress row would overtake the text preceding it.
            if buffer:
                yield streaming.sse("output", {"content": buffer})
                buffer = ""

            if event:
                yield streaming.sse(event, data)

        if buffer:
            yield streaming.sse("output", {"content": buffer})

    except Exception as exc:
        print(f"CHAT STREAM FAILED | {type(exc).__name__}: {exc}")

        await turns.save(
            turn_id=turn_id,
            session_id=session_id,
            question=question,
            mode=final.get("mode", "unknown"),
            answer="",
            error=str(exc),
        )

        yield streaming.sse("error", {"message": "The request could not be completed."})
        yield streaming.sse("done", {"turn_id": str(turn_id)})

        return

    explain = _explain(final)

    payload = {
        "turn_id": str(turn_id),
        "session_id": str(session_id),
        "question": question,
        "mode": final.get("mode", "direct"),
        "answer": final.get("answer", ""),
        "rows": final.get("rows", []),
        "chart": final.get("chart_config"),
        "chart_skipped": final.get("chart_skipped", ""),
        "explain": explain,
    }

    yield streaming.sse("complete", payload)

    # Persisted after the client has the payload. A failure to write history
    # should not cost the user an answer they have already been given.
    try:
        await turns.save(
            turn_id=turn_id,
            session_id=session_id,
            question=question,
            mode=payload["mode"],
            answer=payload["answer"],
            sql_text=final.get("sql") or None,
            rows=payload["rows"],
            chart=payload["chart"],
            explain=explain,
            error=final.get("sql_error") or None,
        )
    except Exception as exc:
        print(f"TURN NOT SAVED | {type(exc).__name__}: {exc}")

    yield streaming.sse("done", {"turn_id": str(turn_id)})


@router.post("")
async def chat(request: ChatRequest) -> StreamingResponse:
    question = (request.question or "").strip()

    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    session_id = request.session_id or uuid4()

    return StreamingResponse(
        _run(question, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Without this an intermediate proxy will buffer the whole
            # response and deliver it as one block, which looks exactly like a
            # server that does not stream.
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{turn_id}")
async def get_turn(turn_id: UUID) -> Dict[str, Any]:
    turn = await turns.get(turn_id)

    if turn is None:
        raise HTTPException(status_code=404, detail="turn not found")

    return turn


@router.get("/session/{session_id}/history")
async def get_history(session_id: UUID) -> Dict[str, Any]:
    return {"turns": await turns.history(session_id)}
