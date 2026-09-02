"""Events on the wire.

Graph nodes never touch the socket. They call the emit helpers below, which go
through LangGraph's custom stream writer; the route handler consumes that
stream and turns each event into an SSE frame. A node called outside a stream
context - a unit test, the non-streaming chart endpoint - simply emits nothing.

Event vocabulary:

  progress        one step of the pipeline changed state
  output          a chunk of the answer text
  chart           a finished ChartConfig for the report panel
  chart_skipped   the result was not chartable, with the reason
  complete        the full turn payload, including turn_id
  error           the turn failed
  done            terminal; the client closes the stream
"""

import json
from typing import Any, Dict, Optional

from langgraph.config import get_stream_writer

RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
SKIPPED = "skipped"


def emit(event: str, data: Dict[str, Any]) -> None:
    try:
        get_stream_writer()({"event": event, "data": data})
    except RuntimeError:
        # Not inside graph.astream(). Expected when a node is called directly.
        pass


def progress(
    step: str,
    message: str,
    status: str = RUNNING,
    detail: Optional[Dict[str, Any]] = None,
) -> None:
    """
    One row in the frontend's analysis panel.

    Flat, not a tree. The reference this was adapted from builds a nested
    progress tree keyed by dotted paths, which earns its complexity when a
    pipeline fans out across several data sources; this one has at most five
    steps in a line.

    The rows are also the explainability surface the assignment asks for - the
    user sees which tool was chosen, the SQL that ran, and how many rows came
    back, rather than being told to trust the answer.
    """
    print(f"PROGRESS | {step} | {status} | {message}")

    emit(
        "progress",
        {
            "step": step,
            "message": message,
            "status": status,
            "detail": detail or {},
        },
    )


def output(content: str) -> None:
    if not content:
        return

    emit("output", {"content": content})


def sse(event: str, data: Any) -> str:
    """Serialise one Server-Sent Event frame."""
    payload = json.dumps(data, ensure_ascii=False, default=str)

    return f"event: {event}\ndata: {payload}\n\n"
