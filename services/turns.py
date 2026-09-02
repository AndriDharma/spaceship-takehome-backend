"""Reading and writing chat_turn.

The table serves three readers and this module is all of them: the chart
endpoint fetching a turn's SQL back, the router fetching recent turns for
conversational context, and the history endpoint.
"""

import json
from typing import Any, Dict, List, Optional
from uuid import UUID

from core import config, db

_INSERT = """
INSERT INTO chat_turn (
    id, session_id, question, mode, sql_text, answer,
    row_json, chart_config, explain_json, error
)
VALUES (
    :id, :session_id, :question, :mode, :sql_text, :answer,
    CAST(:row_json AS jsonb), CAST(:chart_config AS jsonb),
    CAST(:explain_json AS jsonb), :error
)
"""

_BY_ID = """
SELECT id, session_id, question, mode, sql_text, answer,
       row_json, chart_config, explain_json, error, created_at
FROM   chat_turn
WHERE  id = :id
"""

# Only the columns the router needs. The answer text is deliberately excluded:
# the router resolves what a follow-up refers to, and feeding it a previous
# answer invites it to reuse those numbers instead of querying for new ones.
_RECENT = """
SELECT question, mode, sql_text
FROM   chat_turn
WHERE  session_id = :session_id
  AND  error IS NULL
ORDER  BY created_at DESC
LIMIT  :limit
"""

_HISTORY = """
SELECT id, question, mode, created_at
FROM   chat_turn
WHERE  session_id = :session_id
ORDER  BY created_at DESC
LIMIT  50
"""


def _dumps(value: Any) -> Optional[str]:
    if value is None:
        return None

    return json.dumps(value, ensure_ascii=False, default=str)


async def save(
    turn_id: UUID,
    session_id: UUID,
    question: str,
    mode: str,
    answer: str,
    sql_text: Optional[str] = None,
    rows: Optional[List[Dict[str, Any]]] = None,
    chart: Optional[Dict[str, Any]] = None,
    explain: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    await db.execute(
        _INSERT,
        {
            "id": str(turn_id),
            "session_id": str(session_id),
            "question": question,
            "mode": mode,
            "sql_text": sql_text,
            "answer": answer,
            "row_json": _dumps(rows),
            "chart_config": _dumps(chart),
            "explain_json": _dumps(explain),
            "error": error,
        },
    )


async def update_chart(turn_id: UUID, chart: Optional[Dict[str, Any]]) -> None:
    """Replace a turn's chart after a regenerate."""
    await db.execute(
        "UPDATE chat_turn SET chart_config = CAST(:chart AS jsonb) WHERE id = :id",
        {"id": str(turn_id), "chart": _dumps(chart)},
    )


async def get(turn_id: UUID) -> Optional[Dict[str, Any]]:
    rows = await db.query(_BY_ID, {"id": str(turn_id)})

    return rows[0] if rows else None


async def recent(session_id: UUID) -> List[Dict[str, Any]]:
    """
    The last few turns of this session, oldest first.

    Reversed on the way out because the router prompt reads as a transcript,
    and a transcript that runs backwards is harder for the model to follow than
    one extra list operation is to write.
    """
    rows = await db.query(
        _RECENT,
        {"session_id": str(session_id), "limit": config.MEMORY_TURNS},
    )

    return list(reversed(rows))


async def history(session_id: UUID) -> List[Dict[str, Any]]:
    return await db.query(_HISTORY, {"session_id": str(session_id)})
