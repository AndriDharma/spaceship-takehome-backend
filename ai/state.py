"""The object every node reads from and writes to.

One rule governs the shape of this: after execute_sql the graph fans out to
answer_node and chart_node, which run concurrently in the same superstep.
LangGraph raises on two branches writing the same key without a reducer, so the
parallel pair writes disjoint keys - answer_node owns `answer`, chart_node owns
`chart_config` and `chart_skipped`, and neither touches anything the other
writes.
"""

from typing import Any, Dict, List, Optional, TypedDict


class GraphState(TypedDict, total=False):
    # --- input ---
    question: str
    session_id: str
    history: List[Dict[str, Any]]

    # --- context assembled before the graph runs ---
    schema_text: str
    anchor_date: str
    # Human-readable period, e.g. "2025-01-01 to 2025-12-30". Handed to the
    # answer node so it can tell the user which dates a relative phrase like
    # "last month" actually resolved to.
    data_window: str

    # --- router output ---
    mode: str
    route_reason: str
    forecast_params: Dict[str, Any]

    # --- sql branch ---
    sql: str
    sql_error: str
    sql_error_category: str
    retries: int
    validation: Dict[str, Any]

    # --- result ---
    rows: List[Dict[str, Any]]
    headers: List[str]
    row_count: int
    truncated: bool

    # --- forecast branch ---
    forecast: Dict[str, Any]

    # --- parallel branch outputs, disjoint by construction ---
    answer: str
    chart_config: Optional[Dict[str, Any]]
    chart_skipped: str

    # --- terminal ---
    error: str
