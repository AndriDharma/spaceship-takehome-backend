"""Graph assembly.

    route ─┬─ sql ──→ validate ─┬─ ok ────→ execute ─┬─→ answer ─┬─→ finalize
           │                    ├─ retry ─→ repair          └─→ chart ──┘
           │                    │            ↑                    (parallel)
           │                    │            └────────────────┐
           │                    └─ give up ────────────────→ answer
           ├─ forecast ─→ forecast ─────────────────────────→ answer
           └─ direct ───────────────────────────────────────→ answer

Five working nodes and a join. The fan-out after execute is the only place two
nodes run at once, and it exists so the chart model call overlaps the answer
stream rather than queueing behind it.

finalize does no work. It is a join so that the stream has one unambiguous last
node in every branch, which is what lets the route handler know the turn is
over without tracking which path it took.
"""

from functools import lru_cache
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from ai.nodes.answer import answer_node
from ai.nodes.chart import chart_node
from ai.nodes.forecast import forecast_node
from ai.nodes.router import route_branch, route_node
from ai.nodes.sql import (
    execute_sql_node,
    retry_sql_node,
    validate_branch,
    validate_sql_node,
)
from ai.state import GraphState


async def finalize_node(state: GraphState) -> Dict[str, Any]:
    return {}


def _execute_branch(state: GraphState) -> list:
    """
    Returns the node names to schedule next.

    A list rather than a single name is how LangGraph is told to fan out, so
    the two names come back together and both run in the next superstep.

    A statement that passed validation can still fail at runtime - a timeout,
    a type the planner refuses. There is nothing new for the repair loop to
    work with, so that case schedules only the answer node, which explains
    itself instead of charting an empty result.
    """
    if state.get("sql_error"):
        return ["answer"]

    return ["answer", "chart"]


@lru_cache(maxsize=1)
def compiled():
    builder = StateGraph(GraphState)

    builder.add_node("route", route_node)
    builder.add_node("validate", validate_sql_node)
    builder.add_node("repair", retry_sql_node)
    builder.add_node("execute", execute_sql_node)
    builder.add_node("forecast", forecast_node)
    builder.add_node("answer", answer_node)
    builder.add_node("chart", chart_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "route")

    builder.add_conditional_edges(
        "route",
        route_branch,
        {
            "sql": "validate",
            "forecast": "forecast",
            "direct": "answer",
        },
    )

    builder.add_conditional_edges(
        "validate",
        validate_branch,
        {
            "execute": "execute",
            "retry": "repair",
            # Two failures mean the question cannot be expressed against this
            # schema. The answer node says that, rather than the turn dying.
            "give_up": "answer",
        },
    )

    builder.add_edge("repair", "validate")

    # The fan-out. No path_map: _execute_branch returns node names directly,
    # and returning two of them is what schedules both in one superstep. They
    # then run concurrently because both nodes are async.
    builder.add_conditional_edges("execute", _execute_branch, ["answer", "chart"])

    builder.add_edge("forecast", "answer")

    builder.add_edge("answer", "finalize")
    builder.add_edge("chart", "finalize")

    builder.add_edge("finalize", END)

    return builder.compile()
