"""The one interpretation call.

This node does the whole of "AI interpretation → tool selection → structured
input" in a single model call: it decides which tool answers the question and,
when that tool is the query tool, writes the statement. Splitting the decision
from the generation would double the latency in front of the first token
without making either decision better, because both need exactly the same
context - the schema and the conversation.

What it returns is a proposal, not an instruction. Nothing here is trusted:
the SQL goes to the validator, the forecast parameters go through a Pydantic
model, and an unparseable response degrades to a direct answer rather than
failing the turn.
"""

import json
import re
from typing import Any, Dict

from ai import streaming
from ai.llm import get_llm
from ai.state import GraphState
from core.messages import text_of
from models.schemas import RouteDecision

_PROMPT = """You are the router for a logistics analytics assistant. Decide which tool answers the user's question, and return ONLY a JSON object - no markdown, no prose.

{{
  "mode": "sql" | "forecast" | "direct",
  "sql": string or null,
  "forecast": object or null,
  "reason": string
}}

Choosing a mode:
- "sql" - the question is about what has already happened: counts, totals, rates, rankings, breakdowns, or trends over the recorded data. Write one PostgreSQL SELECT into "sql".
- "forecast" - the question is about future demand, projected volume, or how much inventory to plan. Fill "forecast" and leave "sql" null.
- "direct" - a greeting, a question about what you can do, or a question this dataset cannot answer. Leave both null and explain in "reason".

Rules for "sql":
- Exactly one SELECT statement. No semicolon, no markdown fence, no explanation.
- Query only v_orders. It is a view that already contains every metric - never compute a delay flag or a delivery duration yourself, use the columns provided.
- Use only the column names listed in the schema, spelled exactly as shown.
- Rates are SUM(is_on_time) / NULLIF(SUM(is_completed), 0), never COUNT(*), because COUNT(*) would include orders that are still in transit.
- Group by order_month or order_week for a time trend. Do not call date_trunc yourself.
- Alias every aggregate to a readable name: AS delay_rate_pct, AS total_orders.
- ORDER BY whatever the question ranks or sequences.
- Return the columns needed to answer and to plot, and nothing else. Two or three columns is usually right.

Rules for "forecast":
{{
  "level": "overall" | "category" | "region",
  "key": string or null,
  "horizon": integer between 1 and 12,
  "method": "moving_average" | "linear_regression",
  "requested_sku": string or null
}}
- If the user names a specific SKU, put it verbatim in "requested_sku" and set level to "overall". The application resolves it - do not guess a category.
- "key" is the category or region name when level is "category" or "region".
- Default horizon is 3 when the user does not say.
- Use "linear_regression" when the user asks about a trend or direction, "moving_average" otherwise.

Schema:
{schema}

{history}

User question: {question}"""


def _history_block(history: list) -> str:
    if not history:
        return "This is the first question in the conversation."

    lines = ["Earlier in this conversation (oldest first):"]

    for turn in history:
        lines.append(f"- Q: {turn.get('question', '')}")

        if turn.get("sql_text"):
            compact = " ".join(str(turn["sql_text"]).split())
            lines.append(f"  SQL: {compact}")

    lines.append(
        "Use these only to resolve what a follow-up refers to. "
        "Do not reuse their numbers - every answer must come from a fresh query."
    )

    return "\n".join(lines)


def _parse(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object in the router response.")

    return json.loads(cleaned[start : end + 1])


async def route_node(state: GraphState) -> Dict[str, Any]:
    streaming.progress("route", "Interpreting the question")

    prompt = _PROMPT.format(
        schema=state.get("schema_text", ""),
        history=_history_block(state.get("history") or []),
        question=state.get("question", ""),
    )

    try:
        response = await get_llm().ainvoke(prompt)
        decision = RouteDecision(**_parse(text_of(response)))

    except Exception as exc:
        print(f"ROUTER FAILED | {type(exc).__name__}: {exc}")

        streaming.progress(
            "route",
            "Could not interpret the question as a data query",
            status=streaming.FAILED,
        )

        # A router that fails is not a turn that fails. The answer node can
        # still say something useful about what this assistant can do.
        return {
            "mode": "direct",
            "route_reason": "The question could not be interpreted as a data query.",
        }

    label = {
        "sql": "Query tool selected",
        "forecast": "Forecasting tool selected",
        "direct": "Answering directly - no computation needed",
    }.get(decision.mode, decision.mode)

    streaming.progress(
        "route",
        label,
        status=streaming.COMPLETED,
        detail={"mode": decision.mode, "reason": decision.reason},
    )

    return {
        "mode": decision.mode,
        "route_reason": decision.reason,
        "sql": decision.sql or "",
        "forecast_params": (
            decision.forecast.model_dump() if decision.forecast else {}
        ),
        "retries": 0,
    }


def route_branch(state: GraphState) -> str:
    mode = state.get("mode", "direct")

    if mode == "sql" and state.get("sql"):
        return "sql"

    if mode == "forecast":
        return "forecast"

    return "direct"
