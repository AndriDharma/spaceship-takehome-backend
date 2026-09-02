"""Turning a computed result into a sentence.

The one rule this node exists to enforce: it is given the rows and nothing
else. It has no database access, no tools, and no conversation history - the
router got the history, because resolving "break that down by region" is an
interpretation problem, whereas narrating a result is not. A model that cannot
see previous answers cannot quote a number from one, which is what keeps "the
AI must not generate answers without computation" true rather than merely
requested.
"""

import json
from typing import Any, Dict, List

from ai import streaming
from ai.llm import get_llm
from ai.state import GraphState
from core.messages import text_of

# The model has to recognise the shape of the result and read the notable
# values out of it, not recite all of it. The full set goes to the frontend.
_MAX_ROWS_IN_PROMPT = 100

_SQL_PROMPT = """Answer the user's question using ONLY the query result below. Every number you state must appear in these rows.

Question: {question}

Result ({row_count} row(s)){truncation}:
{rows}

Rules:
- Two to four sentences. Use markdown sparingly: **bold** for the figures and names that answer the question, and `backticks` for order IDs and other codes.
- Use a short bullet list only when there are three or more findings of equal weight. For one or two, write prose.
- No headings and no tables. The rows are already shown beside your answer, so a table here is the same data twice.
- Quote the concrete figures that answer the question: the value, the rank, the extreme, the gap.
- Percentages and rates in the data are already computed. Do not recompute or re-derive anything.
- If the result is empty, say so plainly and suggest what to ask instead. Do not speculate about what the numbers might have been.
- The data covers {window}. If the question referred to a relative period, say which dates that resolved to.
- Do not mention SQL, queries, tables or columns. The user asked a business question."""

_FORECAST_PROMPT = """Present this demand forecast to the user.

Question: {question}

Forecast:
{forecast}

Rules:
- If there are notes, lead with them. A note explaining that a SKU was too sparse to forecast, or that a category averages only a few orders a month, is the most important thing on the page - burying it would misrepresent what was computed.
- State the total over the horizon as a range, not as a single number: "around X units, most likely between Y and Z". The interval is the honest part of this forecast and an answer that quotes only the midpoint throws it away.
- State the recommended inventory quantity and, in one clause, how safety stock was derived.
- Name the method in plain language.
- If the projection is flat, say why in one clause - a moving average estimates the level and does not project a trend - so a flat line reads as a decision rather than a failure.
- Close with one sentence on how much confidence twelve months of history supports.
- Four to six sentences. Use markdown sparingly: **bold** for the projected range and the recommended quantity, since those are the two numbers the reader acts on. No headings and no tables."""

_DIRECT_PROMPT = """You are a logistics analytics assistant. Answer the user briefly and honestly.

Question: {question}

Context: {reason}

You can answer questions about orders, deliveries, carriers, regions, product categories and warehouses in a dataset covering {window}, and you can forecast demand by category or region. Say what you can do that is relevant to what they asked. Two or three sentences, or a short bullet list when you are listing several kinds of question - that is the one place a list reads better than prose. **Bold** sparingly. No headings, no tables. Do not invent any figures."""

_ERROR_PROMPT = """A user's question could not be answered against the data.

Question: {question}
What went wrong: {error}

Tell them plainly that this one could not be answered, why in one clause, and suggest a close question that would work against a dataset of orders with carrier, region, product category, warehouse, status and delivery dates covering {window}. Three sentences at most. Put the suggested question in **bold** so it is the thing they see; no other markdown, no headings, no tables. Do not apologise more than once."""


def _rows_for_prompt(rows: List[Dict[str, Any]]) -> str:
    return json.dumps(
        rows[:_MAX_ROWS_IN_PROMPT],
        ensure_ascii=False,
        default=str,
    )


def _build_prompt(state: GraphState) -> str:
    window = state.get("data_window") or "the recorded period"
    question = state.get("question", "")

    mode = state.get("mode", "direct")
    error = state.get("sql_error", "")

    if error and mode == "sql":
        return _ERROR_PROMPT.format(question=question, error=error, window=window)

    if mode == "forecast":
        forecast = state.get("forecast") or {}

        if not forecast.get("ok"):
            return _ERROR_PROMPT.format(
                question=question,
                error=forecast.get("reason", "the forecast could not be computed"),
                window=window,
            )

        return _FORECAST_PROMPT.format(
            question=question,
            forecast=json.dumps(
                {
                    key: value
                    for key, value in forecast.items()
                    # The full history is on the chart; repeating it in the
                    # prompt costs tokens and invites the model to narrate
                    # every month one at a time.
                    if key not in ("history", "chart")
                },
                ensure_ascii=False,
                default=str,
            ),
        )

    if mode == "sql":
        return _SQL_PROMPT.format(
            question=question,
            row_count=state.get("row_count", 0),
            truncation=(
                ", truncated to the row limit" if state.get("truncated") else ""
            ),
            rows=_rows_for_prompt(state.get("rows") or []),
            window=window,
        )

    return _DIRECT_PROMPT.format(
        question=question,
        reason=state.get("route_reason", ""),
        window=window,
    )


async def answer_node(state: GraphState) -> Dict[str, Any]:
    streaming.progress("answer", "Writing the answer")

    prompt = _build_prompt(state)
    collected: List[str] = []

    try:
        async for chunk in get_llm(temperature=0.0).astream(prompt):
            # A streamed chunk carries the same block structure as a complete
            # message, so it needs the same flattening - and the same dropping
            # of reasoning blocks, which would otherwise stream the model's
            # working straight into the answer panel.
            text = text_of(chunk)

            if text:
                collected.append(text)
                streaming.output(text)

    except Exception as exc:
        print(f"ANSWER FAILED | {type(exc).__name__}: {exc}")

        streaming.progress("answer", "The answer could not be written", status=streaming.FAILED)

        fallback = (
            "The result was computed but could not be summarised. "
            "The data is shown in the table beside this message."
        )

        streaming.output(fallback)

        return {"answer": fallback}

    streaming.progress("answer", "Answer complete", status=streaming.COMPLETED)

    return {"answer": "".join(collected)}
