"""Turning a computed result into a written answer.

How long that answer is follows from the result rather than from a fixed
budget. A single figure gets two sentences; a result with several dimensions
gets a titled, sectioned note with the findings in bullets. The prompt states
those tiers explicitly, because a model given room to write and no rule about
when to use it will pad a one-number answer into a report.

The counterpart to that freedom is the instruction to characterise rather than
recite. The full table is already rendered beside the answer, so prose that
walks the rows one at a time is the same data twice - and the longer the answer
is allowed to be, the more inviting that failure becomes.

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

Match the depth of your answer to the shape of the result:

- A single value, or a single row: two or three sentences, with the figure in **bold**. No headings and no bullets - there is one finding, and a section header over one number is a shape imposed on data that does not have it.
- One dimension (one grouping column and one measure): open with a **bold title line** naming what is shown and the period it covers, then a sentence introducing it, then two to four bullets. Each bullet leads with its **bold** figure.
- Two or more dimensions, or two or more measures: the same opening, then one `### Section` per dimension or measure. Under each, a sentence describing what that part of the data does, then two or three bullets leading with **bold** figures.

Writing it:
- Characterise the data, do not recite it. The highest and the lowest, the gap between them, the range, where a trend turns, which groups sit together, which one is the outlier. Never walk the rows one at a time - the full table is already on screen beside you, and repeating it in prose is the one thing that makes a long answer worse than a short one.
- Percentages, rates and averages in the result are already computed. Report them as they are; never recompute or re-derive.
- Use `backticks` for order IDs, SKUs, warehouse codes and similar identifiers.
- No tables. `###` is the only heading level you may use - never `#` or `##`.
- Do not use citation markers of any kind.
- The data covers {window}. When the question used a relative period like "last month", name the dates it resolved to in the title.
- If the result is empty, say so plainly in a sentence or two and suggest what to ask instead. Do not speculate about what the numbers might have been.
- Do not mention SQL, queries, tables or columns. The user asked a business question.

Tone: a helpful analyst talking to a colleague. Warm and direct. Open by telling them what you are showing, and close with a sentence on what the pattern might reflect when the data supports one - seasonality, a procurement cycle, a carrier that behaves differently from the rest. Skip the empty parts of politeness: no "great question", no "I hope this helps", no offers of further assistance."""

_FORECAST_PROMPT = """Present this demand forecast to the user.

Question: {question}

Forecast:
{forecast}

Structure:
- Open with a **bold title line** naming what is being forecast and over what horizon, then a sentence introducing it.
- If there are notes, they come immediately after that sentence and before anything else. A note explaining that a SKU was too sparse to forecast, or that a category averages only a few orders a month, is the most important thing on the page - burying it below the numbers would misrepresent what was actually computed.
- `### Projection` - the monthly figures, and the total over the horizon stated as a range rather than a single number: "around X units, most likely between Y and Z". The interval is the honest part of this forecast, and an answer that quotes only the midpoint throws it away. **Bold** the range.
- `### Inventory Recommendation` - the recommended quantity in **bold**, and in one clause how safety stock was derived.
- `### Method and Confidence` - the method in plain language, and how much confidence twelve months of history supports. If the projection is flat, say why here: a moving average estimates the level and does not project a trend, so a flat line is a decision rather than a failure.

Writing it:
- No tables. `###` is the only heading level you may use - never `#` or `##`.
- Do not use citation markers of any kind.
- Every figure you state must come from the forecast above. Do not compute your own interval, total or recommendation.

Tone: a helpful analyst talking to a colleague. Warm and direct, and plain about what the numbers can and cannot support. Skip the empty parts of politeness: no "great question", no "I hope this helps", no offers of further assistance."""

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
        # A larger output budget than the other calls. A sectioned answer over
        # a multi-dimension result runs several hundred tokens, and on a
        # thinking model the reasoning is drawn from the same budget - so the
        # default 2048 is close enough to the ceiling to risk truncating an
        # answer mid-section. It is a cap, not a reservation; short answers
        # cost the same as before.
        async for chunk in get_llm(temperature=0.0, max_tokens=4096).astream(prompt):
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
