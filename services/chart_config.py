"""Choosing a chart for a result set.

The model picks a configuration; it never writes rendering code and never sees
the rendering. Its output is validated against the actual column names before
it goes anywhere, and if it is unusable a deterministic fallback covers the
result instead - a chart is a presentation detail, and no chart failure should
ever cost the user their answer.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from core import config
from core.messages import text_of
from core.serialize import column_kind
from models.schemas import FullChartConfig

_VALID_TYPES = {"bar", "line", "area", "stacked_bar", "pie", "doughnut"}

_SYSTEM_PROMPT = """You are a data visualization expert. Given a dataset and the question it answers, respond with ONLY a JSON object - no markdown, no prose.

Schema:
{
  "chartType": "bar" | "line" | "area" | "stacked_bar" | "pie" | "doughnut",
  "title": string,
  "description": string,
  "xKey": string,
  "yKeys": string[],
  "seriesKey": string | null,
  "stacked": boolean,
  "insight": string
}

Choosing a type:
- A date or month column on the x axis means "line", or "area" for a single cumulative measure.
- A categorical x axis means "bar".
- "pie" or "doughnut" only for parts of one whole, and only with 5 or fewer rows.
- "stacked_bar" only when seriesKey is set AND the measure is additive - a count, a total, a quantity, an amount.

Series rules:
- yKeys are columns. seriesKey is a column whose VALUES each become a series. They are not the same thing.
- Set seriesKey only when the same xKey value appears on several rows, once per category.
- When seriesKey is set, yKeys must contain exactly one column.
- seriesKey must differ from xKey and must not appear in yKeys.
- Never stack an average, a rate, a percentage or a ratio. Those do not sum to anything real - use "line".

Insight rules:
- One sentence, about what the numbers show: a rank, an extreme, a gap, a trend.
- Cite concrete values.
- Do not describe the chart. Never write "this chart shows" or "makes it easy to see".

Rules:
- xKey and yKeys must be exact column names from the dataset, unchanged.
- yKeys must be numeric columns and must not include xKey.
- Do not invent column names.
- Write the title and insight in readable prose: "On-Time Rate by Carrier", not "on_time_rate".
- Return JSON only."""


def _numeric_columns(rows: List[Dict[str, Any]], headers: List[str]) -> List[str]:
    return [
        header
        for header in headers
        if column_kind([row.get(header) for row in rows]) == "number"
    ]


def is_chartable(rows: List[Dict[str, Any]], headers: List[str]) -> Tuple[bool, str]:
    """
    Decide whether a chart is worth an LLM call, deterministically and before
    spending one.

    The reference this was adapted from discovers the same three failures the
    expensive way - it generates a config, tries to render, and drops the chart
    when it throws. Checking first costs nothing and means a single-number
    answer like "how many orders were delivered late last month" does not
    quietly burn a model call to produce nothing.
    """
    if len(rows) < config.CHART_MIN_ROWS:
        return False, f"Only {len(rows)} row(s) - a single value is not a chart."

    if len(headers) > config.CHART_MAX_COLUMNS:
        return False, f"{len(headers)} columns is too wide to plot meaningfully."

    if not _numeric_columns(rows, headers):
        return False, "No numeric column to plot."

    return True, ""


def column_kinds(rows: List[Dict[str, Any]], headers: List[str]) -> Dict[str, str]:
    return {
        header: column_kind([row.get(header) for row in rows]) for header in headers
    }


def _parse_json(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object in the response.")

    return json.loads(cleaned[start : end + 1])


def _coerce(
    raw: Dict[str, Any],
    rows: List[Dict[str, Any]],
    headers: List[str],
) -> Dict[str, Any]:
    """
    Force the model's choice to be renderable, or raise.

    Everything here is a rule the prompt already states. It is repeated as code
    because a prompt is a request and this is a guarantee: the frontend renders
    whatever comes back, so a yKey that is not a column produces an empty chart
    with no error anywhere.
    """
    lookup = {header.lower(): header for header in headers}
    numeric = set(_numeric_columns(rows, headers))

    chart_type = str(raw.get("chartType", "")).strip().lower()

    if chart_type not in _VALID_TYPES:
        raise ValueError(f"Unsupported chartType {chart_type!r}.")

    x_key = lookup.get(str(raw.get("xKey", "")).strip().lower())

    if not x_key:
        raise ValueError("xKey is not a column of the result.")

    y_keys = [
        lookup[key]
        for key in (
            str(item).strip().lower() for item in (raw.get("yKeys") or [])
        )
        if key in lookup and lookup[key] != x_key and lookup[key] in numeric
    ]

    if not y_keys:
        raise ValueError("No usable numeric yKeys.")

    series_key_raw = raw.get("seriesKey")
    series_key = (
        lookup.get(str(series_key_raw).strip().lower()) if series_key_raw else None
    )

    if series_key in (x_key, *y_keys):
        series_key = None

    # A series column splits one measure. With more than one measure there is
    # nothing coherent to split, so the split is dropped rather than the
    # measures.
    if series_key and len(y_keys) > 1:
        series_key = None

    if chart_type in ("pie", "doughnut"):
        y_keys = y_keys[:1]
        series_key = None

    stacked = bool(raw.get("stacked", False)) and chart_type == "stacked_bar"

    if chart_type == "stacked_bar" and not series_key:
        # Nothing to stack. A plain bar is what was actually asked for.
        chart_type = "bar"
        stacked = False

    return {
        "chartType": chart_type,
        "title": str(raw.get("title") or "Result"),
        "description": str(raw.get("description") or ""),
        "xKey": x_key,
        "yKeys": y_keys,
        "seriesKey": series_key,
        "stacked": stacked,
        "insight": str(raw.get("insight") or ""),
    }


def fallback(
    rows: List[Dict[str, Any]],
    headers: List[str],
    title: str,
) -> Optional[Dict[str, Any]]:
    """
    A chart chosen without a model, used when the model's config will not
    validate.

    Deliberately dull: first non-numeric column across the x axis, first
    numeric column up it, line if the axis is a date and bar otherwise. It is
    not the best chart for the data, but it is always a correct one.
    """
    kinds = column_kinds(rows, headers)

    numeric = [header for header in headers if kinds[header] == "number"]
    other = [header for header in headers if kinds[header] != "number"]

    if not numeric or not other:
        return None

    x_key = other[0]

    return {
        "chartType": "line" if kinds[x_key] == "date" else "bar",
        "title": title,
        "description": "Chart selected automatically.",
        "xKey": x_key,
        "yKeys": [numeric[0]],
        "seriesKey": None,
        "stacked": False,
        "insight": "",
    }


async def build(
    rows: List[Dict[str, Any]],
    headers: List[str],
    question: str,
    llm: Any,
) -> Tuple[Optional[FullChartConfig], str]:
    """
    Returns (chart, skip_reason). Exactly one is meaningful.
    """
    chartable, reason = is_chartable(rows, headers)

    if not chartable:
        return None, reason

    sample = rows[: config.CHART_SAMPLE_ROWS]

    user_message = (
        f"Dataset columns: {', '.join(headers)}\n"
        f"First {len(sample)} of {len(rows)} rows: "
        f"{json.dumps(sample, ensure_ascii=False, default=str)}\n\n"
        f"The question this data answers: {question}"
    )

    chosen: Optional[Dict[str, Any]] = None

    try:
        response = await llm.ainvoke(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", user_message),
            ]
        )

        chosen = _coerce(_parse_json(text_of(response)), rows, headers)

    except Exception as exc:
        print(f"CHART CONFIG REJECTED | {type(exc).__name__}: {exc}")
        chosen = fallback(rows, headers, question[:80])

    if chosen is None:
        return None, "Could not select a chart for this result."

    return (
        FullChartConfig(
            **chosen,
            data=rows,
            headers=headers,
            columnKinds=column_kinds(rows, headers),
        ),
        "",
    )
