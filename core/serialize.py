"""JSON-safe conversion for database rows.

pg8000 returns Decimal for numeric columns and date objects for date ones, and
json.dumps refuses both. Converting once here means the same rows can go into a
prompt, into a chart config, into the SSE payload and into a JSONB column
without four separate encoders.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Any


def jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        # float, not str: every consumer of these rows is arithmetic or a
        # chart axis, and a quoted number breaks both.
        return float(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]

    return value


def column_kind(values: list) -> str:
    """
    Classify a column so the frontend can build the right filter control.

    Returned alongside the data rather than inferred in the browser, because
    the database already knows and a column of all-integers is otherwise
    indistinguishable from a categorical code.
    """
    present = [value for value in values if value is not None]

    if not present:
        return "category"

    sample = present[0]

    if isinstance(sample, bool):
        return "category"

    if isinstance(sample, (int, float, Decimal)):
        return "number"

    if isinstance(sample, (date, datetime)):
        return "date"

    # Dates arrive as ISO strings once jsonable() has run over them.
    if isinstance(sample, str) and len(sample) == 10 and sample[4] == "-":
        return "date"

    return "category"
