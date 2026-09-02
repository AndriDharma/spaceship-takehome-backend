"""What the model is told about the data, loaded once at startup.

Three things are read from the live database rather than hard-coded:

  * the column list, which becomes the validator's whitelist as well as the
    schema block in the prompt;
  * the column comments, which is where the metric semantics live - see
    db/01_v_orders.sql. Keeping them in the catalog means the rules cannot
    drift from the view that implements them;
  * the data window and the distinct values of the low-cardinality columns.

The last one matters more than it looks. Without it the model guesses at
literals - 'DHL Express' for a carrier stored as 'DHL', 'Delivered' for
'delivered' - and returns an empty result that looks like a correct answer.
"""

from typing import Any, Dict, List, Optional

from core import db

# Columns whose values are worth listing in full. All are small enough that
# the entire domain fits in a few tokens.
_ENUM_COLUMNS = [
    "status",
    "carrier",
    "region",
    "product_category",
    "warehouse",
]

_COLUMN_SQL = """
SELECT a.attname                                  AS name,
       format_type(a.atttypid, a.atttypmod)       AS data_type,
       col_description(a.attrelid, a.attnum)      AS comment
FROM   pg_attribute a
WHERE  a.attrelid = 'v_orders'::regclass
  AND  a.attnum > 0
  AND  NOT a.attisdropped
ORDER  BY a.attnum
"""

_WINDOW_SQL = """
SELECT MIN(order_date) AS min_date,
       MAX(order_date) AS max_date,
       COUNT(*)        AS row_count
FROM   v_orders
"""

_cache: Optional[Dict[str, Any]] = None


async def load() -> Dict[str, Any]:
    """Read the schema once. Called from the application lifespan."""
    global _cache

    if _cache is not None:
        return _cache

    columns = await db.query(_COLUMN_SQL)
    window = (await db.query(_WINDOW_SQL))[0]

    enums: Dict[str, List[str]] = {}

    for column in _ENUM_COLUMNS:
        # The column name is from the fixed list above, never from user input.
        rows = await db.query(
            f"SELECT DISTINCT {column} AS value FROM v_orders "
            f"WHERE {column} IS NOT NULL ORDER BY 1"
        )
        enums[column] = [row["value"] for row in rows]

    _cache = {
        "columns": columns,
        "column_names": {column["name"] for column in columns},
        "min_date": window["min_date"],
        "max_date": window["max_date"],
        "row_count": window["row_count"],
        "enums": enums,
    }

    return _cache


def cached() -> Dict[str, Any]:
    if _cache is None:
        raise RuntimeError("schema_service.load() has not run yet")

    return _cache


def anchor_date() -> str:
    """
    The date every relative period is resolved against.

    Not today. The dataset ends in December 2025 and the wall clock is well
    past that, so resolving "last month" against now() would return nothing on
    exactly the questions the assignment gives as examples. The latest
    order_date is what "now" means for this data, and the answer says so.
    """
    return cached()["max_date"]


def schema_block() -> str:
    """The schema as the router prompt sees it."""
    meta = cached()

    lines = [
        "Table: v_orders",
        "(the only table you may query - it already contains every metric you need)",
        "",
    ]

    for column in meta["columns"]:
        comment = column["comment"] or ""
        suffix = f"  -- {comment}" if comment else ""
        lines.append(f"  {column['name']} {column['data_type']}{suffix}")

    lines.append("")
    lines.append("Exact values for the categorical columns:")

    for name, values in meta["enums"].items():
        lines.append(f"  {name}: {', '.join(str(value) for value in values)}")

    lines.append("")
    lines.append(
        f"Data window: {meta['min_date']} to {meta['max_date']} "
        f"({meta['row_count']} orders)."
    )
    lines.append(
        f"Treat {meta['max_date']} as today. "
        "\"Last month\", \"the last 3 months\" and similar phrases are relative "
        "to that date, not to the current calendar date."
    )

    return "\n".join(lines)
