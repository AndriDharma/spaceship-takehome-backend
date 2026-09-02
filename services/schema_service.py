"""What the model is told about the data, read from a file at startup.

This used to run seven queries against the live database on every process
start. It now reads schema/v_orders.yaml instead, for two reasons:

  * Cloud Run runs this with min-instances=0, so every cold start paid a Cloud
    SQL connect plus seven round-trips before the first token. The file removes
    the database from the boot path entirely.
  * The metric semantics - that a delay rate divides by completed orders, that
    order_value_usd is already extended - are prompt content. Keeping them in a
    version-controlled file means a change to what the model is told shows up
    in a diff, instead of living invisibly inside a COMMENT ON COLUMN.

The trade is that the file is now authoritative for the validator's column
whitelist, so it has to be regenerated whenever the view changes. _validate()
below is what makes that failure loud rather than silent.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core import config


class SchemaFileError(RuntimeError):
    """The schema file is missing, malformed, or internally inconsistent."""


# Columns whose complete value domain is listed in the prompt. Anything with a
# "values" key in the file qualifies; the file decides, not this module.
_cache: Optional[Dict[str, Any]] = None


def _collapse(text: Any) -> str:
    """YAML folded scalars keep their line breaks. The prompt does not want them."""
    return " ".join(str(text or "").split())


def _validate(table: str, body: Dict[str, Any], path: Path) -> None:
    """
    Reject a file that cannot safely drive the validator.

    The check that matters is the last one. Every other failure here is a typo
    that shows up immediately; a `columns` block that has drifted out of step
    with `features` produces a whitelist that silently disagrees with the view,
    and the symptom would be arbitrary queries being rejected days later.
    """
    if table not in config.ALLOWED_TABLES:
        raise SchemaFileError(
            f"{path} documents table {table!r}, which is not in "
            f"ALLOWED_TABLES ({sorted(config.ALLOWED_TABLES)}). "
            "Generated SQL against it would be rejected by the validator."
        )

    columns = body.get("columns")

    if not isinstance(columns, dict) or not columns:
        raise SchemaFileError(f"{path}: tables.{table}.columns is missing or empty.")

    window = body.get("data_window")

    if not isinstance(window, dict):
        raise SchemaFileError(f"{path}: tables.{table}.data_window is missing.")

    missing = [
        key
        for key in ("min_date", "max_date", "row_count", "anchor_date")
        if window.get(key) in (None, "")
    ]

    if missing:
        raise SchemaFileError(
            f"{path}: data_window is missing {', '.join(missing)}. "
            "anchor_date is what relative dates resolve against - without it "
            "the assistant would answer 'last month' against the wall clock."
        )

    features = (body.get("semantics") or {}).get("features") or {}

    undocumented = sorted(set(columns) - set(features))
    orphaned = sorted(set(features) - set(columns))

    if undocumented or orphaned:
        parts = []

        if undocumented:
            parts.append(f"columns with no feature entry: {', '.join(undocumented)}")

        if orphaned:
            parts.append(f"features with no column: {', '.join(orphaned)}")

        raise SchemaFileError(
            f"{path} is internally inconsistent - {'; '.join(parts)}. "
            "Regenerate it against the current view."
        )


async def load() -> Dict[str, Any]:
    """
    Read the schema file once. Called from the application lifespan.

    Async only so main.py's call site did not have to change. The read is a
    single small file on local disk before the server accepts traffic, so
    pushing it to a thread would buy nothing.
    """
    global _cache

    if _cache is not None:
        return _cache

    path = Path(config.SCHEMA_FILE)

    if not path.is_file():
        raise SchemaFileError(f"Schema file not found at {path}.")

    document = yaml.safe_load(path.read_text(encoding="utf-8"))

    tables = (document or {}).get("tables")

    if not isinstance(tables, dict) or len(tables) != 1:
        raise SchemaFileError(
            f"{path}: expected exactly one table under 'tables', "
            f"found {len(tables) if isinstance(tables, dict) else 0}."
        )

    table, body = next(iter(tables.items()))

    _validate(table, body, path)

    declared: Dict[str, str] = body["columns"]
    features: Dict[str, Any] = body["semantics"]["features"]
    window: Dict[str, Any] = body["data_window"]

    columns: List[Dict[str, Any]] = []
    enums: Dict[str, List[str]] = {}

    # Declaration order is the file's order, which is the view's order. Python
    # dicts preserve it and so does yaml.safe_load, so the prompt reads
    # top-to-bottom like the DDL does.
    for name, data_type in declared.items():
        feature = features.get(name) or {}
        values = feature.get("values")

        columns.append(
            {
                "name": name,
                "data_type": data_type,
                "description": _collapse(feature.get("description")),
                "metric_semantics": _collapse(feature.get("metric_semantics")),
                "values": values,
            }
        )

        if values:
            enums[name] = list(values)

    _cache = {
        "table": table,
        "schema": body.get("schema", "public"),
        "description": _collapse(body.get("description")),
        "columns": columns,
        "column_names": set(declared),
        "min_date": window["min_date"],
        "max_date": window["max_date"],
        "row_count": window["row_count"],
        "anchor": window["anchor_date"],
        "anchor_note": _collapse(window.get("anchor_note")),
        "enums": enums,
        "rules": [
            _collapse(rule)
            for rule in (body["semantics"].get("computation_rules") or {}).values()
        ],
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
    return cached()["anchor"]


def schema_block() -> str:
    """
    The schema as the router prompt sees it.

    Deliberately narrower than the file. The file also carries synonyms,
    dimensions and filters, which exist so a retrieval layer can pick a table
    out of many - there is one table here and all of its columns go into every
    prompt, so that metadata would cost tokens per turn and change nothing.
    """
    meta = cached()

    lines = [
        f"Table: {meta['schema']}.{meta['table']}",
        "(the only table you may query - it already contains every metric you need)",
        "",
        meta["description"],
        "",
        f"Data window: {meta['min_date']} to {meta['max_date']} "
        f"({meta['row_count']} orders).",
        meta["anchor_note"],
        "",
        "Columns:",
    ]

    for column in meta["columns"]:
        lines.append(f"  {column['name']}  {column['data_type']}")

        detail = " ".join(
            part
            for part in (column["description"], column["metric_semantics"])
            if part
        )

        if detail:
            lines.append(f"    {detail}")

        # The complete domain, not a sample. A near-miss on a literal returns
        # zero rows rather than an error, which reads to the user as a valid
        # answer of "none" - listing every value is what makes that impossible.
        if column["values"]:
            rendered = ", ".join(str(value) for value in column["values"])
            lines.append(f"    Values: {rendered}")

    if meta["rules"]:
        lines.append("")
        lines.append("Rules for computing metrics from these columns:")

        for rule in meta["rules"]:
            lines.append(f"  - {rule}")

    return "\n".join(lines)
