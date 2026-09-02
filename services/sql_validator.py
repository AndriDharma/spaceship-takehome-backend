"""Deterministic validation of model-generated SQL.

No LLM runs here and nothing is executed. A statement either parses into a
single read-only SELECT over v_orders using columns that exist, or it is
rejected with a reason specific enough for one repair attempt to act on.

This is the layer the assignment's architecture guidance is about: generated
SQL is never executed unvalidated. The database grants and the read-only
transaction in core/db.py sit behind it, so a gap here is contained rather
than fatal.
"""

import re
from typing import Any, Dict, List, Set

import sqlglot
from sqlglot import exp

from core import config

DIALECT = "postgres"

# Any of these appearing anywhere in the tree ends validation. exp.Command is
# the important one: sqlglot parses statements it has no node for - SET, COPY,
# CALL, VACUUM, GRANT - into Command, so listing it catches the whole class
# rather than an enumeration that goes stale.
#
# Resolved by name rather than imported directly. sqlglot moves node classes
# between releases, and a name that no longer exists would otherwise be an
# ImportError at startup - the validator failing closed by refusing to load at
# all. Every name that does resolve is still enforced.
_FORBIDDEN = tuple(
    node
    for node in (
        getattr(exp, name, None)
        for name in (
            "Insert",
            "Update",
            "Delete",
            "Drop",
            "Create",
            "Alter",
            "Merge",
            "Command",
            "Into",
            "Grant",
            "TruncateTable",
        )
    )
    if isinstance(node, type)
)

# Functions that read outside the database. None are reachable through the
# grants, but rejecting them here produces a clear message instead of a
# privilege error surfacing as a failed turn.
_FORBIDDEN_FUNCTIONS = {
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "lo_import",
    "lo_export",
    "dblink",
    "pg_sleep",
}


def _fail(category: str, reason: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "category": category,
        "reason": reason,
        "sql": "",
        "tables": [],
        "columns": [],
        "filters": "",
        "group_by": [],
        "limit_injected": False,
    }


def clean(sql: str) -> str:
    """Strip the wrapping a model adds even when told not to."""
    text = (sql or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:sql)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    return text.strip().rstrip(";").strip()


def _local_names(tree: exp.Expression) -> Set[str]:
    """
    Names the query defines for itself: CTEs, table aliases, and every output
    alias.

    A column referring to one of these is legitimate even though it is not a
    column of v_orders - `SELECT carrier, COUNT(*) AS n FROM v_orders GROUP BY
    carrier ORDER BY n` is the ordinary case. Collecting them is what keeps
    the unknown-column check from rejecting correct SQL.
    """
    names: Set[str] = set()

    for cte in tree.find_all(exp.CTE):
        if cte.alias:
            names.add(cte.alias.lower())

    for alias in tree.find_all(exp.Alias):
        if alias.alias:
            names.add(alias.alias.lower())

    for table in tree.find_all(exp.Table):
        if table.alias:
            names.add(table.alias.lower())

    return names


def validate(sql: str, allowed_columns: Set[str]) -> Dict[str, Any]:
    statement = clean(sql)

    if not statement:
        return _fail("empty", "No SQL was produced.")

    try:
        parsed = sqlglot.parse(statement, read=DIALECT)
    except Exception as exc:
        return _fail("parse_error", f"The SQL did not parse: {exc}")

    parsed = [tree for tree in parsed if tree is not None]

    if len(parsed) != 1:
        return _fail(
            "multiple_statements",
            f"Exactly one statement is allowed; {len(parsed)} were found.",
        )

    tree = parsed[0]

    # WITH ... SELECT parses as a Select carrying a `with` argument, so this
    # covers CTEs without a separate branch.
    if not isinstance(tree, (exp.Select, exp.Union, exp.Subquery)):
        return _fail(
            "not_a_select",
            f"Only SELECT queries are allowed; this is {type(tree).__name__.upper()}.",
        )

    for node_type in _FORBIDDEN:
        found = tree.find(node_type)

        if found is not None:
            return _fail(
                "forbidden_statement",
                f"{type(found).__name__.upper()} is not allowed. "
                "The query must only read data.",
            )

    for function in tree.find_all(exp.Anonymous):
        name = str(function.this or "").lower()

        if name in _FORBIDDEN_FUNCTIONS:
            return _fail("forbidden_function", f"The function {name} is not allowed.")

    local = _local_names(tree)

    # ------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------

    tables: List[str] = []
    unknown_tables: List[str] = []

    for table in tree.find_all(exp.Table):
        name = str(table.name or "").lower()

        if not name or name in local:
            continue

        tables.append(name)

        if name not in config.ALLOWED_TABLES:
            unknown_tables.append(name)

    if unknown_tables:
        allowed = ", ".join(sorted(config.ALLOWED_TABLES))
        return _fail(
            "unknown_table",
            f"These tables are not available: {', '.join(sorted(set(unknown_tables)))}. "
            f"Only {allowed} may be queried, and it already contains every metric.",
        )

    if not tables:
        return _fail("no_table", "The query does not read from v_orders.")

    # ------------------------------------------------------------
    # Columns
    # ------------------------------------------------------------

    lowered = {name.lower() for name in allowed_columns}

    columns: List[str] = []
    unknown_columns: List[str] = []

    for column in tree.find_all(exp.Column):
        name = str(column.name or "").lower()

        if not name or name == "*":
            continue

        columns.append(name)

        if name not in lowered and name not in local:
            unknown_columns.append(name)

    if unknown_columns:
        return _fail(
            "unknown_column",
            f"These columns do not exist: {', '.join(sorted(set(unknown_columns)))}. "
            "Use only the columns listed in the schema.",
        )

    # ------------------------------------------------------------
    # Row cap
    # ------------------------------------------------------------

    limit_injected = False

    if tree.args.get("limit") is None:
        try:
            tree = tree.limit(config.SQL_MAX_ROWS)
            limit_injected = True
        except Exception:
            # Some shapes have no limit builder. The executor caps the fetch
            # regardless, so this is a lost optimisation, not a lost guarantee.
            pass

    where = tree.find(exp.Where)

    group_by = [
        expression.sql(dialect=DIALECT)
        for expression in (tree.args.get("group") or exp.Group()).expressions
    ]

    return {
        "ok": True,
        "category": "valid",
        "reason": "",
        "sql": tree.sql(dialect=DIALECT, pretty=True),
        "tables": sorted(set(tables)),
        "columns": sorted(set(columns)),
        "filters": where.this.sql(dialect=DIALECT) if where else "",
        "group_by": group_by,
        "limit_injected": limit_injected,
    }
