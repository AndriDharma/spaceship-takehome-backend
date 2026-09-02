"""Validate, repair once, execute.

The repair loop is capped at one attempt on purpose. In practice a second
failure is almost never the same kind of mistake as the first - it means the
question cannot be expressed against this schema - and each extra attempt is a
full model call sitting in front of the user's first token. One retry catches
the typo case; beyond that, saying so honestly is the better answer.
"""

from typing import Any, Dict

from ai import streaming
from ai.llm import get_llm
from ai.state import GraphState
from core import db
from services import schema_service, sql_validator

MAX_RETRIES = 1

_REPAIR_PROMPT = """The PostgreSQL query you wrote was rejected.

Query:
{sql}

Reason:
{error}

Rewrite it so it passes. Return ONLY the corrected SELECT statement - no markdown, no explanation, no semicolon.

Schema:
{schema}

The question it must answer: {question}"""


async def validate_sql_node(state: GraphState) -> Dict[str, Any]:
    streaming.progress("validate", "Validating the query")

    result = sql_validator.validate(
        state.get("sql", ""),
        schema_service.cached()["column_names"],
    )

    if result["ok"]:
        streaming.progress(
            "validate",
            "Query validated",
            status=streaming.COMPLETED,
            detail={
                "sql": result["sql"],
                "tables": result["tables"],
                "columns": result["columns"],
                "filters": result["filters"],
                "limit_injected": result["limit_injected"],
            },
        )

        return {
            "sql": result["sql"],
            "validation": result,
            "sql_error": "",
            "sql_error_category": "",
        }

    streaming.progress(
        "validate",
        f"Query rejected: {result['reason']}",
        status=streaming.FAILED,
        detail={"category": result["category"]},
    )

    return {
        "validation": result,
        "sql_error": result["reason"],
        "sql_error_category": result["category"],
    }


def validate_branch(state: GraphState) -> str:
    if not state.get("sql_error"):
        return "execute"

    if state.get("retries", 0) < MAX_RETRIES:
        return "retry"

    return "give_up"


async def retry_sql_node(state: GraphState) -> Dict[str, Any]:
    attempt = state.get("retries", 0) + 1

    streaming.progress("retry", f"Rewriting the query (attempt {attempt + 1})")

    prompt = _REPAIR_PROMPT.format(
        sql=state.get("sql", ""),
        error=state.get("sql_error", ""),
        schema=state.get("schema_text", ""),
        question=state.get("question", ""),
    )

    try:
        response = await get_llm().ainvoke(prompt)
        repaired = sql_validator.clean(response.content)

    except Exception as exc:
        print(f"SQL REPAIR FAILED | {type(exc).__name__}: {exc}")
        repaired = ""

    return {"sql": repaired, "retries": attempt}


async def execute_sql_node(state: GraphState) -> Dict[str, Any]:
    streaming.progress("execute", "Running the query")

    try:
        rows, truncated = await db.run_generated_sql(state["sql"])

    except Exception as exc:
        # The validator passed it, so this is a runtime failure - a timeout, a
        # type mismatch, a division the planner would not accept. Reported as
        # an error rather than retried, because the repair loop has no new
        # information to work with.
        print(f"SQL EXECUTION FAILED | {type(exc).__name__}: {exc}")

        streaming.progress(
            "execute",
            "The query failed to run",
            status=streaming.FAILED,
            detail={"error": str(exc)},
        )

        return {
            "rows": [],
            "headers": [],
            "row_count": 0,
            "truncated": False,
            "sql_error": str(exc),
            "sql_error_category": "execution_error",
        }

    headers = list(rows[0].keys()) if rows else []

    streaming.progress(
        "execute",
        f"{len(rows)} row(s) returned" + (" (truncated)" if truncated else ""),
        status=streaming.COMPLETED,
        detail={"row_count": len(rows), "truncated": truncated},
    )

    return {
        "rows": rows,
        "headers": headers,
        "row_count": len(rows),
        "truncated": truncated,
    }
