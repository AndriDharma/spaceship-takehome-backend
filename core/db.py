"""Cloud SQL connection and the two ways SQL reaches the database.

There is one database role and one connection pool. What separates trusted
application SQL from model-generated SQL is not the credential, it is which of
the two functions below is used - and the guardrails one of them applies.

Authentication is Application Default Credentials in both environments. Locally
GOOGLE_APPLICATION_CREDENTIALS points at the service account file; on Cloud Run
the attached service account is served from the metadata server. The connector
calls google.auth.default() either way, so there is no branch here.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import sqlalchemy
from google.cloud.sql.connector import Connector, IPTypes
from sqlalchemy import text

from core import config
from core.serialize import jsonable

_connector: Optional[Connector] = None
_engine: Optional[sqlalchemy.Engine] = None


def _getconn():
    return _connector.connect(
        config.INSTANCE_CONNECTION_NAME,
        "pg8000",
        user=config.DB_USER,
        password=config.DB_PASS,
        db=config.DB_NAME,
        # Public IP keeps the deployment to one Cloud Run service with no VPC
        # connector. The connector still tunnels over TLS with an ephemeral
        # client certificate, so nothing is exposed by this.
        ip_type=IPTypes.PUBLIC,
    )


def init_engine() -> sqlalchemy.Engine:
    global _connector, _engine

    if _engine is not None:
        return _engine

    _connector = Connector()

    _engine = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=_getconn,
        # Cloud Run scales to a small number of concurrent requests and the
        # dataset is 400 rows; a large pool would only hold idle connections
        # open against the smallest Cloud SQL tier.
        pool_size=5,
        max_overflow=2,
        pool_pre_ping=True,
        # Cloud SQL drops idle connections, and a container that has been warm
        # for a while will otherwise hand out a dead one on the next request.
        pool_recycle=1800,
    )

    return _engine


def close_engine() -> None:
    global _connector, _engine

    if _engine is not None:
        _engine.dispose()
        _engine = None

    if _connector is not None:
        _connector.close()
        _connector = None


def engine() -> sqlalchemy.Engine:
    if _engine is None:
        return init_engine()

    return _engine


# ------------------------------------------------------------
# Trusted application SQL
# ------------------------------------------------------------


def _run(sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    with engine().connect() as conn:
        result = conn.execute(text(sql), params or {})

        return [
            {key: jsonable(value) for key, value in row._mapping.items()}
            for row in result
        ]


def _write(sql: str, params: Optional[Dict[str, Any]] = None) -> None:
    with engine().begin() as conn:
        conn.execute(text(sql), params or {})


async def query(sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Run SQL this codebase wrote: dashboard aggregates, schema introspection,
    conversation history. Parameterised, never interpolated.

    The driver is synchronous, so every call is pushed to a worker thread. At
    this data size the alternative - an async driver plus an async connector -
    would be more moving parts for no measurable gain, and blocking the event
    loop would stall the SSE stream that is running concurrently.
    """
    return await asyncio.to_thread(_run, sql, params)


async def execute(sql: str, params: Optional[Dict[str, Any]] = None) -> None:
    await asyncio.to_thread(_write, sql, params)


# ------------------------------------------------------------
# Model-generated SQL
# ------------------------------------------------------------


def _run_generated(sql: str) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Execute a statement the model wrote, inside a transaction that cannot
    write and cannot run long.

    This is the third of three independent layers. The validator has already
    rejected anything that is not a single SELECT over v_orders, and the
    database grants make the base table unreachable. This layer exists so that
    a mistake in either of the other two still lands in a session where a write
    is refused by the transaction itself.

    SET LOCAL is used rather than ALTER ROLE because the same role writes
    chat_turn; scoping the setting to this transaction applies it to exactly
    the statements that need it.
    """
    with engine().begin() as conn:
        conn.execute(
            text(f"SET LOCAL statement_timeout = '{config.SQL_STATEMENT_TIMEOUT}'")
        )
        conn.execute(text("SET LOCAL transaction_read_only = on"))

        result = conn.execute(text(sql))

        # One row beyond the cap, so the caller can tell a result that happens
        # to be exactly at the limit from one that was cut short.
        fetched = result.fetchmany(config.SQL_MAX_ROWS + 1)

        truncated = len(fetched) > config.SQL_MAX_ROWS
        fetched = fetched[: config.SQL_MAX_ROWS]

        rows = [
            {key: jsonable(value) for key, value in row._mapping.items()}
            for row in fetched
        ]

    return rows, truncated


async def run_generated_sql(sql: str) -> Tuple[List[Dict[str, Any]], bool]:
    return await asyncio.to_thread(_run_generated, sql)
