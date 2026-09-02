"""Environment configuration, read once at import."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# ------------------------------------------------------------
# Vertex AI
# ------------------------------------------------------------

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")

# Defaults to global because that is where gemini-3.7-flash is served, and it
# is the only place it is served. A regional endpoint returns 404 for this
# model regardless of project permissions.
VERTEX_REGION = os.getenv("VERTEX_REGION", "global")

# Deliberately the only place a model id appears, so switching models is an
# environment change rather than a code change.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

# Local only. Unset on Cloud Run, where the attached service account supplies
# Application Default Credentials from the metadata server.
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")


# ------------------------------------------------------------
# Cloud SQL
# ------------------------------------------------------------

INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME", "")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "")
DB_PASS = os.getenv("DB_PASS", "")


# ------------------------------------------------------------
# Behaviour
# ------------------------------------------------------------

SQL_MAX_ROWS = _int("SQL_MAX_ROWS", 500)
SQL_STATEMENT_TIMEOUT = os.getenv("SQL_STATEMENT_TIMEOUT", "10s")

MEMORY_TURNS = _int("MEMORY_TURNS", 3)

# The only relation generated SQL may reference. The database grants enforce
# this too; this set is what lets the validator reject a bad statement with a
# readable message instead of a privilege error from the driver.
ALLOWED_TABLES = {"v_orders"}

# Column names, metric semantics and the date anchor, read at startup instead
# of queried. Absolute by default so it resolves the same from the repo root,
# from uvicorn's reloader, and from the container's WORKDIR.
SCHEMA_FILE = os.getenv("SCHEMA_FILE", str(_ROOT / "schema" / "v_orders.yaml"))

# Rows shown to the chart model. It only has to recognise the shape of the
# result, not read all of it, and the full set is sent to the frontend anyway.
CHART_SAMPLE_ROWS = 30

# A result narrower than this cannot produce a meaningful chart, so the chart
# branch exits before spending an LLM call.
CHART_MIN_ROWS = 2
CHART_MAX_COLUMNS = 6

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
