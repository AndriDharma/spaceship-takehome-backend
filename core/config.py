"""Environment configuration, read once at import."""

import os

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# ------------------------------------------------------------
# Vertex AI
# ------------------------------------------------------------

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
VERTEX_REGION = os.getenv("VERTEX_REGION", "us-central1")

# Deliberately the only place a model id appears. Google versions Gemini as
# 1.5 / 2.0 / 2.5 / 3, so if this value 404s at Vertex the fix is one line in
# .env rather than a code change.
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
