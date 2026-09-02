# Logistics Analytics API — Backend

Backend for an AI-powered logistics analytics dashboard. It serves a
deterministic KPI dashboard, a streaming natural-language interface over the
order data, automatic chart selection, and demand forecasting.

The frontend is a separate application deployed as its own Cloud Run service;
this repository is the backend only.

| | |
|---|---|
| **Runtime** | Python 3.12, FastAPI, Uvicorn |
| **AI** | Gemini `gemini-3.7-flash` on Vertex AI, orchestrated with LangGraph |
| **Database** | PostgreSQL on Cloud SQL, reached through the Cloud SQL Python Connector |
| **Deployment** | Cloud Run (container), Artifact Registry, Cloud Build |
| **Streaming** | Server-Sent Events over a single `POST` |

---

## 1. Setup

### Prerequisites

- Python 3.12
- A Google Cloud project with **Vertex AI**, **Cloud SQL Admin**, **Cloud Run**,
  **Cloud Build** and **Artifact Registry** APIs enabled
- A Cloud SQL for PostgreSQL instance
- A service account with `roles/cloudsql.client` and `roles/aiplatform.user`,
  and a downloaded JSON key for local development

### 1.1 Database

The dataset is loaded once, as the instance's admin user, into a table named
`logistics_orders` with one column per CSV field. Loading is a one-off
operation performed outside the application — the application never writes to
the order data.

Then create the metric view and the conversation table:

```bash
psql "$CONN" -f db/01_v_orders.sql -f db/02_chat_turn.sql
```

`db/01_v_orders.sql` creates **`v_orders`**, which is the only relation the
application reads. It adds the delivery-outcome flags, the delivery duration
and the pre-truncated date grains described in
[section 5](#5-data-model-and-metric-definitions).

### 1.2 Grants

The application connects as a single role. What it may touch is deliberately
narrow:

```sql
GRANT USAGE ON SCHEMA public TO your_app_role;
GRANT SELECT ON v_orders TO your_app_role;
GRANT SELECT, INSERT, UPDATE ON chat_turn TO your_app_role;

-- The important one. Reading the base table directly would let a generated
-- query define "delayed" or "on time" for itself and bypass the metric view.
REVOKE ALL ON logistics_orders FROM your_app_role;
```

### 1.3 Environment variables

Create a `.env` file in the repository root. One file serves both the running
application and `deploy.sh`.

**Read by the application:**

| Variable | Required | Default | Notes |
|---|---|---|---|
| `GCP_PROJECT_ID` | yes | — | Project for Vertex AI |
| `VERTEX_REGION` | no | `global` | **Must be `global`.** `gemini-3.7-flash` is served only from the global endpoint; a regional value returns a 404 that reads like a permissions error but is not |
| `GEMINI_MODEL` | no | `gemini-3.7-flash` | The only place a model id appears in the project |
| `GOOGLE_APPLICATION_CREDENTIALS` | local only | — | Path to the service account JSON. **Leave unset on Cloud Run**, where the attached service account supplies credentials automatically |
| `INSTANCE_CONNECTION_NAME` | yes | — | `project:region:instance` |
| `DB_NAME` | yes | `postgres` | |
| `DB_USER` | yes | — | The role from 1.2 |
| `DB_PASS` | yes | — | |
| `SQL_MAX_ROWS` | no | `500` | Ceiling on rows from a generated query, and the `LIMIT` the validator injects when the model omits one |
| `SQL_STATEMENT_TIMEOUT` | no | `10s` | Applied per transaction around generated SQL only |
| `MEMORY_TURNS` | no | `3` | Prior turns given to the router for follow-up resolution |
| `SCHEMA_FILE` | no | `schema/v_orders.yaml` | |
| `CORS_ORIGINS` | no | *(allows all)* | Comma-separated. Set this to the frontend's URL in production |

**Read by `deploy.sh` only:**

| Variable | Notes |
|---|---|
| `PROJECT_ID` | Same value as `GCP_PROJECT_ID`. Both names are needed — the deploy script passes `PROJECT_ID` into the container *as* `GCP_PROJECT_ID` |
| `REGION` | Cloud Run and Artifact Registry region (default `asia-southeast2`). Independent of `VERTEX_REGION` |
| `REPOSITORY`, `IMAGE_NAME`, `SERVICE_NAME` | Artifact Registry and Cloud Run names |
| `SERVICE_ACCOUNT` | Runtime service account email |
| `MIN_INSTANCES` | Default `1` — see [deployment](#7-deployment) |
| `MAX_INSTANCES` | Default `10` |

### 1.4 Run locally

```bash
conda activate spaceship-test && pip install -r requirements.txt
```

```bash
python main.py
```

The server listens on `:8080` with reload enabled. Confirm it came up correctly:

```bash
curl http://localhost:8080/api/health
```

`schema_loaded` must be `true` and `row_count` must be `400`. Interactive API
docs are at `http://localhost:8080/docs`.

---

## 2. API

| Method | Path | Streams | AI | Purpose |
|---|---|---|---|---|
| `GET` | `/api/health` | no | no | Liveness, model and data window. Also a warm-up target |
| `GET` | `/api/dashboard` | no | no | All KPIs and all three dashboard charts in one payload |
| `POST` | `/api/chat` | **SSE** | yes | Natural-language question |
| `GET` | `/api/chat/{turn_id}` | no | no | Re-open a stored turn |
| `GET` | `/api/chat/session/{session_id}/history` | no | no | Query history for a session |
| `POST` | `/api/chart` | no | yes | Regenerate the chart for a stored turn |
| `POST` | `/api/forecast` | no | no | Forecast directly, for a dashboard widget |

### `POST /api/chat`

```json
{ "question": "Which carrier has the highest delay rate?", "session_id": "optional-uuid" }
```

Responds with `text/event-stream`. `POST` rather than `GET` means the browser's
native `EventSource` cannot be used — the frontend reads the body with `fetch`
and a `ReadableStream`.

**Event sequence:**

| Event | Payload | When |
|---|---|---|
| `start` | `turn_id`, `session_id`, `data_window` | Immediately, before any model call |
| `progress` | `step`, `message`, `status`, `detail` | Each pipeline step. `detail` carries the validated SQL, the tables and columns touched, the filters, and the row count |
| `output` | `content` | Answer text, flushed in ~30-character blocks |
| `chart` | Full `ChartConfig` | When a chart was selected |
| `chart_skipped` | `reason` | When the result was not chartable |
| `complete` | Full turn payload | Once, at the end |
| `error` | `message` | Only on an unrecoverable failure |
| `done` | `turn_id` | Terminal — the client closes the stream |

`progress` events are not decoration. Together with the SQL and row count in
their `detail`, they are the explainability surface required by §4.4 of the
specification: the user can see which tool was chosen, what ran, and how many
rows came back.

---

## 3. Architecture

### 3.1 Overview

```
                    ┌──────────────────────────────────────────┐
  POST /api/chat ──►│              LangGraph                   │
                    │                                          │
                    │  route ─┬─ sql ──► validate ─┬─ ok ──► execute ─┬─► answer ─┐
                    │         │           ▲        │                  │           │
                    │         │           └ repair ┘ (max 1)          └─► chart ──┤
                    │         ├─ forecast ──────────────────────────────► answer ─┤
                    │         └─ direct ───────────────────────────────► answer ──┤
                    │                                                   finalize ◄┘
                    └──────────────────────────────────────────┘
                          │                              │
                          ▼                              ▼
                   Vertex AI (Gemini)          PostgreSQL (v_orders, chat_turn)

  GET /api/dashboard ────────────────────────► PostgreSQL          (no AI)
  POST /api/forecast ────────────────────────► PostgreSQL          (no AI)
```

### 3.2 Data flow for one question

1. The last three turns of the session are read from `chat_turn`.
2. **`route`** makes one model call returning JSON: the tool to use and, for the
   query tool, a PostgreSQL `SELECT`.
3. **`validate`** parses that statement with `sqlglot` and checks it
   deterministically — no model involved. On failure, one repair attempt is made
   with the rejection reason fed back.
4. **`execute`** runs the statement inside a read-only, time-limited
   transaction.
5. **`answer`** and **`chart`** are scheduled together and run concurrently.
   The answer streams token by token while the chart model call is in flight, so
   the chart typically lands as the last token arrives.
6. **`finalize`** joins the branches; the turn is persisted to `chat_turn`
   *after* the client already has the payload.

### 3.3 Key design decisions

**Generated SQL runs against a metric view, never the base table.** `v_orders`
precomputes `is_delayed`, `is_on_time`, `is_completed`, `delivery_days` and the
date grains. The model cannot invent its own definition of "delayed" because it
never sees the columns needed to build one, and the dashboard and the AI path
therefore compute an on-time rate identically — they read the same column.

**Three independent layers protect the database.** Any one of them failing is
contained by the other two:

| Layer | Enforces |
|---|---|
| `sqlglot` validator | One statement; `SELECT`/`WITH` only; no `INSERT`/`UPDATE`/`DROP`/`COPY`/`Command` nodes; tables restricted to `v_orders`; every column must exist; a `LIMIT` is injected if absent |
| Database grants | `SELECT` on `v_orders` only. The base table is revoked, so a validator bypass still cannot reach it |
| Read-only transaction | `SET LOCAL transaction_read_only = on` and `SET LOCAL statement_timeout` around every generated statement |

The transaction settings are `SET LOCAL` rather than `ALTER ROLE` because the
same role writes `chat_turn`; a role-wide read-only setting would block the
application's own inserts.

**One model call does interpretation and generation together.** Splitting
"choose a tool" from "write the query" would double the latency before the first
token without improving either decision — both need exactly the same context.

**The chart runs in parallel with the answer, and can never block it.** A bad
config falls back to a deterministic chart choice; an impossible one emits
`chart_skipped` with a reason; an exception is caught. The answer is the
product, the chart is presentation.

**The schema is a version-controlled file, not database introspection.**
`schema/v_orders.yaml` holds the column list, the full value domain of every
categorical column, and the metric computation rules. Two reasons: startup no
longer touches Cloud SQL, which matters on a cold start; and what the model is
told is prompt content, so a change to it belongs in a diff rather than hidden
in a `COMMENT ON COLUMN`. The trade-off is that the file must be regenerated
when the view changes — `schema_service._validate()` fails loudly at boot if
`columns` and `features` drift apart.

**Single-hop SSE.** One request, one connection, one graph run. No message
queue, no Redis relay, no worker pool. Those solve multi-tenant concurrency and
cross-worker resumability, neither of which exists here, and each would add
infrastructure to the deployment for no user-visible gain.

---

## 4. AI approach

### 4.1 How questions are interpreted

One call to `gemini-3.7-flash` at temperature 0 receives the question, the
schema block, and the last three turns, and returns:

```json
{
  "mode": "sql" | "forecast" | "direct",
  "sql": "SELECT ...",
  "forecast": { "level": "...", "key": "...", "horizon": 3, "method": "...", "requested_sku": null },
  "reason": "..."
}
```

Temperature is set explicitly because Gemini 3 defaults to `1.0`, and a router
that picks a different tool for the same question on consecutive runs is not a
router.

The schema block given to the model includes the **complete value domain** of
every categorical column — all nine carriers, all five regions, all forty-seven
destination cities. This is the single highest-value thing in the prompt: a
near-miss on a literal (`'DHL Express'` for `'DHL'`, `'In Transit'` for
`'in_transit'`) returns zero rows rather than an error, which reads to the user
as a valid answer of "none".

### 4.2 How tools are selected

| Mode | Chosen when | Path |
|---|---|---|
| `sql` | The question is about what has happened — counts, rates, rankings, breakdowns, trends | validate → execute → answer + chart |
| `forecast` | The question is about future demand or inventory planning | forecast → answer |
| `direct` | A greeting, a capability question, or something the data cannot answer | answer |

Routing failures degrade rather than fail: an unparseable router response
becomes `direct`, two failed validations become an answer explaining that the
question cannot be expressed against this data, and a runtime SQL error becomes
an answer explaining what went wrong.

### 4.3 What the AI is not allowed to do

The specification requires that the AI never answer without computation. Two
structural properties enforce that rather than merely requesting it:

- **The answer node has no tools and no database access.** It receives the
  result rows and nothing else.
- **The answer node never sees conversation history.** Only the router does.
  Resolving "now break that down by region" is interpretation; narrating a
  result is not. A model that cannot see previous answers cannot quote a number
  from one.

Reasoning blocks returned by Gemini 3 are stripped in `core/messages.py`, so
the model's working never reaches the user or the JSON parser.

### 4.4 Forecasting

`services/forecast.py` does the arithmetic; no model is involved in producing a
number. Two methods are available — a 3-month moving average (default) and
least-squares linear regression, chosen by the router from the phrasing of the
question. Months with no orders are filled with zero before fitting, because a
gap left as a gap turns a flat series into a rising one.

The inventory recommendation is `total forecast + 1.65 × σ × √horizon`, roughly
a 95% service level, and the answer states the formula rather than presenting
the number as an oracle.

The forecast chart is built in Python, not chosen by a model — its shape is
known before the data is read — but it is emitted in the same `ChartConfig`
schema the query path produces, so the frontend has exactly one renderer for
every chart in the product.

### 4.5 Chart selection

The model receives the column names and the first 30 rows, and returns a
`ChartConfig`: type, axes, series, title, and a one-sentence insight. It never
writes rendering code and never sees the rendering.

A deterministic gate runs **before** the model call — a result must have at
least 2 rows, at most 6 columns, and at least one numeric column. This means a
single-value answer such as *"how many orders were delivered late last month?"*
never spends a model call to discover it has nothing to plot.

Whatever the model returns is then coerced against the real column names: an
axis that is not a column, a non-numeric measure, or a series key that collides
with the axis is corrected or the config is rejected in favour of the fallback.

---

## 5. Data model and metric definitions

The source data has **no promised-delivery or SLA column**, so on-time
performance can only be derived from `status`. These definitions live in
`v_orders` and are used identically by the dashboard and the AI path:

| Metric | Definition |
|---|---|
| `is_delayed` | `1` when `status = 'delayed'` |
| `is_on_time` | `1` when `status = 'delivered'` |
| `is_completed` | `1` when `status IN ('delivered','delayed')` — **always the rate denominator** |
| on-time rate | `SUM(is_on_time) / NULLIF(SUM(is_completed), 0)` |
| delay rate | `SUM(is_delayed) / NULLIF(SUM(is_completed), 0)` |
| `delivery_days` | `delivery_date - order_date`; `NULL` when undelivered, so `AVG` is correct without filtering |
| `net_value_usd` | `order_value_usd` after `promo_discount_pct` |

Orders that are `in_transit`, `canceled` or `exception` have no delivery
outcome and are excluded from every rate denominator. Using `COUNT(*)` instead
of `SUM(is_completed)` would count them as on time and understate the delay
rate — this is the single most likely correctness error in the whole product,
which is why the denominator is a column rather than a convention.

The flags are `smallint` rather than `boolean` deliberately: PostgreSQL has no
`AVG(boolean)`, so a rate over a boolean flag fails at execution and costs a
retry.

---

## 6. Assumptions and simplifications

**The dataset's last order date is "today".** The data covers 2025-01-01 to
2025-12-30 while the wall clock is well past that. Relative expressions —
"last month", "the last 3 months" — resolve against **2025-12-30**, not the
current date. Resolving against the real clock would return zero rows on
exactly the questions the specification gives as examples. The anchor is
declared in `schema/v_orders.yaml`, surfaced in `/api/health` and
`/api/dashboard`, and the answer states which dates a relative phrase resolved
to.

**"Late" means `status = 'delayed'` and nothing else.** With no promised
delivery date in the source, there is no threshold against which lateness could
otherwise be computed.

**Forecasting is not meaningful per SKU.** There are 355 distinct SKUs across
400 orders and 313 of them appear exactly once. A question naming a SKU is
resolved to that SKU's `product_category`, and the answer leads with the fact
that it was — silently answering a different question would be worse than
saying so.

**No authentication.** The API is public. There are no accounts, and
`session_id` is generated by the client purely to group turns.

**Order data is read-only.** The application never writes to
`logistics_orders`; the only table it writes is `chat_turn`.

**One repair attempt for invalid SQL.** A second failure almost always means
the question cannot be expressed against this schema rather than that the model
made a typo, and each attempt is a full model call in front of the user's first
token.

**Results are capped at 500 rows** and generated queries at a 10-second
statement timeout.

---

## 7. Limitations and unsupported queries

**Not supported by design:**

- **Joins and multi-table queries.** `v_orders` is the only readable relation.
- **Anything requiring data not in the dataset** — shipping cost, weather,
  customer names, carrier contracts, promised delivery dates.
- **Chart revision.** "Make that a bar chart" is not supported;
  `POST /api/chart` regenerates from scratch rather than modifying an existing
  chart.
- **Cross-year comparison.** Every row is 2025, so any year filter other than
  2025 returns nothing.
- **Write operations of any kind** through the natural-language interface.

**Known weaknesses:**

- **Small-sample aggregates.** 400 orders spread across 47 destination cities
  and 30 clients means per-city and per-client figures rest on very few rows.
  The API reports them without a significance warning.
- **Forecast confidence.** Twelve monthly observations with visible lumpiness
  (75 orders in January, 21 in June) support a trend estimate, not a
  confidence-bounded projection. No prediction interval is produced, and the
  answer says so.
- **Causal questions.** "Why did delays increase in July?" will be answered
  with what the data shows, not with a cause. The data contains no explanatory
  variables.
- **Conversation memory is three turns**, router-only, and scoped to a
  `session_id` the client is responsible for.
- **No caching.** Every question is a fresh model call and a fresh query.
- **No automated tests.** Verification was manual against the running service.
- **Cold starts.** `min-instances` defaults to `1` in `deploy.sh` precisely
  because at `0` the first request waits on a container start plus the
  LangGraph and Vertex client imports, which reads as a broken application.
- **Data residency.** `gemini-3.7-flash` is served only from the `global`
  endpoint, which carries no residency guarantee. For a real client with EU
  orders this would need revisiting; Gemini 3.5 Flash is the newest model that
  supports EU residency.

---

## 8. Deployment

```bash
./deploy.sh
```

Reads `.env`, builds through Cloud Build into Artifact Registry, and deploys to
Cloud Run with the Cloud SQL instance attached and the runtime service account
bound. The one-time API enablement and IAM bindings are included as commented
commands at the top of the script.

The deployed service uses the **attached service account** for both Cloud SQL
and Vertex AI — `GOOGLE_APPLICATION_CREDENTIALS` is not set in the container,
and no key file is baked into the image (`.dockerignore` excludes
`service_account/`). Locally the same code path reads the key file instead;
the presence of the file is what decides, not an environment flag.

After deploying, verify:

```bash
curl "$SERVICE_URL/api/health"
```

---

## 9. Future improvements

In the order I would actually do them:

1. **Tests.** A fixture set of question → expected-SQL-shape cases for the
   router, and unit tests for the validator's rejection paths. The validator is
   the security boundary and currently has no automated coverage.
2. **Response caching** keyed on the normalised question, which would make the
   demo noticeably faster on repeated queries and cut model spend.
3. **A structured query IR as a first path**, falling back to generated SQL.
   Emitting a validated `{metric, dimensions, filters, grain}` object for the
   common question shapes would make the majority of turns provably correct
   rather than validated-after-the-fact.
4. **Chart revision**, carrying the previous config forward so "make it a bar
   chart" works.
5. **Clarifying questions for ambiguous input.** "Show me the bad ones" should
   ask what "bad" means rather than guessing.
6. **Confidence intervals on forecasts**, once there is enough history to
   support them honestly.
7. **Streaming the chart config incrementally** so the chart panel can render a
   skeleton before the model call returns.
8. **Observability** — per-turn latency and token accounting, currently only
   visible in stdout logs.

---

## 10. AI assistance disclosure

This project was built with the assistance of an AI coding assistant (Claude).
It was used for code generation, architectural discussion, and debugging. All
architectural decisions — the metric-view approach, the three-layer SQL
guardrails, the parallel chart branch, the date-anchoring strategy, and the
decision to degrade SKU-level forecasts — were reviewed and directed by me, and
the resulting code was read, tested and corrected before being committed.
