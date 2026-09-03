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

**Contents** — [Setup](#1-setup) · [API](#2-api) · [Architecture](#3-architecture) ·
[Where the LLM is used](#4-where-the-llm-is-used) · [SQL route](#5-the-sql-route-in-detail) ·
[Forecast route](#6-the-forecast-route-in-detail) · [Metrics](#7-data-model-and-metric-definitions) ·
[Assumptions](#8-assumptions-and-simplifications) · [Limitations](#9-limitations-and-unsupported-queries) ·
[Deployment](#10-deployment) · [Future work](#11-future-improvements) · [AI disclosure](#12-ai-assistance-disclosure)

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

`db/01_v_orders.sql` creates **`v_orders`**, the only relation the application
reads. It adds the delivery-outcome flags, the delivery duration and the
pre-truncated date grains described in [section 7](#7-data-model-and-metric-definitions).

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
| `CORS_ORIGINS` | no | *(allows all)* | Comma-separated. Set to the frontend's URL in production |

**Read by `deploy.sh` only:**

| Variable | Notes |
|---|---|
| `PROJECT_ID` | Same value as `GCP_PROJECT_ID`. Both names are needed — the deploy script passes `PROJECT_ID` into the container *as* `GCP_PROJECT_ID` |
| `REGION` | Cloud Run and Artifact Registry region (default `asia-southeast2`). Independent of `VERTEX_REGION` |
| `REPOSITORY`, `IMAGE_NAME`, `SERVICE_NAME` | Artifact Registry and Cloud Run names |
| `SERVICE_ACCOUNT` | Runtime service account email |
| `MIN_INSTANCES` | Script default `1`, but set to `0` in the current `.env` to avoid paying for an idle instance during development. **Set to `1` before review** — see [deployment](#10-deployment) |
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

| Method | Path | Streams | LLM calls | Purpose |
|---|---|---|---|---|
| `GET` | `/api/health` | no | **0** | Liveness, model and data window. Also a warm-up target |
| `GET` | `/api/dashboard` | no | **0** | All KPIs and all three dashboard charts in one payload |
| `POST` | `/api/chat` | **SSE** | 2–4 | Natural-language question |
| `GET` | `/api/chat/{turn_id}` | no | **0** | Re-open a stored turn |
| `GET` | `/api/chat/session/{session_id}/history` | no | **0** | Query history for a session |
| `POST` | `/api/chart` | no | 1 | Regenerate the chart for a stored turn |
| `POST` | `/api/forecast` | no | **0** | Forecast directly, for a dashboard widget |

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
                    ┌──────────────────────────────────────────────────────┐
  POST /api/chat ──►│                      LangGraph                       │
                    │                                                      │
                    │  route ─┬─ sql ──► validate ─┬─ ok ──► execute ─┬─► answer ─┐
                    │   LLM 1 │            ▲       │                  │    LLM 3  │
                    │         │      LLM 2 └ repair┘ (max 1)          └─► chart ──┤
                    │         │                                            LLM 4  │
                    │         ├─ forecast ────────────────────────────────► answer┤
                    │         │  (no LLM — pure Python)                    LLM 3  │
                    │         └─ direct ────────────────────────────────► answer ─┤
                    │                                                      LLM 3  │
                    │                                            finalize ◄───────┘
                    └──────────────────────────────────────────────────────┘
                          │                              │
                          ▼                              ▼
                   Vertex AI (Gemini)          PostgreSQL (v_orders, chat_turn)

  GET /api/dashboard ────────────────────────► PostgreSQL          (no LLM)
  POST /api/forecast ────────────────────────► PostgreSQL          (no LLM)
```

### 3.2 Data flow for one question

1. The last three turns of the session are read from `chat_turn`.
2. **`route`** makes one model call returning JSON: the tool to use and, for the
   query tool, a PostgreSQL `SELECT`.
3. **`validate`** parses that statement with `sqlglot` and checks it
   deterministically — no model involved. On failure, one repair attempt is made
   with the rejection reason fed back.
4. **`execute`** runs the statement inside a read-only, time-limited transaction.
5. **`answer`** and **`chart`** are scheduled together and run concurrently. The
   answer streams token by token while the chart model call is in flight, so the
   chart typically lands as the last token arrives.
6. **`finalize`** joins the branches; the turn is persisted to `chat_turn`
   *after* the client already has the payload.

### 3.3 Key design decisions

**Generated SQL runs against a metric view, never the base table.** `v_orders`
precomputes `is_delayed`, `is_on_time`, `is_completed`, `delivery_days` and the
date grains. The model cannot invent its own definition of "delayed" because it
never sees the columns needed to build one, and the dashboard and the AI path
therefore compute an on-time rate identically — they read the same column.

**Three independent layers protect the database.** Any one failing is contained
by the other two:

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
`chart_skipped` with a reason; an exception is caught.

**The schema is a version-controlled file, not database introspection.**
`schema/v_orders.yaml` holds the column list, the full value domain of every
categorical column, and the metric computation rules. Startup no longer touches
Cloud SQL, which matters on a cold start; and what the model is told is prompt
content, so a change to it belongs in a diff rather than hidden in a
`COMMENT ON COLUMN`. The trade-off is that the file must be regenerated when the
view changes — `schema_service._validate()` fails loudly at boot if `columns`
and `features` drift apart.

**Single-hop SSE.** One request, one connection, one graph run. No message
queue, no Redis relay, no worker pool. Those solve multi-tenant concurrency and
cross-worker resumability, neither of which exists here.

---

## 4. Where the LLM is used

There are exactly **four** places a model is called. Everything else —
validation, execution, forecasting arithmetic, dashboard aggregation, chart
rendering data — is deterministic Python and SQL.

### 4.1 Inventory

| # | Call site | Runs when | Input | Output | Temp / max tokens | On failure |
|---|---|---|---|---|---|---|
| **1** | `ai/nodes/router.py` → `route_node` | **Every** chat turn | Schema block + last 3 turns + question | JSON `RouteDecision` | 0 / 2048 | Degrades to `direct` mode |
| **2** | `ai/nodes/sql.py` → `retry_sql_node` | Only when validation failed, **max once** | Rejected SQL + rejection reason + schema + question | Raw `SELECT` text | 0 / 2048 | Empty SQL → give up → answer explains |
| **3** | `ai/nodes/answer.py` → `answer_node` | **Every** chat turn | One of four prompts, by mode | Streamed markdown | 0 / **4096** | Static fallback sentence |
| **4** | `services/chart_config.py` → `build` | Only when the result passed the chartability gate | Column names + first 30 rows + question | JSON `ChartConfig` | 0 / 2048 | Deterministic fallback chart |

**Calls per turn:**

| Scenario | Calls | Which |
|---|---|---|
| Greeting / capability question | **2** | router + answer |
| SQL question, chartable result | **3** | router + answer + chart |
| SQL question, single-value result | **2** | router + answer (chart gated out before the call) |
| SQL question needing one repair | **4** | router + repair + answer + chart |
| SQL question, two failed validations | **2** | router + repair + answer *(3)* |
| Forecast question | **2** | router + answer — the forecast itself and its chart are pure Python |

Temperature is `0` on all four. Gemini 3 defaults to `1.0`, so it is passed
explicitly — a router that picks a different tool for the same question on
consecutive runs is not a router.

The answer call gets a **4096-token budget** where the others take the 2048
default. A sectioned answer over a multi-dimension result runs several hundred
tokens, and on a thinking model the reasoning is drawn from the same budget, so
2048 sits close enough to the ceiling to risk truncating an answer mid-section.
It is a cap, not a reservation — short answers cost exactly what they did before.

### 4.2 Call 1 — Router (interpretation + tool selection)

**Input:** one prompt string containing

- The **schema block** rendered from `schema/v_orders.yaml`: every column with
  its type, description and metric semantics; the **complete value domain** of
  every categorical column (all 9 carriers, all 5 regions, all 47 destination
  cities); the data window and anchor-date note; and the cross-column
  computation rules.
- The **last 3 turns** of the session as `question` + `SQL` pairs — no answers,
  see [4.6](#46-what-the-llm-is-never-allowed-to-do).
- The user's question.

**Output:**

```json
{
  "mode": "sql" | "forecast" | "direct",
  "sql": "SELECT carrier, SUM(is_delayed)::numeric / NULLIF(SUM(is_completed),0) AS delay_rate ...",
  "forecast": { "level": "category", "key": "CRAYON", "horizon": 4,
                "method": "moving_average", "requested_sku": null },
  "reason": "Ranking carriers by delay rate over recorded data."
}
```

Parsed leniently (code fences stripped, first `{` to last `}` extracted) then
validated by the `RouteDecision` Pydantic model. **Nothing in this output is
trusted** — the SQL goes to the validator, the forecast parameters are
type-checked and range-bounded, and an unparseable response becomes `direct`
rather than failing the turn.

The complete value domain is the highest-value part of this prompt. A near-miss
on a literal — `'DHL Express'` for `'DHL'`, `'In Transit'` for `'in_transit'` —
returns zero rows rather than an error, which reads to the user as a valid
answer of "none".

### 4.3 Call 2 — SQL repair

**Runs only on validation failure, and only once.** Input is the rejected
statement, the validator's specific reason (`"These columns do not exist:
delay_pct"`), the schema block, and the original question. Output is a bare
`SELECT` — no JSON wrapper — which is cleaned of fences and semicolons and sent
straight back to the validator.

A second failure almost always means the question cannot be expressed against
this schema rather than that the model made a typo, and each attempt is a full
model call sitting in front of the user's first token.

### 4.4 Call 3 — Answer

The only streaming call. It selects one of **four** prompts:

| Prompt | Used when | Input |
|---|---|---|
| `_SQL_PROMPT` | `mode=sql`, query succeeded | Question, row count, truncation flag, **first 100 rows as JSON**, data window |
| `_FORECAST_PROMPT` | `mode=forecast`, forecast succeeded | Question + the forecast result **minus `history` and `chart`** |
| `_DIRECT_PROMPT` | `mode=direct` | Question, router's `reason`, data window |
| `_ERROR_PROMPT` | SQL failed, or forecast failed | Question, the error, data window |

**Output:** streamed markdown, emitted as `output` SSE events buffered to ~30
characters.

**Answer depth follows the shape of the result, not a fixed budget.** The SQL
prompt states three tiers explicitly, because a model given room to write and no
rule about when to use it will pad a one-number answer into a report:

| Result shape | Format |
|---|---|
| A single value, or one row | Two or three sentences, figure in **bold**. No headings, no bullets — a section header over one number imposes a shape the data does not have |
| One dimension (one grouping column, one measure) | **Bold title line** naming what is shown and the period, a sentence of introduction, then 2–4 bullets each leading with its **bold** figure |
| Two or more dimensions or measures | Same opening, then one `### Section` per dimension or measure, each with a sentence and 2–3 bullets |

The counterpart to that freedom is the instruction to **characterise rather than
recite** — the highest and lowest, the gap, the range, where a trend turns,
which group is the outlier. The full table is already rendered beside the
answer, so prose walking the rows one at a time is the same data twice, and the
longer an answer is allowed to be the more inviting that failure becomes.

**Markdown contract for the frontend.** The answer is markdown, and the prompts
constrain it to a narrow subset the client must render:

- `###` is the **only** heading level permitted — never `#` or `##`
- **Bold**, bullet lists, and `backticks` for identifiers
- **No tables** — the result table is a separate panel
- No citation markers of any kind

Tone is specified too: a helpful analyst talking to a colleague, closing on what
a pattern might reflect when the data supports one, and skipping the empty parts
of politeness — no "great question", no "I hope this helps", no offers of
further assistance.

`history` and `chart` are stripped from the forecast payload deliberately —
twelve months of data in the prompt costs tokens and invites the model to
narrate every month one at a time.

### 4.5 Call 4 — Chart selection

**Input:** the result's column names, the **first 30 rows** as JSON, the total
row count, and the question the data answers.

**Output:**

```json
{ "chartType": "bar", "title": "Delay Rate by Carrier", "description": "...",
  "xKey": "carrier", "yKeys": ["delay_rate_pct"], "seriesKey": null,
  "stacked": false, "insight": "DHL leads at 22.4%, nine points above UPS." }
```

The model **never writes rendering code and never sees the rendering**. Its
output is coerced against the real column names: an axis that is not a column,
a non-numeric measure, or a series key colliding with the axis is corrected, or
the config is rejected in favour of a deterministic fallback.

A gate runs **before** this call — at least 2 rows, at most 6 columns, at least
one numeric column — so a single-value answer such as *"how many orders were
delivered late last month?"* never spends a model call to discover it has
nothing to plot.

### 4.6 What the LLM is never allowed to do

The specification requires that the AI never answer without computation. Two
structural properties enforce that rather than merely requesting it:

- **The answer node has no tools and no database access.** It receives the
  result rows and nothing else.
- **The answer node never sees conversation history.** Only the router does.
  Resolving "now break that down by region" is interpretation; narrating a
  result is not. A model that cannot see previous answers cannot quote a number
  from one.

Reasoning blocks returned by Gemini 3 are stripped in `core/messages.py`, so the
model's working never reaches the user or a JSON parser.

---

## 5. The SQL route in detail

### 5.1 Pipeline

```
router SQL ──► validate ──┬── ok ────────────────► execute ──► answer + chart
                 ▲        │
                 │        ├── invalid, retries<1 ─► repair (LLM 2) ──┐
                 │        │                                          │
                 └────────┴── invalid, retries≥1 ─► answer (explains why)
                                                                     │
                 └───────────────────────────────────────────────────┘
```

### 5.2 Validation — `services/sql_validator.py`

Runs with **no model and no execution**. In order:

| Step | Rejects |
|---|---|
| **Clean** | Strips ``` fences and trailing semicolons |
| **Parse** | `sqlglot.parse(dialect="postgres")` — unparseable SQL |
| **Single statement** | Anything with more than one statement (stacked-query injection) |
| **Statement type** | Anything not `Select` / `Union` / `Subquery`. `WITH … SELECT` parses as a `Select` carrying a `with` arg, so CTEs pass without a special case |
| **Forbidden nodes** | `Insert`, `Update`, `Delete`, `Drop`, `Create`, `Alter`, `Merge`, `Into`, `Grant`, `TruncateTable`, and **`Command`** — sqlglot's catch-all for `SET`, `COPY`, `CALL`, `VACUUM`, so the whole class is caught rather than an enumeration that goes stale |
| **Forbidden functions** | `pg_read_file`, `pg_ls_dir`, `lo_import`, `dblink`, `pg_sleep`, … |
| **Table whitelist** | Any table not in `ALLOWED_TABLES` (`{v_orders}`). CTE names are collected first and exempted |
| **Column whitelist** | Any column not in the YAML's column list. Output aliases, CTE names and table aliases are collected first and exempted, so `… COUNT(*) AS n … ORDER BY n` passes |
| **Row cap** | Injects `LIMIT 500` when the statement has none |

The forbidden-node list is resolved by `getattr` rather than imported directly:
sqlglot moves node classes between releases, and a missing name would otherwise
be an `ImportError` at startup — the validator failing so closed it refuses to
load at all.

On success it also returns the **explainability payload**: `tables`, `columns`,
`filters` (the `WHERE` clause as SQL), `group_by`, and `limit_injected` — which
is what populates the `progress` event's `detail` and the analysis panel.

### 5.3 Execution — `core/db.run_generated_sql`

```sql
BEGIN;
SET LOCAL statement_timeout = '10s';
SET LOCAL transaction_read_only = on;
<the validated statement>
```

`fetchmany(SQL_MAX_ROWS + 1)` — one row past the cap, so a result that lands
exactly at the limit can be distinguished from one that was cut short, and
`truncated` is reported honestly to both the user and the answer prompt.
`Decimal` and `date` values are converted to JSON-safe types on the way out.

The call is pushed to a worker thread (`asyncio.to_thread`), because pg8000 is
synchronous and blocking the event loop would stall the SSE stream running
concurrently.

### 5.4 Failure modes

| Failure | Handling | User sees |
|---|---|---|
| Router returns unparseable JSON | Degrade to `direct` | An answer about what the assistant can do |
| SQL fails validation, 1st time | One repair call with the reason | A `retry` progress row, then normal flow |
| SQL fails validation, 2nd time | `give_up` → answer node with `_ERROR_PROMPT` | Plain explanation + a suggested workable question |
| SQL passes validation but fails at runtime | No retry — the repair loop has no new information | Same error explanation |
| Query returns 0 rows | Normal path | Answer says so plainly and suggests an alternative |
| Result not chartable | Gate exits before the model call | `chart_skipped` with a reason |

### 5.5 Worked example

> **"Which carrier has the highest delay rate?"**

1. Router → `mode: "sql"` with
   `SELECT carrier, SUM(is_delayed)::numeric / NULLIF(SUM(is_completed),0) * 100 AS delay_rate_pct FROM v_orders GROUP BY carrier ORDER BY delay_rate_pct DESC`
2. Validator → passes; tables `[v_orders]`, columns `[carrier, is_delayed, is_completed]`, `LIMIT 500` injected
3. Execute → 9 rows, 2 columns
4. Gate → chartable (9 rows ≥ 2, 2 cols ≤ 6, one numeric)
5. Answer + chart run concurrently → prose naming the top carrier, and a bar
   chart of `carrier` × `delay_rate_pct`

---

## 6. The forecast route in detail

Reached when the router returns `mode: "forecast"`. **No model is involved in
producing any number** — `services/forecast.py` is pure Python and SQL. The
model is called twice on this route: once to route, once to narrate.

### 6.1 Pipeline

```
router ForecastParams
   │
   ├─ 1. Resolve SKU → category            (SQL)
   ├─ 2. Fetch monthly history             (SQL: GROUP BY order_month)
   ├─ 3. Densify — fill empty months with 0
   ├─ 4. Sparsity check → note if < 10 orders/month
   ├─ 5. Fit → predictions + residual spread + degrees of freedom
   ├─ 6. Prediction interval (Student's t, 80%)
   ├─ 7. Inventory recommendation
   └─ 8. Build ChartConfig in Python
   │
   └──► answer node narrates (LLM 3)
```

### 6.2 SKU resolution

The dataset has 355 distinct SKUs across 400 orders; 313 appear exactly once.
Per-SKU forecasting is not difficult here, it is impossible.

The router puts a named SKU verbatim into `requested_sku` and is explicitly told
**not** to guess a category. The application then looks up that SKU's real
category and order count. Below `MIN_SKU_ORDERS = 8`, the level is switched to
`category` and a **note** is added:

> *SKU CRAYON-0008 has only 3 order(s) in the dataset, which is too little
> history to forecast. Forecasting the CRAYON category instead.*

The answer prompt is instructed to **lead with the notes**. Silently answering a
different question than the one asked would be worse than saying so.

### 6.3 History and densification

```sql
SELECT order_month AS month, SUM(quantity)::int AS demand, COUNT(*)::int AS orders
FROM v_orders [WHERE product_category = :key | WHERE region = :key]
GROUP BY order_month ORDER BY order_month
```

Demand is **units ordered**, not order count. Months with no orders are then
filled with zero — a gap left as a gap turns a flat series into a rising one,
because both methods would treat twelve months as though every observation were
consecutive.

### 6.4 Methods

| Method | Chosen when | Behaviour |
|---|---|---|
| `moving_average` *(default)* | Anything not phrased as a trend | Mean of the last 3 months, projected **flat**. A moving average has no trend component; projecting one would be inventing information |
| `linear_regression` | The question mentions a trend or direction | Least squares over the full history, **clamped at 0** — an unclamped downward trend goes negative within a few months on a series this short |

Each returns `(predictions, residual spread σ, degrees of freedom)`.

### 6.5 Prediction interval

σ is the spread of the **method's own one-step-ahead errors**, not the spread of
the observations — the latter counts the trend as though it were error. It falls
back to the observation spread only on series too short to leave residuals,
which errs conservatively.

The interval is **80%**, using a **Student's t** multiplier at the 0.90 quantile
looked up by degrees of freedom from a small table. The normal multiplier
(1.282) would be roughly 8% too narrow at these sample sizes — exactly the
direction a small sample should not err in. A `scipy` dependency for one lookup
was not worth it.

Width grows with `√horizon`, matching the accumulation the safety stock already
assumes, so the band and the stock recommendation tell one story.

80% rather than 95% because the wider band on a series this noisy spans nearly
the whole plot area, which tells the reader less than a narrower one does.

### 6.6 Inventory recommendation

```
recommended = total_forecast + 1.65 × σ × √horizon
```

1.65 standard deviations is the usual one-sided ~95% service level. **This is
deliberately not the same confidence as the displayed 80% band** — stocking to
95% while showing an 80% planning range is normal practice, because you hold
more than you expect to need. Side by side and unexplained it looks like a
mistake, so the generated `explanation` string says so explicitly.

### 6.7 Sparsity note

When a level averages fewer than 10 orders per month, a second note is added
saying month-to-month movement is mostly sampling noise rather than demand
behaviour. The answer prompt also requires that a **flat projection be explained
as a decision** — a moving average estimates level and does not project trend —
so a flat line does not read as the method having given up.

### 6.8 Chart

Built in Python, not by a model: its shape is known before the data is read.
Emitted in the **same `ChartConfig` schema** as the query path so the frontend
has one renderer, with `actual` and `forecast` as separate series (the actual
series is `null` from the split onward) plus a `ChartBand` shading the interval.

The band bounds are in `data` but **not** in `headers`, so they shade the chart
without appearing as two extra columns in the table beside it. The band is
pinned to zero width at the last observation — that month is measured, not
estimated, so the envelope opens from a point rather than a step.

### 6.9 Output

`total_forecast_units`, `total_forecast_low` / `_high`, `interval_pct`,
`residual_stdev`, `safety_stock_units`, `recommended_units`,
`orders_per_month`, `months_of_history`, `method_label`, `notes`, the history
and forecast rows, a prose `explanation`, and the chart.

The answer prompt gives this route a **fixed three-section structure**, since a
forecast always has the same parts to report:

| Section | Contents |
|---|---|
| *(opening)* | Bold title line, a sentence of introduction, then **the notes** — before any number |
| `### Projection` | Monthly figures and the horizon total **as a range**, never a point |
| `### Inventory Recommendation` | The recommended quantity, and in one clause how safety stock was derived |
| `### Method and Confidence` | The method in plain language, what twelve months supports, and — if the projection is flat — why that is a decision rather than a failure |

The range requirement is the important one: the interval is the honest part of
this forecast, and an answer quoting only the midpoint throws it away. The
prompt also forbids the model computing its own interval, total or
recommendation — every figure must come from the payload.

---

## 7. Data model and metric definitions

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

Orders that are `in_transit`, `canceled` or `exception` have no delivery outcome
and are excluded from every rate denominator. Using `COUNT(*)` instead of
`SUM(is_completed)` would count them as on time and understate the delay rate —
the single most likely correctness error in the product, which is why the
denominator is a column rather than a convention.

The flags are `smallint` rather than `boolean` deliberately: PostgreSQL has no
`AVG(boolean)`, so a rate over a boolean flag fails at execution and costs a
retry. `is_promo` is a real boolean and is documented as such in the schema file.

---

## 8. Assumptions and simplifications

**The dataset's last order date is "today".** The data covers 2025-01-01 to
2025-12-30 while the wall clock is well past that. Relative expressions resolve
against **2025-12-30**, not the current date — resolving against the real clock
would return zero rows on exactly the questions the specification gives as
examples. The anchor is declared in `schema/v_orders.yaml`, surfaced in
`/api/health` and `/api/dashboard`, and the answer states which dates a relative
phrase resolved to.

**"Late" means `status = 'delayed'` and nothing else.** With no promised
delivery date in the source, there is no threshold against which lateness could
otherwise be computed.

**Forecasting is not meaningful per SKU.** A question naming a SKU is resolved
to that SKU's `product_category`, and the answer leads with the fact that it was.

**Demand is measured as units ordered** (`SUM(quantity)`), not order count.

**No authentication.** The API is public. There are no accounts, and
`session_id` is generated by the client purely to group turns.

**Order data is read-only.** The application never writes to `logistics_orders`;
the only table it writes is `chat_turn`.

**One repair attempt for invalid SQL**, capped for latency reasons described in
[5.4](#54-failure-modes).

**Results are capped at 500 rows** and generated queries at a 10-second
statement timeout.

---

## 9. Limitations and unsupported queries

**Not supported by design:**

- **Joins and multi-table queries.** `v_orders` is the only readable relation.
- **Anything requiring data not in the dataset** — shipping cost, weather,
  customer names, carrier contracts, promised delivery dates.
- **Chart revision.** "Make that a bar chart" is not supported;
  `POST /api/chart` regenerates from scratch rather than modifying a chart.
- **Cross-year comparison.** Every row is 2025, so any other year filter returns
  nothing.
- **Seasonal forecasting.** Twelve months means exactly one observation per
  calendar month, so a seasonal index would be fitting each month to itself.
- **Write operations of any kind** through the natural-language interface.

**Known weaknesses:**

- **Small-sample aggregates.** 400 orders across 47 destination cities and 30
  clients means per-city and per-client figures rest on very few rows. Forecasts
  carry a sparsity note; ad-hoc query results do not.
- **Forecast confidence is an error bound, not a seasonal model.** The interval
  reflects the spread of the method's own residuals over twelve lumpy months.
  It is honest about the level; it cannot anticipate a seasonal peak.
- **Causal questions.** "Why did delays increase in July?" is answered with what
  the data shows, not a cause — the dataset has no explanatory variables.
- **Conversation memory is three turns**, router-only, scoped to a `session_id`
  the client is responsible for.
- **No caching.** Every question is a fresh model call and a fresh query.
- **No automated tests.** Verification was manual against the running service.
- **Cold starts.** The service currently runs at `min-instances=0` so an idle
  deployment costs nothing. The first request after an idle period therefore
  waits on a container start plus the LangGraph and Vertex client imports —
  perhaps 10–25 seconds, which reads as a broken application rather than a slow
  one. `--cpu-boost` shortens it but does not remove it. Setting
  `MIN_INSTANCES=1` eliminates it, and that is the intended configuration for
  review; `/api/health` is also a cheap warm-up target.
- **Data residency.** `gemini-3.7-flash` is served only from the `global`
  endpoint, which carries no residency guarantee. For a real client with EU
  orders this would need revisiting; Gemini 3.5 Flash is the newest model
  supporting EU residency.

---

## 10. Deployment

```bash
./deploy.sh
```

Reads `.env`, builds through Cloud Build into Artifact Registry, and deploys to
Cloud Run with the Cloud SQL instance attached and the runtime service account
bound. One-time API enablement and IAM bindings are included as commented
commands at the top of the script.

### Runtime configuration

| Setting | Value | Why |
|---|---|---|
| `--memory` | `512Mi` | The dataset is 400 rows and results are capped at 500, so nothing large is ever held in memory. The floor is the Python process plus the LangGraph and Vertex imports, which fits comfortably. Halved from 1Gi after observing actual usage |
| `--cpu` | `1` | The workload is I/O bound — database round trips and Vertex calls |
| `--cpu-boost` | on | Extra CPU during container start, which is where the only real latency is |
| `--min-instances` | `0` *(currently)* | No cost while idle. **Set to `1` for review** — see the cold-start note in [limitations](#9-limitations-and-unsupported-queries) |
| `--max-instances` | `10` | |
| `--timeout` | `300` | An SSE connection stays open for the whole turn |
| `--allow-unauthenticated` | on | There is no auth layer; the frontend is a separate public service |

One worker per container (`--workers 1` in the Dockerfile): the workload is I/O
bound, and a second worker would double the connection pool against the smallest
Cloud SQL tier for no throughput.

The deployed service uses the **attached service account** for both Cloud SQL
and Vertex AI — `GOOGLE_APPLICATION_CREDENTIALS` is not set in the container and
no key file is baked into the image (`.dockerignore` excludes
`service_account/`). Locally the same code path reads the key file instead; the
presence of the file is what decides, not an environment flag.

After deploying:

```bash
curl "$SERVICE_URL/api/health"
```

---

## 11. Future improvements

In the order I would actually do them:

1. **Tests.** Fixture cases of question → expected-SQL-shape for the router, and
   unit tests for every validator rejection path. The validator is the security
   boundary and currently has no automated coverage.
2. **Response caching** keyed on the normalised question — faster demos, lower
   model spend.
3. **A structured query IR as the first path**, falling back to generated SQL.
   Emitting a validated `{metric, dimensions, filters, grain}` object for common
   question shapes would make most turns provably correct rather than
   validated-after-the-fact.
4. **Chart revision**, carrying the previous config forward.
5. **Clarifying questions for ambiguous input.** "Show me the bad ones" should
   ask what "bad" means rather than guessing.
6. **Significance warnings on thin ad-hoc aggregates**, reusing the sparsity
   logic the forecast route already has.
7. **Streaming the chart config incrementally** so the panel can render a
   skeleton before the model call returns.
8. **Observability** — per-turn latency and token accounting, currently only in
   stdout logs.

---

## 12. AI assistance disclosure

This project was built with the assistance of an AI coding assistant (Claude).
It was used for code generation, architectural discussion, and debugging. All
architectural decisions — the metric-view approach, the three-layer SQL
guardrails, the parallel chart branch, the date-anchoring strategy, the decision
to degrade SKU-level forecasts, and the residual-based prediction interval —
were reviewed and directed by me, and the resulting code was read, tested and
corrected before being committed.
