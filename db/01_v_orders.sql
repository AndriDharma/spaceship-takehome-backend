-- ============================================================
-- v_orders : the metric layer
-- ============================================================
--
-- This view is the ONLY relation the text2sql path is allowed to read, and it
-- is the only schema the model is ever shown. Everything the assignment calls
-- a metric is computed here, once, so a generated query cannot invent its own
-- definition of "delayed" or "on time" and quietly answer two questions two
-- different ways.
--
-- The flags are SMALLINT rather than BOOLEAN on purpose. Postgres has no
-- AVG(boolean), so a rate over a boolean flag fails at execution time and
-- costs a retry; over 0/1 it is just AVG(is_delayed). Naming them is_* keeps
-- them readable in a generated SELECT list.
--
-- Column comments are not decoration: the prompt builder reads them out of
-- the catalog and inlines them next to each column, which is how the model
-- learns that in_transit rows are excluded from the rate denominators without
-- that rule having to live in prose in the system prompt.

DROP VIEW IF EXISTS v_orders;

CREATE VIEW v_orders AS
SELECT
    o.client_id,
    o.order_id,
    o.order_date,
    o.delivery_date,
    o.carrier,
    o.origin_city,
    o.destination_city,
    o.status,
    o.sku,
    o.product_category,
    o.quantity,
    o.unit_price_usd,
    o.order_value_usd,
    o.is_promo,
    o.promo_discount_pct,
    o.region,
    o.warehouse,

    -- Outcome flags. There is no promised-delivery or SLA column anywhere in
    -- the source data, so on-time can only be read off the status. delivered
    -- is on time, delayed is not, and the three statuses that describe an
    -- order which has not finished - in_transit, canceled, exception - are
    -- counted in neither.
    CASE WHEN o.status = 'delayed'   THEN 1 ELSE 0 END::smallint AS is_delayed,
    CASE WHEN o.status = 'delivered' THEN 1 ELSE 0 END::smallint AS is_on_time,

    -- The denominator for both rates, so that a rate is always
    -- SUM(is_on_time) / NULLIF(SUM(is_completed), 0) and never a COUNT(*)
    -- that silently includes orders still in transit.
    CASE WHEN o.status IN ('delivered', 'delayed') THEN 1 ELSE 0 END::smallint AS is_completed,

    -- NULL for the 30 rows with no delivery_date, which is what keeps AVG()
    -- over this column correct without anyone having to write a WHERE clause.
    (o.delivery_date - o.order_date) AS delivery_days,

    -- Pre-truncated grains. Generated SQL gets these wrong more often than it
    -- gets anything else wrong - date_trunc returns a timestamp, which then
    -- groups and sorts inconsistently against a date axis - so the truncation
    -- happens here and casts back to date.
    date_trunc('month', o.order_date)::date AS order_month,
    date_trunc('week',  o.order_date)::date AS order_week,
    EXTRACT(YEAR FROM o.order_date)::int    AS order_year,

    ROUND(o.order_value_usd * (1 - o.promo_discount_pct / 100.0), 2) AS net_value_usd

FROM logistics_orders o;


COMMENT ON VIEW v_orders IS
    'One row per purchase order, with delivery outcome flags and date grains precomputed. The only relation exposed to generated SQL.';

COMMENT ON COLUMN v_orders.order_date    IS 'Date the order was placed. Data covers 2025-01-01 to 2025-12-30; treat the latest order_date as "today" when resolving relative periods.';
COMMENT ON COLUMN v_orders.delivery_date IS 'Date the order was delivered. NULL while an order is in_transit or canceled.';
COMMENT ON COLUMN v_orders.status        IS 'One of: delivered, delayed, in_transit, exception, canceled.';
COMMENT ON COLUMN v_orders.is_delayed    IS '1 when status = delayed, else 0. Delay rate is SUM(is_delayed) / NULLIF(SUM(is_completed), 0).';
COMMENT ON COLUMN v_orders.is_on_time    IS '1 when status = delivered, else 0. On-time rate is SUM(is_on_time) / NULLIF(SUM(is_completed), 0).';
COMMENT ON COLUMN v_orders.is_completed  IS '1 when the order finished as delivered or delayed. Always the denominator for a rate; never use COUNT(*), which would include in_transit and canceled orders.';
COMMENT ON COLUMN v_orders.delivery_days IS 'delivery_date - order_date, in days. NULL when undelivered, so AVG() is already correct without filtering.';
COMMENT ON COLUMN v_orders.order_month   IS 'order_date truncated to the first of its month, as a date. Use this for monthly grouping.';
COMMENT ON COLUMN v_orders.order_week    IS 'order_date truncated to the start of its ISO week, as a date. Use this for weekly grouping.';
COMMENT ON COLUMN v_orders.net_value_usd IS 'order_value_usd after promo_discount_pct is applied. Use order_value_usd for gross revenue.';
COMMENT ON COLUMN v_orders.region        IS 'One of: US-E, US-W, US-C, EU, UK.';
COMMENT ON COLUMN v_orders.sku           IS 'Individual product code. 355 distinct values over 400 rows, most appearing once - too sparse to forecast; forecast on product_category instead.';
