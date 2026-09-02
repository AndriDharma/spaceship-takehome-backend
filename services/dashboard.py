"""The descriptive layer: KPIs and the three required charts.

No model runs anywhere in this file. The dashboard is deterministic SQL over
the metric view, which is the point - it is the part of the product that has to
be right every time, and the part a reviewer checks first.

The charts come back as the same FullChartConfig the AI path produces, so the
frontend renders all of them through one component. That is also why the metric
definitions live in v_orders rather than here: the dashboard and a generated
query compute an on-time rate the same way because they read the same column.
"""

from typing import Any, Dict, List

from core import db
from models.schemas import FullChartConfig
from services import chart_config, schema_service

_KPI_SQL = """
SELECT COUNT(*)::int                                            AS total_orders,
       SUM(is_on_time)::int                                     AS delivered_orders,
       SUM(is_delayed)::int                                     AS delayed_orders,
       SUM(is_completed)::int                                   AS completed_orders,
       ROUND(
           SUM(is_on_time)::numeric / NULLIF(SUM(is_completed), 0) * 100,
           1
       )                                                        AS on_time_rate_pct,
       ROUND(AVG(delivery_days)::numeric, 2)                    AS avg_delivery_days,
       ROUND(SUM(order_value_usd), 2)                           AS total_value_usd
FROM   v_orders
"""

_VOLUME_SQL = """
SELECT order_month::text          AS month,
       COUNT(*)::int              AS orders,
       ROUND(SUM(order_value_usd), 2) AS value_usd
FROM   v_orders
GROUP  BY order_month
ORDER  BY order_month
"""

_PERFORMANCE_SQL = """
SELECT order_month::text     AS month,
       SUM(is_on_time)::int  AS on_time,
       SUM(is_delayed)::int  AS delayed
FROM   v_orders
WHERE  is_completed = 1
GROUP  BY order_month
ORDER  BY order_month
"""

_CARRIER_SQL = """
SELECT carrier,
       COUNT(*)::int        AS orders,
       ROUND(
           SUM(is_delayed)::numeric / NULLIF(SUM(is_completed), 0) * 100,
           1
       )                    AS delay_rate_pct
FROM   v_orders
GROUP  BY carrier
ORDER  BY orders DESC
"""


def _chart(
    rows: List[Dict[str, Any]],
    headers: List[str],
    chart_type: str,
    title: str,
    description: str,
    x_key: str,
    y_keys: List[str],
    stacked: bool = False,
) -> FullChartConfig:
    return FullChartConfig(
        chartType=chart_type,
        title=title,
        description=description,
        xKey=x_key,
        yKeys=y_keys,
        seriesKey=None,
        stacked=stacked,
        insight="",
        data=rows,
        headers=headers,
        columnKinds=chart_config.column_kinds(rows, headers),
    )


async def build() -> Dict[str, Any]:
    kpis = (await db.query(_KPI_SQL))[0]

    volume = await db.query(_VOLUME_SQL)
    performance = await db.query(_PERFORMANCE_SQL)
    carrier = await db.query(_CARRIER_SQL)

    meta = schema_service.cached()

    return {
        "kpis": kpis,
        # Shown in the header so a reviewer is never confused about why "last
        # month" does not mean the current calendar month.
        "data_window": {
            "from": str(meta["min_date"]),
            "to": str(meta["max_date"]),
            "row_count": meta["row_count"],
        },
        "charts": [
            _chart(
                volume,
                ["month", "orders", "value_usd"],
                "line",
                "Order Volume Over Time",
                "Orders placed and their total value, by month.",
                "month",
                ["orders"],
            ),
            _chart(
                performance,
                ["month", "on_time", "delayed"],
                "stacked_bar",
                "Delivery Performance",
                "Completed orders each month, split into on-time and delayed.",
                "month",
                ["on_time", "delayed"],
                stacked=True,
            ),
            _chart(
                carrier,
                ["carrier", "orders", "delay_rate_pct"],
                "bar",
                "Orders by Carrier",
                "Order count per carrier, with each carrier's delay rate.",
                "carrier",
                ["orders"],
            ),
        ],
    }
