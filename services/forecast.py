"""Demand forecasting.

Two honest constraints shape everything here.

The first is that the dataset holds 355 distinct SKUs across 400 orders, and
313 of those SKUs appear exactly once. SKU-level forecasting is not hard here,
it is impossible, and producing a number anyway would be the one failure this
assignment is most explicit about - the AI must not answer without computation
that supports the answer. So a SKU question is resolved to that SKU's product
category, and the answer says so in the first sentence.

The second is twelve monthly observations with visible lumpiness (75 orders in
January, 21 in June). Moving average and least-squares regression are the two
methods that stay honest at that sample size. Anything seasonal would be
fitting noise.

The chart is built here rather than by the model. Its shape - months across,
actual and forecast up - is known before the data is read, so there is nothing
for a model to decide and a model call would be pure latency and pure risk. It
is emitted in the same ChartConfig the SQL path produces, so the frontend has
one renderer.
"""

import statistics
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from core import db
from models.schemas import FullChartConfig

# Below this many orders a SKU has no history worth extrapolating, and the
# forecast is computed for its category instead.
MIN_SKU_ORDERS = 8

# Months of history averaged by the moving-average method.
MOVING_AVERAGE_WINDOW = 3

# 1.65 standard deviations is the usual one-sided 95% service level. Named
# rather than inlined because it is the one number in the inventory
# recommendation a reader might want to argue with.
SAFETY_STOCK_Z = 1.65


def _add_months(anchor: date, count: int) -> date:
    month_index = anchor.month - 1 + count

    return date(anchor.year + month_index // 12, month_index % 12 + 1, 1)


async def _history(level: str, key: Optional[str]) -> List[Dict[str, Any]]:
    """Monthly demand, as units ordered."""
    where = ""
    params: Dict[str, Any] = {}

    if level == "category" and key:
        where = "WHERE product_category = :key"
        params["key"] = key
    elif level == "region" and key:
        where = "WHERE region = :key"
        params["key"] = key

    return await db.query(
        f"""
        SELECT order_month              AS month,
               SUM(quantity)::int       AS demand,
               COUNT(*)::int            AS orders
        FROM   v_orders
        {where}
        GROUP  BY order_month
        ORDER  BY order_month
        """,
        params,
    )


def _densify(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Insert the months that have no orders.

    A gap left as a gap turns a flat series into a rising one, because both
    the moving average and the regression would treat twelve months of data as
    though every observation were consecutive.
    """
    if not rows:
        return []

    parsed = [(date.fromisoformat(str(row["month"])), row) for row in rows]

    first, last = parsed[0][0], parsed[-1][0]
    present = {month: row for month, row in parsed}

    dense: List[Dict[str, Any]] = []
    cursor = first

    while cursor <= last:
        row = present.get(cursor)

        dense.append(
            {
                "month": cursor.isoformat(),
                "demand": int(row["demand"]) if row else 0,
                "orders": int(row["orders"]) if row else 0,
            }
        )

        cursor = _add_months(cursor, 1)

    return dense


def _moving_average(values: List[float], horizon: int) -> List[float]:
    window = values[-MOVING_AVERAGE_WINDOW:] or values
    level = statistics.fmean(window)

    # Flat by construction. A moving average has no trend component, and
    # projecting one from it would be inventing information.
    return [round(level, 2)] * horizon


def _linear_regression(values: List[float], horizon: int) -> List[float]:
    count = len(values)

    if count < 2:
        return [round(values[0] if values else 0.0, 2)] * horizon

    xs = list(range(count))
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(values)

    denominator = sum((x - mean_x) ** 2 for x in xs)

    if denominator == 0:
        return [round(mean_y, 2)] * horizon

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values)) / denominator
    intercept = mean_y - slope * mean_x

    # Demand cannot be negative, and an unclamped downward trend goes there
    # within a few months on a series this short.
    return [
        round(max(0.0, intercept + slope * (count + step)), 2)
        for step in range(horizon)
    ]


def _chart(
    history: List[Dict[str, Any]],
    forecast: List[Dict[str, Any]],
    title: str,
) -> FullChartConfig:
    """
    History and forecast as two series on one axis.

    Separate columns rather than one, so the frontend can style the projected
    half differently without having to know where the split is - the actual
    series is simply null from the split onward.
    """
    data = [
        {"month": row["month"], "actual": row["demand"], "forecast": None}
        for row in history
    ]

    if data:
        # Join the two lines at the last observation, otherwise the forecast
        # series starts detached from the history it came from.
        data[-1]["forecast"] = history[-1]["demand"]

    data.extend(
        {"month": row["month"], "actual": None, "forecast": row["demand"]}
        for row in forecast
    )

    return FullChartConfig(
        chartType="line",
        title=title,
        description="Monthly units ordered, with the projected period shown separately.",
        xKey="month",
        yKeys=["actual", "forecast"],
        seriesKey=None,
        stacked=False,
        insight="",
        data=data,
        headers=["month", "actual", "forecast"],
        columnKinds={"month": "date", "actual": "number", "forecast": "number"},
    )


async def resolve_sku(sku: str) -> Tuple[Optional[str], int]:
    """The category a SKU belongs to, and how many orders it has."""
    rows = await db.query(
        """
        SELECT product_category AS category,
               COUNT(*)::int    AS orders
        FROM   v_orders
        WHERE  UPPER(sku) = UPPER(:sku)
        GROUP  BY product_category
        ORDER  BY orders DESC
        LIMIT  1
        """,
        {"sku": sku},
    )

    if not rows:
        return None, 0

    return rows[0]["category"], rows[0]["orders"]


async def run(
    level: str = "overall",
    key: Optional[str] = None,
    horizon: int = 3,
    method: str = "moving_average",
    requested_sku: Optional[str] = None,
) -> Dict[str, Any]:
    notes: List[str] = []

    # ------------------------------------------------------------
    # Resolve a SKU request to something the data can actually support
    # ------------------------------------------------------------

    if requested_sku:
        category, order_count = await resolve_sku(requested_sku)

        if category is None:
            notes.append(f"SKU {requested_sku} does not appear in the data.")
        elif order_count < MIN_SKU_ORDERS:
            level, key = "category", category
            notes.append(
                f"SKU {requested_sku} has only {order_count} order(s) in the dataset, "
                f"which is too little history to forecast. Forecasting the "
                f"{category} category instead."
            )
        else:
            level, key = "category", category

    history = _densify(await _history(level, key))

    if len(history) < 2:
        return {
            "ok": False,
            "reason": "Not enough history to forecast at this level.",
            "notes": notes,
        }

    values = [float(row["demand"]) for row in history]

    predicted = (
        _linear_regression(values, horizon)
        if method == "linear_regression"
        else _moving_average(values, horizon)
    )

    last_month = date.fromisoformat(history[-1]["month"])

    forecast_rows = [
        {"month": _add_months(last_month, step + 1).isoformat(), "demand": value}
        for step, value in enumerate(predicted)
    ]

    # ------------------------------------------------------------
    # Inventory recommendation
    # ------------------------------------------------------------

    total_forecast = sum(predicted)
    deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
    safety_stock = SAFETY_STOCK_Z * deviation * (horizon**0.5)

    label = {
        "overall": "all orders",
        "category": f"the {key} category",
        "region": f"the {key} region",
    }.get(level, "all orders")

    method_label = (
        "least-squares linear regression over the full history"
        if method == "linear_regression"
        else f"{MOVING_AVERAGE_WINDOW}-month moving average"
    )

    return {
        "ok": True,
        "level": level,
        "key": key,
        "horizon": horizon,
        "method": method,
        "method_label": method_label,
        "notes": notes,
        "history": history,
        "forecast": forecast_rows,
        "total_forecast_units": round(total_forecast, 2),
        "safety_stock_units": round(safety_stock, 2),
        "recommended_units": round(total_forecast + safety_stock, 2),
        "monthly_stdev": round(deviation, 2),
        "explanation": (
            f"Forecast for {label} over the next {horizon} month(s), using a "
            f"{method_label} on {len(history)} months of history "
            f"({history[0]['month']} to {history[-1]['month']}). "
            f"Recommended stock is the {round(total_forecast, 2)} forecast units plus "
            f"{round(safety_stock, 2)} units of safety stock, computed as "
            f"{SAFETY_STOCK_Z} standard deviations of historical monthly demand "
            f"scaled over the horizon - roughly a 95% service level. "
            f"With twelve months of lumpy history this is a trend estimate, "
            f"not a confidence-bounded projection."
        ),
        "chart": _chart(
            history,
            forecast_rows,
            f"Demand Forecast: {label.title()}",
        ),
    }
