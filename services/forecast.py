"""Demand forecasting.

Three honest constraints shape everything here.

The first is that the dataset holds 355 distinct SKUs across 400 orders, and
313 of those SKUs appear exactly once. SKU-level forecasting is not hard here,
it is impossible, and producing a number anyway would be the one failure this
assignment is most explicit about - the AI must not answer without computation
that supports the answer. So a SKU question is resolved to that SKU's product
category, and the answer says so in the first sentence.

The second is twelve monthly observations with visible lumpiness (75 orders in
January, 21 in June). Moving average and least-squares regression are the two
methods that stay honest at that sample size. Anything seasonal would be
fitting noise - with twelve months there is exactly one observation per
calendar month, so a seasonal index would be fitting each month to itself.

The third is that a point estimate on data this thin is misleading on its own.
A flat line with nothing around it reads as though the method gave up. The
same flat line inside a widening 80% interval reads as what it actually is:
the level is known, the month-to-month movement is not. So every forecast
carries an interval, and the same residual spread that draws it also sizes the
safety stock - one number telling one story in both places.

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
from models.schemas import ChartBand, FullChartConfig

# Below this many orders a SKU has no history worth extrapolating, and the
# forecast is computed for its category instead.
MIN_SKU_ORDERS = 8

# Months of history averaged by the moving-average method.
MOVING_AVERAGE_WINDOW = 3

# 1.65 standard deviations is the usual one-sided 95% service level. Named
# rather than inlined because it is the one number in the inventory
# recommendation a reader might want to argue with.
#
# Note this is deliberately not the same confidence as the displayed interval
# below. Stocking to 95% while showing an 80% planning range is normal
# practice - you hold more than you expect to need - and the explanation text
# says so, because side by side and unexplained it looks like a mistake.
SAFETY_STOCK_Z = 1.65

# The displayed interval. 80% rather than 95% because the wider band on a
# series this noisy spans nearly the whole plot area, which tells the reader
# less than a narrower one does.
INTERVAL_PCT = 80

# Below this many orders per month, month-to-month movement is mostly sampling
# noise rather than demand behaviour, and the answer should say so rather than
# letting the reader infer a trend from four-order months.
SPARSE_ORDERS_PER_MONTH = 10

# Student's t at the 0.90 quantile - the multiplier for a two-sided 80%
# interval - by degrees of freedom.
#
# A table rather than scipy, which would be a large dependency for one lookup.
# The normal 1.282 is what you get without it, and at the sample sizes here
# that interval is about 8% too narrow, which is exactly the direction a small
# sample should not err in.
_T_90 = {
    1: 3.078, 2: 1.886, 3: 1.638, 4: 1.533, 5: 1.476,
    6: 1.440, 7: 1.415, 8: 1.397, 9: 1.383, 10: 1.372,
    11: 1.363, 12: 1.356, 13: 1.350, 14: 1.345, 15: 1.341,
    16: 1.337, 17: 1.333, 18: 1.330, 19: 1.328, 20: 1.325,
    21: 1.323, 22: 1.321, 23: 1.319, 24: 1.318, 25: 1.316,
    26: 1.315, 27: 1.314, 28: 1.313, 29: 1.311, 30: 1.310,
}

_T_90_LARGE = 1.282


def _t90(df: int) -> float:
    if df < 1:
        return _T_90[1]

    return _T_90.get(df, _T_90_LARGE)


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


def _sigma(residuals: List[float], df: int, fallback: List[float]) -> Tuple[float, int]:
    """
    Spread of the method's own errors, which is what an interval is made of.

    Falls back to the spread of the observations themselves when the series is
    too short to leave any residuals behind. That is a worse estimate - it
    counts the trend as though it were error - but it is the conservative
    direction to be wrong in, and it only fires on series of two or three
    months where nothing better exists.
    """
    if df >= 1 and len(residuals) >= 2:
        total = sum(value * value for value in residuals)

        return (total / df) ** 0.5, df

    if len(fallback) >= 2:
        return statistics.pstdev(fallback), max(len(fallback) - 1, 1)

    return 0.0, 1


def _moving_average(
    values: List[float],
    horizon: int,
) -> Tuple[List[float], float, int]:
    """
    Returns (predictions, residual spread, degrees of freedom).

    The predictions are flat by construction. A moving average has no trend
    component, and projecting one from it would be inventing information - the
    interval is where the uncertainty goes, not the line.
    """
    window = values[-MOVING_AVERAGE_WINDOW:] or values
    level = statistics.fmean(window)

    # One-step-ahead errors: what this method would have predicted for each
    # month, judged against what actually happened. One parameter is estimated
    # (the level), so the residual count loses one degree of freedom.
    residuals = [
        values[index]
        - statistics.fmean(values[index - MOVING_AVERAGE_WINDOW : index])
        for index in range(MOVING_AVERAGE_WINDOW, len(values))
    ]

    spread, df = _sigma(residuals, len(residuals) - 1, values)

    return [round(level, 2)] * horizon, spread, df


def _linear_regression(
    values: List[float],
    horizon: int,
) -> Tuple[List[float], float, int]:
    count = len(values)

    if count < 2:
        return [round(values[0] if values else 0.0, 2)] * horizon, 0.0, 1

    xs = list(range(count))
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(values)

    denominator = sum((x - mean_x) ** 2 for x in xs)

    if denominator == 0:
        return [round(mean_y, 2)] * horizon, statistics.pstdev(values), count - 1

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values)) / denominator
    intercept = mean_y - slope * mean_x

    # Two parameters estimated - slope and intercept - so two degrees of
    # freedom are spent.
    residuals = [value - (intercept + slope * x) for x, value in zip(xs, values)]

    spread, df = _sigma(residuals, count - 2, values)

    # Demand cannot be negative, and an unclamped downward trend goes there
    # within a few months on a series this short.
    return (
        [
            round(max(0.0, intercept + slope * (count + step)), 2)
            for step in range(horizon)
        ],
        spread,
        df,
    )


def _interval(
    predicted: List[float],
    spread: float,
    df: int,
) -> List[Tuple[float, float]]:
    """
    Bounds for each projected month.

    Width grows with the square root of the horizon. For a strictly stationary
    series it arguably should not grow at all, but the level estimate itself is
    uncertain over several months - and the safety stock below already assumes
    the same sqrt(h) accumulation, so using it here keeps the band and the
    stock recommendation telling one story instead of two.
    """
    multiplier = _t90(df)

    return [
        (
            round(max(0.0, value - multiplier * spread * ((step + 1) ** 0.5)), 2),
            round(value + multiplier * spread * ((step + 1) ** 0.5), 2),
        )
        for step, value in enumerate(predicted)
    ]


def _chart(
    history: List[Dict[str, Any]],
    forecast: List[Dict[str, Any]],
    title: str,
) -> FullChartConfig:
    """
    History and forecast as two series on one axis, with the interval shaded
    behind the projected half.

    Separate columns rather than one, so the frontend can style the projected
    half differently without having to know where the split is - the actual
    series is simply null from the split onward.

    The bounds are in data but not in headers, so they shade the chart without
    turning into two extra columns in the table beside it.
    """
    data = [
        {
            "month": row["month"],
            "actual": row["demand"],
            "forecast": None,
            "forecast_low": None,
            "forecast_high": None,
        }
        for row in history
    ]

    if data:
        # Join the two lines at the last observation, otherwise the forecast
        # series starts detached from the history it came from. The band is
        # pinned to that same point with zero width: the last month is an
        # observation, not an estimate, so there is nothing uncertain about it
        # and the envelope should open from a point rather than a step.
        last = history[-1]["demand"]

        data[-1]["forecast"] = last
        data[-1]["forecast_low"] = last
        data[-1]["forecast_high"] = last

    data.extend(
        {
            "month": row["month"],
            "actual": None,
            "forecast": row["demand"],
            "forecast_low": row["low"],
            "forecast_high": row["high"],
        }
        for row in forecast
    )

    return FullChartConfig(
        chartType="line",
        title=title,
        description=(
            "Monthly units ordered, with the projected period shown separately. "
            f"The shaded band is the {INTERVAL_PCT}% interval."
        ),
        xKey="month",
        yKeys=["actual", "forecast"],
        seriesKey=None,
        stacked=False,
        insight="",
        band=ChartBand(
            lowKey="forecast_low",
            highKey="forecast_high",
            ofKey="forecast",
            label=f"{INTERVAL_PCT}% interval",
        ),
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

    label = {
        "overall": "all orders",
        "category": f"the {key} category",
        "region": f"the {key} region",
    }.get(level, "all orders")

    # ------------------------------------------------------------
    # How much data is actually behind each monthly point
    # ------------------------------------------------------------

    total_orders = sum(int(row["orders"]) for row in history)
    orders_per_month = total_orders / len(history)

    if orders_per_month < SPARSE_ORDERS_PER_MONTH:
        notes.append(
            f"{label.capitalize()} averages {orders_per_month:.1f} orders per month "
            f"across this history, so month-to-month movement is mostly sampling "
            f"noise rather than demand behaviour. The forecast reflects the average "
            f"level; the interval reflects how wide that noise is."
        )

    # ------------------------------------------------------------
    # Fit, project, and put an interval around the projection
    # ------------------------------------------------------------

    fit = (
        _linear_regression if method == "linear_regression" else _moving_average
    )

    predicted, spread, df = fit(values, horizon)
    bounds = _interval(predicted, spread, df)

    last_month = date.fromisoformat(history[-1]["month"])

    forecast_rows = [
        {
            "month": _add_months(last_month, step + 1).isoformat(),
            "demand": value,
            "low": low,
            "high": high,
        }
        for step, (value, (low, high)) in enumerate(zip(predicted, bounds))
    ]

    # ------------------------------------------------------------
    # Inventory recommendation
    # ------------------------------------------------------------

    total_forecast = sum(predicted)

    # Safety stock covers forecast error, so it is sized from the same residual
    # spread that draws the band - not from the spread of the observations,
    # which counts the trend as though it were error and overstates the stock.
    safety_stock = SAFETY_STOCK_Z * spread * (horizon**0.5)

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
        "orders_per_month": round(orders_per_month, 1),
        "months_of_history": len(history),
        "total_forecast_units": round(total_forecast, 2),
        "total_forecast_low": round(sum(low for low, _ in bounds), 2),
        "total_forecast_high": round(sum(high for _, high in bounds), 2),
        "interval_pct": INTERVAL_PCT,
        "residual_stdev": round(spread, 2),
        "safety_stock_units": round(safety_stock, 2),
        "recommended_units": round(total_forecast + safety_stock, 2),
        "explanation": (
            f"Forecast for {label} over the next {horizon} month(s), using a "
            f"{method_label} on {len(history)} months of history "
            f"({history[0]['month']} to {history[-1]['month']}). "
            f"The projection totals {round(total_forecast, 2)} units, with an "
            f"{INTERVAL_PCT}% interval of {round(sum(low for low, _ in bounds), 2)} to "
            f"{round(sum(high for _, high in bounds), 2)} units, widened over the "
            f"horizon from a residual spread of {round(spread, 2)} units per month. "
            f"Recommended stock is the forecast plus {round(safety_stock, 2)} units of "
            f"safety stock at {SAFETY_STOCK_Z} standard deviations - roughly a 95% "
            f"service level, deliberately higher than the {INTERVAL_PCT}% display "
            f"interval, because holding stock protects against the bad tail while the "
            f"band describes the likely range. "
            f"With twelve months of lumpy history this is a level estimate with an "
            f"error bound, not a seasonal model."
        ),
        "chart": _chart(
            history,
            forecast_rows,
            f"Demand Forecast: {label.title()}",
        ),
    }
