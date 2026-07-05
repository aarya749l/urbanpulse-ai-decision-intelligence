"""
tools.py
--------
Predictive-analytics tools exposed as Gemini "Function Calling" tools for the
UrbanPulse mobility Decision Intelligence agent.

Each tool simulates a forecasting model that would, in production, be trained
on historical data warehoused in BigQuery (traffic sensors, transit smart-card
taps, parking meter feeds, event calendars, etc.). Here we generate
deterministic, realistic-looking synthetic time series so the prototype is
fully self-contained and reproducible without external data access -- while
keeping the exact same function signature / docstring pattern you would use
to swap in a real BigQuery ML or Vertex AI Forecasting model later.

Add new tools by:
  1. Writing a new function below with type hints + a clear docstring.
  2. Registering it in the `AVAILABLE_TOOLS` list at the bottom of this file.
  3. Adding it to the `TOOL_DISPATCH` dict so app.py can execute it by name.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Deterministic seeding helpers
# ---------------------------------------------------------------------------
def _stable_seed(*parts: str) -> int:
    """Builds a deterministic integer seed from one or more strings so the
    same route/zone/date combination always produces the same synthetic
    forecast within and across sessions (unlike Python's randomized
    string hash())."""
    combined = "|".join(p.lower().strip() for p in parts)
    return sum((i + 1) * ord(c) for i, c in enumerate(combined)) % (2**31)


def _parse_date(date_str: str) -> datetime:
    """Parses a date string in a forgiving way, accepting 'YYYY-MM-DD' or
    'YYYY/MM/DD'. Falls back to today's date if parsing fails."""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return datetime.utcnow()


def _is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5  # Saturday=5, Sunday=6


# ---------------------------------------------------------------------------
# Synthetic baseline curve generators (0.0-1.0 normalized shape per hour)
# ---------------------------------------------------------------------------
def _base_congestion_shape(is_weekend: bool) -> List[float]:
    """Returns a normalized 24-value hourly congestion shape (0-1)."""
    if is_weekend:
        # Flatter midday bump, no sharp commute peaks.
        return [
            0.10, 0.08, 0.06, 0.05, 0.05, 0.08, 0.15, 0.25,
            0.35, 0.45, 0.55, 0.60, 0.62, 0.60, 0.58, 0.55,
            0.52, 0.50, 0.48, 0.42, 0.35, 0.28, 0.20, 0.14,
        ]
    # Weekday: sharp morning + evening commute peaks.
    return [
        0.08, 0.05, 0.04, 0.04, 0.08, 0.20, 0.45, 0.75,
        0.90, 0.70, 0.50, 0.45, 0.48, 0.50, 0.48, 0.55,
        0.72, 0.95, 0.85, 0.60, 0.40, 0.28, 0.18, 0.12,
    ]


def _apply_noise_and_modifier(
    shape: List[float], seed: int, modifier: float = 0.0, noise_amp: float = 0.06
) -> List[float]:
    """Applies a deterministic per-route/zone offset and small hourly noise
    to a normalized base shape, then returns the adjusted 0-1 series."""
    rng = random.Random(seed)
    adjusted = []
    for value in shape:
        noise = rng.uniform(-noise_amp, noise_amp)
        adjusted.append(max(0.0, min(1.0, value + modifier + noise)))
    return adjusted


def _maybe_inject_event_spike(
    shape: List[float], seed: int, spike_probability: float = 0.35
) -> List[float]:
    """Deterministically decides (based on the seed) whether a local event
    (concert, roadwork, festival, incident) perturbs 2-3 consecutive hours
    of the forecast, simulating a real-world anomaly for demo purposes."""
    rng = random.Random(seed + 999)
    if rng.random() > spike_probability:
        return shape
    start_hour = rng.randint(9, 20)
    duration = rng.randint(2, 3)
    boost = rng.uniform(0.25, 0.4)
    spiked = shape.copy()
    for h in range(start_hour, min(start_hour + duration, 24)):
        spiked[h] = max(0.0, min(1.0, spiked[h] + boost))
    return spiked


def _categorize(value_0_to_1: float, labels: List[str]) -> str:
    """Buckets a normalized 0-1 value into one of 4 category labels."""
    if value_0_to_1 < 0.30:
        return labels[0]
    if value_0_to_1 < 0.55:
        return labels[1]
    if value_0_to_1 < 0.80:
        return labels[2]
    return labels[3]


# ---------------------------------------------------------------------------
# Tool 1: Traffic congestion forecasting
# ---------------------------------------------------------------------------
def forecast_traffic_congestion(
    route_name: str, forecast_date: str, forecast_hour: Optional[int] = None
) -> Dict[str, Any]:
    """Forecasts road traffic congestion for a named route or corridor on a
    given date, using a predictive model trained on historical traffic
    sensor patterns (commute peaks, day-of-week effects, and local events).

    Use this tool whenever the user asks about expected traffic, congestion
    levels, travel delays, or the best time to drive on a specific road,
    highway, or corridor.

    Args:
        route_name: The name of the road, highway, or corridor to forecast
            (e.g. "Marine Drive", "Western Express Highway", "MG Road").
        forecast_date: The date to forecast for, formatted "YYYY-MM-DD"
            (e.g. "2026-07-08").
        forecast_hour: Optional hour of day in 24-hour format (0-23) to
            highlight as the primary answer (e.g. 8 for 8 AM). If omitted,
            the current hour is used.

    Returns:
        A dictionary containing:
            - "route_name", "forecast_date", "requested_hour"
            - "predicted_congestion_percent": congestion at the requested hour (0-100)
            - "congestion_category": one of "Light", "Moderate", "Heavy", "Severe"
            - "confidence_percent": model confidence for this forecast
            - "contributing_factors": list of short factors driving the forecast
            - "hourly_series": 24 entries of {"hour", "congestion_percent"} for charting
    """
    dt = _parse_date(forecast_date)
    hour = forecast_hour if forecast_hour is not None else datetime.utcnow().hour
    hour = max(0, min(23, hour))

    seed = _stable_seed(route_name, forecast_date)
    shape = _base_congestion_shape(_is_weekend(dt))
    route_modifier = (seed % 21 - 10) / 100.0  # +/- 0.10 route-specific bias
    shape = _apply_noise_and_modifier(shape, seed, modifier=route_modifier)
    shape = _maybe_inject_event_spike(shape, seed)

    congestion_at_hour = shape[hour]
    category = _categorize(congestion_at_hour, ["Light", "Moderate", "Heavy", "Severe"])

    factors = ["Historical commute pattern for this corridor", "Day-of-week effect"]
    if shape[hour] > _base_congestion_shape(_is_weekend(dt))[hour] + 0.2:
        factors.append("Possible local event or incident detected near this time")

    return {
        "route_name": route_name,
        "forecast_date": dt.date().isoformat(),
        "requested_hour": hour,
        "predicted_congestion_percent": round(congestion_at_hour * 100, 1),
        "congestion_category": category,
        "confidence_percent": round(78 + (seed % 15), 1),
        "contributing_factors": factors,
        "hourly_series": [
            {"hour": h, "congestion_percent": round(v * 100, 1)}
            for h, v in enumerate(shape)
        ],
    }


# ---------------------------------------------------------------------------
# Tool 2: Public transit ridership demand forecasting
# ---------------------------------------------------------------------------
def forecast_transit_ridership(
    line_name: str, forecast_date: str, forecast_hour: Optional[int] = None
) -> Dict[str, Any]:
    """Forecasts expected public transit ridership (bus or metro line
    passenger demand) for a given line and date, to help transit operators
    plan capacity, scheduling, and crowd management.

    Use this tool whenever the user asks about expected ridership, crowding,
    passenger demand, or the best time to travel on a specific bus or metro
    line.

    Args:
        line_name: The name or number of the transit line (e.g. "Metro Line 3",
            "Route 42 Bus", "Central Line").
        forecast_date: The date to forecast for, formatted "YYYY-MM-DD".
        forecast_hour: Optional hour of day in 24-hour format (0-23) to
            highlight as the primary answer. If omitted, the current hour is
            used.

    Returns:
        A dictionary containing:
            - "line_name", "forecast_date", "requested_hour"
            - "predicted_passengers_per_hour": estimated passenger count at the requested hour
            - "demand_category": one of "Low", "Moderate", "High", "Very High"
            - "confidence_percent": model confidence for this forecast
            - "recommended_action": a short operational suggestion
            - "hourly_series": 24 entries of {"hour", "passengers_per_hour"} for charting
    """
    dt = _parse_date(forecast_date)
    hour = forecast_hour if forecast_hour is not None else datetime.utcnow().hour
    hour = max(0, min(23, hour))

    seed = _stable_seed(line_name, forecast_date, "transit")
    shape = _base_congestion_shape(_is_weekend(dt))  # commute-shaped demand curve
    line_modifier = (seed % 17 - 8) / 100.0
    shape = _apply_noise_and_modifier(shape, seed, modifier=line_modifier, noise_amp=0.05)
    shape = _maybe_inject_event_spike(shape, seed, spike_probability=0.25)

    max_capacity_per_hour = 2000
    passengers = shape[hour] * max_capacity_per_hour
    demand_fraction = shape[hour]
    category = _categorize(demand_fraction, ["Low", "Moderate", "High", "Very High"])

    if category in ("High", "Very High"):
        action = "Consider deploying additional vehicles or shorter headways during this window."
    elif category == "Moderate":
        action = "Standard scheduling should be sufficient; monitor for changes."
    else:
        action = "Off-peak schedule is appropriate; opportunity for maintenance windows."

    return {
        "line_name": line_name,
        "forecast_date": dt.date().isoformat(),
        "requested_hour": hour,
        "predicted_passengers_per_hour": round(passengers),
        "demand_category": category,
        "confidence_percent": round(75 + (seed % 18), 1),
        "recommended_action": action,
        "hourly_series": [
            {"hour": h, "passengers_per_hour": round(v * max_capacity_per_hour)}
            for h, v in enumerate(shape)
        ],
    }


# ---------------------------------------------------------------------------
# Tool 3: Parking availability prediction
# ---------------------------------------------------------------------------
def predict_parking_availability(
    zone_name: str, forecast_date: str, forecast_hour: Optional[int] = None
) -> Dict[str, Any]:
    """Predicts the percentage of available parking spots in a named zone or
    facility at a given date and time, to help commuters and city planners
    anticipate parking pressure.

    Use this tool whenever the user asks about parking availability, whether
    a lot or zone will be full, or the best time to find parking somewhere.

    Args:
        zone_name: The name of the parking zone, lot, or facility (e.g.
            "Downtown Zone C", "Central Station Parking", "Mall Basement Lot").
        forecast_date: The date to forecast for, formatted "YYYY-MM-DD".
        forecast_hour: Optional hour of day in 24-hour format (0-23) to
            highlight as the primary answer. If omitted, the current hour is
            used.

    Returns:
        A dictionary containing:
            - "zone_name", "forecast_date", "requested_hour"
            - "predicted_availability_percent": estimated free capacity at the requested hour (0-100)
            - "availability_category": one of "Full", "Limited", "Moderate", "Plentiful"
            - "confidence_percent": model confidence for this forecast
            - "hourly_series": 24 entries of {"hour", "availability_percent"} for charting
    """
    dt = _parse_date(forecast_date)
    hour = forecast_hour if forecast_hour is not None else datetime.utcnow().hour
    hour = max(0, min(23, hour))

    seed = _stable_seed(zone_name, forecast_date, "parking")
    demand_shape = _base_congestion_shape(_is_weekend(dt))
    zone_modifier = (seed % 15 - 7) / 100.0
    demand_shape = _apply_noise_and_modifier(demand_shape, seed, modifier=zone_modifier, noise_amp=0.05)

    # Availability is inversely related to nearby demand/congestion.
    availability_shape = [max(0.0, min(1.0, 1.0 - d)) for d in demand_shape]

    availability_at_hour = availability_shape[hour]
    category = _categorize(
        1.0 - availability_at_hour, ["Plentiful", "Moderate", "Limited", "Full"]
    )

    return {
        "zone_name": zone_name,
        "forecast_date": dt.date().isoformat(),
        "requested_hour": hour,
        "predicted_availability_percent": round(availability_at_hour * 100, 1),
        "availability_category": category,
        "confidence_percent": round(76 + (seed % 16), 1),
        "hourly_series": [
            {"hour": h, "availability_percent": round(v * 100, 1)}
            for h, v in enumerate(availability_shape)
        ],
    }


# ---------------------------------------------------------------------------
# Tool 4: Mobility anomaly detection
# ---------------------------------------------------------------------------
def detect_mobility_anomalies(area_name: str, forecast_date: str) -> Dict[str, Any]:
    """Compares the predicted traffic/demand pattern for an area against its
    typical historical baseline for the same day-of-week, and flags hours
    where a significant deviation (anomaly) is expected -- such as a local
    event, incident, or unusual surge in activity.

    Use this tool whenever the user asks about unusual patterns, anomalies,
    expected disruptions, or "is anything unusual happening" in a given area
    on a given date.

    Args:
        area_name: The name of the area, corridor, or district to analyze
            (e.g. "Downtown District", "Airport Corridor", "Riverside Zone").
        forecast_date: The date to analyze, formatted "YYYY-MM-DD".

    Returns:
        A dictionary containing:
            - "area_name", "forecast_date"
            - "anomalies_detected": count of anomalous hours found
            - "anomaly_details": list of {"hour", "typical_percent", "predicted_percent", "deviation_percent"}
            - "summary": a short natural-language-friendly summary string
    """
    dt = _parse_date(forecast_date)
    seed = _stable_seed(area_name, forecast_date, "anomaly")

    typical_shape = _base_congestion_shape(_is_weekend(dt))
    modifier = (seed % 21 - 10) / 100.0
    baseline = _apply_noise_and_modifier(typical_shape, seed, modifier=modifier, noise_amp=0.03)
    predicted = _maybe_inject_event_spike(baseline.copy(), seed, spike_probability=0.55)

    anomalies = []
    for hour, (base_val, pred_val) in enumerate(zip(baseline, predicted)):
        deviation = (pred_val - base_val) * 100
        if abs(deviation) >= 15:
            anomalies.append(
                {
                    "hour": hour,
                    "typical_percent": round(base_val * 100, 1),
                    "predicted_percent": round(pred_val * 100, 1),
                    "deviation_percent": round(deviation, 1),
                }
            )

    if anomalies:
        summary = (
            f"{len(anomalies)} anomalous hour(s) detected in {area_name} on "
            f"{dt.date().isoformat()}, likely due to a local event, incident, "
            f"or unusual surge in activity."
        )
    else:
        summary = (
            f"No significant anomalies detected in {area_name} on "
            f"{dt.date().isoformat()}. Patterns are expected to follow the "
            f"typical baseline for this day of week."
        )

    return {
        "area_name": area_name,
        "forecast_date": dt.date().isoformat(),
        "anomalies_detected": len(anomalies),
        "anomaly_details": anomalies,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
AVAILABLE_TOOLS = [
    forecast_traffic_congestion,
    forecast_transit_ridership,
    predict_parking_availability,
    detect_mobility_anomalies,
]

TOOL_DISPATCH = {
    "forecast_traffic_congestion": forecast_traffic_congestion,
    "forecast_transit_ridership": forecast_transit_ridership,
    "predict_parking_availability": predict_parking_availability,
    "detect_mobility_anomalies": detect_mobility_anomalies,
}
