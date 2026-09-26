"""Weather acquisition.

Notebook section 2 used the ERA5 *archive* API, which lags real time by days and
never contains the future. A live system needs the *forecast* API: it takes the
same hourly variables, returns recent days (`past_days`) plus the forecast
(`forecast_days`), and accepts many locations in one request.

The daily collapse is identical to the notebook: take every variable AT the hour
of peak temperature, so the humidity is the humidity that co-occurred with the
heat (daily-max humidity happens at dawn and would be meaningless).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd

from .config import Settings

log = logging.getLogger("heatward.weather")

HOURLY = "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation,direct_radiation"
Cell = tuple  # (lat, lon) snapped


def snap(v: float, grid: float) -> float:
    return round(round(v / grid) * grid, 4)


def cell_of(lat: float, lon: float, grid: float) -> Cell:
    return (snap(lat, grid), snap(lon, grid))


# --------------------------------------------------------------------------- fetch
async def fetch_open_meteo(cells: list[Cell], s: Settings,
                           client: httpx.AsyncClient | None = None, tries: int = 4) -> list[dict]:
    """One request for all cells. Returns one payload dict per cell, in request order."""
    params = {
        "latitude": ",".join(str(c[0]) for c in cells),
        "longitude": ",".join(str(c[1]) for c in cells),
        "hourly": HOURLY,
        "timezone": s.tz,
        "past_days": s.history_days,
        "forecast_days": s.forecast_days,
    }
    own = client is None
    client = client or httpx.AsyncClient(timeout=60)
    try:
        for attempt in range(tries):
            try:
                r = await client.get(s.forecast_url, params=params)
                if r.status_code == 429:                       # rate limited -> back off
                    await asyncio.sleep(min(20 * (attempt + 1), 60))
                    continue
                r.raise_for_status()
                data = r.json()
                data = [data] if isinstance(data, dict) else data   # 1 location -> dict, N -> list
                if len(data) != len(cells):
                    raise ValueError(f"asked for {len(cells)} locations, got {len(data)}")
                return data
            except (httpx.HTTPError, ValueError) as e:
                if attempt == tries - 1:
                    raise
                log.warning("open-meteo attempt %d failed: %s", attempt + 1, e)
                await asyncio.sleep(3 * (attempt + 1))
        raise RuntimeError("open-meteo: retries exhausted (rate limited)")
    finally:
        if own:
            await client.aclose()


# --------------------------------------------------------------------------- collapse
def collapse_daily(payload: dict) -> pd.DataFrame:
    """Hourly -> one row per day, values at the hour of peak temperature."""
    h = pd.DataFrame(payload["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    h["date"] = h["time"].dt.normalize()
    h = h.dropna(subset=["temperature_2m"])
    if h.empty:
        return pd.DataFrame()
    pk = h.loc[h.groupby("date")["temperature_2m"].idxmax()].copy()
    out = pk.rename(columns={
        "temperature_2m": "temperature_c",
        "relative_humidity_2m": "relative_humidity",
        "wind_speed_10m": "wind_speed_ms",
        "shortwave_radiation": "solar_radiation"})
    out["wind_speed_ms"] = out["wind_speed_ms"] / 3.6            # km/h -> m/s
    out["peak_hour"] = out["time"].dt.hour
    cols = ["date", "peak_hour", "temperature_c", "relative_humidity",
            "wind_speed_ms", "solar_radiation", "direct_radiation"]
    out = out[cols].dropna().reset_index(drop=True)              # a half-filled day is dropped, not guessed
    return out


def payloads_to_daily(cells: list[Cell], payloads: list[dict]) -> pd.DataFrame:
    frames = []
    for (lat, lon), p in zip(cells, payloads):
        d = collapse_daily(p)
        if d.empty:
            continue
        d.insert(0, "cell_lon", lon)
        d.insert(0, "cell_lat", lat)
        frames.append(d)
    if not frames:
        raise ValueError("no usable weather rows in provider response")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- demo source
# Monthly mean of daily-max temperature and the RH at that hour for Kolkata,
# roughly IMD normals. This is DEMO DATA, not observations: it exists so the
# stack can run offline and so tests are deterministic. The API labels it.
_TMAX = [27.0, 30.0, 34.0, 36.0, 36.5, 34.5, 32.5, 32.5, 32.5, 32.0, 29.5, 26.5]
_RHPK = [45, 40, 40, 50, 58, 68, 76, 77, 76, 68, 55, 48]


def _monthly(table, ts):
    x = ts.month - 1 + (ts.day - 1) / 30.0
    i0 = int(np.floor(x)) % 12
    i1 = (i0 + 1) % 12
    return table[i0] + (table[i1] - table[i0]) * (x - np.floor(x))


def demo_payloads(cells: list[Cell], s: Settings, now: datetime | None = None,
                  start: datetime | None = None, n_days: int | None = None) -> list[dict]:
    """Synthetic hourly weather in Open-Meteo's response shape. `start`/`n_days` let the
    Tier B training script ask for a multi-year 'archive' instead of the live window."""
    tz = ZoneInfo(s.tz)
    now = now or datetime.now(tz)
    if start is None:
        start = (now - timedelta(days=s.history_days)).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    n_days = n_days or (s.history_days + s.forecast_days)
    out = []
    for lat, lon in cells:
        rng = np.random.default_rng(int(abs(lat * 1000) + abs(lon * 100)) + start.toordinal())
        times, T, RH, WS, SW, DR = [], [], [], [], [], []
        noise = 0.0
        for k in range(n_days):
            day = start + timedelta(days=k)
            noise = 0.7 * noise + rng.normal(0, 1.2)                        # persistent anomaly
            tmax = _monthly(_TMAX, day) + noise + s.demo_temp_offset_c + (lat - 22.57) * -1.0
            rhpk = float(np.clip(_monthly(_RHPK, day) - 2.0 * noise, 20, 95))
            cloud = float(np.clip(rng.beta(2, 3) + (0.25 if 6 <= day.month <= 9 else 0), 0, 0.95))
            wind = float(np.clip(rng.normal(11, 3), 3, 25))
            for hr in range(24):
                c = np.cos(2 * np.pi * (hr - 14) / 24)
                times.append((day + timedelta(hours=hr)).strftime("%Y-%m-%dT%H:%M"))
                T.append(round(tmax - 4.5 + 4.5 * c, 1))
                RH.append(int(np.clip(rhpk + 15 - 15 * c, 5, 100)))
                WS.append(round(wind * (0.7 + 0.3 * max(c, 0)), 1))
                sun = max(0.0, np.sin(np.pi * (hr - 6) / 12))
                sw = 900 * sun * (1 - 0.75 * cloud)
                SW.append(round(sw, 1)); DR.append(round(sw * (0.75 - 0.6 * cloud), 1))
        out.append({"latitude": lat, "longitude": lon, "timezone": s.tz,
                    "hourly": {"time": times, "temperature_2m": T, "relative_humidity_2m": RH,
                               "wind_speed_10m": WS, "shortwave_radiation": SW, "direct_radiation": DR}})
    return out
