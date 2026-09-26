"""HEATWARD API.

  GET  /api/health            liveness + whether data is loaded
  GET  /api/meta              engine parameters, provenance, freshness
  GET  /api/timeline          city-level series: past days, today, forecast
  GET  /api/risk?date=        every ward for one day (defaults to today)
  GET  /api/wards             ward metadata (centroid, demographics, source flags)
  GET  /api/wards/{ward_id}   one ward: metadata + its daily series
  POST /api/refresh           force a re-fetch (rate-limited; token if REFRESH_TOKEN is set)
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from .config import load_settings
from .risk import UTCI_TIERS
from .service import RiskService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

settings = load_settings()
svc = RiskService(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    svc.load_cache()                                        # instant data after a restart
    task = asyncio.create_task(svc.run_forever())           # first live fetch happens in the background
    yield
    task.cancel()


app = FastAPI(title="HEATWARD API", version="1.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                   allow_methods=["GET", "POST"], allow_headers=["*"])

PROVENANCE = {
    "weather": "REAL — Open-Meteo forecast API (recent days + forecast), value taken at the hour of peak temperature",
    "thermal_index": "COMPUTED — NOAA HI, Stull wet bulb, BOM shade WBGT, outdoor WBGT, UTCI (pythermalcomfort)",
    "ward_geometry": "REAL — supplied KMC 141-ward GeoJSON (centroids computed from it)",
    "demographics": "SYNTHETIC — Census-2011-shaped ranges, not extracted from Census tables",
    "excess_mortality": "TRANSFERRED — published exposure-response curve; NOT trained, NOT validated",
    "tier_b": "LEARNED — gradient-boosting model trained on SYNTHETIC NCRB-anchored outcomes; lag inputs simulated; pipeline check only",
}


def today_ist() -> date:
    return datetime.now(ZoneInfo(settings.tz)).date()


def need_data():
    if svc.snapshot is None:
        raise HTTPException(503, detail={"status": "warming_up", "last_error": svc.last_error})
    return svc.snapshot


def offset_of(iso: str) -> int:
    return (date.fromisoformat(iso) - today_ist()).days


def cached(resp: Response, seconds: int = 60):
    resp.headers["Cache-Control"] = f"public, max-age={seconds}"


@app.get("/api/health")
def health():
    return {"status": "ok", "has_data": svc.snapshot is not None, "stale": svc.is_stale()}


@app.get("/api/meta")
def meta(resp: Response):
    cached(resp, 30)
    snap = svc.snapshot
    return {
        "city": settings.city, "today": today_ist().isoformat(),
        "source": snap.source if snap else None,
        "demo": bool(snap and snap.source == "demo"),
        "generated_at": snap.fetched_at.isoformat() if snap else None,
        "age_minutes": None if svc.age_minutes() is None else round(svc.age_minutes(), 1),
        "stale": svc.is_stale(), "last_error": svc.last_error,
        "refresh_minutes": settings.refresh_minutes,
        "tierb": svc.tierb.describe() if svc.tierb else {
            "available": False,
            "how_to_enable": "python -m app.train   (see backend/README) then restart the server"},
        "engine": {"tier": "A", "index": "wbgt_outdoor_c", "mmt_c": round(svc.s.mmt_c, 2), "mmt_source": svc.mmt_source,
                   "beta": settings.beta, "beta_ci": [settings.beta_lo, settings.beta_hi],
                   "cdr_per_1000": settings.cdr_per_1000},
        "tiers": [{"name": n, "index": k, "label": lab, "utci_below": None if hi == float("inf") else hi}
                  for hi, k, n, lab in UTCI_TIERS],
        "coverage": {"wards_total": int(len(svc.all_wards)), "wards_modeled": int(len(svc.wards)),
                     "weather_cells": len(svc.cells), "grid_deg": settings.grid_deg},
        "provenance": PROVENANCE,
        "unmeasured": "Accuracy against observed ward mortality. No public ward-level heat-mortality dataset exists in India.",
    }


@app.get("/api/timeline")
def timeline(resp: Response):
    cached(resp)
    snap = need_data()
    return {"today": today_ist().isoformat(), "generated_at": snap.fetched_at.isoformat(),
            "days": [{"date": d, "offset": offset_of(d), "is_forecast": offset_of(d) > 0,
                      "city": snap.city_by_day[d]} for d in snap.dates]}


@app.get("/api/risk")
def risk(resp: Response, date: str | None = None):
    cached(resp)
    snap = need_data()
    if date is None:
        t = today_ist().isoformat()
        date = t if t in snap.wards_by_day else snap.dates[-1]
    if date not in snap.wards_by_day:
        raise HTTPException(404, detail={"error": "date not available", "available": snap.dates})
    return {"date": date, "offset": offset_of(date), "generated_at": snap.fetched_at.isoformat(),
            "demo": snap.source == "demo", "city": snap.city_by_day[date], "wards": snap.wards_by_day[date]}


def _ward_meta(row):
    return {"ward_id": row.ward_id, "ward_no": int(row.ward_no), "latitude": row.latitude,
            "longitude": row.longitude, "population": int(row.population),
            "elderly_pct": row.elderly_pct, "outdoor_worker_pct": row.outdoor_worker_pct,
            "informal_housing_pct": row.informal_housing_pct, "modeled": bool(row.modeled),
            "demographics_source": row.demographics_source, "geometry_source": row.geometry_source}


@app.get("/api/wards")
def wards(resp: Response):
    cached(resp, 3600)
    return {"wards": [_ward_meta(r) for r in svc.all_wards.itertuples(index=False)]}


@app.get("/api/wards/{ward_id}")
def ward(ward_id: str, resp: Response):
    cached(resp)
    snap = need_data()
    ward_id = ward_id.upper()
    m = svc.all_wards[svc.all_wards.ward_id == ward_id]
    if m.empty:
        raise HTTPException(404, detail="unknown ward")
    return {**_ward_meta(next(m.itertuples(index=False))),
            "series": snap.series_by_ward.get(ward_id, [])}


@app.post("/api/refresh")
async def refresh(x_refresh_token: str | None = Header(default=None)):
    if settings.refresh_token and x_refresh_token != settings.refresh_token:
        raise HTTPException(401, detail="bad token")
    ok = await svc.refresh(force=True)
    return {"ok": ok, "last_error": svc.last_error}
