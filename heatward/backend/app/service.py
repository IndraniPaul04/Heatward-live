"""Orchestration: fetch -> indices -> risk -> an immutable in-memory snapshot.

The API never calls the weather provider per request. A background task refreshes
the snapshot every REFRESH_MINUTES and requests read from memory, so the frontend
can poll as often as it likes without touching Open-Meteo's quota. If a refresh
fails, the previous snapshot keeps being served and is flagged `stale`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import Settings
from .indices import add_indices
from .risk import TIER_NAMES, compute_risk, tier_of, vulnerability_index
from .tierb import DEMOGRAPHICS, TierBModel
from .weather import cell_of, demo_payloads, fetch_open_meteo, payloads_to_daily

log = logging.getLogger("heatward.service")

WARD_FIELDS = ["temperature_c", "relative_humidity", "wind_speed_ms", "wet_bulb_c", "heat_index_c",
               "wbgt_shade_c", "wbgt_outdoor_c", "utci_c", "tmrt_c", "vulnerability",
               "excess_deaths", "excess_deaths_lo", "excess_deaths_hi", "excess_ed_visits",
               "tierb_burden", "tierb_clim", "tierb_ratio"]
ROUND = {"temperature_c": 1, "relative_humidity": 0, "wind_speed_ms": 1, "wet_bulb_c": 1,
         "heat_index_c": 1, "wbgt_shade_c": 1, "wbgt_outdoor_c": 2, "utci_c": 1, "tmrt_c": 1,
         "vulnerability": 3, "excess_deaths": 3, "excess_deaths_lo": 3, "excess_deaths_hi": 3,
         "excess_ed_visits": 2, "tierb_burden": 5, "tierb_clim": 5, "tierb_ratio": 2}


def _num(v, nd):
    """NaN/inf -> None so the JSON encoder never chokes."""
    if v is None:
        return None
    v = float(v)
    return round(v, nd) if np.isfinite(v) else None


@dataclass
class Snapshot:
    fetched_at: datetime
    source: str
    n_cells: int
    dates: list
    wards_by_day: dict = field(default_factory=dict)     # iso date -> [ward dict]
    city_by_day: dict = field(default_factory=dict)      # iso date -> city dict
    series_by_ward: dict = field(default_factory=dict)   # ward_id -> [day dict]


class RiskService:
    def __init__(self, s: Settings):
        self.s = s
        allw = pd.read_csv(s.wards_csv)
        self.all_wards = allw
        self.wards = allw[allw.modeled == 1].set_index("ward_id")
        self.vulnerability = vulnerability_index(self.wards)
        self.snapshot: Snapshot | None = None
        self.last_error: str | None = None
        self.last_attempt: float = 0.0
        self.tierb = TierBModel.load(s.tierb_path)
        # MMT precedence: explicit MMT_C env > the p80 stored with the trained model > back-solved default
        self.mmt_source = "env" if "MMT_C" in os.environ else "default (back-solved from the original dashboard)"
        if self.tierb and "MMT_C" not in os.environ:
            self.s = replace(self.s, mmt_c=self.tierb.mmt)
            self.mmt_source = f"tier B training run ({self.tierb.a['trained_on']})"
        self._lock = asyncio.Lock()
        self._cache_file = s.cache_dir / "last_weather.json"
        s.cache_dir.mkdir(parents=True, exist_ok=True)

        # ward -> weather cell (fixed for the life of the process)
        self.ward_cell = {w: cell_of(r.latitude, r.longitude, s.grid_deg) for w, r in self.wards.iterrows()}
        self.cells = sorted(set(self.ward_cell.values()))

    # ------------------------------------------------------------------ build
    def build(self, payloads: list, source: str, fetched_at: datetime) -> Snapshot:
        daily = payloads_to_daily(self.cells, payloads)
        daily = add_indices(daily)                       # indices depend only on weather -> compute per cell

        cell_df = pd.DataFrame([(w, c[0], c[1]) for w, c in self.ward_cell.items()],
                               columns=["ward_id", "cell_lat", "cell_lon"])
        stress = cell_df.merge(daily, on=["cell_lat", "cell_lon"], how="inner")
        r = compute_risk(stress, self.wards, self.vulnerability, self.s)
        r["ward_no"] = r.ward_id.map(self.wards.ward_no)

        # Tier B needs the whole fetched history (rolling/lag features), so run it BEFORE trimming.
        for c in ("tierb_burden", "tierb_clim", "tierb_ratio"):
            r[c] = np.nan
        if self.tierb is not None:
            try:
                tb_in = stress.merge(self.wards[DEMOGRAPHICS], left_on="ward_id", right_index=True)
                tb = self.tierb.predict(tb_in)
                r = r.drop(columns=["tierb_burden", "tierb_clim", "tierb_ratio"]).merge(
                    tb, on=["ward_id", "date"], how="left")
            except Exception as e:                       # Tier B must never take Tier A down
                log.error("tier B prediction failed, serving Tier A only: %s", e)

        cutoff = pd.Timestamp(datetime.now(ZoneInfo(self.s.tz)).date()) - pd.Timedelta(days=self.s.past_days)
        r = r[r.date >= cutoff]                          # history beyond the display window stays internal

        snap = Snapshot(fetched_at=fetched_at, source=source, n_cells=len(self.cells),
                        dates=sorted({d.strftime("%Y-%m-%d") for d in r.date}))
        for date, g in r.groupby(r.date.dt.strftime("%Y-%m-%d")):
            g = g.sort_values("excess_deaths", ascending=False).reset_index(drop=True)
            rows = []
            for rank, row in enumerate(g.itertuples(index=False), 1):
                d = {"ward_id": row.ward_id, "ward_no": int(row.ward_no), "rank": rank,
                     "tier": row.tier, "tier_index": None if row.tier_index is None else int(row.tier_index),
                     "tier_label": row.tier_label}
                for f in WARD_FIELDS:
                    d[f] = _num(getattr(row, f), ROUND[f])
                rows.append(d)
            snap.wards_by_day[date] = rows
            snap.city_by_day[date] = self._city(g)
        for wid, g in r.groupby("ward_id"):
            g = g.sort_values("date")
            snap.series_by_ward[wid] = [
                {"date": row.date.strftime("%Y-%m-%d"), "tier": row.tier,
                 **{f: _num(getattr(row, f), ROUND[f]) for f in WARD_FIELDS}}
                for row in g.itertuples(index=False)]
        return snap

    def _city(self, g: pd.DataFrame) -> dict:
        ok = g[g.utci_c.notna()]
        w = ok.population.to_numpy(float)
        wavg = lambda col: float(np.average(ok[col], weights=w)) if len(ok) else float("nan")
        utci = wavg("utci_c")
        k, name, label = tier_of(utci)
        counts = {n: int((ok.tier == n).sum()) for n in TIER_NAMES}
        return {"wbgt_outdoor_c": _num(wavg("wbgt_outdoor_c"), 2), "utci_c": _num(utci, 1),
                "temperature_c": _num(wavg("temperature_c"), 1),
                "relative_humidity": _num(wavg("relative_humidity"), 0),
                "tier": name, "tier_index": k, "tier_label": label,
                "excess_deaths": _num(g.excess_deaths.sum(), 2),
                "excess_deaths_lo": _num(g.excess_deaths_lo.sum(), 2),
                "excess_deaths_hi": _num(g.excess_deaths_hi.sum(), 2),
                "excess_ed_visits": _num(g.excess_ed_visits.sum(), 1),
                "tierb_burden": _num(g.tierb_burden.sum(min_count=1), 5),
                "tierb_clim": _num(g.tierb_clim.sum(min_count=1), 5),
                "tierb_ratio": (_num(g.tierb_burden.sum() / max(g.tierb_clim.sum(), 1e-12), 2)
                                if g.tierb_burden.notna().all() else None),
                "wards_by_tier": counts, "n_wards": int(len(ok))}

    # ------------------------------------------------------------------ refresh
    async def refresh(self, force: bool = True) -> bool:
        async with self._lock:
            now = time.monotonic()
            if force and self.last_attempt and now - self.last_attempt < self.s.min_manual_refresh_s:
                return self.last_error is None
            self.last_attempt = now
            try:
                if self.s.weather_source == "demo":
                    payloads, source = demo_payloads(self.cells, self.s), "demo"
                else:
                    payloads, source = await fetch_open_meteo(self.cells, self.s), "open-meteo"
                fetched = datetime.now(timezone.utc)
                snap = self.build(payloads, source, fetched)     # build first: never publish a half-built snapshot
                self.snapshot, self.last_error = snap, None
                self._save_cache(payloads, source, fetched)
                log.info("refreshed: %s, %d cells, %d days", source, len(self.cells), len(snap.dates))
                return True
            except Exception as e:                                # keep serving the old snapshot
                self.last_error = f"{type(e).__name__}: {e}"
                log.error("refresh failed: %s", self.last_error)
                return False

    async def run_forever(self):
        while True:
            ok = await self.refresh(force=False)
            await asyncio.sleep((self.s.refresh_minutes if ok else self.s.retry_minutes) * 60)

    # ------------------------------------------------------------------ disk cache
    def _save_cache(self, payloads, source, fetched):
        try:
            self._cache_file.write_text(json.dumps({
                "fetched_at": fetched.isoformat(), "source": source,
                "cells": self.cells, "payloads": payloads}))
        except OSError as e:
            log.warning("cache write failed: %s", e)

    def load_cache(self) -> bool:
        """Serve something immediately after a restart, even offline."""
        try:
            c = json.loads(self._cache_file.read_text())
            if [tuple(x) for x in c["cells"]] != self.cells or c["source"] != self.s.weather_source:
                return False            # different ward file / grid / source -> don't trust it
            self.snapshot = self.build(c["payloads"], c["source"], datetime.fromisoformat(c["fetched_at"]))
            log.info("loaded cached snapshot from %s", c["fetched_at"])
            return True
        except Exception as e:
            log.info("no usable cache: %s", e)
            return False

    # ------------------------------------------------------------------ status
    def age_minutes(self) -> float | None:
        if not self.snapshot:
            return None
        return (datetime.now(timezone.utc) - self.snapshot.fetched_at).total_seconds() / 60

    def is_stale(self) -> bool:
        a = self.age_minutes()
        return a is None or a > self.s.refresh_minutes * 2.5
