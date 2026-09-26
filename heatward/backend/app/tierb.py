"""Tier B -- the trained model from notebook sections 4-6, served live.

WHAT IT IS.  A HistGradientBoostingRegressor (Poisson loss) that predicts health burden
4 days ahead (hospitalisations + 3 x reported deaths) from same-day weather/indices,
short rolling weather stats, ward demographics and LAGGED burden.

WHAT IT IS NOT.  It is trained on SYNTHETIC outcomes (notebook section 4: NCRB-anchored
Poisson counts driven by outdoor WBGT). Its metrics validate the pipeline; they are not
epidemiological accuracy.  The notebook says so, and so does every surface that shows it.

THE LAG PROBLEM.  burden_lag1/3/7 and burden_roll7_lag1 are past health outcomes.  In
real time there are none, so at serving time they are filled with the generator's
EXPECTED burden for those past days (computed from the real weather).  This is stated in
/api/meta.  When a health department shares real daily counts, replace `burden_proxy`
with them -- nothing else changes.

TRAIN/SERVE PARITY.  `build_features` is the single function used by both the training
script and the live service, so the two cannot drift apart.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("heatward.tierb")

HORIZON = 4                    # days ahead -- midpoint of the PS's 3-5 day requirement
STRESS_C = 30.0
STREAK_CAP = 14                # consecutive_stress_days is capped so a rolling window
MIN_HISTORY = STREAK_CAP       # of >=14 prior days reproduces training features exactly
HOSP_MULT, DEATH_WEIGHT = 18, 3

# Real NCRB West Bengal heat-stroke deaths (MoES Lok Sabha reply, 20 Aug 2025) -- notebook section 4
NCRB_WB_DEATHS = {2018: 46, 2019: 49, 2020: 6, 2021: 11, 2022: 18}
KOLKATA_SHARE = 0.25           # an ASSUMPTION, stated in the notebook

DEMOGRAPHICS = ["population", "elderly_pct", "outdoor_worker_pct", "informal_housing_pct", "population_density"]
WEATHER_INDEX = ["temperature_c", "relative_humidity", "wind_speed_ms", "solar_radiation", "direct_radiation",
                 "tmrt_c", "heat_index_c", "wet_bulb_c", "wbgt_shade_c", "wbgt_outdoor_c", "utci_c"]
ENGINEERED = ["burden_lag1", "burden_lag3", "burden_lag7", "burden_roll7_lag1",
              "temperature_c_roll3", "temperature_c_roll7max", "wbgt_outdoor_c_roll3", "wbgt_outdoor_c_roll7max",
              "utci_c_roll3", "utci_c_roll7max", "consecutive_stress_days", "month", "doy",
              "thermal_x_elderly", "thermal_x_outdoor"]
FEATURES = WEATHER_INDEX + DEMOGRAPHICS + ENGINEERED       # 31, same set the notebook derives (all are BUILT;
                                                           # a trained artifact may use a subset)
# Calendar features let trees memorise dates from a handful of synthetic events (event rate ~0.2%).
# Observed: a hard step in predictions at a specific day-of-year. Dropped by default; --keep-doy restores the notebook.
DEFAULT_DROP = ("doy",)


def anchor_annual() -> float:
    return float(np.mean(list(NCRB_WB_DEATHS.values())) * KOLKATA_SHARE)


def generator_raw(d: pd.DataFrame) -> pd.Series:
    """Notebook section 4, channel 1: relative daily hazard before scaling to the NCRB anchor."""
    vuln = (d.elderly_pct / 100 * 1.6 + d.outdoor_worker_pct / 100 * 1.2 + d.informal_housing_pct / 100 * 0.8)
    load = np.clip((d.wbgt_outdoor_c - 30.0) / 5.0, 0, None) ** 1.5
    return load * (1 + vuln)


def expected_burden(d: pd.DataFrame, scale: float) -> pd.Series:
    """E[hospitalisations + 3*deaths] = lambda*(18+3).  Used for the lag features at serving time."""
    return generator_raw(d) * scale * (HOSP_MULT + DEATH_WEIGHT)


def build_features(d: pd.DataFrame, burden_col: str) -> pd.DataFrame:
    """Notebook section 5 feature engineering. Input needs ward_id, date, WEATHER_INDEX,
    DEMOGRAPHICS and `burden_col`. Adds ENGINEERED columns and `_pos` (rows of history)."""
    d = d.sort_values(["ward_id", "date"]).reset_index(drop=True)
    g = d.groupby("ward_id")
    d["_pos"] = g.cumcount()
    for lag in (1, 3, 7):
        d[f"burden_lag{lag}"] = g[burden_col].shift(lag)
    d["burden_roll7_lag1"] = g[burden_col].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    for col in ("temperature_c", "wbgt_outdoor_c", "utci_c"):
        d[f"{col}_roll3"] = g[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
        d[f"{col}_roll7max"] = g[col].transform(lambda s: s.rolling(7, min_periods=1).max())
    d["consecutive_stress_days"] = g["wbgt_outdoor_c"].transform(
        lambda s: (s > STRESS_C).groupby((s <= STRESS_C).cumsum()).cumsum()).clip(upper=STREAK_CAP)
    d["month"], d["doy"] = d.date.dt.month, d.date.dt.dayofyear
    d["thermal_x_elderly"] = d.wbgt_outdoor_c * d.elderly_pct / 100
    d["thermal_x_outdoor"] = d.wbgt_outdoor_c * d.outdoor_worker_pct / 100
    return d


class TierBModel:
    def __init__(self, artifact: dict):
        self.a = artifact
        self.model = artifact["model"]
        self.features = artifact["features"]
        self.mmt = float(artifact["mmt"])

    @classmethod
    def load(cls, path: Path) -> "TierBModel | None":
        path = Path(path)
        if not path.exists():
            return None
        try:
            import joblib
            art = joblib.load(path)
            if not set(art.get("features", [])) <= set(FEATURES):
                log.error("tierb artifact uses features this code does not build -- retrain it")
                return None
            return cls(art)
        except Exception as e:                       # corrupt file, sklearn version skew, ...
            log.error("could not load tierb model %s: %s", path, e)
            return None

    def describe(self) -> dict:
        a = self.a
        return {"available": True, "trained_on": a["trained_on"], "created_at": a["created_at"],
                "train_range": a["train_range"], "horizon_days": a["horizon"], "n_features": len(self.features),
                "n_wards": a["n_wards"], "mmt_c": round(self.mmt, 2), "metrics": a["metrics"],
                "dropped_features": [f for f in FEATURES if f not in self.features],
                "lag_features": "simulated: generator-expected burden from real weather (no real outcome feed)",
                "outcome_data": "SYNTHETIC — NCRB-anchored Poisson counts (notebook section 4)"}

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """frame: one row per ward-day with WEATHER_INDEX + DEMOGRAPHICS columns, ~28+ days of
        history.  Returns predictions keyed by the TARGET date (feature date + 4 days)."""
        d = frame[["ward_id", "date"] + WEATHER_INDEX + DEMOGRAPHICS].copy()
        d["burden_proxy"] = expected_burden(d, self.a["scale"])
        d = build_features(d, "burden_proxy")
        d = d[d._pos >= MIN_HISTORY].copy()           # only rows whose history matches training
        if d.empty:
            return pd.DataFrame(columns=["ward_id", "date", "tierb_burden", "tierb_clim", "tierb_ratio"])
        mu = np.clip(self.model.predict(d[self.features].to_numpy(float)), 1e-9, None)
        clim_tbl, fb = self.a["clim"], self.a["fallback"]
        clim = np.array([clim_tbl.get((w, m), fb) for w, m in zip(d.ward_id, d.month)], float)
        ratio = mu / np.maximum(clim, self.a["ratio_floor"])
        return pd.DataFrame({"ward_id": d.ward_id.to_numpy(),
                             "date": (d.date + pd.Timedelta(days=HORIZON)).to_numpy(),
                             "tierb_burden": mu, "tierb_clim": clim, "tierb_ratio": ratio})
