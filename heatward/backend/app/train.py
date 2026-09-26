"""Train Tier B -- a faithful port of notebook sections 1-6, runnable on your machine.

    python -m app.train                       # real ERA5 via Open-Meteo archive (needs internet)
    python -m app.train --source demo         # offline, synthetic weather (pipeline check only)
    python -m app.train --wards pilot         # only W01-W10, closest to the notebook

Writes models/tierb.joblib. The live service loads it at startup. It also stores the local
80th-percentile MMT (notebook section 4), which the service then uses for Tier A unless
MMT_C is set explicitly.

Everything downstream of the weather is SYNTHETIC by construction (notebook section 4):
NCRB-anchored Poisson outcomes driven by outdoor WBGT. Read the metrics as pipeline checks.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from .config import Settings, load_settings
from .indices import add_indices
from .tierb import (DEFAULT_DROP, FEATURES, HORIZON, HOSP_MULT, DEATH_WEIGHT, anchor_annual, build_features,
                    expected_burden, generator_raw)
from .weather import cell_of, collapse_daily, demo_payloads

log = logging.getLogger("heatward.train")
SEED = 42
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation,direct_radiation"


# ----------------------------------------------------------------------------- weather (section 2)
def _fetch_cell_era5(lat, lon, start, end, s: Settings, cache: Path, tries=4) -> pd.DataFrame:
    f = cache / f"era5_{lat}_{lon}_{start}_{end}.csv"
    if f.exists():
        return pd.read_csv(f, parse_dates=["date"])
    for attempt in range(tries):
        try:
            r = httpx.get(ARCHIVE, timeout=180, params={
                "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                "hourly": HOURLY, "timezone": s.tz})
            if r.status_code == 429:
                time.sleep(20 * (attempt + 1)); continue
            r.raise_for_status()
            out = collapse_daily(r.json())
            out.to_csv(f, index=False)
            return out
        except httpx.HTTPError as e:
            if attempt == tries - 1:
                raise
            log.warning("archive fetch failed (%s), retrying", e); time.sleep(10 * (attempt + 1))
    raise RuntimeError("archive: retries exhausted")


def load_weather(source: str, cells, start: str, end: str, s: Settings, cache: Path) -> pd.DataFrame:
    frames = []
    if source == "demo":
        n = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
        payloads = demo_payloads(cells, s, start=dt.datetime.fromisoformat(start), n_days=n)
        for (lat, lon), p in zip(cells, payloads):
            d = collapse_daily(p); d.insert(0, "cell_lon", lon); d.insert(0, "cell_lat", lat); frames.append(d)
    else:
        cache.mkdir(parents=True, exist_ok=True)
        for lat, lon in cells:
            d = _fetch_cell_era5(lat, lon, start, end, s, cache)
            d.insert(0, "cell_lon", lon); d.insert(0, "cell_lat", lat); frames.append(d)
            log.info("cell (%s, %s): %d days, Tmax %.1f C", lat, lon, len(d), d.temperature_c.max())
            time.sleep(1.2)
    return add_indices(pd.concat(frames, ignore_index=True))


# ----------------------------------------------------------------------------- metrics (section 6)
def poisson_dev(y, mu):
    y = np.asarray(y, float); mu = np.clip(np.asarray(mu, float), 1e-9, None)
    r = np.divide(y, mu, out=np.ones_like(y), where=y > 0)
    return float(2 * np.mean(np.where(y > 0, y * np.log(r), 0.0) - (y - mu)))


def pr_auc(y, sc):
    from sklearn.metrics import average_precision_score
    b = (np.asarray(y) > 0).astype(int)
    return float(average_precision_score(b, sc)) if b.sum() else float("nan")


# ----------------------------------------------------------------------------- train
def train(source="era5", start="2019-01-01", end="2025-12-31", wards_mode="all",
          out: Path | None = None, test_years=(2025,), val_years=(2024,), perm_repeats=8,
          settings: Settings | None = None, keep_doy: bool = False) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    s = settings or load_settings()
    out = Path(out or s.tierb_path)

    allw = pd.read_csv(s.wards_csv)
    wards = allw[allw.modeled == 1]
    if wards_mode == "pilot":
        wards = wards[wards.ward_no <= 10]
    wards = wards.set_index("ward_id")
    ward_cell = {w: cell_of(r.latitude, r.longitude, s.grid_deg) for w, r in wards.iterrows()}
    cells = sorted(set(ward_cell.values()))
    log.info("%d wards, %d weather cells, source=%s, %s..%s", len(wards), len(cells), source, start, end)

    cw = load_weather(source, cells, start, end, s, s.cache_dir)
    cm = pd.DataFrame([(w, c[0], c[1]) for w, c in ward_cell.items()], columns=["ward_id", "cell_lat", "cell_lon"])
    d = cm.merge(cw, on=["cell_lat", "cell_lon"]).merge(
        wards[["population", "elderly_pct", "outdoor_worker_pct", "informal_housing_pct", "population_density"]],
        left_on="ward_id", right_index=True)
    d["date"] = pd.to_datetime(d["date"])

    # -- section 4: NCRB-anchored synthetic outcomes, and the local MMT ---------------------
    years = d.date.dt.year.nunique()
    raw = generator_raw(d)
    scale = anchor_annual() * years / max(float(raw.sum()), 1e-9)
    lam = raw * scale
    rng = np.random.default_rng(SEED)
    d["reported_deaths"] = rng.poisson(np.clip(lam, 0, None))
    d["hospitalizations"] = rng.poisson(np.clip(lam * HOSP_MULT, 0, None))
    d["health_burden"] = d.hospitalizations + d.reported_deaths * DEATH_WEIGHT
    mmt = float(np.percentile(d.wbgt_outdoor_c, 80))
    log.info("MMT = %.2f C outdoor WBGT (local p80)", mmt)

    # -- section 5: features, target, split by YEAR -----------------------------------------
    feats = [f for f in FEATURES if keep_doy or f not in DEFAULT_DROP]
    d = build_features(d, "health_burden")
    g = d.groupby("ward_id")
    d["target"] = g["health_burden"].shift(-HORIZON)
    d = d.dropna(subset=["target"]).reset_index(drop=True)
    d = d[d._pos >= 14].reset_index(drop=True)             # same eligibility rule as serving
    yr = d.date.dt.year
    d["split"] = np.where(yr.isin(test_years), "test", np.where(yr.isin(val_years), "validation", "train"))
    heat = d[d.date.dt.month.isin([4, 5, 6])].groupby("split").size()
    gate3 = bool((heat > 0).all() and len(heat) == 3)
    tr, va, te = (d[d.split == k] for k in ("train", "validation", "test"))
    if min(len(tr), len(va), len(te)) == 0:
        raise SystemExit("A split is empty -- widen --start/--end or adjust test/val years.")

    # -- section 6: train, evaluate vs ward-month climatology -------------------------------
    model = HistGradientBoostingRegressor(
        loss="poisson", max_depth=3, learning_rate=.05, max_iter=400, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=.15, random_state=SEED)
    model.fit(tr[feats].to_numpy(float), tr["target"].to_numpy(float))

    clim = tr.assign(m=tr.date.dt.month).groupby(["ward_id", "m"])["target"].mean()
    fallback = float(tr["target"].mean())
    clim_dict = {(w, int(m)): float(v) for (w, m), v in clim.items()}

    def clim_pred(p):
        idx = pd.MultiIndex.from_arrays([p.ward_id, p.date.dt.month])
        return np.clip(clim.reindex(idx).fillna(fallback).to_numpy(float), 1e-9, None)

    metrics = {"splits": {}}
    for name, part in (("validation", va), ("test", te)):
        p = np.clip(model.predict(part[feats].to_numpy(float)), 1e-9, None)
        b, y = clim_pred(part), part["target"].to_numpy(float)
        dm, db = poisson_dev(y, p), poisson_dev(y, b)
        metrics["splits"][name] = {"dev_model": dm, "dev_clim": db, "skill_vs_clim": (1 - dm / db) if db > 0 else None,
                                   "pr_auc_model": pr_auc(y, p), "pr_auc_clim": pr_auc(y, b),
                                   "event_rate": float((y > 0).mean()), "rows": int(len(y))}
    metrics["gate3_heat_season_in_every_split"] = gate3

    # -- Gate 4: do the indices beat raw temperature? --------------------------------------
    y_te = te["target"].to_numpy(float)
    imp = permutation_importance(model, te[feats].to_numpy(float), y_te, n_repeats=perm_repeats,
                                 random_state=SEED, scoring="neg_mean_poisson_deviance")
    sr = pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False)
    thermal = [c for c in ("wbgt_outdoor_c", "utci_c", "heat_index_c", "temperature_c") if c in sr.index]
    best = sr[thermal].idxmax()
    gate4 = ("INCONCLUSIVE" if sr[best] <= 0 else "FAIL" if best == "temperature_c" else "PASS")
    metrics["gate4"] = {"result": gate4, "strongest_thermal_feature": best, "value": float(sr[best]),
                        "top_features": {k: float(v) for k, v in sr.head(8).items()}}

    # "normal" is the ward-month mean, but never below the all-year average: with ~0.2% event rates the
    # ward-month mean is ~0 outside the heat season and any ratio against it explodes.
    floor = max(fallback, 1e-9)
    art = {"model": model, "features": feats, "clim": clim_dict, "fallback": fallback, "ratio_floor": floor,
           "scale": float(scale), "mmt": mmt, "trained_on": source, "horizon": HORIZON,
           "train_range": [start, end], "n_wards": int(len(wards)), "metrics": metrics,
           "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "wards": list(wards.index)}
    out.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(art, out)
    log.info("saved %s", out)
    return art


def _report(art: dict):
    m = art["metrics"]
    print("\n=== Tier B trained ===")
    print(f"trained on : {art['trained_on'].upper()} weather, {art['train_range'][0]} .. {art['train_range'][1]}, {art['n_wards']} wards")
    print(f"features   : {len(art['features'])} of 31 (dropped: {', '.join(f for f in FEATURES if f not in art['features']) or 'none'})")
    print(f"MMT        : {art['mmt']:.2f} C outdoor WBGT (local p80)  <- Tier A will use this unless MMT_C is set")
    for k, v in m["splits"].items():
        sk = v["skill_vs_clim"]
        print(f"{k:<11}: skill vs ward-month climatology = {sk:+.4f}   PR-AUC model {v['pr_auc_model']:.4f} vs clim {v['pr_auc_clim']:.4f}   event rate {v['event_rate']:.4%}")
    print(f"GATE 3     : {'PASS' if m['gate3_heat_season_in_every_split'] else 'FAIL'}")
    g = m["gate4"]
    print(f"GATE 4     : {g['result']}  (strongest thermal feature: {g['strongest_thermal_feature']}, {g['value']:+.5f})")
    print("\nOutcomes are SYNTHETIC (NCRB-anchored). Skill <= 0 means climatology already explains the target;")
    print("that is a finding about sparse official counts, not a bug. Do not quote these as epidemiological accuracy.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["era5", "demo"], default="era5")
    ap.add_argument("--start", default="2019-01-01"); ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--wards", choices=["all", "pilot"], default="all")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fast", action="store_true", help="fewer permutation-importance repeats")
    ap.add_argument("--keep-doy", action="store_true", help="keep day-of-year as a feature (notebook-exact; overfits calendar dates)")
    a = ap.parse_args()
    _report(train(a.source, a.start, a.end, a.wards, a.out, perm_repeats=2 if a.fast else 8, keep_doy=a.keep_doy))


if __name__ == "__main__":
    main()
