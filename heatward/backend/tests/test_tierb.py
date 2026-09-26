"""Tier B: leakage, train/serve parity, end-to-end serving, and graceful absence."""
import datetime as dt
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from app.config import load_settings
from app.indices import add_indices
from app.tierb import (DEMOGRAPHICS, FEATURES, MIN_HISTORY, WEATHER_INDEX, TierBModel,
                       build_features, expected_burden)
from app.weather import cell_of, collapse_daily, demo_payloads

S = load_settings()


def _history(n_days=60, wards=("W01", "W02"), seed=0):
    """A ward-day frame shaped like the service's `stress` frame, from the demo weather."""
    cells = [(22.6, 88.4)]
    p = demo_payloads(cells, replace(S, history_days=n_days, forecast_days=1), start=dt.datetime(2025, 4, 1), n_days=n_days)
    w = add_indices(collapse_daily(p[0]))
    rng = np.random.default_rng(seed)
    frames = []
    for i, wid in enumerate(wards):
        x = w.copy(); x["ward_id"] = wid
        for c, lo, hi in (("population", 3e4, 9e4), ("elderly_pct", 6, 16), ("outdoor_worker_pct", 8, 32),
                          ("informal_housing_pct", 3, 40), ("population_density", 5e3, 3e4)):
            x[c] = rng.uniform(lo, hi)
        frames.append(x)
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    return d


def test_feature_list_is_the_notebooks_31():
    assert len(FEATURES) == 31 and len(set(FEATURES)) == 31


def test_no_lookahead_features_at_t_ignore_burden_after_t():
    """Leakage guard: changing burden at t+1... must not change any feature at t."""
    d = _history()
    d["b"] = np.random.default_rng(1).poisson(0.5, len(d)).astype(float)
    base = build_features(d, "b")
    for t in (20, 30, 45):
        pert = d.copy()
        m = pert.groupby("ward_id").cumcount() > t
        pert.loc[m, "b"] += 1000.0                                   # wildly change the future of every ward
        alt = build_features(pert, "b")
        at_or_before = base._pos <= t
        pd.testing.assert_frame_equal(base.loc[at_or_before, FEATURES], alt.loc[at_or_before, FEATURES])


def test_serving_window_reproduces_training_features_exactly():
    """Train/serve parity: features from a 28-day window == features from the long history
    (this is why streak length is capped and rows need >=14 days of history)."""
    d = _history(n_days=90)
    d["b"] = expected_burden(d, 0.001)
    full = build_features(d, "b")
    last = d.groupby("ward_id").tail(28).copy()
    win = build_features(last, "b")
    fe = full.merge(win[["ward_id", "date", "_pos"] + FEATURES], on=["ward_id", "date"], suffixes=("", "_w"))
    ok = fe._pos_w >= MIN_HISTORY
    assert ok.sum() > 0
    for c in FEATURES:
        np.testing.assert_allclose(fe.loc[ok, c].to_numpy(float), fe.loc[ok, c + "_w"].to_numpy(float),
                                   rtol=1e-9, atol=1e-9, err_msg=c)


# ---------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def artifact_path(tmp_path_factory):
    from app.train import train
    out = tmp_path_factory.mktemp("tb") / "tierb.joblib"
    s = replace(S, cache_dir=tmp_path_factory.mktemp("cache"))
    train("demo", "2022-01-01", "2025-12-31", "pilot", out, perm_repeats=1, settings=s)
    return out


def test_trained_artifact_loads_and_describes_itself(artifact_path):
    m = TierBModel.load(artifact_path)
    assert m is not None and m.a["trained_on"] == "demo"
    assert "doy" not in m.a["features"] and len(m.a["features"]) == 30      # calendar-date feature dropped by default
    assert m.describe()["dropped_features"] == ["doy"]
    desc = m.describe()
    assert "SYNTHETIC" in desc["outcome_data"] and "simulated" in desc["lag_features"]
    assert set(desc["metrics"]["splits"]) == {"validation", "test"}
    assert desc["metrics"]["gate4"]["result"] in ("PASS", "FAIL", "INCONCLUSIVE")


def test_missing_or_corrupt_artifact_returns_none(tmp_path):
    assert TierBModel.load(tmp_path / "nope.joblib") is None
    bad = tmp_path / "bad.joblib"; bad.write_bytes(b"not a model")
    assert TierBModel.load(bad) is None


def test_service_serves_tierb_for_today_onward_and_null_before(artifact_path):
    from app.service import RiskService
    s = replace(S, tierb_path=artifact_path)
    svc = RiskService(s)
    assert svc.tierb is not None
    assert svc.s.mmt_c == pytest.approx(svc.tierb.mmt)                       # Tier A adopts the trained MMT
    assert svc.mmt_source.startswith("tier B training run")
    snap = svc.build(demo_payloads(svc.cells, s), "demo", pd.Timestamp.now(tz="UTC").to_pydatetime())
    today = snap.dates[S.past_days]
    rows = snap.wards_by_day[today]
    assert all(r["tierb_ratio"] is not None and np.isfinite(r["tierb_ratio"]) and r["tierb_ratio"] > 0 for r in rows)
    # the ratio's denominator is floored at the all-year average, so ward ratios must stay sane (no /~0 blow-ups)
    assert max(r["tierb_ratio"] for r in rows) < 200
    assert snap.city_by_day[today]["tierb_ratio"] is not None
    assert snap.city_by_day[snap.dates[-1]]["tierb_ratio"] is not None       # +5 days: features come from today+1
    # Tier A is untouched by Tier B being present
    assert all(r["tier"] in ("Green", "Yellow", "Orange", "Red") for r in rows)


def test_tierb_failure_does_not_take_tier_a_down(artifact_path, monkeypatch):
    from app.service import RiskService
    s = replace(S, tierb_path=artifact_path)
    svc = RiskService(s)
    monkeypatch.setattr(svc.tierb, "predict", lambda f: (_ for _ in ()).throw(RuntimeError("boom")))
    snap = svc.build(demo_payloads(svc.cells, s), "demo", pd.Timestamp.now(tz="UTC").to_pydatetime())
    r = snap.wards_by_day[snap.dates[S.past_days]][0]
    assert r["tier"] is not None and r["tierb_ratio"] is None


def test_without_artifact_tierb_fields_are_null_and_meta_says_how_to_enable(client_no_tb):
    m = client_no_tb.get("/api/meta").json()
    assert m["tierb"]["available"] is False and "app.train" in m["tierb"]["how_to_enable"]
    w = client_no_tb.get("/api/risk").json()["wards"][0]
    assert w["tierb_ratio"] is None and w["excess_deaths"] is not None


@pytest.fixture(scope="module")
def client_no_tb():
    import time
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        for _ in range(60):
            if c.get("/api/health").json()["has_data"]: break
            time.sleep(0.1)
        yield c
