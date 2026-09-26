"""Run with:  pytest -q   (from backend/)"""
import asyncio, json, math

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.indices import add_indices
from app.risk import compute_risk, tier_of, vulnerability_index
from app.weather import collapse_daily, fetch_open_meteo, cell_of, demo_payloads

S = load_settings()

# Peak-day (17 Jun 2023) outputs from the notebook, as hard-coded in the original
# frontend: ward -> (wbgt, utci, vulnerability, excess, low, high).
NOTEBOOK = {
    "W06": (35.776, 42.6, 0.595, 0.307, 0.177, 0.448), "W08": (35.971, 43.0, 0.652, 0.300, 0.172, 0.440),
    "W02": (35.752, 42.6, 0.587, 0.283, 0.163, 0.412), "W05": (35.971, 43.0, 0.560, 0.209, 0.120, 0.305),
    "W03": (35.786, 42.6, 0.272, 0.177, 0.103, 0.255), "W09": (35.786, 42.6, 0.710, 0.167, 0.096, 0.245),
    "W04": (35.752, 42.6, 0.331, 0.152, 0.088, 0.219), "W07": (35.752, 42.6, 0.742, 0.139, 0.079, 0.203),
    "W10": (35.776, 42.6, 0.542, 0.117, 0.068, 0.170), "W01": (35.786, 42.6, 0.463, 0.106, 0.062, 0.154),
}


# ---------------------------------------------------------------- parity with the notebook
def test_tier_a_reproduces_notebook_peak_day():
    """Same inputs the notebook had -> same outputs the frontend was built on."""
    wards = pd.read_csv(S.wards_csv).set_index("ward_id")
    pilot = wards.loc[list(NOTEBOOK)]
    # vulnerability is min-max normalised over the 10 pilot wards in the notebook
    vul = vulnerability_index(pilot)
    stress = pd.DataFrame({"ward_id": list(NOTEBOOK),
                           "wbgt_outdoor_c": [v[0] for v in NOTEBOOK.values()],
                           "utci_c": [v[1] for v in NOTEBOOK.values()]})
    r = compute_risk(stress, pilot, vul, S).set_index("ward_id")
    for w, (_, _, v, ex, lo, hi) in NOTEBOOK.items():
        assert r.vulnerability[w] == pytest.approx(v, abs=6e-4), w
        assert r.excess_deaths[w] == pytest.approx(ex, abs=2e-3), w
        assert r.excess_deaths_lo[w] == pytest.approx(lo, abs=2e-3), w
        assert r.excess_deaths_hi[w] == pytest.approx(hi, abs=2e-3), w
        assert r.tier[w] == "Orange"
    assert r.excess_deaths.sum() == pytest.approx(1.96, abs=0.01)          # city total on the old dashboard
    assert r.excess_deaths_lo.sum() == pytest.approx(1.13, abs=0.01)
    assert r.excess_deaths_hi.sum() == pytest.approx(2.85, abs=0.01)


def test_tier_boundaries():
    assert tier_of(31.99)[1] == "Green" and tier_of(32.0)[1] == "Yellow"
    assert tier_of(37.99)[1] == "Yellow" and tier_of(38.0)[1] == "Orange"
    assert tier_of(45.99)[1] == "Orange" and tier_of(46.0)[1] == "Red"
    assert tier_of(float("nan")) == (None, None, None)


def test_no_excess_below_mmt():
    wards = pd.read_csv(S.wards_csv).set_index("ward_id")
    st = pd.DataFrame({"ward_id": ["W01"], "wbgt_outdoor_c": [S.mmt_c - 3], "utci_c": [30.0]})
    r = compute_risk(st, wards, vulnerability_index(wards), S)
    assert r.excess_deaths.iloc[0] == 0 and r.tier.iloc[0] == "Green"


# ---------------------------------------------------------------- indices
def _row(t, rh, wind=2.0, sw=800, dr=600):
    return pd.DataFrame({"temperature_c": [t], "relative_humidity": [rh], "wind_speed_ms": [wind],
                         "solar_radiation": [sw], "direct_radiation": [dr]})


def test_gate2_humidity_moves_index_at_constant_temperature():
    """Notebook Gate 2, as a unit test."""
    dry, wet = add_indices(_row(34, 35)), add_indices(_row(34, 80))
    assert wet.wbgt_outdoor_c[0] - dry.wbgt_outdoor_c[0] > 1.0
    assert wet.utci_c[0] > dry.utci_c[0]


def test_indices_shapes_and_ranges():
    d = add_indices(pd.concat([_row(24, 50, 3, 300, 100), _row(38, 60, 1, 900, 700)], ignore_index=True))
    assert d.utci_c.notna().all()
    assert (d.tmrt_c - d.temperature_c).between(0, 22.001).all()
    assert d.utci_c[1] > d.utci_c[0]


# ---------------------------------------------------------------- weather collapse
def test_collapse_takes_values_at_peak_temperature_hour():
    times = pd.date_range("2026-05-01", periods=48, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    T = [25.0] * 48; RH = [90] * 48; WS = [3.6] * 48; SW = [0.0] * 48; DR = [0.0] * 48
    T[14] = 40.0; RH[14] = 33; WS[14] = 18.0; SW[14] = 850.0; DR[14] = 700.0    # day 1 peak
    RH[3] = 99                                                                  # dawn humidity must NOT be used
    T[24 + 15] = 39.0; RH[24 + 15] = 41
    out = collapse_daily({"hourly": {"time": times, "temperature_2m": T, "relative_humidity_2m": RH,
                                     "wind_speed_10m": WS, "shortwave_radiation": SW, "direct_radiation": DR}})
    assert len(out) == 2
    a = out.iloc[0]
    assert (a.peak_hour, a.temperature_c, a.relative_humidity) == (14, 40.0, 33)
    assert a.wind_speed_ms == pytest.approx(5.0)          # 18 km/h -> 5 m/s
    assert out.iloc[1].peak_hour == 15


def test_collapse_drops_half_filled_days_instead_of_guessing():
    times = pd.date_range("2026-05-01", periods=24, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    T = [30.0] * 24; T[14] = 35.0
    RH = [50] * 24; RH[14] = None                                                # missing at the peak hour
    out = collapse_daily({"hourly": {"time": times, "temperature_2m": T, "relative_humidity_2m": RH,
                                     "wind_speed_10m": [5.0] * 24, "shortwave_radiation": [1.0] * 24,
                                     "direct_radiation": [1.0] * 24}})
    assert out.empty


# ---------------------------------------------------------------- Open-Meteo client (mocked transport)
def _mock(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_fetch_sends_one_multilocation_request_and_handles_list_response():
    cells = [(22.5, 88.3), (22.6, 88.4)]
    seen = {}

    def handler(req):
        seen["q"] = dict(req.url.params)
        return httpx.Response(200, json=demo_payloads(cells, S))

    async def go():
        async with _mock(handler) as c:
            return await fetch_open_meteo(cells, S, client=c)
    out = asyncio.run(go())
    assert len(out) == 2
    assert seen["q"]["latitude"] == "22.5,22.6" and seen["q"]["longitude"] == "88.3,88.4"
    assert seen["q"]["timezone"] == "Asia/Kolkata" and seen["q"]["past_days"] == str(S.history_days)


def test_fetch_single_location_dict_response_is_wrapped():
    cell = [(22.5, 88.3)]
    async def go():
        async with _mock(lambda r: httpx.Response(200, json=demo_payloads(cell, S)[0])) as c:
            return await fetch_open_meteo(cell, S, client=c)
    assert len(asyncio.run(go())) == 1


def test_fetch_retries_on_server_error_then_succeeds(monkeypatch):
    import app.weather as W
    async def nosleep(_): pass
    monkeypatch.setattr(W.asyncio, "sleep", nosleep)
    cell = [(22.5, 88.3)]; calls = {"n": 0}
    def handler(req):
        calls["n"] += 1
        return httpx.Response(500) if calls["n"] < 3 else httpx.Response(200, json=demo_payloads(cell, S))
    async def go():
        async with _mock(handler) as c:
            return await fetch_open_meteo(cell, S, client=c)
    assert len(asyncio.run(go())) == 1 and calls["n"] == 3


def test_fetch_gives_up_after_retries(monkeypatch):
    import app.weather as W
    async def nosleep(_): pass
    monkeypatch.setattr(W.asyncio, "sleep", nosleep)
    async def go():
        async with _mock(lambda r: httpx.Response(500)) as c:
            return await fetch_open_meteo([(22.5, 88.3)], S, client=c, tries=2)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(go())


# ---------------------------------------------------------------- HTTP API
@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        for _ in range(50):                      # background first-fetch
            if c.get("/api/health").json()["has_data"]:
                break
            import time; time.sleep(0.1)
        yield c


def _no_nan(x):
    s = json.dumps(x, allow_nan=False)           # raises on NaN / Infinity
    return s


def test_health_and_meta(client):
    assert client.get("/api/health").json()["has_data"] is True
    m = client.get("/api/meta").json()
    assert m["demo"] is True and m["engine"]["tier"] == "A"
    assert m["coverage"]["wards_total"] == 141 and m["coverage"]["weather_cells"] == 4
    assert "SYNTHETIC" in m["provenance"]["demographics"]
    _no_nan(m)


def test_timeline_has_past_today_and_forecast(client):
    t = client.get("/api/timeline").json()
    offs = [d["offset"] for d in t["days"]]
    assert 0 in offs and min(offs) < 0 and max(offs) > 0
    assert all(d["is_forecast"] == (d["offset"] > 0) for d in t["days"])
    _no_nan(t)


def test_risk_defaults_to_today_and_covers_every_ward(client):
    r = client.get("/api/risk").json()
    assert r["offset"] == 0 and len(r["wards"]) == 141
    ids = [w["ward_id"] for w in r["wards"]]
    assert len(set(ids)) == 141 and "W141" in ids
    ex = [w["excess_deaths"] for w in r["wards"]]
    assert ex == sorted(ex, reverse=True)                                   # ranked
    for w in r["wards"]:
        assert w["excess_deaths_lo"] <= w["excess_deaths"] + 1e-9 <= w["excess_deaths_hi"] + 2e-9
        assert w["tier"] in ("Green", "Yellow", "Orange", "Red")
    assert sum(r["city"]["wards_by_tier"].values()) == 141
    _no_nan(r)


def test_risk_specific_date_and_bad_date(client):
    dates = [d["date"] for d in client.get("/api/timeline").json()["days"]]
    assert client.get(f"/api/risk?date={dates[-1]}").status_code == 200
    bad = client.get("/api/risk?date=1999-01-01")
    assert bad.status_code == 404 and bad.json()["detail"]["available"] == dates


def test_ward_endpoint(client):
    w = client.get("/api/wards/w06").json()
    assert w["ward_id"] == "W06" and len(w["series"]) >= 10
    assert client.get("/api/wards/W999").status_code == 404
    assert len(client.get("/api/wards").json()["wards"]) == 141


def test_heavier_heat_means_more_excess():
    """Monotonicity through the whole stack: hotter demo weather -> not less risk."""
    from app.service import RiskService
    from dataclasses import replace
    cool, hot = RiskService(replace(S, demo_temp_offset_c=0.0)), RiskService(replace(S, demo_temp_offset_c=6.0))
    for s in (cool, hot):
        s.snapshot = s.build(demo_payloads(s.cells, s.s), "demo", pd.Timestamp.now(tz="UTC").to_pydatetime())
    d = cool.snapshot.dates[6]
    assert hot.snapshot.city_by_day[d]["excess_deaths"] > cool.snapshot.city_by_day[d]["excess_deaths"]
    assert hot.snapshot.city_by_day[d]["utci_c"] > cool.snapshot.city_by_day[d]["utci_c"] + 3


def test_failed_refresh_keeps_serving_previous_snapshot(client, monkeypatch):
    import app.weather as W
    async def nosleep(_): pass
    monkeypatch.setattr(W.asyncio, "sleep", nosleep)
    from app.main import svc
    before = svc.snapshot
    orig = svc.s
    from dataclasses import replace
    svc.s = replace(orig, weather_source="open-meteo", forecast_url="http://127.0.0.1:9/none")
    try:
        ok = asyncio.run(svc.refresh(force=True))
    finally:
        svc.s = orig
    assert ok is False and svc.last_error
    assert svc.snapshot is before                                              # nothing was published
    assert client.get("/api/risk").status_code == 200
    asyncio.run(svc.refresh(force=True))                                       # recover
    assert svc.last_error is None
