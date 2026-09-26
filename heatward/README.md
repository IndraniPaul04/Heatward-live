# HEATWARD — live stack

```
Open-Meteo forecast API ──►  backend/ (FastAPI)  ──►  frontend/ (React + MapLibre)
 hourly T, RH, wind, SW/DR     fetch every 60 min        polls /api/* every 5 min
                               WBGT · UTCI · Tier A      day strip: past 6 days → +5
                               in-memory snapshot        map, charts, CSV export
```

The browser never talks to the weather provider and the API never fetches per request, so
polling is free and Open-Meteo's quota is safe (one request of ~4 locations per hour).

## Layout

```
heatward/
  notebook/   the original research notebook (Colab) -- not run by the app; ported, not imported
  backend/    FastAPI service: live weather -> indices -> Tier A -> Tier B -> API
  frontend/   React dashboard
```

`notebook/HEATWARD_Model2_Colab.ipynb` is included for reference and grading. It documents the
methodology (ERA5 checks, the NCRB anchor, the gates) that `backend/app/*.py` ports into code that
runs continuously on live weather. Nothing in the backend imports or executes the notebook.

## Run it

**Python 3.10–3.13 required (3.12 recommended).** Python 3.14 does not work yet: `pythermalcomfort`
pins `numpy<2.3`, which has no 3.14 wheels, so pip tries to compile NumPy and fails on Windows.
Use a venv: `py -3.12 -m venv .venv` then activate it (`.\.venv\Scripts\Activate.ps1` on Windows,
`source .venv/bin/activate` elsewhere), and start the server with `python -m uvicorn ...`.

```bash
# terminal 1 — backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --port 8000          # live weather
# WEATHER_SOURCE=demo DEMO_TEMP_OFFSET_C=5 uvicorn app.main:app --port 8000   # offline demo

# terminal 2 — frontend
cd frontend
npm install
npm run dev                                # http://localhost:5173  (proxies /api -> :8000)
```

Check the backend on its own: `curl localhost:8000/api/meta` (`"demo": false` = live) and
`pytest -q` in `backend/`.

## API

| Endpoint | Returns |
|---|---|
| `GET /api/health` | liveness, whether data is loaded, staleness |
| `GET /api/meta` | engine parameters (MMT, β, CI), provenance of each layer, freshness, coverage |
| `GET /api/timeline` | city-level series: 6 past days, today, 5 forecast days |
| `GET /api/risk?date=YYYY-MM-DD` | all wards for one day (default today), ranked by excess burden |
| `GET /api/wards`, `/api/wards/{id}` | ward metadata; one ward's daily series |
| `POST /api/refresh` | force a re-fetch (rate-limited; `X-Refresh-Token` if configured) |

Every burden figure ships with `_lo` / `_hi`. If a refresh fails the last good snapshot keeps
being served and `/api/meta` reports `stale` + `last_error`; the UI shows it.

## The two models

**Tier A — exposure-response engine (always on).** Notebook §3 + §7: peak-hour weather → WBGT/UTCI
→ published excess-mortality curve × ward population × vulnerability. Needs no training.
`app/indices.py` and `app/risk.py` are the notebook code ported line-for-line; `tests/` proves it by
reproducing the notebook's peak-day numbers (city total 1.96 [1.13–2.85]).

**Tier B — the trained predictive model (optional, needs one training run).** Notebook §4–§6: a
`HistGradientBoostingRegressor` (Poisson loss) that predicts health burden **4 days ahead**.
The notebook never saved its model, so `app/train.py` ports §1–§6 and saves `models/tierb.joblib`:

```bash
cd backend
python -m app.train                 # real ERA5 2019-2025 via Open-Meteo archive (needs internet, ~1-2 min)
python -m app.train --source demo   # offline pipeline check on synthetic weather
python -m app.train --wards pilot   # W01-W10 only (closest to the notebook)
python -m app.train --keep-doy      # notebook-exact feature set (see below)
```

Restart the API and the UI gains: a **Tier B** map mode, a "Tier B vs normal" line, ward detail rows,
and a model-skill panel on Analytics. Without the file everything still works, Tier A only.

Training also stores the local 80th-percentile **MMT** (notebook §4). Tier A uses it automatically
unless you set `MMT_C` yourself, so you no longer have to copy it from the notebook by hand.

### Read this before presenting Tier B

* **Outcomes are synthetic.** The notebook has no real ward-level health data; it generates NCRB-anchored
  Poisson counts from outdoor WBGT. The metrics validate the *pipeline*, not epidemiological accuracy.
  Every surface that shows Tier B says so.
* **Its lag inputs are simulated in real time.** `burden_lag1/3/7` and `burden_roll7_lag1` are past health
  outcomes; live there are none, so they are filled with the generator's *expected* burden computed from
  the real weather. When a health department shares real daily counts, substitute them in
  `TierBModel.predict` and nothing else changes.
* **One deliberate deviation from the notebook:** `doy` (day-of-year) is dropped by default. With ~0.2%
  of ward-days being events, the trees memorised calendar dates and predictions jumped at a specific date
  regardless of weather (measured: ratio ×2.1 → ×0.8 between adjacent days). Skill was unchanged without
  it. `--keep-doy` restores the notebook exactly.
* **Gate 4 may read FAIL** (dry-bulb temperature beating the compound indices). That is the notebook's own
  consistency check; report it as a finding, don't hide it.
* "vs normal" divides by the ward-month mean, floored at the all-year average, because ward-month means
  are ~0 outside the heat season and any ratio against ~0 explodes.
* Predictions exist for days whose weather (4 days earlier) is inside the fetched window, i.e. from
  ~6 days ago through +5 days.
* Train/serve parity is enforced: one `build_features` function serves both, and tests check that a
  28-day serving window reproduces the training features exactly and that no future outcome leaks into
  any feature.

## Things to do before you rely on the numbers

1. **MMT.** Until you train Tier B, `31.98` is used — back-solved from the original dashboard's hard-coded
   outputs (all ten wards agree within ±0.013 °C). After `python -m app.train` on real ERA5 the trained
   p80 is used instead (`/api/meta` → `engine.mmt_source`). If it lands far from 31.98, that's worth a look.
2. **Source mismatch.** MMT was calibrated on ERA5; live data is the forecast API's blend of
   weather models. Compare the two WBGT distributions over a shared period before trusting
   absolute burden numbers.
3. **Demographics are synthetic** (as in the notebook). `python scripts/build_wards.py <geojson>`
   regenerates `data/wards.csv`; swap in real Census 2011 columns there and nothing else changes.
   `--pilot-only` restricts modelling to W01–W10.
4. **Weather resolution.** The 141 wards fall into 4 weather cells at 0.1°, so the *Risk* tier
   (weather-only) is blocky by design. Ward-level differences live in the *Burden* map mode.
5. **Ward IDs W01–W10 changed meaning.** In the notebook they were placeholder centroids; here
   they are KMC wards 1–10, located from the GeoJSON.
6. Open-Meteo's free tier is for non-commercial use; a commercial deployment needs their paid plan.

## Deploying

Run **one** uvicorn worker (each worker would run its own refresh loop). Serve `frontend/dist`
and reverse-proxy `/api` to it, or build with `VITE_API_BASE=https://api.example.com` and set
`CORS_ORIGINS` on the backend.
