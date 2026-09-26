"""Runtime configuration. Everything is overridable by environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    city: str = "Kolkata"
    tz: str = "Asia/Kolkata"

    # ---- weather source ------------------------------------------------
    # "open-meteo" = live forecast API.  "demo" = deterministic synthetic
    # weather in Open-Meteo's response format, for offline dev and tests.
    weather_source: str = os.getenv("WEATHER_SOURCE", "open-meteo")
    forecast_url: str = os.getenv("FORECAST_URL", "https://api.open-meteo.com/v1/forecast")
    past_days: int = _i("PAST_DAYS", 6)                    # days SHOWN before today
    history_days: int = _i("HISTORY_DAYS", 28)             # days FETCHED (Tier B needs >=14 + 7 of history)
    forecast_days: int = _i("FORECAST_DAYS", 6)          # includes today
    refresh_minutes: int = _i("REFRESH_MINUTES", 60)
    retry_minutes: int = _i("RETRY_MINUTES", 5)
    min_manual_refresh_s: int = _i("MIN_MANUAL_REFRESH_S", 60)
    # Wards are snapped to this grid before fetching. NWP/reanalysis grids are
    # ~10-25 km, so finer than 0.1 deg only produces duplicate cells.
    grid_deg: float = _f("GRID_DEG", 0.1)
    demo_temp_offset_c: float = _f("DEMO_TEMP_OFFSET_C", 0.0)   # push demo weather into Orange/Red

    # ---- Tier A exposure-response (notebook section 4 / 7) --------------
    # MMT = 80th percentile of local peak-hour outdoor WBGT over the training
    # period. 31.98 was BACK-SOLVED from the peak-day outputs baked into the
    # frontend (all ten wards agree to +/-0.013 C). Replace it with the value your
    # notebook prints ("MMT = ... C outdoor WBGT") once you re-run section 4.
    mmt_c: float = _f("MMT_C", 31.98)
    beta: float = _f("BETA", 0.0476)
    beta_lo: float = _f("BETA_LO", 0.0286)
    beta_hi: float = _f("BETA_HI", 0.0666)
    cdr_per_1000: float = _f("CDR_PER_1000", 6.0)
    ed_ratio: float = _f("ED_RATIO", 20.0)

    # ---- files ----------------------------------------------------------
    tierb_path: Path = Path(os.getenv("TIERB_PATH", BASE_DIR / "models" / "tierb.joblib"))
    wards_csv: Path = Path(os.getenv("WARDS_CSV", BASE_DIR / "data" / "wards.csv"))
    cache_dir: Path = Path(os.getenv("CACHE_DIR", BASE_DIR / "cache"))

    # ---- http -----------------------------------------------------------
    cors_origins: tuple = tuple(
        o.strip() for o in os.getenv(
            "CORS_ORIGINS", "http://localhost:5173,http://localhost:4173,http://127.0.0.1:5173"
        ).split(",") if o.strip()
    )
    refresh_token: str = os.getenv("REFRESH_TOKEN", "")


def load_settings() -> Settings:
    return Settings()
