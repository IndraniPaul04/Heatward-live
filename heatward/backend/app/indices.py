"""Thermal indices -- ported line-for-line from notebook section 3.

The only change is that UTCI is computed on the whole array at once instead of
in a Python loop (pythermalcomfort accepts arrays), which is what makes a
refresh take milliseconds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pythermalcomfort.models import utci as utci_model


def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    """df needs: temperature_c, relative_humidity, wind_speed_ms, solar_radiation, direct_radiation."""
    d = df.copy()
    t = d.temperature_c.to_numpy(float)
    rh = d.relative_humidity.to_numpy(float)
    v = np.clip(d.wind_speed_ms.to_numpy(float), 0.5, 17.0)      # UTCI valid range
    sw = d.solar_radiation.to_numpy(float)
    dr = d.direct_radiation.to_numpy(float)

    # Mean radiant temperature proxy (see notebook for the reasoning on the 0.022 / 22 C cap)
    tmrt = t + np.clip(0.022 * (0.6 * sw + 0.4 * dr), 0, 22)
    tmrt = np.clip(tmrt, t - 29.9, t + 69.9)                     # UTCI valid range
    d["tmrt_c"] = tmrt

    # NOAA Rothfusz heat index
    tf = t * 9 / 5 + 32
    simple = 0.5 * (tf + 61.0 + (tf - 68.0) * 1.2 + rh * 0.094)
    full = (-42.379 + 2.04901523 * tf + 10.14333127 * rh - 0.22475541 * tf * rh
            - 0.00683783 * tf ** 2 - 0.05481717 * rh ** 2 + 0.00122874 * tf ** 2 * rh
            + 0.00085282 * tf * rh ** 2 - 0.00000199 * tf ** 2 * rh ** 2)
    d["heat_index_c"] = (np.where((tf >= 80) & (rh >= 40), full, simple) - 32) * 5 / 9

    # Stull (2011) wet bulb
    tnw = (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh)
           - np.arctan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
           - 4.686035)
    d["wet_bulb_c"] = tnw

    e = (rh / 100.0) * 6.105 * np.exp(17.27 * t / (237.7 + t))
    d["wbgt_shade_c"] = 0.567 * t + 0.393 * e + 3.94              # BOM shade approximation
    d["wbgt_outdoor_c"] = 0.7 * tnw + 0.2 * tmrt + 0.1 * t        # outdoor form

    d["utci_c"] = np.asarray(utci_model(tdb=t, tr=tmrt, v=v, rh=rh).utci, dtype=float)
    return d
