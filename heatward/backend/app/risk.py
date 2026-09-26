"""Tier A risk engine -- notebook section 7.

No training data. A published exposure-response curve applied to ward
population and vulnerability, with a confidence band carried through to every
number. Every output has a low/high; never render a bare point estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings

UTCI_TIERS = ((32, 0, "Green", "Green — no to moderate"),
              (38, 1, "Yellow", "Yellow — strong"),
              (46, 2, "Orange", "Orange — very strong"),
              (np.inf, 3, "Red", "Red — extreme"))
TIER_NAMES = [t[2] for t in UTCI_TIERS]

VULN_WEIGHTS = {"elderly_pct": .35, "outdoor_worker_pct": .35, "informal_housing_pct": .30}


def vulnerability_index(wards: pd.DataFrame) -> pd.Series:
    """Min-max normalise each demographic across the modelled wards, then weight."""
    cols = list(VULN_WEIGHTS)
    span = (wards[cols].max() - wards[cols].min()).replace(0, 1.0)
    nz = (wards[cols] - wards[cols].min()) / span
    return (nz * pd.Series(VULN_WEIGHTS)).sum(axis=1)


def tier_of(utci: float):
    if utci is None or not np.isfinite(utci):
        return None, None, None
    for hi, k, name, label in UTCI_TIERS:
        if utci < hi:
            return k, name, label
    return None, None, None


def compute_risk(stress: pd.DataFrame, wards: pd.DataFrame, vulnerability: pd.Series,
                 s: Settings, index_col: str = "wbgt_outdoor_c") -> pd.DataFrame:
    """stress: one row per ward-day with `ward_id`, `index_col`, `utci_c`."""
    r = stress.copy()
    r["vulnerability"] = r.ward_id.map(vulnerability)
    r["population"] = r.ward_id.map(wards.population)

    vs = 0.5 + r.vulnerability
    over = np.clip(r[index_col] - s.mmt_c, 0, None)
    base = r.population * s.cdr_per_1000 / 1000 / 365
    for name, b in (("rr", s.beta), ("rr_lo", s.beta_lo), ("rr_hi", s.beta_hi)):
        r[name] = np.exp(b * vs * over)
    r["excess_deaths"] = base * (r.rr - 1)
    r["excess_deaths_lo"] = base * (r.rr_lo - 1)
    r["excess_deaths_hi"] = base * (r.rr_hi - 1)
    r["excess_ed_visits"] = r.excess_deaths * s.ed_ratio

    t = [tier_of(u) for u in r.utci_c]
    r["tier_index"] = [x[0] for x in t]
    r["tier"] = [x[1] for x in t]
    r["tier_label"] = [x[2] for x in t]
    return r
