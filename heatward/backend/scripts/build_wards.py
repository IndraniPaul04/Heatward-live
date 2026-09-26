"""Build data/wards.csv from the KMC ward GeoJSON.

Geometry is REAL (centroids come from the supplied 141-ward file).
Demographics are SYNTHETIC, Census-2011-shaped -- exactly as in the notebook (section 1).

W01-W10 reuse the notebook's seeded draws (SEED=42, same call order) so the
peak-day numbers the frontend was built against still reproduce. W11+ come from
a second seeded stream. Replace the demographic columns with real Census 2011
ward tables when you have them; nothing else in the backend needs to change.

    python scripts/build_wards.py ../frontend/public/wards_kolkata.geojson
    python scripts/build_wards.py <geojson> --pilot-only     # only W01-W10 are modelled
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd


def _polys(geom):
    """Yield every polygon's exterior ring, recursing into collections."""
    t = geom["type"]
    if t == "Polygon":
        yield geom["coordinates"][0]
    elif t == "MultiPolygon":
        for p in geom["coordinates"]:
            yield p[0]
    elif t == "GeometryCollection":
        for g in geom["geometries"]:
            yield from _polys(g)


def _ring_centroid(ring):
    x = np.array([p[0] for p in ring]); y = np.array([p[1] for p in ring])
    a = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = a.sum() / 2
    if abs(area) < 1e-14:
        return x.mean(), y.mean(), 0.0
    cx = ((x[:-1] + x[1:]) * a).sum() / (6 * area)
    cy = ((y[:-1] + y[1:]) * a).sum() / (6 * area)
    return cx, cy, abs(area)


def ward_centroids(geojson_path):
    g = json.loads(Path(geojson_path).read_text())
    out = {}
    for f in g["features"]:
        wid = int(str(f["properties"]["WARD"]).strip())
        # largest polygon wins, so a stray island can't drag the centroid
        lon, lat, _ = max((_ring_centroid(r) for r in _polys(f["geometry"])), key=lambda t: t[2])
        out[wid] = (lat, lon)
    return out


def synthetic_demographics(n, seed=42):
    rng = np.random.default_rng(seed)
    # --- identical call order to the notebook, n = 10 ---
    pilot = pd.DataFrame({
        "population":           rng.integers(28_000, 95_000, 10),
        "elderly_pct":          rng.uniform(6, 16, 10).round(1),
        "outdoor_worker_pct":   rng.uniform(8, 32, 10).round(1),
        "informal_housing_pct": rng.uniform(3, 40, 10).round(1),
    })
    pilot["population_density"] = (pilot.population / rng.uniform(2.0, 8.0, 10)).round(0)
    if n <= 10:
        return pilot.iloc[:n]
    m = n - 10
    r2 = np.random.default_rng(seed + 1)
    rest = pd.DataFrame({
        "population":           r2.integers(28_000, 95_000, m),
        "elderly_pct":          r2.uniform(6, 16, m).round(1),
        "outdoor_worker_pct":   r2.uniform(8, 32, m).round(1),
        "informal_housing_pct": r2.uniform(3, 40, m).round(1),
    })
    rest["population_density"] = (rest.population / r2.uniform(2.0, 8.0, m)).round(0)
    return pd.concat([pilot, rest], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("geojson")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "wards.csv"))
    ap.add_argument("--pilot-only", action="store_true",
                    help="model only W01-W10; the rest stay 'no output' in the UI")
    a = ap.parse_args()

    cen = ward_centroids(a.geojson)
    ids = sorted(cen)
    demo = synthetic_demographics(len(ids))
    df = pd.DataFrame({
        "ward_id":  [f"W{i:02d}" for i in ids],
        "ward_no":  ids,
        "latitude": [round(cen[i][0], 5) for i in ids],
        "longitude": [round(cen[i][1], 5) for i in ids],
    })
    df = pd.concat([df, demo.reset_index(drop=True)], axis=1)
    df["modeled"] = 1
    if a.pilot_only:
        df.loc[df.ward_no > 10, "modeled"] = 0
    df["demographics_source"] = "synthetic"
    df["geometry_source"] = "kmc_geojson"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} wards, {int(df.modeled.sum())} modelled")


if __name__ == "__main__":
    main()
