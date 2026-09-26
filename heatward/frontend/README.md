# HEATWARD Frontend — Neon Green / Black Build

React + Vite dashboard over the supplied 141-ward Kolkata GeoJSON. It now reads live data from
the HEATWARD API (`../backend`); see the root README.

```bash
npm install
npm run dev        # proxies /api to http://localhost:8000 (override with HEATWARD_API=...)
npm run build      # production; set VITE_API_BASE if the API is on another origin
```

## Visual system
- Black foundation, neon green `#39FF14` brand accent; risk colours stay conventional
  (green/yellow/orange/red) and are never replaced by the brand colour.
- No Google Maps or external basemap — the map is drawn from the ward GeoJSON.
- Inter = UI, Roboto = data, Montserrat = headings, Impact = risk emphasis.

## Data states
The header strip always says what you're looking at: **LIVE** (with age), **DEMO DATA**,
**STALE**, or **CONNECTION LOST**. Weather is real; ward demographics are labelled synthetic.
