import React, { Component, createContext, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { Activity, AlertTriangle, BarChart3, Bell, ChevronRight, Database, FileText, Gauge, Layers3, Map as MapIcon, Menu, Moon, Search, Settings as SettingsIcon, ShieldCheck, Sun, Thermometer, Users, X, ArrowUpRight, CheckCircle2, Info, Download, SlidersHorizontal } from 'lucide-react';
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis, BarChart, Bar } from 'recharts';
import './styles.css';

/* ───────────────────────────── data layer ─────────────────────────────
   Everything the UI shows now comes from the HEATWARD API (backend/).
   Weather is refreshed server-side; the browser just polls the cheap,
   in-memory endpoints. Dev: Vite proxies /api -> :8000.                    */
const API = import.meta.env.VITE_API_BASE || '';
const POLL_MS = 5 * 60 * 1000;

const TIER_COLOR = { Green: '#39ff14', Yellow: '#f2c94c', Orange: '#f2994a', Red: '#e05252' };
const TIER_DESC = { Green: 'Low thermal stress', Yellow: 'Strong thermal stress', Orange: 'Very strong thermal stress', Red: 'Extreme thermal stress' };
const UNKNOWN = '#263029';

async function get(path) {
  const r = await fetch(API + path);
  if (!r.ok) { const e = new Error(`HTTP ${r.status}`); e.status = r.status; throw e; }
  return r.json();
}

const normWard = w => ({
  id: w.ward_id, no: w.ward_no, rank: w.rank, tier: w.tier,
  wbgt: w.wbgt_outdoor_c, utci: w.utci_c, temp: w.temperature_c, rh: w.relative_humidity,
  vuln: w.vulnerability, excess: w.excess_deaths, low: w.excess_deaths_lo, high: w.excess_deaths_hi,
  tbRatio: w.tierb_ratio, tbBurden: w.tierb_burden
});

const DataCtx = createContext(null);
const useData = () => useContext(DataCtx);

function useHeatward() {
  const [meta, setMeta] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [risk, setRisk] = useState(null);
  const [picked, setPicked] = useState(null);           // day the user chose; null = follow "today"
  const [status, setStatus] = useState('loading');      // loading | live | warming | offline

  useEffect(() => {
    let alive = true, timer;
    const load = async () => {
      let delay = POLL_MS;
      try {
        const [m, t] = await Promise.all([get('/api/meta'), get('/api/timeline')]);
        if (!alive) return;
        setMeta(m); setTimeline(t); setStatus('live');
      } catch (e) {
        if (!alive) return;
        setStatus(e.status === 503 ? 'warming' : 'offline');
        delay = e.status === 503 ? 4000 : 15000;
      }
      if (alive) timer = setTimeout(load, delay);
    };
    load();
    return () => { alive = false; clearTimeout(timer); };
  }, []);

  const day = useMemo(() => {
    if (!timeline) return null;
    const dates = timeline.days.map(d => d.date);
    if (picked && dates.includes(picked)) return picked;
    return dates.includes(timeline.today) ? timeline.today : dates[dates.length - 1];
  }, [timeline, picked]);

  // Re-fetch the ward table when the day changes or the server produced a new snapshot.
  useEffect(() => {
    if (!day || !timeline) return;
    let alive = true;
    get(`/api/risk?date=${day}`).then(r => alive && setRisk(r)).catch(() => {});
    return () => { alive = false; };
  }, [day, timeline?.generated_at]);

  const rows = useMemo(() => (risk ? risk.wards.map(normWard) : []), [risk]);
  const byId = useMemo(() => Object.fromEntries(rows.map(r => [r.id, r])), [rows]);
  const ready = !!(meta && timeline && risk);
  return { status, meta, timeline, day, pickDay: setPicked, risk, rows, byId, city: risk?.city, ready };
}

/* ───────────────────────────── helpers ───────────────────────────── */
const d0 = iso => new Date(iso + 'T00:00:00');
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const longDate = iso => { const d = d0(iso); return `${String(d.getDate()).padStart(2, '0')} ${MON[d.getMonth()].toUpperCase()} ${d.getFullYear()}`; };
const shortDate = iso => { const d = d0(iso); return `${d.getDate()} ${MON[d.getMonth()]}`; };
const weekday = iso => d0(iso).toLocaleDateString('en-GB', { weekday: 'short' });
const f1 = v => (v == null ? '—' : Number(v).toFixed(1));
const f2 = v => (v == null ? '—' : Number(v).toFixed(2));
const xr = v => (v == null ? '—' : `×${Number(v).toFixed(v >= 10 ? 0 : 1)}`);
const pct = v => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);
const tierColor = t => TIER_COLOR[t] || UNKNOWN;
const horizon = o => (o === 0 ? 'TODAY' : o > 0 ? `T + ${o} DAYS` : `T − ${-o} DAYS`);

function useNow(ms = 30000) {
  const [n, setN] = useState(Date.now());
  useEffect(() => { const t = setInterval(() => setN(Date.now()), ms); return () => clearInterval(t); }, [ms]);
  return n;
}
function ago(iso, now) {
  const m = Math.max(0, Math.round((now - new Date(iso)) / 60000));
  if (m < 1) return 'just now';
  if (m < 60) return `${m} min ago`;
  return `${Math.floor(m / 60)} h ${m % 60} min ago`;
}

/* Map colouring. Wards are matched on the GeoJSON's numeric WARD property. */
function colorFor(w, mode, mmt, maxExcess, maxTb) {
  if (mode === 'risk') return tierColor(w.tier);
  if (mode === 'tierb') {          // learned model: predicted burden as a share of the day's highest ward
    if (w.tbBurden == null || !(maxTb > 0)) return UNKNOWN;
    const r = w.tbBurden / maxTb;
    return r >= 0.75 ? TIER_COLOR.Red : r >= 0.5 ? TIER_COLOR.Orange : r >= 0.25 ? TIER_COLOR.Yellow : TIER_COLOR.Green;
  }
  if (mode === 'burden') {         // hazard x vulnerability: share of the day's highest ward
    const r = maxExcess > 0 ? w.excess / maxExcess : 0;
    return r >= 0.75 ? TIER_COLOR.Red : r >= 0.5 ? TIER_COLOR.Orange : r >= 0.25 ? TIER_COLOR.Yellow : TIER_COLOR.Green;
  }
  if (mode === 'wbgt') {           // bands relative to the model's MMT, where modelled excess begins
    if (w.wbgt == null) return UNKNOWN;
    return w.wbgt < mmt - 2 ? TIER_COLOR.Green : w.wbgt < mmt ? TIER_COLOR.Yellow : w.wbgt < mmt + 2 ? TIER_COLOR.Orange : TIER_COLOR.Red;
  }
  return w.vuln >= 0.6 ? '#39ff14' : w.vuln >= 0.4 ? '#8fdc63' : '#3f713f';
}
function fillExpr(rows, mode, mmt) {
  if (!rows.length) return UNKNOWN;             // MapLibre `match` needs at least one case
  const e = ['match', ['to-string', ['get', 'WARD']]];
  const maxExcess = Math.max(...rows.map(r => r.excess || 0));
  const maxTb = Math.max(...rows.map(r => r.tbBurden || 0));
  rows.forEach(w => e.push(String(w.no), colorFor(w, mode, mmt, maxExcess, maxTb)));
  e.push(UNKNOWN);
  return e;
}

const nav = [['Dashboard', Gauge], ['Risk Map', MapIcon], ['Analytics', BarChart3], ['Alerts', Bell], ['Reports', FileText], ['Data Layers', Layers3]];
const pageMeta = {
  Dashboard: ['CITY HEAT MONITORING', 'Heat Risk Overview', 'Live thermal stress and the forecast for the days ahead.'],
  'Risk Map': ['SPATIAL INTELLIGENCE', 'Risk Map', 'Explore ward-level thermal stress across Kolkata.'],
  Analytics: ['MODEL ANALYTICS', 'Analytics', 'Inspect thermal trends, vulnerability and forecast behaviour.'],
  Alerts: ['RESPONSE CENTER', 'Alerts', 'Priority conditions requiring attention.'],
  Reports: ['EVIDENCE & OUTPUTS', 'Reports', 'Review model outputs, methodology and exportable summaries.'],
  'Data Layers': ['DATA CONTROL', 'Data Layers', 'Choose which HEATWARD data layer is visualized on the map.'],
  Settings: ['SYSTEM', 'Settings', 'Control interface preferences and model display behaviour.']
};

class AppErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  render() {
    if (this.state.error) return <div className="fatal"><div className="fatal-box"><div className="eyebrow">HEATWARD RUNTIME ERROR</div><h2>The dashboard could not render.</h2><p>Open the browser console and share the red error with me.</p><pre>{String(this.state.error?.stack || this.state.error)}</pre></div></div>;
    return this.props.children;
  }
}

/* ───────────────────────────── shell ───────────────────────────── */
function App() {
  const hw = useHeatward();
  const [dark, setDark] = useState(true);
  const [active, setActive] = useState('Dashboard');
  const [picked, setPicked] = useState(null);
  const [mapMode, setMapMode] = useState('risk');
  const [mobileOpen, setMobileOpen] = useState(false);
  const now = useNow();

  const selectedWard = picked && hw.byId[picked] ? picked : hw.rows[0]?.id || null;     // default: highest-burden ward
  const selected = hw.byId[selectedWard] || null;
  const navigate = page => { setActive(page); setMobileOpen(false); };
  const chip = !hw.meta ? 'Connecting…' : hw.meta.demo ? 'Demo weather' : hw.status === 'offline' ? 'Offline · cached' : `Live · ${ago(hw.meta.generated_at, now)}`;

  return <DataCtx.Provider value={{ ...hw, now }}>
    <div className={dark ? 'app dark' : 'app light'}>
      <aside className={mobileOpen ? 'sidebar open' : 'sidebar'}>
        <div className="brand-row"><div className="brand-mark">H</div><div><div className="brand">HEATWARD</div><div className="brand-sub">THERMAL INTELLIGENCE</div></div><button className="icon-btn mobile-close" onClick={() => setMobileOpen(false)}><X size={18} /></button></div>
        <div className="nav-label">MONITOR</div>
        <nav>{nav.map(([label, Icon]) => <button key={label} className={active === label ? 'nav-item active' : 'nav-item'} onClick={() => navigate(label)}><Icon size={17} /><span>{label}</span></button>)}</nav>
        <div className="sidebar-bottom"><button className={active === 'Settings' ? 'nav-item active' : 'nav-item'} onClick={() => navigate('Settings')}><SettingsIcon size={17} /><span>Settings</span></button><div className="model-chip"><ShieldCheck size={16} /><div><b>TIER A ENGINE</b><span>{chip}</span></div></div></div>
      </aside>
      <main className="main">
        <header className="topbar"><button className="icon-btn mobile-menu" onClick={() => setMobileOpen(true)}><Menu size={20} /></button><div className="search"><Search size={16} /><input placeholder="Search wards, reports, data..." /></div><div className="top-actions"><div className="location"><span className="dot"></span>Kolkata</div><button className="theme-btn" onClick={() => setDark(!dark)} title="Toggle theme">{dark ? <Sun size={17} /> : <Moon size={17} />}</button><div className="avatar">SS</div></div></header>
        <div className="content">
          {!hw.ready ? <Splash status={hw.status} /> : <>
            <PageHead active={active} />
            <StatusBar />
            <DayStrip />
            {active === 'Dashboard' && <Dashboard selectedWard={selectedWard} setSelectedWard={setPicked} selected={selected} mapMode={mapMode} setMapMode={setMapMode} navigate={navigate} />}
            {active === 'Risk Map' && <RiskPage selectedWard={selectedWard} setSelectedWard={setPicked} mapMode={mapMode} setMapMode={setMapMode} />}
            {active === 'Analytics' && <AnalyticsPage />}
            {active === 'Alerts' && <AlertsPage navigate={navigate} />}
            {active === 'Reports' && <ReportsPage />}
            {active === 'Data Layers' && <LayersPage mapMode={mapMode} setMapMode={setMapMode} navigate={navigate} />}
            {active === 'Settings' && <SettingsPage dark={dark} setDark={setDark} />}
          </>}
        </div>
      </main>
    </div>
  </DataCtx.Provider>;
}

function Splash({ status }) {
  const msg = { loading: 'Connecting to the HEATWARD API…', warming: 'The API is fetching its first weather snapshot…', offline: 'Cannot reach the HEATWARD API.' }[status] || 'Loading…';
  return <div className="splash panel"><span className="section-kicker">{status === 'offline' ? 'CONNECTION' : 'STARTING UP'}</span><h2>{msg}</h2>
    {status === 'offline' && <p>Start the backend (<code>uvicorn app.main:app --port 8000</code> in <code>backend/</code>) — this page retries automatically every 15 seconds.</p>}
    {status !== 'offline' && <p>This usually takes a few seconds on first launch.</p>}</div>;
}

function PageHead({ active }) {
  const { risk } = useData();
  const [eyebrow, title, desc] = pageMeta[active];
  return <div className="page-head"><div><div className="eyebrow">{eyebrow} · {longDate(risk.date)}</div><h1>{title}</h1><p>{desc}</p></div><div className="forecast-badge"><span>{risk.offset > 0 ? 'FORECAST HORIZON' : 'REFERENCE DAY'}</span><b>{horizon(risk.offset)}</b></div></div>;
}

function StatusBar() {
  const { meta, status, now } = useData();
  const cls = meta.demo ? 'status-bar warn' : status === 'offline' || meta.stale ? 'status-bar bad' : 'status-bar';
  let left;
  if (meta.demo) left = <><b>DEMO DATA</b> Synthetic weather for offline development — not a live feed.</>;
  else if (status === 'offline') left = <><b>CONNECTION LOST</b> Showing data from {ago(meta.generated_at, now)}. Retrying…</>;
  else if (meta.stale) left = <><b>STALE</b> Last successful weather fetch was {ago(meta.generated_at, now)}{meta.last_error ? ` — ${meta.last_error}` : ''}.</>;
  else left = <><span className="dot"></span><b>LIVE</b> Open-Meteo · updated {ago(meta.generated_at, now)} · refreshes every {meta.refresh_minutes} min</>;
  return <div className={cls}><span className="live">{left}</span><span>Weather real · ward demographics <b>synthetic</b></span></div>;
}

function DayStrip() {
  const { timeline, day, pickDay } = useData();
  const box = useRef(null);
  useEffect(() => {       // keep the selected day visible on narrow screens (scrolls the strip only, not the page)
    const el = box.current?.querySelector('.day-btn.on');
    if (el && box.current) box.current.scrollTo({ left: el.offsetLeft - box.current.clientWidth / 2 + el.clientWidth / 2, behavior: 'auto' });
  }, [day, timeline.generated_at]);
  return <div className="day-strip" ref={box} role="tablist" aria-label="Select day">{timeline.days.map(d => {
    const on = d.date === day;
    return <button key={d.date} role="tab" aria-selected={on} className={`day-btn${on ? ' on' : ''}${d.offset < 0 ? ' past' : ''}`} onClick={() => pickDay(d.date)} title={`${d.city.tier || '—'} · WBGT ${f1(d.city.wbgt_outdoor_c)}°C`}>
      <span className="day-wk">{d.offset === 0 ? 'TODAY' : weekday(d.date).toUpperCase()}</span>
      <span className="day-num">{d0(d.date).getDate()}</span>
      <span className="day-tier" style={{ background: tierColor(d.city.tier) }}></span>
      <span className="day-off">{d.offset === 0 ? 'NOW' : d.offset > 0 ? `+${d.offset}` : d.offset}</span>
    </button>;
  })}</div>;
}

/* ───────────────────────────── dashboard ───────────────────────────── */
function Dashboard({ selectedWard, setSelectedWard, selected, mapMode, setMapMode, navigate }) {
  const { city } = useData();
  return <>
    <section className="hero-grid"><RiskCard /><Metric title="WBGT" value={`${f1(city.wbgt_outdoor_c)}°C`} sub="Outdoor thermal stress · city" icon={<Thermometer size={17} />} /><Metric title="UTCI" value={`${f1(city.utci_c)}°C`} sub="Physiological stress · city" icon={<Activity size={17} />} /><Metric title="VULNERABILITY" value={pct(selected?.vuln)} sub={`Selected ward · ${selectedWard}`} icon={<Users size={17} />} /></section>
    <section className="map-section panel"><MapHeader mapMode={mapMode} setMapMode={setMapMode} /><RiskMap selectedWard={selectedWard} setSelectedWard={setSelectedWard} mapMode={mapMode} /></section>
    <section className="lower-grid"><Upcoming selectedWard={selectedWard} setSelectedWard={setSelectedWard} navigate={navigate} /><Trend /><WardDetail selectedWard={selectedWard} selected={selected} /></section>
    <ModelPanel />
  </>;
}

function RiskCard() {
  const { city, risk, meta } = useData();
  const c = tierColor(city.tier);
  const kicker = risk.offset === 0 ? 'CURRENT CITY RISK' : risk.offset > 0 ? 'FORECAST CITY RISK' : 'RECENT CITY RISK';
  return <div className="risk-card panel"><div className="card-top"><span className="section-kicker">{kicker}</span><span className="status-dot" style={{ background: c, boxShadow: `0 0 8px ${c}55` }}></span></div>
    <div className="risk-word" style={{ color: c, textShadow: `0 0 18px ${c}22` }}>{(city.tier || 'N/A').toUpperCase()}</div>
    <div className="risk-desc">{TIER_DESC[city.tier] || 'No data'}</div>
    <div className="risk-meta"><span><Thermometer size={14} /> Day</span><b>{longDate(risk.date)}</b></div>
    <div className="risk-meta"><span><Activity size={14} /> City excess burden</span><b>{f2(city.excess_deaths)} <small>[{f2(city.excess_deaths_lo)}–{f2(city.excess_deaths_hi)}]</small></b></div>
    {meta.tierb.available && <div className="risk-meta"><span><Gauge size={14} /> Tier B · vs normal</span><b>{xr(city.tierb_ratio)} <small>(synthetic-trained)</small></b></div>}</div>;
}
function Metric({ title, value, sub, icon }) { return <div className="metric-card panel"><div className="metric-icon">{icon}</div><span className="section-kicker">{title}</span><strong>{value}</strong><small>{sub}</small></div>; }
function MapHeader({ mapMode, setMapMode }) { const { meta } = useData(); const modes = [['risk', 'Risk'], ['burden', 'Burden'], ['wbgt', 'WBGT'], ['vulnerability', 'Vulnerability'], ...(meta.tierb.available ? [['tierb', 'Tier B']] : [])]; return <div className="map-head"><div><span className="section-kicker">SPATIAL RISK</span><h2>Kolkata Ward Distribution</h2></div><div className="map-controls">{modes.map(([id, label]) => <button key={id} className={mapMode === id ? 'seg active' : 'seg'} onClick={() => setMapMode(id)}>{label}</button>)}</div></div>; }

function Upcoming({ selectedWard, setSelectedWard, navigate }) {
  const { rows } = useData();
  return <div className="panel upcoming"><div className="panel-head"><div><span className="section-kicker">PRIORITY WARDS</span><h2>Highest burden</h2></div><button className="text-btn" onClick={() => navigate('Risk Map')}>Open map <ChevronRight size={15} /></button></div>
    <div className="ward-list">{rows.slice(0, 5).map(d => <button className={selectedWard === d.id ? 'ward-row selected' : 'ward-row'} key={d.id} onClick={() => setSelectedWard(d.id)}><span className="ward-id">{d.id}</span><span className="ward-risk"><i className="risk-dot" style={{ background: tierColor(d.tier) }}></i>{d.tier}</span><span className="ward-wbgt">{f1(d.wbgt)}°C</span><ChevronRight size={14} /></button>)}</div></div>;
}

function Trend() {
  const { timeline } = useData();
  const data = timeline.days.map(d => ({ day: shortDate(d.date), wbgt: d.city.wbgt_outdoor_c, utci: d.city.utci_c, off: d.offset }));
  const today = data.find(d => d.off === 0)?.day;
  return <div className="panel trend"><div className="panel-head"><div><span className="section-kicker">THERMAL TREND</span><h2>Stress outlook</h2></div><span className="date-pill">{data[0].day.toUpperCase()} – {data[data.length - 1].day.toUpperCase()}</span></div>
    <div className="chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={data}><defs><linearGradient id="fillGreen" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#39ff14" stopOpacity={0.24} /><stop offset="100%" stopColor="#39ff14" stopOpacity={0} /></linearGradient></defs><CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.12} /><XAxis dataKey="day" tickLine={false} axisLine={false} fontSize={10} interval="preserveStartEnd" /><YAxis tickLine={false} axisLine={false} fontSize={10} width={30} domain={['auto', 'auto']} />
      <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid var(--border)', background: 'var(--surface)', color: 'var(--text)' }} />
      {today && <ReferenceLine x={today} stroke="#8f9a94" strokeDasharray="3 3" label={{ value: 'today', fill: '#8f9a94', fontSize: 9, position: 'insideTopLeft' }} />}
      <Area type="monotone" dataKey="wbgt" stroke="#39ff14" strokeWidth={2.2} fill="url(#fillGreen)" /><Area type="monotone" dataKey="utci" stroke="#8f9a94" strokeWidth={1.5} fill="none" /></AreaChart></ResponsiveContainer></div>
    <div className="legend-line"><span><i className="line green"></i>WBGT</span><span><i className="line gray"></i>UTCI</span></div></div>;
}

function WardDetail({ selectedWard, selected }) {
  const { meta } = useData();
  if (!selected) return <div className="panel ward-detail"><span className="section-kicker">SELECTED WARD</span><h2>{selectedWard || '—'}</h2></div>;
  const c = tierColor(selected.tier);
  return <div className="panel ward-detail"><div className="panel-head"><div><span className="section-kicker">SELECTED WARD</span><h2>{selectedWard}</h2></div><span className="risk-badge" style={{ color: c, borderColor: `${c}88` }}>{(selected.tier || 'N/A').toUpperCase()}</span></div>
    <div className="detail-grid"><Detail label="WBGT" value={`${f1(selected.wbgt)}°C`} /><Detail label="UTCI" value={`${f1(selected.utci)}°C`} /><Detail label="Vulnerability" value={pct(selected.vuln)} /><Detail label="Excess burden" value={f2(selected.excess)} />{meta.tierb.available && <><Detail label="Tier B vs normal" value={xr(selected.tbRatio)} /><Detail label="Tier B burden" value={selected.tbBurden == null ? '—' : selected.tbBurden.toFixed(4)} /></>}</div>
    <div className="confidence"><div><span>Estimated excess burden</span><b>{f2(selected.excess)}</b></div><p>Range {f2(selected.low)}–{f2(selected.high)}</p></div></div>;
}
function Detail({ label, value }) { return <div className="detail"><span>{label}</span><b>{value}</b></div>; }

function ModelPanel() {
  const { meta } = useData();
  const e = meta.engine, tb = meta.tierb;
  return <section className="transparency panel"><div><span className="section-kicker">MODEL TRANSPARENCY</span><h2>Forecast integrity</h2>
    <p>Live engine: Tier A exposure–response. Peak-hour weather → WBGT / UTCI → excess-mortality curve (minimum-mortality threshold {e.mmt_c} °C WBGT, β {e.beta}, CI {e.beta_ci[0]}–{e.beta_ci[1]}) applied to ward population and vulnerability. It is not trained on outcomes, and its accuracy against observed ward mortality is unmeasured — no public ward-level dataset exists.</p>
    <p>{tb.available ? `Tier B (learned): a gradient-boosting model that predicts burden ${tb.horizon_days} days ahead, trained on ${tb.trained_on === 'demo' ? 'DEMO weather' : `ERA5 ${tb.train_range[0].slice(0, 4)}–${tb.train_range[1].slice(0, 4)}`} with synthetic NCRB-anchored outcomes. Its lagged-outcome inputs are simulated, so it is a pipeline demonstration, not a validated forecast.` : 'Tier B (learned model) is not loaded — train it with python -m app.train, then restart the API.'}</p></div>
    <div className="checks"><Check label="Confidence band on every estimate" /><Check label="Peak-hour co-occurring humidity" /><Check label="Demographics synthetic" warn />{tb.available && <Check label="Tier B trained on synthetic outcomes" warn />}</div></section>;
}
function Check({ label, warn }) { return <div className={warn ? 'check warn' : 'check'}>{warn ? <AlertTriangle size={13} /> : <CheckCircle2 size={13} />}{label}</div>; }

/* ───────────────────────────── pages ───────────────────────────── */
function RiskPage({ selectedWard, setSelectedWard, mapMode, setMapMode }) {
  const { city } = useData();
  const t = city.wards_by_tier, n = city.n_wards;
  const top = ['Red', 'Orange', 'Yellow', 'Green'].find(k => t[k] > 0);
  return <><section className="map-section panel full-map"><MapHeader mapMode={mapMode} setMapMode={setMapMode} /><RiskMap selectedWard={selectedWard} setSelectedWard={setSelectedWard} mapMode={mapMode} tall /><div className="map-explainer"><Info size={15} /><span>Weather is resolved on a coarse grid (a few cells cover the whole city), so ward-to-ward differences mostly reflect population and vulnerability. Only wards with model outputs are colored.</span></div></section>
    <div className="two-col"><div className="panel info-panel"><span className="section-kicker">DISTRIBUTION</span><h2>Heat-risk distribution</h2><p>{top ? `${t[top]} of ${n} wards sit in the highest active tier (${top}).` : 'No wards with model output.'}</p>
      <div className="distribution">{[['Red', 'red-bg'], ['Orange', 'orange-bg'], ['Yellow', 'yellow-bg'], ['Green', 'green-bg']].map(([k, cls]) => <DistributionBar key={k} label={k === 'Green' ? 'Low' : k} value={t[k]} total={n} cls={cls} />)}</div></div>
      <div className="panel info-panel"><span className="section-kicker">MAP READING</span><h2>How to use the map</h2><ul className="clean-list"><li>Pick a day above — today, or up to five days ahead.</li><li>Hover a ward to inspect its tier; click to open its metrics.</li><li>Switch between Risk, WBGT and Vulnerability.</li><li>Neutral wards indicate missing model output.</li></ul></div></div></>;
}
function DistributionBar({ label, value, total, cls }) { return <div className="dist-row"><div><span>{label}</span><b>{value}</b></div><div className="dist-track"><i className={cls} style={{ width: `${(value / Math.max(total, 1)) * 100}%` }}></i></div></div>; }

function AnalyticsPage() {
  const { timeline, rows } = useData();
  const data = timeline.days.map(d => ({ day: shortDate(d.date), wbgt: d.city.wbgt_outdoor_c, utci: d.city.utci_c, off: d.offset }));
  const today = data.find(d => d.off === 0)?.day;
  const bars = rows.slice(0, 15).map(d => ({ ward: String(d.no), excess: d.excess }));
  const tip = { borderRadius: 8, border: '1px solid var(--border)', background: 'var(--surface)' };
  return <><div className="analytics-grid"><div className="panel analytics-chart"><div className="panel-head"><div><span className="section-kicker">THERMAL OUTLOOK</span><h2>WBGT and UTCI</h2></div><span className="date-pill">{data.length} DAYS</span></div>
    <div className="chart large"><ResponsiveContainer width="100%" height="100%"><AreaChart data={data}><CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.12} /><XAxis dataKey="day" tickLine={false} axisLine={false} /><YAxis tickLine={false} axisLine={false} domain={['auto', 'auto']} /><Tooltip contentStyle={tip} />{today && <ReferenceLine x={today} stroke="#8f9a94" strokeDasharray="3 3" />}<Area type="monotone" dataKey="wbgt" stroke="#39ff14" fill="none" strokeWidth={2.5} /><Area type="monotone" dataKey="utci" stroke="#8f9a94" fill="none" strokeWidth={2} /></AreaChart></ResponsiveContainer></div></div>
    <div className="panel analytics-chart"><div className="panel-head"><div><span className="section-kicker">WARD COMPARISON</span><h2>Excess burden · top 15 wards</h2></div></div>
      <div className="chart large"><ResponsiveContainer width="100%" height="100%"><BarChart data={bars}><CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.1} /><XAxis dataKey="ward" tickLine={false} axisLine={false} /><YAxis tickLine={false} axisLine={false} /><Tooltip contentStyle={tip} /><Bar dataKey="excess" fill="#39ff14" radius={[3, 3, 0, 0]} /></BarChart></ResponsiveContainer></div></div></div>
    <TierBStrip /></>;
}

function TierBStrip() {
  const { meta } = useData();
  const tb = meta.tierb;
  if (!tb.available) return <section className="panel method-strip"><div><span className="section-kicker">TIER B · LEARNED MODEL</span><h2>Not loaded</h2><p>Train it once on your machine with <code>python -m app.train</code> (real ERA5) and restart the API. Tier A is unaffected.</p></div><span className="warning-chip"><Info size={14} /> OPTIONAL</span></section>;
  const t = tb.metrics.splits.test, g = tb.metrics.gate4;
  return <section className="panel method-strip"><div><span className="section-kicker">TIER B · LEARNED MODEL · T + {tb.horizon_days}</span>
    <h2>Skill vs climatology: {t.skill_vs_clim == null ? '—' : `${t.skill_vs_clim >= 0 ? '+' : ''}${t.skill_vs_clim.toFixed(3)}`} (test)</h2>
    <p>Trained on {tb.trained_on === 'demo' ? 'DEMO weather' : `ERA5 ${tb.train_range[0]} → ${tb.train_range[1]}`} across {tb.n_wards} wards with SYNTHETIC outcomes. Gate 4 (do compound indices beat raw temperature?): <b>{g.result}</b> — strongest thermal feature {g.strongest_thermal_feature}. Skill at or below zero means a ward-month climatology already explains the target. These are pipeline checks, not epidemiological accuracy.</p></div>
    <span className="warning-chip"><AlertTriangle size={14} /> {g.result === 'PASS' ? 'SYNTHETIC-TRAINED' : g.result}</span></section>;
}

function AlertsPage({ navigate }) {
  const { rows, city, risk } = useData();
  const hot = (city.wards_by_tier.Orange || 0) + (city.wards_by_tier.Red || 0);
  return <><div className="alert-grid"><div className="alert-primary panel"><div className="alert-icon" style={{ color: tierColor(city.tier), borderColor: `${tierColor(city.tier)}66` }}><AlertTriangle size={21} /></div><span className="section-kicker">ACTIVE PRIORITY</span><h2>{TIER_DESC[city.tier] || 'No data'}</h2>
    <p>{hot > 0 ? `${hot} of ${city.n_wards} wards are in the Orange or Red tier on ${longDate(risk.date)}.` : `No wards reach the Orange or Red tier on ${longDate(risk.date)}.`}</p>
    <button className="primary-btn" onClick={() => navigate('Risk Map')}>Open risk map <ArrowUpRight size={15} /></button></div>
    <div className="panel alert-list"><div className="panel-head"><div><span className="section-kicker">WARD ALERTS</span><h2>Priority queue</h2></div><span className="date-pill">{rows.length} MODELED</span></div>
      {rows.slice(0, 6).map(d => <div className="alert-row" key={d.id}><span className="risk-dot" style={{ background: tierColor(d.tier) }}></span><div><b>{d.id}</b><span>WBGT {f1(d.wbgt)}°C · UTCI {f1(d.utci)}°C</span></div><strong style={{ color: tierColor(d.tier) }}>{(d.tier || 'N/A').toUpperCase()}</strong></div>)}</div></div></>;
}

function exportCsv(rows, day, meta) {
  const head = ['ward_id', 'date', 'tier', 'wbgt_outdoor_c', 'utci_c', 'temperature_c', 'relative_humidity', 'vulnerability', 'excess_deaths', 'excess_deaths_lo', 'excess_deaths_hi', 'weather_source', 'demographics_source'];
  const body = rows.map(r => [r.id, day, r.tier, r.wbgt, r.utci, r.temp, r.rh, r.vuln, r.excess, r.low, r.high, meta.demo ? 'demo' : meta.source, 'synthetic'].join(','));
  const url = URL.createObjectURL(new Blob([[head.join(','), ...body].join('\n')], { type: 'text/csv' }));
  const a = Object.assign(document.createElement('a'), { href: url, download: `heatward_${day}.csv` });
  document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
}

function ReportsPage() {
  const { rows, day, meta, risk } = useData();
  const [showProv, setShowProv] = useState(false);
  return <><div className="reports-grid"><div className="panel report-card"><FileText size={20} /><span className="section-kicker">TIER A</span><h2>Ward assessment</h2><p>Ward-level heat burden summary for {longDate(risk.date)}, including uncertainty bands. Exported as CSV with provenance columns.</p><button className="secondary-btn" onClick={() => exportCsv(rows, day, meta)}><Download size={14} /> Export summary</button></div>
    <div className="panel report-card"><ShieldCheck size={20} /><span className="section-kicker">METHODOLOGY</span><h2>What is and isn't validated</h2><p>{meta.unmeasured}</p></div>
    <div className="panel report-card"><Database size={20} /><span className="section-kicker">PROVENANCE</span><h2>Data sources</h2><p>Every layer is labelled real, computed, transferred or synthetic.</p><button className="secondary-btn" onClick={() => setShowProv(!showProv)}>{showProv ? 'Hide provenance' : 'View provenance'} <ArrowUpRight size={14} /></button></div></div>
    {showProv && <section className="panel method-strip prov"><div><span className="section-kicker">LAYER PROVENANCE</span>{Object.entries(meta.provenance).map(([k, v]) => <div className="setting-row" key={k}><span>{k.replace('_', ' ')}</span><b>{v}</b></div>)}</div></section>}</>;
}

function LayersPage({ mapMode, setMapMode, navigate }) {
  const { meta, rows } = useData();
  const c = meta.coverage;
  return <><div className="layer-layout"><div className="panel layer-controls"><div className="panel-head"><div><span className="section-kicker">VISIBLE LAYERS</span><h2>Map controls</h2></div><SlidersHorizontal size={17} /></div>{[['risk', 'Heat risk', 'Tiered hazard classification (UTCI, weather only)'], ['burden', 'Excess burden', 'Hazard × vulnerability × population, vs the day\'s highest ward'], ...(meta.tierb.available ? [['tierb', 'Tier B forecast', 'Learned model (T+4): predicted burden, vs the day\'s highest ward']] : []), ['wbgt', 'WBGT', 'Outdoor thermal stress, banded around the MMT'], ['vulnerability', 'Vulnerability', 'Ward vulnerability weight']].map(([id, title, desc]) => <button className={mapMode === id ? 'layer-option active' : 'layer-option'} key={id} onClick={() => setMapMode(id)}><span className="layer-swatch"></span><div><b>{title}</b><small>{desc}</small></div><span className="layer-state">{mapMode === id ? 'ON' : 'OFF'}</span></button>)}</div>
    <div className="panel layer-note"><span className="section-kicker">DATA COVERAGE</span><h2>{c.wards_total} wards in geometry</h2><p>Model outputs cover {rows.length} wards. Weather is fetched for {c.weather_cells} grid cells ({c.grid_deg}° resolution) and shared by the wards inside each.</p>
      <div className="coverage"><div><b>{c.wards_total}</b><span>Mapped wards</span></div><div><b>{rows.length}</b><span>Modeled wards</span></div><div><b>{c.wards_total - rows.length}</b><span>No output</span></div></div><button className="primary-btn" onClick={() => navigate('Risk Map')}>Apply to map <ArrowUpRight size={15} /></button></div></div></>;
}

function SettingsPage({ dark, setDark }) {
  const { meta, timeline } = useData();
  const max = Math.max(...timeline.days.map(d => d.offset));
  return <div className="settings-grid"><div className="panel settings-card"><span className="section-kicker">APPEARANCE</span><h2>Interface theme</h2><p>Switch between the command-center dark interface and a high-contrast light interface.</p><div className="theme-choice"><button className={!dark ? 'theme-choice active' : ''} onClick={() => setDark(false)}><Sun size={17} /><span>Light</span></button><button className={dark ? 'theme-choice active' : ''} onClick={() => setDark(true)}><Moon size={17} /><span>Dark</span></button></div></div>
    <div className="panel settings-card"><span className="section-kicker">MODEL DISPLAY</span><h2>Live feed</h2><p>HEATWARD serves the Tier A engine on live weather. The PS asks for a 3–5 day lead; pick any day in the strip.</p>
      <div className="setting-row"><span>Forecast horizon</span><b>UP TO T + {max} DAYS</b></div><div className="setting-row"><span>Weather source</span><b>{meta.demo ? 'DEMO (SYNTHETIC)' : meta.source.toUpperCase()}</b></div><div className="setting-row"><span>Server refresh</span><b>EVERY {meta.refresh_minutes} MIN</b></div><div className="setting-row"><span>Minimum-mortality threshold</span><b>{meta.engine.mmt_c} °C WBGT</b></div></div></div>;
}

/* ───────────────────────────── map ───────────────────────────── */
const LEGENDS = {
  tierb: [['< 25% of max', TIER_COLOR.Green], ['25–50%', TIER_COLOR.Yellow], ['50–75%', TIER_COLOR.Orange], ['≥ 75%', TIER_COLOR.Red]],
  burden: [['< 25% of max', TIER_COLOR.Green], ['25–50%', TIER_COLOR.Yellow], ['50–75%', TIER_COLOR.Orange], ['≥ 75%', TIER_COLOR.Red]],
  risk: [['Lower', TIER_COLOR.Green], ['Strong', TIER_COLOR.Yellow], ['Very strong', TIER_COLOR.Orange], ['Extreme', TIER_COLOR.Red]],
  wbgt: [['< MMT−2', TIER_COLOR.Green], ['< MMT', TIER_COLOR.Yellow], ['< MMT+2', TIER_COLOR.Orange], ['≥ MMT+2', TIER_COLOR.Red]],
  vulnerability: [['Lower', '#3f713f'], ['Mid', '#8fdc63'], ['Higher', '#39ff14']]
};

function RiskMap({ selectedWard, setSelectedWard, mapMode, tall = false }) {
  const { rows, byId, meta } = useData();
  const ref = useRef(null), mapRef = useRef(null);
  const [loaded, setLoaded] = useState(false), [hover, setHover] = useState(null);
  const byIdRef = useRef(byId); byIdRef.current = byId;
  const mmt = meta.engine.mmt_c;
  const selNo = selectedWard ? String(parseInt(selectedWard.slice(1), 10)) : '';

  useEffect(() => {
    if (!ref.current || mapRef.current) return;
    const map = new maplibregl.Map({ container: ref.current, style: { version: 8, sources: {}, layers: [{ id: 'background', type: 'background', paint: { 'background-color': '#0b110f' } }] }, center: [88.36, 22.57], zoom: 10.3, attributionControl: false });
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right');
    map.on('load', async () => {
      const geo = await fetch('/wards_kolkata.geojson').then(r => r.json());
      map.addSource('wards', { type: 'geojson', data: geo });
      map.addLayer({ id: 'ward-fill', type: 'fill', source: 'wards', paint: { 'fill-color': UNKNOWN, 'fill-opacity': 0.88 } });   // real colours arrive from the effect below
      map.addLayer({ id: 'ward-line', type: 'line', source: 'wards', paint: { 'line-color': '#3b523e', 'line-width': 0.55, 'line-opacity': 0.55 } });
      map.addLayer({ id: 'ward-selected', type: 'line', source: 'wards', paint: { 'line-color': '#39ff14', 'line-width': 2.2 }, filter: ['==', ['get', 'WARD'], ''] });
      map.fitBounds([[88.18, 22.43], [88.52, 22.72]], { padding: 28, duration: 0 });
      map.on('mousemove', 'ward-fill', e => { map.getCanvas().style.cursor = 'pointer'; const f = e.features?.[0]; if (f) setHover({ x: e.point.x, y: e.point.y, no: String(parseInt(f.properties?.WARD, 10)) }); });
      map.on('mouseleave', 'ward-fill', () => { map.getCanvas().style.cursor = ''; setHover(null); });
      map.on('click', 'ward-fill', e => { const no = parseInt(e.features?.[0]?.properties?.WARD, 10); if (!Number.isNaN(no)) { const id = `W${String(no).padStart(2, '0')}`; if (byIdRef.current[id]) setSelectedWard(id); } });
      setLoaded(true);
    });
    return () => { map.remove(); mapRef.current = null; };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loaded) return;
    map.setPaintProperty('ward-fill', 'fill-color', fillExpr(rows, mapMode, mmt));
    map.setFilter('ward-selected', ['==', ['to-string', ['get', 'WARD']], selNo]);
  }, [mapMode, rows, selNo, loaded, mmt]);

  const hw = hover ? byId[`W${hover.no.padStart(2, '0')}`] : null;
  return <div className={tall ? 'map-wrap tall' : 'map-wrap'}><div ref={ref} className="map-canvas"></div>
    <div className="map-legend"><b>{mapMode === 'risk' ? 'RISK' : mapMode === 'tierb' ? 'TIER B · SHARE OF MAX' : mapMode.toUpperCase()}</b>{LEGENDS[mapMode].map(([l, c]) => <span key={l}><i style={{ background: c }}></i>{l}</span>)}<span><i className="legend-none"></i>No output</span></div>
    {hover && <div className="map-tooltip" style={{ left: Math.min(hover.x + 14, (ref.current?.clientWidth || 400) - 170), top: Math.max(hover.y - 18, 12) }}><b>WARD {hover.no.padStart(3, '0')}</b><span>{hw ? `${hw.tier} · ${f1(hw.wbgt)}°C WBGT` : 'No model output'}</span></div>}
    <div className="map-note">{meta.coverage.wards_total} ward boundaries · no external basemap</div></div>;
}

createRoot(document.getElementById('root')).render(<AppErrorBoundary><App /></AppErrorBoundary>);
