// app.module.js
// Odds Genius — Globe Fixtures UI (ESM + local vendor imports)

import * as THREE from 'three';
import { OrbitControls }   from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer }  from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass }      from 'three/examples/jsm/postprocessing/RenderPass.js';
import { ShaderPass }      from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { FXAAShader }      from 'three/examples/jsm/shaders/FXAAShader.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

// PapaParse (UMD) is loaded in index.html
const Papa = window.Papa;
if (!Papa) throw new Error('PapaParse missing from window');

// ----------------------------
// DOM refs
// ----------------------------
const el = {
  globeWrap: document.getElementById('globe-container'),
  insights: document.getElementById('insights-content'),
  fixtureTitle: document.getElementById('fixture-title'),
  fixtureContext: document.getElementById('fixture-context'),
  matchList: document.getElementById('match-intelligence'),
  watchlist: document.getElementById('player-watchlist'),
  market: document.getElementById('market-snapshot'),
  deepBtn: document.getElementById('deep-dive-btn'),
  homeBadge: document.getElementById('home-badge'),
  awayBadge: document.getElementById('away-badge'),
  upload: document.getElementById('bet-upload'),
  prevBtn: document.getElementById('nav-prev'),
  nextBtn: document.getElementById('nav-next')
};

// ----------------------------
// three-globe loader (CDN-first; ESM → UMD)
// ----------------------------
async function importScriptUMD(src) {
  await new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.onload = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
  });
  if (window.Globe) return window.Globe;
  throw new Error('UMD three-globe loaded but window.Globe was not defined');
}

async function loadThreeGlobe() {
  for (const v of ['2.31.3','2.31.1','2.30.1','2.29.3']) {
    const url = `https://esm.sh/three-globe@${v}?bundle&external=three`;
    try {
      const m = await import(url);
      console.warn('[three-globe] using esm.sh ESM:', url);
      return m.default ?? m;
    } catch (e) {
      console.warn('[three-globe] esm.sh failed:', url, e);
    }
  }
  for (const url of [
    'https://cdn.jsdelivr.net/npm/three-globe@2.31.1/dist/three-globe.min.js',
    'https://unpkg.com/three-globe@2.31.1/dist/three-globe.min.js',
    './vendor/three-globe.min.js'
  ]) {
    try {
      const ctor = await importScriptUMD(url);
      console.warn('[three-globe] using UMD:', url);
      return ctor;
    } catch (e) {
      console.warn('[three-globe] UMD failed:', url, e);
    }
  }
  if (el.globeWrap) {
    el.globeWrap.innerHTML = `<div class="globe-error">
      <strong>three-globe failed to load.</strong><br/>
      Check your network or add <code>vendor/three-globe.min.js</code>.
    </div>`;
  }
  return Promise.reject(new Error('three-globe could not be loaded from any source'));
}

// ----------------------------
// Globals / tuning
// ----------------------------
let ThreeGlobeCtor;
let globe;
let renderer, scene, camera, controls, composer, bloomPass;
let fixtures = [];
let filtered = [];       // <- current filtered list (date/league)
let activeIdx = 0;
let selectedId = null;
let pulseRing;
let htmlTabsData = []; // [{lat,lng,altitude,idx,el}]

// Central UI state (A)
const UI = {
  date: null,       // start date (00:00)
  rangeDays: 1,     // number of days
  league: 'ALL'     // league selector
};

// Color/tuning
const COLORS = {
  marker:        '#A7FFF6',
  markerInactive:'#8CEFE5',
  markerActive:  '#CFFFFA',
  ring:          '#9EE7E3'
};

const SURFACE_EPS   = 0.009;
const RADIUS_BASE   = 0.014;
const RADIUS_ACTIVE = 0.040;
const CAMERA_ALT    = 2.0;
const BLOOM = { strength: 0.9, radius: 0.6, threshold: 0.75 };

// ----------------------------
// Utilities
// ----------------------------
const clamp01 = v => Math.max(0, Math.min(1, v));
const pct = n => `${Math.round(clamp01(n) * 100)}%`;

// Demo "today" (Patch 1)
const DEMO_TODAY = new Date('2023-11-28T00:00:00Z');

function showToast(type, text, ms=2600){
  const t = document.createElement('div');
  t.className = `og-toast ${type}`;
  t.textContent = text;
  document.body.appendChild(t);
  requestAnimationFrame(()=>t.classList.add('show'));
  setTimeout(()=>{ t.classList.remove('show'); setTimeout(()=>t.remove(),250); }, ms);
}

function clearNode(node) { if (!node) return; while (node.firstChild) node.removeChild(node.firstChild); }

function pick(row, keys) {
  for (const k of keys) {
    const v = (row[k] ?? '').toString().trim();
    if (v) return v;
  }
  return '';
}

function getGlobeRadius() {
  if (globe?.getGlobeRadius) return globe.getGlobeRadius();
  try {
    const m = globe.children?.find(c => c.geometry?.parameters?.radius);
    return m?.geometry?.parameters?.radius || 100;
  } catch { return 100; }
}

function hasHtmlElementsApi(g){
  return g && typeof g.htmlElementsData === 'function' &&
               typeof g.htmlElement      === 'function' &&
               typeof g.htmlLat          === 'function';
}

function initials(name = '') {
  const words = String(name).trim().split(/\s+/).filter(Boolean);
  if (!words.length) return '';
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + (words[1][0] || '')).toUpperCase();
}
function stripDiacritics(s) { try { return s.normalize('NFD').replace(/\p{Diacritic}/gu, ''); } catch { return s; } }
function slugLocal(team) {
  return stripDiacritics(String(team))
    .toLowerCase()
    .replace(/&/g,'and')
    .replace(/[\u2019'’]/g,'')
    .replace(/[^a-z0-9]+/g,'-')
    .replace(/^-+|-+$/g,'');
}
function normalizeBasicUrl(raw) {
  if (!raw) return '';
  let u = String(raw).trim();
  if (!u) return '';
  if (u.startsWith('//')) u = `https:${u}`;
  if (/^http:\/\//i.test(u) && location.protocol === 'https:') u = u.replace(/^http:\/\//i, 'https://');
  return u;
}

// ---- Date helpers (E)
const startOfDay = d => new Date(d.getFullYear(), d.getMonth(), d.getDate());
const addDays = (d, n) => { const x = new Date(d); x.setDate(x.getDate()+n); return x; };
const dayKey = d => `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
const fromISOtoLocalDay = iso => startOfDay(new Date(iso));
const inDayRange = (iso, sDay, nDays) => {
  if (!iso || !sDay || !Number.isFinite(nDays)) return false;
  const d = new Date(iso);
  const s = startOfDay(sDay);
  const e = addDays(s, nDays); // exclusive
  return d >= s && d < e;
};
function extractFixtureDays(list){
  const map = new Map();
  (list||[]).forEach(r=>{
    const d = fromISOtoLocalDay(r.date_utc);
    map.set(dayKey(d), d);
  });
  return [...map.values()].sort((a,b)=>a-b);
}

// ------- Globe lat/lon helpers (corrected mapping)
function latLngToVec3(latDeg, lonDeg, altFrac = 0) {
  const R = getGlobeRadius() * (1 + (altFrac || 0));
  // three-globe uses: phi = (90 - lat), theta = (180 - lon)
  const phi   = THREE.MathUtils.degToRad(90 - latDeg);
  const theta = THREE.MathUtils.degToRad(180 - lonDeg);
  const x = R * Math.sin(phi) * Math.cos(theta);
  const y = R * Math.cos(phi);
  const z = R * Math.sin(phi) * Math.sin(theta);
  return new THREE.Vector3(x, y, z);
}
function surfaceNormalAt(latDeg, lonDeg) { return latLngToVec3(latDeg, lonDeg, 0).normalize(); }

// ---- Stadium image candidates (local assets only)
const STADIUM_BASE = './assets/stadiums';
function stadiumCandidates(f) {
  const names = [
    f.stadium, f.venue, f.stadium_name, f.venue_name,
    `${f.home_team} ${f.city||''}`, f.home_team, f.away_team,
    f.fixture_id
  ].filter(Boolean).map(v => String(v));
  const slugs = Array.from(new Set(names.map(n => slugLocal(n)).filter(Boolean)));
  const exts = ['.jpg','.jpeg','.png','.webp'];
  const out = [];
  for (const s of slugs) for (const ext of exts) out.push(`${STADIUM_BASE}/${s}${ext}`);
  return out;
}

// ---------- Logo system (local only; no remote)
const LOGO_LOCAL_BASE = './assets/assets/logos';
const TEAM_LOGO_OVERRIDES = {
  'Celtic': `${LOGO_LOCAL_BASE}/celtic.svg`,
  'Lazio': `${LOGO_LOCAL_BASE}/lazio.svg`,
  'Royal Antwerp FC': `${LOGO_LOCAL_BASE}/royal-antwerp.svg`,
  'Shakhtar Donetsk': `${LOGO_LOCAL_BASE}/shakhtar-donetsk.svg`,
  'Atlético Madrid': `${LOGO_LOCAL_BASE}/atletico-madrid.svg`,
  'Atletico Madrid': `${LOGO_LOCAL_BASE}/atletico-madrid.svg`,
  'Feyenoord': `${LOGO_LOCAL_BASE}/feyenoord.svg`,
  'PSG': `${LOGO_LOCAL_BASE}/paris-saint-germain.svg`,
  'Paris Saint-Germain': `${LOGO_LOCAL_BASE}/paris-saint-germain.svg`,
  'Newcastle United': `${LOGO_LOCAL_BASE}/newcastle-united.svg`,
  'AC Milan': `${LOGO_LOCAL_BASE}/ac-milan.svg`,
  'Borussia Dortmund': `${LOGO_LOCAL_BASE}/borussia-dortmund.svg`,
  'Manchester City': `${LOGO_LOCAL_BASE}/manchester-city.svg`,
  'RB Leipzig': `${LOGO_LOCAL_BASE}/rb-leipzig.svg`,
  'Young Boys': `${LOGO_LOCAL_BASE}/young-boys.svg`,
  'Red Star Belgrade': `${LOGO_LOCAL_BASE}/red-star-belgrade.svg`,
  'Crvena Zvezda': `${LOGO_LOCAL_BASE}/red-star-belgrade.svg`,
  'FC Barcelona': `${LOGO_LOCAL_BASE}/fc-barcelona.svg`,
  'Barcelona': `${LOGO_LOCAL_BASE}/fc-barcelona.svg`,
  'Porto': `${LOGO_LOCAL_BASE}/fc-porto.svg`,
  'FC Porto': `${LOGO_LOCAL_BASE}/fc-porto.svg`,
  'Galatasaray': `${LOGO_LOCAL_BASE}/galatasaray.svg`,
  'Manchester United': `${LOGO_LOCAL_BASE}/manchester-united.svg`,
  'Sevilla FC': `${LOGO_LOCAL_BASE}/sevilla-fc.svg`,
  'PSV': `${LOGO_LOCAL_BASE}/psv.svg`,
  'PSV Eindhoven': `${LOGO_LOCAL_BASE}/psv.svg`,
  'København': `${LOGO_LOCAL_BASE}/fc-kobenhavn.svg`,
  'FC Copenhagen': `${LOGO_LOCAL_BASE}/fc-kobenhavn.svg`,
  'Arsenal': `${LOGO_LOCAL_BASE}/arsenal.svg`,
  'Lens': `${LOGO_LOCAL_BASE}/lens.svg`,
  'Real Madrid': `${LOGO_LOCAL_BASE}/real-madrid.svg`,
  'Napoli': `${LOGO_LOCAL_BASE}/ssc-napoli.svg`,
  'SSC Napoli': `${LOGO_LOCAL_BASE}/ssc-napoli.svg`,
  'Benfica': `${LOGO_LOCAL_BASE}/sl-benfica.svg`,
  'SL Benfica': `${LOGO_LOCAL_BASE}/sl-benfica.svg`,
  'Inter Milan': `${LOGO_LOCAL_BASE}/inter-milan.svg`,
  'Inter': `${LOGO_LOCAL_BASE}/inter-milan.svg`,
  'Real Sociedad': `${LOGO_LOCAL_BASE}/real-sociedad.svg`,
  'Salzburg': `${LOGO_LOCAL_BASE}/rb-salzburg.svg`,
  'Bayern München': `${LOGO_LOCAL_BASE}/bayern-munich.svg`,
  'Bayern Munich': `${LOGO_LOCAL_BASE}/bayern-munich.svg`,
  'Sporting Braga': `${LOGO_LOCAL_BASE}/sporting-braga.svg`,
  'Union Berlin': `${LOGO_LOCAL_BASE}/union-berlin.svg`
};
const LOGO_CACHE_KEY  = 'og_logo_cache_v_preload';
let LOGO_CACHE = {};
try { LOGO_CACHE = JSON.parse(localStorage.getItem(LOGO_CACHE_KEY) || '{}'); } catch {}
function saveLogoCache(){ try { localStorage.setItem(LOGO_CACHE_KEY, JSON.stringify(LOGO_CACHE)); } catch {} }
const LOGO_STORE = new Map(); // teamName -> { img, url }
function localLogoCandidates(team) {
  const slug = slugLocal(team);
  return [`${LOGO_LOCAL_BASE}/${slug}.png`, `${LOGO_LOCAL_BASE}/${slug}.svg`];
}
function guessLogoSources(teamName = '') {
  const list = [];
  if (TEAM_LOGO_OVERRIDES[teamName]) list.push(TEAM_LOGO_OVERRIDES[teamName]);
  for (const loc of localLogoCandidates(teamName)) list.push(loc);
  return [...new Set(list.map(normalizeBasicUrl))];
}
async function tryLoadDirect(src, teamName, timeoutMs) {
  return new Promise(resolve => {
    let done = false;
    const img = new Image();
    img.alt = teamName || '';
    img.loading = 'lazy';
    img.decoding = 'async';
    if (!/^data:/i.test(src)) { img.crossOrigin = 'anonymous'; img.referrerPolicy = 'no-referrer'; }
    const t = setTimeout(() => { if (!done){ done = true; resolve(null); } }, timeoutMs);
    img.onload  = () => { if (!done){ clearTimeout(t); done = true; resolve({ ok:true, img, src }); } };
    img.onerror = () => { if (!done){ clearTimeout(t); done = true; resolve(null); } };
    img.src = `${src}${src.includes('?') ? '&' : '?'}v=${Date.now().toString(36)}`;
  });
}
async function blobToDataURL(blob) {
  return new Promise((resolve, reject)=>{
    const fr = new FileReader();
    fr.onload  = () => resolve(fr.result);
    fr.onerror = reject;
    fr.readAsDataURL(blob);
  });
}
async function tryLoadViaDataURL(src, teamName, timeoutMs) {
  try {
    const ac = new AbortController();
    const to = setTimeout(() => ac.abort(), timeoutMs);
    const res = await fetch(`${src}${src.includes('?') ? '&' : '?'}v=${Date.now().toString(36)}`, { cache: 'no-store', signal: ac.signal });
    clearTimeout(to);
    if (!res.ok) return null;
    const blob = await res.blob();
    const dataURL = await blobToDataURL(blob);
    return await new Promise(resolve => {
      let done = false;
      const img = new Image();
      img.alt = teamName || '';
      img.onload  = () => { if (!done){ done = true; resolve({ ok:true, img, src: dataURL }); } };
      img.onerror = () => { if (!done){ done = true; resolve(null); } };
      img.src = dataURL;
    });
  } catch { return null; }
}
async function tryLoad(src, teamName, timeoutMs = 15000) {
  let hit = await tryLoadDirect(src, teamName, Math.min(timeoutMs, 7000));
  if (hit) return hit;
  hit = await tryLoadViaDataURL(src, teamName, timeoutMs);
  return hit;
}
async function prefetchAllLogos(teamMap) {
  const items = [...teamMap.keys()];
  const CONCURRENCY = 4;
  let idx = 0;
  async function worker() {
    while (idx < items.length) {
      const i = idx++;
      const name = items[i];
      if (!name || LOGO_STORE.has(name)) continue;
      const sources = guessLogoSources(name);
      let hit = null;
      for (const src of sources) { hit = await tryLoad(src, name, 15000); if (hit) break; }
      if (hit) LOGO_STORE.set(name, { img: hit.img, url: hit.src });
      else LOGO_STORE.set(name, { img: null, url: null });
    }
  }
  await Promise.all(Array.from({length: CONCURRENCY}, worker));
}
const badgeTokens = new WeakMap();
async function setBadge(elm, urlFromCsv, teamName='') {
  if (!elm) return;
  const prevTeam = elm.dataset.teamName || '';
  if (prevTeam === teamName && elm.querySelector('img')) return;
  const token = {}; badgeTokens.set(elm, token);
  elm.dataset.teamName = teamName;
  elm.textContent = initials(teamName) || '';
  elm.classList.remove('has-logo');
  const pre = LOGO_STORE.get(teamName);
  if (pre?.img) {
    if (badgeTokens.get(elm) !== token) return;
    elm.innerHTML = ''; elm.appendChild(pre.img.cloneNode(true)); elm.classList.add('has-logo');
    if (pre.url) { LOGO_CACHE[teamName] = pre.url; saveLogoCache(); }
    return;
  }
  const sources = guessLogoSources(teamName);
  let hit = null;
  for (const src of sources) { hit = await tryLoad(src, teamName, 15000); if (hit) break; }
  if (!hit) return;
  if (badgeTokens.get(elm) !== token) return;
  elm.innerHTML = ''; elm.appendChild(hit.img); elm.classList.add('has-logo');
  LOGO_STORE.set(teamName, { img: hit.img.cloneNode(true), url: hit.src });
  LOGO_CACHE[teamName] = hit.src; saveLogoCache();
}

// ----------------------------
// COMPETITION LOGOS + ACCURACY STRIP
// ----------------------------
const COMP_LOGO_BASE = './assets/assets/leagues';
const COMP_LOGO_MAP = {
  'UEFA Champions League':      `${COMP_LOGO_BASE}/uefa-champions-league.svg`,
  'UEFA Europa League':         `${COMP_LOGO_BASE}/uefa-europa-league.svg`,
  'UEFA Europa Conference League': `${COMP_LOGO_BASE}/uefa-europa-conference-league.svg`,
  'Premier League':             `${COMP_LOGO_BASE}/premier-league.svg`,
  'LaLiga':                     `${COMP_LOGO_BASE}/la-liga.svg`,
  'La Liga':                    `${COMP_LOGO_BASE}/la-liga.svg`,
  'Bundesliga':                 `${COMP_LOGO_BASE}/bundesliga.svg`,
  'Serie A':                    `${COMP_LOGO_BASE}/serie-a.svg`,
  'Ligue 1':                    `${COMP_LOGO_BASE}/ligue-1.svg`,
  'Liga Portugal':              `${COMP_LOGO_BASE}/liga-nos.png`,
  'Liga NOS':                   `${COMP_LOGO_BASE}/liga-nos.png`,
  'MLS':                        `${COMP_LOGO_BASE}/usa-mls.svg`,
  'Major League Soccer':        `${COMP_LOGO_BASE}/usa-mls.svg`,
};
function findCompLogoSrc(name = '') {
  if (!name) return '';
  if (COMP_LOGO_MAP[name]) return COMP_LOGO_MAP[name];
  const key = Object.keys(COMP_LOGO_MAP).find(k => name.toLowerCase().includes(k.toLowerCase()));
  return key ? COMP_LOGO_MAP[key] : '';
}
function getCompetitionSnapshot(compName) {
  const rows = (compName && compName.trim())
    ? filtered.filter(r => (r?.competition || '').toLowerCase() === compName.toLowerCase())
    : [];
  const avg = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
  const toPct = x => Math.round((x || 0) * 100);
  return {
    n: rows.length,
    ftr:    toPct(avg(rows.map(r => +r?.confidence_ftr || 0))),
    over25: toPct(avg(rows.map(r => +r?.over25_prob   || 0))),
    btts:   toPct(avg(rows.map(r => +r?.btts_prob     || 0)))
  };
}
function renderCompetitionAccuracy(compName) {
  const wrap = document.getElementById('comp-accuracy'); if (!wrap) return;
  const nameEl = document.getElementById('comp-name');
  const logoEl = document.getElementById('comp-logo');
  const fill   = wrap.querySelector('.gauge-fill');
  const val    = wrap.querySelector('.gauge-val');
  const chips  = document.getElementById('comp-traffic');
  if (nameEl) nameEl.textContent = compName || '—';
  if (logoEl) {
    const logoSrc = findCompLogoSrc(compName);
    if (logoSrc) { logoEl.src = logoSrc; logoEl.alt = `${compName} logo`; logoEl.style.display = 'inline-block'; }
    else { logoEl.removeAttribute('src'); logoEl.style.display = 'none'; }
  }
  const stats = getCompetitionSnapshot(compName);
  const ftrPct = 87; // demo
  if (fill) fill.style.width = `${Math.max(0, Math.min(100, ftrPct))}%`;
  if (val)  val.textContent  = `${ftrPct}%`;
  if (chips) {
    chips.innerHTML = '';
    const add = (cls, label) => { const s = document.createElement('span'); s.className = `light ${cls}`; s.textContent = label; return s; };
    chips.append(
      add('light--green', `FTR ${ftrPct}%`),
      add('light--blue',  `O2.5 ${stats.over25||'—'}%`),
      add('light--amber', `BTTS ${stats.btts||'—'}%`)
    );
  }
}

// ----------------------------
// Scene init
// ----------------------------
async function init() {
  ThreeGlobeCtor = await loadThreeGlobe();
  scene = new THREE.Scene();

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.domElement.style.pointerEvents = 'auto'; // ensure controls work
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(el.globeWrap.clientWidth, el.globeWrap.clientHeight);
  el.globeWrap.innerHTML = '';
  el.globeWrap.appendChild(renderer.domElement);

  if ('outputColorSpace' in renderer) renderer.outputColorSpace = THREE.SRGBColorSpace;
  if ('toneMapping' in renderer) { renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.15; }

  camera = new THREE.PerspectiveCamera(45, el.globeWrap.clientWidth / el.globeWrap.clientHeight, 0.1, 5000);
  camera.position.set(0, 0, 300);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = true;
  controls.enableZoom = true;
  controls.autoRotate = false;
  controls.minDistance = 140;
  controls.maxDistance = 1200;

  scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  scene.add(new THREE.HemisphereLight(0xddeeff, 0x223344, 0.6));

  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));

  const fxaaPass = new ShaderPass(FXAAShader);
  const setFXAA = () => {
    const px = renderer.getPixelRatio();
    fxaaPass.material.uniforms['resolution'].value.set(1/(el.globeWrap.clientWidth*px), 1/(el.globeWrap.clientHeight*px));
  };
  setFXAA();
  composer.addPass(fxaaPass);

  bloomPass = new UnrealBloomPass(new THREE.Vector2(el.globeWrap.clientWidth, el.globeWrap.clientHeight), BLOOM.strength, BLOOM.radius, BLOOM.threshold);
  composer.addPass(bloomPass);

  globe = new ThreeGlobeCtor({ waitForGlobeReady: true })
    .showAtmosphere(true)
    .atmosphereColor('#9ef9e3')
    .atmosphereAltitude(0.28)
    .globeImageUrl('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
    .bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png')
    .pointAltitude(() => SURFACE_EPS)
    .pointRadius(d => (d.__active ? RADIUS_ACTIVE : RADIUS_BASE))
    .pointColor(d => (d.__active ? COLORS.markerActive : COLORS.marker))
    .pointResolution(12)
    .pointsMerge(true);

  scene.add(globe);

  if (typeof globe.onPointHover === 'function') globe.onPointHover(handleHover);
  globe.onPointClick?.(pt => { if (!pt) return; const idx = filtered.findIndex(f => f.fixture_id === pt.fixture_id); if (idx >= 0) selectIndex(idx, { fly: true }); });

  // ---------- ACTIVE MARKER (radar + beam + billboard)
  {
    const group = new THREE.Group();
    scene.add(group);

    const radarMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0x80ffe6), transparent: true, opacity: 0.35, depthWrite: false, side: THREE.DoubleSide });
    const beamMat  = new THREE.MeshBasicMaterial({ color: 0x7df9c4, transparent: true, opacity: 0.22, blending: THREE.AdditiveBlending, depthWrite: false });

    const ringGeo = new THREE.RingGeometry(1.0, 1.06, 128);
    const cylGeo  = new THREE.CylinderGeometry(0.18, 0.28, 30, 32, 1, true);

    const radarGroup = new THREE.Group();
    const rings = [];
    for (let i=0;i<3;i++){ const m = new THREE.Mesh(ringGeo, radarMat.clone()); m.scale.setScalar(1); m.material.opacity = 0; rings.push(m); radarGroup.add(m); }
    group.add(radarGroup);

    const beam = new THREE.Mesh(cylGeo, beamMat); beam.visible = false; group.add(beam);

    const billboard = new THREE.Sprite(new THREE.SpriteMaterial({ transparent:true, opacity:0 }));
    billboard.scale.set(14, 8, 1);
    group.add(billboard);

    const markerState = { lat:0, lon:0, t0:0, active:false };
    window.__OG_MARKER__ = { group, radarGroup, rings, beam, billboard, markerState };
  }

  window.addEventListener('resize', () => {
    const { clientWidth, clientHeight } = el.globeWrap;
    renderer.setSize(clientWidth, clientHeight);
    camera.aspect = clientWidth / clientHeight;
    camera.updateProjectionMatrix();
    const px = renderer.getPixelRatio();
    fxaaPass.material.uniforms['resolution'].value.set(1/(clientWidth*px), 1/(clientHeight*px));
    bloomPass.setSize?.(clientWidth, clientHeight);
  });

  window.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight') { e.preventDefault(); step(+1); }
    if (e.key === 'ArrowLeft')  { e.preventDefault(); step(-1); }
  });

  el.prevBtn?.addEventListener('click', () => step(-1));
  el.nextBtn?.addEventListener('click', () => step(+1));

  // Build initial calendar + league UI from data later (after CSV)
  await loadFixturesCSV('./data/fixtures.csv');
  animate();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();

  // marker anims
  const S = window.__OG_MARKER__;
  if (S && S.markerState.active){
    const { rings, radarGroup, beam, billboard, group, markerState } = S;
    const now = performance.now() * 0.001;
    const R = getGlobeRadius();

    // world position + normal
    const pos = latLngToVec3(markerState.lat, markerState.lon, SURFACE_EPS + 0.001);
    const n   = pos.clone().normalize();

    group.position.copy(pos);
    group.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0), n);

    radarGroup.position.set(0,0,0);

    // rings
    const dur = 2.6;
    const t = (now - markerState.t0);
    const baseScale = 1.0;
    const spread    = R * 0.015;
    rings.forEach((r, i) => {
      const k = ((t + i*0.6) % dur) / dur;
      const s = baseScale + k * spread;
      r.scale.setScalar(s);
      r.material.opacity = (1.0 - k) * 0.35;
    });

    if (beam.visible) {
      const flicker = 0.18 + 0.06*Math.sin(now*7.0) + 0.04*Math.sin(now*13.0);
      beam.material.opacity = flicker;
    }
    billboard.quaternion.copy(camera.quaternion);
  }

  composer.render();
}

// ----------------------------
// Calendar rail + League chips (B, C, D) + Patches 2/3
// ----------------------------
function buildLeagueChips() {
  const host = document.getElementById('league-chips'); if (!host) return;
  const leagues = ['ALL','UEFA Champions League','Premier League','LaLiga','Bundesliga','Serie A','Ligue 1','Liga Portugal','MLS'];
  host.innerHTML = '';
  leagues.forEach(name=>{
    const b = document.createElement('button');
    b.className = `chip${name==='ALL'?' is-active':''}`;
    b.dataset.league = name;
    b.textContent = name==='ALL' ? 'All' : name.replace('UEFA ','').replace('Premier ','EPL ');
    b.addEventListener('click', ()=>{
      host.querySelectorAll('.chip').forEach(x=>x.classList.remove('is-active'));
      b.classList.add('is-active');
      UI.league = name;
      refillVisibleFixtures();
    });
    host.appendChild(b);
  });
}

function buildCalendarRail(fixtureDays, baseDay = DEMO_TODAY) {
  const rail = document.getElementById('date-rail'); if (!rail) return;

  // Pills
  const pills = rail.querySelectorAll('[data-range]');
  pills.forEach(btn=>{
    btn.addEventListener('click', ()=>{
      pills.forEach(x=>x.classList.remove('is-active'));
      btn.classList.add('is-active');

      const today = startOfDay(new Date(baseDay));
      const key = btn.getAttribute('data-range');
      if (key === 'today') {
        UI.date = today; UI.rangeDays = 1;
      } else if (key === 'tomorrow') {
        UI.date = addDays(today, 1); UI.rangeDays = 1;
      } else { // weekend
        const dow = today.getDay();
        const toSat = (6 - dow + 7) % 7;
        UI.date = addDays(today, toSat); UI.rangeDays = 2;
      }
      rail.querySelectorAll('.date-pill').forEach(b=>b.classList.remove('is-active'));
      refillVisibleFixtures();
    });
  });

  // Mini calendar from actual fixture days
  const grid = rail.querySelector('#date-grid');
  if (grid) {
    grid.innerHTML = '';
    if (!fixtureDays || !fixtureDays.length) {
      const msg = document.createElement('div'); msg.className='muted'; msg.textContent='No fixture days'; grid.appendChild(msg);
    } else {
      fixtureDays.forEach(d=>{
        const btn = document.createElement('button');
        btn.className = 'date-pill';
        btn.textContent = d.getDate();
        btn.title = d.toDateString();
        btn.addEventListener('click', ()=>{
          rail.querySelectorAll('.date-pill').forEach(b=>b.classList.remove('is-active'));
          btn.classList.add('is-active');
          pills.forEach(b=>b.classList.remove('is-active'));
          UI.date = d; UI.rangeDays = 1;
          refillVisibleFixtures();
        });
        grid.appendChild(btn);
      });
    }
  }
}

// Core filter + rebind (E)
function refillVisibleFixtures(){
  if (!fixtures.length) return;

  // date filter
  const start = UI.date ? startOfDay(UI.date) : startOfDay(new Date(DEMO_TODAY));
  const range = UI.rangeDays || 1;
  const byDate = f => inDayRange(f.date_utc, start, range);

  // league filter
  const byLeague = f => UI.league==='ALL' || (f.competition||'').toLowerCase() === UI.league.toLowerCase();

  filtered = fixtures.filter(byDate).filter(byLeague);

  // Bind globe
  globe
    .pointsData(filtered)
    .pointLat('latitude')
    .pointLng('longitude');

  // Build fixture rail list
  buildRail(filtered);

  // Tabs on-globe
  if (hasHtmlElementsApi(globe)) { renderHtmlTabsFrom(filtered); } else if (typeof globe.labelsData === 'function') { renderLabelsFrom(filtered); }

  // Select first
  if (filtered.length) selectIndex(0, { fly:true });

  // Competition banner
  if (filtered.length) renderCompetitionAccuracy(filtered[0].competition);
}

// lightweight wrappers for filtered set
function renderHtmlTabsFrom(list){
  htmlTabsData = list.map((f,i)=>({ lat:f.latitude, lng:f.longitude, altitude:SURFACE_EPS+0.06, idx:i, el:null }));
  globe.htmlElementsData(htmlTabsData).htmlLat('lat').htmlLng('lng').htmlAltitude('altitude').htmlElement(d=>{
    const w = document.createElement('div');
    w.className = 'fixture-tab' + (d.idx===activeIdx?' is-selected':'');
    w.dataset.idx = String(d.idx);
    const f = filtered[d.idx];
    w.title = `${f.home_team} vs ${f.away_team}${f.city?` — ${f.city}`:''}`;
    w.style.pointerEvents = 'auto';
    const title = document.createElement('div'); title.className='fixture-tab__title'; title.textContent = `${f.home_team} vs ${f.away_team}`;
    const meta  = document.createElement('div'); meta.className='fixture-tab__meta'; meta.textContent = f.city || f.country || '';
    w.append(title, meta);
    w.addEventListener('click', (e)=>{ e.stopPropagation(); selectIndex(d.idx, {fly:true}); });
    d.el = w;
    return w;
  });
  globe.htmlTransitionDuration?.(220);
}
function renderLabelsFrom(list){
  globe.labelsData(list).labelLat('latitude').labelLng('longitude')
    .labelAltitude(()=>SURFACE_EPS+0.06).labelText(f=>`${f.home_team} vs ${f.away_team}`)
    .labelColor(f=>((f.fixture_id||f.re_id)===selectedId ? 'rgba(125,249,196,0.95)':'rgba(255,255,255,0.85)'))
    .labelSize(f=>((f.fixture_id||f.re_id)===selectedId ? 1.4:1.0))
    .labelDotRadius(f=>((f.fixture_id||f.re_id)===selectedId?0.5:0.28))
    .labelResolution(2);
}

// ----------------------------
// CSV ingest & bind
// ----------------------------
async function loadFixturesCSV(url) {
  try {
    const response = await fetch(`${url}?v=${Date.now()}`);
    if (!response.ok) { console.error(`[CSV] HTTP ${response.status} for ${url}`); showToast('error', `Could not load ${url} (HTTP ${response.status}).`); return; }
    const text = await response.text();
    const { data, errors } = Papa.parse(text, { header: true, skipEmptyLines: true });
    if (errors?.length) console.warn('[CSV parse errors]', errors);

    fixtures = (data || []).map(row => {
      const lat = Number(row.latitude ?? row.lat ?? row.Latitude ?? row.lat_deg);
      const lon = Number(row.longitude ?? row.lon ?? row.lng ?? row.Longitude);
      const home_badge_url = pick(row, ['home_badge_url','home_logo_url','home_logo','home_badge']);
      const away_badge_url = pick(row, ['away_badge_url','away_logo_url','away_logo','away_badge']);
      return {
        fixture_id: (row.fixture_id || row.id || `${row.home_team}-${row.away_team}-${row.date_utc || ''}`).trim(),
        home_team: (row.home_team || row.Home || '').trim(),
        away_team: (row.away_team || row.Away || '').trim(),
        home_badge_url, away_badge_url,
        date_utc: row.date_utc || row.date || '',
        competition: row.competition || row.league || '',
        stadium: row.stadium || '',
        city: row.city || '',
        country: row.country || row.venue_country || '',
        latitude: Number.isFinite(lat) ? lat : undefined,
        longitude: Number.isFinite(lon) ? lon : undefined,
        predicted_winner: row.predicted_winner || '',
        confidence_ftr: +row.confidence_ftr || +row.confidence || 0,
        xg_home: +row.xg_home || 0, xg_away: +row.xg_away || 0,
        ppg_home: +row.ppg_home || 0, ppg_away: +row.ppg_away || 0,
        over25_prob: +row.over25_prob || 0, btts_prob: +row.btts_prob || 0,
        key_players_shots: (row.key_players_shots || '').trim(),
        key_players_tackles: (row.key_players_tackles || '').trim(),
        key_players_bookings: (row.key_players_bookings || '').trim(),
        __active: false
      };
    }).filter(f => Number.isFinite(f.latitude) && Number.isFinite(f.longitude));

    if (!fixtures.length) {
      showCsvError('No fixtures with valid latitude/longitude found.');
      document.querySelector('#match-intelligence')?.replaceChildren(elmEmpty('No fixtures available.'));
      document.querySelector('#player-watchlist')?.replaceChildren(elmEmpty('Player insights will appear here.'));
      document.querySelector('#market-snapshot')?.replaceChildren(elmEmpty('Market snapshot will appear here.'));
      return;
    }

    // Prefetch team crests
    const TEAM_MAP = new Map();
    for (const f of fixtures) {
      if (f.home_team) TEAM_MAP.set(f.home_team, f.home_badge_url || f.home_logo_url || '');
      if (f.away_team) TEAM_MAP.set(f.away_team, f.away_badge_url || f.away_logo_url || '');
    }
    prefetchAllLogos(TEAM_MAP).catch(()=>{});

    // Build left rail + league chips (Patch 2/3)
    buildLeagueChips();
    const days = extractFixtureDays(fixtures);
    buildCalendarRail(days, DEMO_TODAY);

    // Default demo filter
    UI.date = startOfDay(new Date(DEMO_TODAY));
    UI.rangeDays = 1;
    UI.league = 'ALL';

    // Fill everything from filter once
    refillVisibleFixtures();

  } catch (err) {
    console.error('[CSV] Failed to fetch/parse]:', err);
    showCsvError(`Failed to load CSV: ${err?.message || err}`);
    showToast('error', 'Failed to load fixtures');
  }
}

function showCsvError(msg) { el.fixtureTitle.textContent = 'Unable to load fixtures'; el.fixtureContext.textContent = msg; }

// ----------------------------
// Selection & nav
// ----------------------------
function step(delta) {
  if (!filtered.length) return;
  const next = (activeIdx + delta + filtered.length) % filtered.length;
  selectIndex(next, { fly: true });
}

function selectIndex(idx, opts = {}) {
  const { fly = false } = opts;
  if (!filtered.length) return;
  activeIdx = Math.max(0, Math.min(idx, filtered.length-1));
  const f = filtered[activeIdx];
  selectedId = f?.fixture_id || f?.re_id || null;

  filtered.forEach(d => (d.__active = (d.fixture_id || d.re_id) === selectedId));
  globe.pointAltitude(()=>SURFACE_EPS).pointRadius(d=>d.__active?RADIUS_ACTIVE:RADIUS_BASE).pointColor(d=>d.__active?COLORS.markerActive:COLORS.marker).pointsData(filtered);
  globe.pointColor(d => d.__active ? '#D7FFF9' : COLORS.marker).pointsTransitionDuration?.(200);
  setTimeout(()=>globe.pointsTransitionDuration?.(0), 220);

  if (fly) flyToFixture(f);
  moveMarkerToFixture(f, { fly });

  renderPanel(f);
  updateSelectionRing(f);
  syncRail();
  if (hasHtmlElementsApi(globe)) updateHtmlTabsSelection(); else if (typeof globe.labelsData === 'function') renderLabelsFrom(filtered);

  const wrap = document.querySelector('.hero__globe'); wrap?.classList.add('glow','glow-pin'); setTimeout(()=>wrap?.classList.remove('glow-pin'), 350);
}

function flyToFixture(f) { if (!f || !globe?.pointOfView) return; globe.pointOfView({ lat:f.latitude, lng:f.longitude, altitude: CAMERA_ALT }, 650); }

// ----------------------------
// Selection halo aligned to surface
// ----------------------------
function createSelectionRing() {
  const R = getGlobeRadius();
  const inner = R * (1 + SURFACE_EPS + 0.001);
  const outer = inner + R * 0.007;
  const ringGeom = new THREE.RingGeometry(inner, outer, 48);
  const ringMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(COLORS.ring), transparent:true, opacity:0.42, side:THREE.DoubleSide, depthWrite:false });
  pulseRing = new THREE.Mesh(ringGeom, ringMat);
  pulseRing.visible = false;
  scene.add(pulseRing);
  pulsePulse();
}

function updateSelectionRing(f) {
  if (!pulseRing || !f) return;
  const R = getGlobeRadius();
  const latRad = THREE.MathUtils.degToRad(90 - f.latitude);
  const lonRad = THREE.MathUtils.degToRad(180 - f.longitude);
  const r = R * (1 + SURFACE_EPS + 0.001);
  const x = r * Math.sin(latRad) * Math.cos(lonRad);
  const y = r * Math.cos(latRad);
  const z = r * Math.sin(latRad) * Math.sin(lonRad);
  pulseRing.position.set(x,y,z);
  const outward = new THREE.Vector3(x,y,z).normalize();
  pulseRing.lookAt(outward.clone().multiplyScalar(R*2));
  pulseRing.visible = true;
}

function pulsePulse() {
  if (!pulseRing) return;
  const T = 1800;
  const t = (performance.now() % T) / T;
  const intensity = 0.15 + 0.35 * Math.sin(t * Math.PI) ** 2;
  pulseRing.material.opacity = intensity;
  requestAnimationFrame(pulsePulse);
}

// ----------------------------
// Hover feedback (size pop)
// ----------------------------
let hoverId = null;
function handleHover(d) {
  hoverId = (d && (d.fixture_id || d.re_id)) || null;
  globe.pointRadius(pt => {
    if ((pt.fixture_id || pt.re_id) === selectedId) return RADIUS_ACTIVE;
    if (hoverId && (pt.fixture_id || pt.re_id) === hoverId) return RADIUS_BASE * 1.6;
    return RADIUS_BASE;
  });
}

// ----------------------------
// Move marker to fixture (beam + radar + billboard)
// ----------------------------
function moveMarkerToFixture(f, { fly=false } = {}){
  const S = window.__OG_MARKER__; if (!S || !f) return;
  const lat = Number(f.latitude), lon = Number(f.longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) { S.group.visible = false; return; }

  S.markerState.lat = lat; S.markerState.lon = lon; S.markerState.t0 = performance.now()*0.001; S.markerState.active = true;

  const R   = getGlobeRadius();
  const pos = latLngToVec3(lat, lon, SURFACE_EPS + 0.001);
  const n   = pos.clone().normalize();

  S.group.visible = true;
  S.group.position.copy(pos);
  S.group.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0), n);

  S.radarGroup.position.set(0,0,0);

  // collapse previous beam quickly is handled in selectIndex before we call here
  S.beam.position.set(0,0,0);
  S.beam.quaternion.identity();
  S.beam.scale.set(1, 0.001, 1);
  S.beam.visible = true;
  const growStart = performance.now();
  (function grow(){
    const t = Math.min(1, (performance.now()-growStart)/550);
    const e = t*t*(3-2*t);
    S.beam.scale.y = 0.001 + e;
    if (t < 1) requestAnimationFrame(grow);
  })();

  S.billboard.position.set(0, R*0.06, 0);
  S.billboard.material.opacity = 0;
  S.billboard.visible = true;

  (async ()=>{
    for (const url of stadiumCandidates(f)){
      try {
        const tex = await new Promise((res, rej)=> new THREE.TextureLoader().load(url, res, undefined, rej));
        S.billboard.material.map = tex;
        S.billboard.material.needsUpdate = true;
        const t0 = performance.now();
        (function fade(){ const k = Math.min(1, (performance.now()-t0)/220); S.billboard.material.opacity = k; if (k<1) requestAnimationFrame(fade); })();
        return;
      } catch {}
    }
    S.billboard.visible = false;
  })();
}

// ----------------------------
// Panel
// ----------------------------
function renderPanel(f) {
  if (!f) return;
  const fmt = iso => { try {
    const d = new Date(iso);
    const date = d.toLocaleDateString(undefined, { weekday: 'short', day: '2-digit', month: 'short' });
    const time = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    return `${date} · ${time} GMT`; } catch { return iso || ''; } };

  el.fixtureTitle.textContent = `${f.home_team} vs ${f.away_team}`;
  el.fixtureContext.textContent = [f.competition, fmt(f.date_utc), f.stadium && `${f.stadium} (${f.city || ''})`, f.country].filter(Boolean).join(' • ');
  renderCompetitionAccuracy(f.competition);

  const homeUrl = TEAM_LOGO_OVERRIDES[f.home_team] || '';
  const awayUrl = TEAM_LOGO_OVERRIDES[f.away_team] || '';
  setBadge(el.homeBadge, homeUrl, f.home_team);
  setBadge(el.awayBadge, awayUrl, f.away_team);

  clearNode(el.matchList);
  const mi = document.createElement('div');
  mi.innerHTML = `
    <div><strong title="Predicted winner">Full-time prediction:</strong> ${f.predicted_winner || '–'} ${f.confidence_ftr ? `(${pct(f.confidence_ftr)})` : ''}</div>
    <div><strong title="Expected Goals">xG edge:</strong> ${num(f.xg_home)} vs ${num(f.xg_away)}</div>
    <div><strong title="Points Per Game">Points momentum:</strong> ${num(f.ppg_home)} PPG • ${num(f.ppg_away)} PPG</div>`;
  el.matchList.appendChild(mi);

  clearNode(el.watchlist);
  const shots = parseKV(f.key_players_shots).slice(0, 6);
  if (shots.length) { shots.forEach(s=>{ const row=document.createElement('div'); row.className='row'; row.textContent=`${s.k} ${s.v}`; el.watchlist.appendChild(row); }); }
  else { el.watchlist.appendChild(elmEmpty('No player highlights available.')); }

  clearNode(el.market);
  el.market.innerHTML = `<div><strong>Over 2.5 goals:</strong> ${pct(f.over25_prob)}</div><div><strong>Both teams to score:</strong> ${pct(f.btts_prob)}</div>`;

  if (window.innerWidth < 720){
    const wrap = document.createElement('div');
    wrap.append(el.matchList.cloneNode(true), el.market.cloneNode(true), el.watchlist.cloneNode(true));
    openSheet(`${f.home_team} vs ${f.away_team}`, wrap);
  }
  el.deepBtn && (el.deepBtn.onclick = () => {
    alert(`Fixture: ${f.home_team} vs ${f.away_team}\nKick-off: ${fmt(f.date_utc)}\nPrediction: ${f.predicted_winner} (${pct(f.confidence_ftr)})\nOver 2.5: ${pct(f.over25_prob)} • BTTS: ${pct(f.btts_prob)}`);
  });
}

function num(x) { const n = Number(x); return Number.isFinite(n) ? n.toFixed(1) : '–'; }
function parseKV(s='') { return s.split(';').map(x=>x.trim()).filter(Boolean).map(pair=>{ const [k,v]=pair.split('|'); return {k:(k||'').trim(), v:(v||'').trim()}; }); }

// ----------------------------
// On-globe HTML fixture tabs (preferred)
// ----------------------------
function renderHtmlTabs(){ renderHtmlTabsFrom(filtered); }
function updateHtmlTabsSelection(){
  if (!htmlTabsData?.length) return;
  htmlTabsData.forEach(d => { const el = d.el || document.querySelector(`.fixture-tab[data-idx="${d.idx}"]`); if (el) el.classList.toggle('is-selected', d.idx === activeIdx); });
}

// ----------------------------
// Fallback: sprite labels if HTML overlay not available
// ----------------------------
function renderLabelSprites(){ renderLabelsFrom(filtered); }

// ----------------------------
// Fixture rail (quick selector)
// ----------------------------
function buildRail(items){
  const rail = document.getElementById('fixture-rail');
  if(!rail) return;
  rail.innerHTML = '';
  items.forEach((f, i)=>{
    const it = document.createElement('button');
    it.className = 'rail-item' + (i===0?' is-active':'');
    it.innerHTML = `<h4>${f.home_team} vs ${f.away_team}</h4><p>${(f.city||f.country||'')}</p>`;
    it.addEventListener('click',()=>selectIndex(i,{fly:true}));
    rail.appendChild(it);
  });
}
function syncRail(){
  const rail = document.getElementById('fixture-rail'); if(!rail) return;
  [...rail.children].forEach((c,idx)=>c.classList.toggle('is-active', idx===activeIdx));
}

// ----------------------------
// Tabs
// ----------------------------
function wireTabs(){
  document.querySelectorAll('.tab')?.forEach(btn=>{
    btn.addEventListener('click',()=>{
      document.querySelectorAll('.tab').forEach(b=>b.classList.remove('is-active'));
      document.querySelectorAll('.tabpane').forEach(p=>p.classList.remove('is-active'));
      btn.classList.add('is-active');
      const id = btn.dataset.tab;
      document.getElementById(`tab-${id}`)?.classList.add('is-active');
    });
  });
}

// ----------------------------
// Bottom sheet (mobile)
// ----------------------------
function openSheet(title, node){
  const s = document.getElementById('sheet'); if(!s) return;
  const titleEl = s.querySelector('#sheet-title'); if(titleEl) titleEl.textContent = title || '';
  const body = s.querySelector('#sheet-body'); if(body){ body.innerHTML=''; body.appendChild(node); }
  s.classList.add('open'); s.setAttribute('aria-hidden','false');
}
function closeSheet(){
  const s = document.getElementById('sheet'); if(!s) return;
  s.classList.remove('open'); s.setAttribute('aria-hidden','true');
}
document.querySelector('.sheet__handle')?.addEventListener('click', closeSheet);

// ----------------------------
// Router
// ----------------------------
const ROUTES = {
  '#/': 'view-home', '#/home':'view-home', '#/bet-checker':'view-betchecker',
  '#/acca-builder':'view-accabuilder', '#/copilot':'view-copilot', '#/login':'view-signin', '#/signup':'view-signup'
};
function showRoute(hash) {
  if (!hash) hash = '#/';
  const id = ROUTES[hash] || 'view-home';
  document.querySelectorAll('.view').forEach(v => { if (v.id === id) { v.classList.add('is-active'); v.removeAttribute('hidden'); } else { v.classList.remove('is-active'); v.setAttribute('hidden',''); } });
  document.querySelectorAll('[data-route]').forEach(a=>{ a.classList.toggle('is-active', a.getAttribute('href') === hash); if (a.classList.contains('side-link')) a.classList.toggle('active', a.getAttribute('href') === hash); });
  const profileMenu = document.getElementById('profile-menu'); const profileBtn  = document.getElementById('btn-profile');
  profileMenu?.classList.remove('show'); profileBtn?.setAttribute('aria-expanded','false');
  const drawer  = document.getElementById('side-drawer'); const scrim   = document.getElementById('scrim');
  drawer?.classList.remove('show'); scrim?.classList.remove('show'); drawer?.setAttribute('aria-hidden','true');
}
window.addEventListener('hashchange', ()=> showRoute(location.hash));
window.addEventListener('DOMContentLoaded', ()=>{ if (!location.hash) location.hash = '#/'; showRoute(location.hash); });

// quick self-test for two logo files
(function verifyLocalLogoSetup(){
  const tests = [`${LOGO_LOCAL_BASE}/arsenal.svg`, `${LOGO_LOCAL_BASE}/fc-barcelona.svg`];
  tests.forEach(src => { const img = new Image(); img.onload = ()=>console.log('%c[LOGOS] OK','color:#22c55e',src); img.onerror=()=>console.warn('%c[LOGOS] 404','color:#f43f5e',src); img.src=src; });
})();

init();
