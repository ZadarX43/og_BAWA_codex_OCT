// app.module.js
// Odds Genius — Globe Fixtures UI (ESM + local vendor imports)
// ------------------------------------------------------------
// - Correct named imports from three/examples/jsm/* (via import map)
// - Ground-level geo markers; selected marker halo
// - Left/right keyboard nav; click-to-select
// - Robust three-globe loader (local → CDN fallback)
// - Club logo badges (with initials fallback)
// - CSV → globe linkage + panel renderer
// - FXAA/Bloom postprocessing sized to container
// ------------------------------------------------------------

import * as THREE from 'three';
import { OrbitControls }   from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer }  from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass }      from 'three/examples/jsm/postprocessing/RenderPass.js';
import { ShaderPass }      from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { FXAAShader }      from 'three/examples/jsm/shaders/FXAAShader.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

// PapaParse (UMD) is loaded in index.html
const Papa = window.Papa;

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
  prevBtn: document.getElementById('nav-prev'), // optional in HTML
  nextBtn: document.getElementById('nav-next')  // optional in HTML
};

// ----------------------------
// three-globe loader (ESM → UMD)
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
  // 1) Local ESM first (place the file in /vendor/)
  const localEsm = [
    './vendor/three-globe.module.js',
    './vendor/three-globe.mjs'
  ];
  for (const p of localEsm) {
    try {
      const m = await import(p);
      console.info('[three-globe] using local ESM:', p);
      return m.default ?? m;
    } catch {}
  }
  // 2) esm.sh ESM (with three externalized)
  for (const v of ['2.30.1', '2.29.3', '2.28.0']) {
    const url = `https://esm.sh/three-globe@${v}?bundle&external=three`;
    try {
      const m = await import(url);
      console.warn('[three-globe] using esm.sh ESM:', url);
      return m.default ?? m;
    } catch (e) {
      console.warn('[three-globe] esm.sh failed:', url, e);
    }
  }
  // 3) UMD fallback
  const umd = [
    './vendor/three-globe.min.js',
    'https://cdn.jsdelivr.net/npm/three-globe@2.29.3/dist/three-globe.min.js',
    'https://unpkg.com/three-globe@2.29.3/dist/three-globe.min.js'
  ];
  for (const url of umd) {
    try {
      const ctor = await importScriptUMD(url);
      console.warn('[three-globe] using UMD:', url);
      return ctor;
    } catch (e) {
      console.warn('[three-globe] UMD failed:', url, e);
    }
  }
  const gc = document.getElementById('globe-container');
  if (gc) {
    gc.innerHTML = `<div class="globe-error">
      <strong>three-globe failed to load.</strong><br/>
      Add a local copy at <code>vendor/three-globe.module.js</code> (ESM)
      or <code>vendor/three-globe.min.js</code> (UMD).
    </div>`;
  }
  throw new Error('three-globe could not be loaded from any source');
}

// ----------------------------
// Globals / tuning
// ----------------------------
let ThreeGlobeCtor;
let globe;
let renderer, scene, camera, controls, composer, bloomPass;
let fixtures = [];
let activeIdx = 0;
let selectedId = null;
let pulseRing;

const COLORS = {
  marker: '#9DEDE6',
  markerInactive: '#79E3D7',
  markerActive: '#BAFFFB',
  ring: '#9EE7E3'
};

const SURFACE_EPS = 0.006;   // marker altitude = SURFACE_EPS × globeRadius
const RADIUS_BASE = 0.010;   // marker radius (× globeRadius)
const RADIUS_ACTIVE = 0.022; // active marker radius
const CAMERA_ALT = 2.0;

const BLOOM = { strength: 0.6, radius: 0.5, threshold: 0.85 };

// ----------------------------
// Utilities
// ----------------------------
const clamp01 = v => Math.max(0, Math.min(1, v));
const pct = n => `${Math.round(clamp01(n) * 100)}%`;

function clearNode(node) {
  if (!node) return;
  while (node.firstChild) node.removeChild(node.firstChild);
}

function setBadge(elm, url, initials) {
  if (!elm) return;
  elm.innerHTML = '';
  elm.classList.remove('has-logo');
  if (url && /^https?:\/\//i.test(url)) {
    const img = document.createElement('img');
    img.src = url;
    img.alt = initials || '';
    img.loading = 'lazy';
    img.decoding = 'async';
    img.onerror = () => {
      elm.textContent = initials || '';
      elm.classList.remove('has-logo');
    };
    elm.appendChild(img);
    elm.classList.add('has-logo');
  } else {
    elm.textContent = initials || '';
  }
}
function initials(name = '') {
  const words = name.split(/\s+/).filter(Boolean);
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] || '').toUpperCase() + (words[1]?.[0]?.toUpperCase() || '');
}

// Helper: robust globe radius (some builds don’t expose getGlobeRadius)
function getGlobeRadius() {
  if (globe?.getGlobeRadius) return globe.getGlobeRadius();
  try {
    const m = globe.children?.find(c => c.geometry?.parameters?.radius);
    return m?.geometry?.parameters?.radius || 100;
  } catch { return 100; }
}

// ----------------------------
// Scene init
// ----------------------------
async function init() {
  ThreeGlobeCtor = await loadThreeGlobe();

  scene = new THREE.Scene();

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(el.globeWrap.clientWidth, el.globeWrap.clientHeight);
  el.globeWrap.innerHTML = '';
  el.globeWrap.appendChild(renderer.domElement);

  camera = new THREE.PerspectiveCamera(
    45,
    el.globeWrap.clientWidth / el.globeWrap.clientHeight,
    0.1,
    5000
  );
  camera.position.set(0, 0, 300);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.enableZoom = true;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.6;
  controls.minDistance = 140;
  controls.maxDistance = 1200;

  scene.add(new THREE.AmbientLight(0xffffff, 0.9));

  // --- Postprocessing (named imports) ---
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));

  const fxaaPass = new ShaderPass(FXAAShader);
  const setFXAA = () => {
    const px = renderer.getPixelRatio();
    fxaaPass.material.uniforms['resolution'].value.set(
      1 / (el.globeWrap.clientWidth  * px),
      1 / (el.globeWrap.clientHeight * px)
    );
  };
  setFXAA();
  composer.addPass(fxaaPass);

  bloomPass = new UnrealBloomPass(
    new THREE.Vector2(el.globeWrap.clientWidth, el.globeWrap.clientHeight),
    BLOOM.strength,
    BLOOM.radius,
    BLOOM.threshold
  );
  composer.addPass(bloomPass);

  // Globe
  globe = new ThreeGlobeCtor({ waitForGlobeReady: true })
    .showAtmosphere(true)
    .atmosphereColor('#94f1ea')
    .atmosphereAltitude(0.2)
    .globeImageUrl('https://unpkg.com/three-globe/example/img/earth-dark.jpg')
    .bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png')
    .pointAltitude(() => SURFACE_EPS)
    .pointRadius(d => (d.__active ? RADIUS_ACTIVE : RADIUS_BASE))
    .pointColor(d => (d.__active ? COLORS.markerActive : COLORS.marker));

  scene.add(globe);

  // Hover (guarded)
  if (typeof globe.onPointHover === 'function') {
    globe.onPointHover(handleHover);
  } else {
    console.info('[three-globe] onPointHover() not available in this build — hover enhancement disabled.');
  }

  // Click select (guarded)
  if (typeof globe.onPointClick === 'function') {
    globe.onPointClick(pt => {
      if (!pt) return;
      const idx = fixtures.findIndex(f => f.fixture_id === pt.fixture_id);
      if (idx >= 0) selectIndex(idx, { fly: true });
    });
  }

  window.addEventListener('resize', () => {
    const { clientWidth, clientHeight } = el.globeWrap;
    renderer.setSize(clientWidth, clientHeight);
    camera.aspect = clientWidth / clientHeight;
    camera.updateProjectionMatrix();
    const px = renderer.getPixelRatio();
    fxaaPass.material.uniforms.resolution.value.set(
      1 / (clientWidth * px),
      1 / (clientHeight * px)
    );
    bloomPass.setSize?.(clientWidth, clientHeight);
  });

  // keyboard nav
  window.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight') { e.preventDefault(); step(+1); }
    if (e.key === 'ArrowLeft')  { e.preventDefault(); step(-1); }
  });

  if (el.prevBtn) el.prevBtn.addEventListener('click', () => step(-1));
  if (el.nextBtn) el.nextBtn.addEventListener('click', () => step(+1));

  // Load fixtures
  await loadFixturesCSV('./data/fixtures.csv');

  animate();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  composer.render();
}

// ----------------------------
// CSV ingest & bind
// ----------------------------
async function loadFixturesCSV(url) {
  try {
    const response = await fetch(`${url}?v=${Date.now()}`);
    if (!response.ok) {
      console.error(`[CSV] HTTP ${response.status} for ${url}`);
      showCsvError(`Could not load ${url} (HTTP ${response.status}). Make sure the file exists on GitHub Pages and path is correct.`);
      return;
    }
    const text = await response.text();
    const { data, errors } = Papa.parse(text, { header: true, skipEmptyLines: true });
    if (errors?.length) console.warn('[CSV parse errors]', errors);

    fixtures = (data || [])
      .map(row => {
        const lat = parseFloat(row.latitude || row.lat || row.Latitude || row.lat_deg);
        const lng = parseFloat(row.longitude || row.lon || row.lng || row.Longitude);
        return {
          fixture_id: (row.fixture_id || row.id || `${row.home_team}-${row.away_team}-${row.date_utc || ''}`).trim(),
          home_team: (row.home_team || row.Home || '').trim(),
          away_team: (row.away_team || row.Away || '').trim(),
          home_logo_url: row.home_logo_url || row.home_logo || '',
          away_logo_url: row.away_logo_url || row.away_logo || '',
          date_utc: row.date_utc || row.date || '',
          competition: row.competition || row.league || '',
          stadium: row.stadium || '',
          city: row.city || '',
          country: row.country || row.venue_country || '',
          latitude: Number.isFinite(lat) ? lat : undefined,
          longitude: Number.isFinite(lng) ? lng : undefined,
          predicted_winner: row.predicted_winner || '',
          confidence_ftr: +row.confidence_ftr || +row.confidence || 0,
          xg_home: +row.xg_home || 0,
          xg_away: +row.xg_away || 0,
          ppg_home: +row.ppg_home || 0,
          ppg_away: +row.ppg_away || 0,
          over25_prob: +row.over25_prob || 0,
          btts_prob: +row.btts_prob || 0,
          key_players_shots: (row.key_players_shots || '').trim(),
          key_players_tackles: (row.key_players_tackles || '').trim(),
          key_players_bookings: (row.key_players_bookings || '').trim(),
          __active: false
        };
      })
      .filter(f => Number.isFinite(f.latitude) && Number.isFinite(f.longitude));

    console.log(`[CSV] Loaded ${fixtures.length} fixtures`);
    if (!fixtures.length) {
      showCsvError('No fixtures with valid latitude/longitude found. Check your CSV columns are named "latitude" and "longitude" (or compatible aliases).');
      return;
    }

    const many = fixtures.length > 250;
    globe.pointsMerge(many).pointResolution(many ? 4 : 8);
    globe
      .pointLat('latitude')
      .pointLng('longitude')
      .pointsData(fixtures);

    // Initial selection
    selectIndex(0, { fly: true });

    // Build selection ring after first selection
    createSelectionRing();
  } catch (err) {
    console.error('[CSV] Failed to fetch/parse:', err);
    showCsvError(`Failed to load CSV: ${err?.message || err}`);
  }
}

function showCsvError(msg) {
  el.fixtureTitle.textContent = 'Unable to load fixtures';
  el.fixtureContext.textContent = msg;
}

// ----------------------------
// Selection & nav
// ----------------------------
function step(delta) {
  if (!fixtures.length) return;
  const next = (activeIdx + delta + fixtures.length) % fixtures.length;
  selectIndex(next, { fly: true });
}

function selectIndex(idx, opts = {}) {
  const { fly = false } = opts;
  activeIdx = idx;
  const f = fixtures[activeIdx];
  selectedId = f?.fixture_id || null;

  fixtures.forEach(d => (d.__active = d.fixture_id === selectedId));
  globe
    .pointAltitude(() => SURFACE_EPS)
    .pointRadius(d => (d.__active ? RADIUS_ACTIVE : RADIUS_BASE))
    .pointColor(d => (d.__active ? COLORS.markerActive : COLORS.marker))
    .pointsData(fixtures);

  if (fly) flyToFixture(f);
  renderPanel(f);
  updateSelectionRing(f);
}

function flyToFixture(f) {
  if (!f || !globe?.pointOfView) return;
  globe.pointOfView(
    { lat: f.latitude, lng: f.longitude, altitude: CAMERA_ALT },
    800
  );
}

// ----------------------------
// Selection halo aligned to surface
// ----------------------------
function createSelectionRing() {
  const R = getGlobeRadius();
  const inner = R * (1 + SURFACE_EPS + 0.001);
  const outer = inner + R * 0.007;
  const ringGeom = new THREE.RingGeometry(inner, outer, 48);
  const ringMat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(COLORS.ring),
    transparent: true,
    opacity: 0.3,
    side: THREE.DoubleSide,
    depthWrite: false
  });
  pulseRing = new THREE.Mesh(ringGeom, ringMat);
  pulseRing.visible = false;
  scene.add(pulseRing);
  pulsePulse();
}

function updateSelectionRing(f) {
  if (!pulseRing || !f) return;
  const R = getGlobeRadius();

  // Convert lat/lng to Cartesian at the ring altitude
  const latRad = THREE.MathUtils.degToRad(90 - f.latitude);
  const lonRad = THREE.MathUtils.degToRad(180 - f.longitude);
  const r = R * (1 + SURFACE_EPS + 0.001);

  const x = r * Math.sin(latRad) * Math.cos(lonRad);
  const y = r * Math.cos(latRad);
  const z = r * Math.sin(latRad) * Math.sin(lonRad);

  pulseRing.position.set(x, y, z);
  // Align ring plane tangent to sphere (normal = radial outward)
  const outward = new THREE.Vector3(x, y, z).normalize();
  const look = outward.clone().multiplyScalar(R * 2);
  pulseRing.lookAt(look);
  pulseRing.visible = true;
}

function pulsePulse() {
  if (!pulseRing) return;
  const T = 1800; // ms
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
  hoverId = d?.fixture_id || null;
  globe.pointRadius(pt => {
    if (pt.fixture_id === selectedId) return RADIUS_ACTIVE;
    if (hoverId && pt.fixture_id === hoverId) return RADIUS_BASE * 1.6;
    return RADIUS_BASE;
  });
}

// ----------------------------
// Panel
// ----------------------------
function renderPanel(f) {
  if (!f) return;

  const fmt = iso => {
    try {
      const d = new Date(iso);
      const date = d.toLocaleDateString(undefined, { weekday: 'short', day: '2-digit', month: 'short' });
      const time = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
      return `${date} · ${time} GMT`;
    } catch { return iso || ''; }
  };

  el.fixtureTitle.textContent = `${f.home_team} vs ${f.away_team}`;
  el.fixtureContext.textContent = [f.competition, fmt(f.date_utc), f.stadium && `${f.stadium} (${f.city || ''})`, f.country]
    .filter(Boolean).join(' • ');

  setBadge(el.homeBadge, f.home_logo_url, initials(f.home_team));
  setBadge(el.awayBadge, f.away_logo_url, initials(f.away_team));

  // Match Intelligence
  clearNode(el.matchList);
  const mi = document.createElement('div');
  mi.innerHTML = `
    <div><strong>Full-time prediction:</strong> ${f.predicted_winner || '–'} ${f.confidence_ftr ? `(${pct(f.confidence_ftr)})` : ''}</div>
    <div><strong>xG edge:</strong> ${num(f.xg_home)} vs ${num(f.xg_away)}</div>
    <div><strong>Points momentum:</strong> ${num(f.ppg_home)} PPG • ${num(f.ppg_away)} PPG</div>
  `;
  el.matchList.appendChild(mi);

  // Player Watchlist (shots)
  clearNode(el.watchlist);
  const shots = parseKV(f.key_players_shots).slice(0, 6);
  if (shots.length) {
    shots.forEach(s => {
      const row = document.createElement('div');
      row.className = 'row';
      row.textContent = `${s.k} ${s.v}`;
      el.watchlist.appendChild(row);
    });
  } else {
    const empty = document.createElement('div');
    empty.className = 'row';
    empty.textContent = 'No player highlights available.';
    el.watchlist.appendChild(empty);
  }

  // Market
  clearNode(el.market);
  el.market.innerHTML = `
    <div><strong>Over 2.5 goals:</strong> ${pct(f.over25_prob)}</div>
    <div><strong>Both teams to score:</strong> ${pct(f.btts_prob)}</div>
  `;

  // Deep-dive
  if (el.deepBtn) {
    el.deepBtn.onclick = () => {
      alert(
        `Fixture: ${f.home_team} vs ${f.away_team}\n` +
        `Kick-off: ${fmt(f.date_utc)}\n` +
        `Prediction: ${f.predicted_winner} (${pct(f.confidence_ftr)})\n` +
        `Over 2.5: ${pct(f.over25_prob)} • BTTS: ${pct(f.btts_prob)}`
      );
    };
  }
}

function num(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n.toFixed(1) : '–';
}

function parseKV(s='') {
  // "Name|Value;Name2|Value2"
  return s.split(';')
    .map(x => x.trim())
    .filter(Boolean)
    .map(pair => {
      const [k, v] = pair.split('|');
      return { k: (k||'').trim(), v: (v||'').trim() };
    });
}

// ----------------------------
// Upload (placeholder)
// ----------------------------
el.upload?.addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  alert(`Bet slip uploaded: ${file.name}\n\nNext steps:\n• OCR the slip\n• Run BetChecker\n• Generate OG Co-Pilot insights`);
  el.upload.value = '';
});

// ----------------------------
// Start
// ----------------------------
init();
