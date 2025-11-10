// app.module.js
// Odds Genius — Globe Fixtures UI (enhanced)
// ------------------------------------------------------------
// - Accurate ground-level markers using globe radius
// - Selected-marker ring highlight + larger sphere
// - Left/right keyboard nav & click to select
// - Robust three-globe loader (ESM → CDN fallback)
// - Club logos in header with fallback initials
// - Performance optimizations for hundreds of fixtures
// - Cleaner panel updates and stat formatting
// ------------------------------------------------------------

import * as THREE from 'three';

// --- Postprocessing / controls (local vendor copies) ---
import 'https://esm.sh/three@0.160.0/examples/js/controls/OrbitControls.js';
import 'https://esm.sh/three@0.160.0/examples/js/postprocessing/EffectComposer.js';
import 'https://esm.sh/three@0.160.0/examples/js/postprocessing/RenderPass.js';
import 'https://esm.sh/three@0.160.0/examples/js/postprocessing/ShaderPass.js';
import 'https://esm.sh/three@0.160.0/examples/js/shaders/FXAAShader.js';
import 'https://esm.sh/three@0.160.0/examples/js/postprocessing/UnrealBloomPass.js';

import Papa from 'https://esm.sh/papaparse@5.4.1';

// ----------------------------
// DOM refs (must exist in HTML)
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
  // Optional: left/right UI buttons (if present in your HTML)
  prevBtn: document.getElementById('nav-prev'),
  nextBtn: document.getElementById('nav-next')
};

// ------------------------------------
// Utility: robust three-globe loader
// ------------------------------------
async function loadThreeGlobe() {
  // 1) Try local ESM if present
  try {
    const mod = await import('./vendor/three-globe.module.js');
    return mod.default || mod;
  } catch (e) {
    console.warn('[three-globe] local ESM not found, falling back to CDN…');
  }
  // 2) Try esm.sh (locks version for stability)
  try {
    const mod = await import('https://esm.sh/three-globe@2.28.0');
    return mod.default || mod;
  } catch (e) {
    console.warn('[three-globe] esm.sh failed, trying jsDelivr UMD…', e);
  }
  // 3) Fallback UMD
  const url = './vendor/three-globe.min.js';
  await new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = url;
    s.onload = () => res();
    s.onerror = rej;
    document.body.appendChild(s);
  });
  // UMD exposes global
  // eslint-disable-next-line no-undef
  return window.ThreeGlobe || window.THREE.Globe;
}

let ThreeGlobeCtor;
let globe;
let renderer, scene, camera, controls, composer, bloomPass;
let fixtures = [];
let activeIdx = 0;
let selectedId = null;
let globeReady = false;
let pulseRing;

// tuneable style constants
const COLORS = {
  marker: '#A1F2EA',
  markerInactive: '#79E3D7',
  markerActive: '#A7FFF6',
  ring: '#9EE7E3',
  selected: '#FFFFFF'
};
const SURFACE_EPS = 0.006;      // fraction of globe radius for marker altitude
const RING_INNER = 0.010;       // relative to globe radius
const RING_OUTER = 0.015;
const RADIUS_BASE = 0.010;      // base sphere radius (× globe radius)
const RADIUS_ACTIVE = 0.022;    // active sphere radius (× globe radius)
const CAMERA_ALT = 2.1;         // camera altitude in radius units
const MAX_LABELS = 80;          // optional future label LOD
const BLOOM = { strength: 0.6, radius: 0.5, threshold: 0.85 };

// helpers
const clamp01 = v => Math.max(0, Math.min(1, v));
const pct = n => `${Math.round(clamp01(n) * 100)}%`;

function clearNode(node) {
  if (!node) return;
  while (node.firstChild) node.removeChild(node.firstChild);
}

function setBadge(el, url, initials) {
  if (!el) return;
  el.innerHTML = '';
  el.classList.remove('has-logo');
  if (url && /^https?:\/\//i.test(url)) {
    const img = document.createElement('img');
    img.src = url;
    img.alt = initials || '';
    img.loading = 'lazy';
    img.decoding = 'async';
    img.onerror = () => {
      el.textContent = initials || '';
      el.classList.remove('has-logo');
    };
    el.appendChild(img);
    el.classList.add('has-logo');
  } else {
    el.textContent = initials || '';
  }
}

function formatTeamInitials(name = '') {
  const words = (name || '').split(/\s+/).filter(Boolean);
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] || '').toUpperCase() + (words[1]?.[0]?.toUpperCase() || '');
}

// --- Build scene ---
async function init() {
  ThreeGlobeCtor = await loadThreeGlobe();

  // Scene
  scene = new THREE.Scene();

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(el.globeWrap.clientWidth, el.globeWrap.clientHeight);
  el.globeWrap.innerHTML = '';
  el.globeWrap.appendChild(renderer.domElement);

  // Camera
  const aspect = el.globeWrap.clientWidth / el.globeWrap.clientHeight;
  camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 10000);
  camera.position.z = 300;

  // Controls
  // @ts-ignore (added globally by import)
  const { OrbitControls } = THREE;
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.enableZoom = true;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.6;
  controls.minDistance = 140;
  controls.maxDistance = 1200;

  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.9));

  // Postprocessing
  // @ts-ignore
  const { EffectComposer } = THREE;
  // @ts-ignore
  const { RenderPass } = THREE;
  // @ts-ignore
  const { ShaderPass } = THREE;
  // @ts-ignore
  const { UnrealBloomPass } = THREE;

  const renderPass = new RenderPass(scene, camera);
  composer = new THREE.EffectComposer(renderer);
  composer.addPass(renderPass);

  const fxaaPass = new THREE.ShaderPass(THREE.FXAAShader);
  fxaaPass.material.uniforms['resolution'].value.set(
    1 / (el.globeWrap.clientWidth * renderer.getPixelRatio()),
    1 / (el.globeWrap.clientHeight * renderer.getPixelRatio())
  );
  composer.addPass(fxaaPass);

  bloomPass = new UnrealBloomPass(
    new THREE.Vector2(el.globeWrap.clientWidth, el.globeWrap.clientHeight),
    BLOOM.strength,
    BLOOM.radius,
    BLOOM.threshold
  );
  composer.addPass(bloomPass);

  // Globe
  globe = new ThreeGlobeCtor()
    .showAtmosphere(true)
    .atmosphereColor('#94f1ea')
    .atmosphereAltitude(0.2)
    .globeImageUrl('https://unpkg.com/three-globe/example/img/earth-dark.jpg')
    .bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png')
    .pointAltitude(() => SURFACE_EPS) // ground-level pins
    .pointRadius(d => (d.__active ? RADIUS_ACTIVE : RADIUS_BASE))
    .pointColor(d => (d.__active ? COLORS.markerActive : COLORS.marker));

  scene.add(globe);

  // Track hover to subtly brighten
  globe.onPointHover(handleHover);

  // Selection (click)
  globe.onPointClick(d => {
    if (!d) return;
    const idx = fixtures.findIndex(f => f.fixture_id === d.fixture_id);
    if (idx >= 0) selectIndex(idx, { fly: true });
  });

  // Window resize
  window.addEventListener('resize', onResize);

  // Keyboard navigation
  window.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight') {
      e.preventDefault();
      step(+1);
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      step(-1);
    }
  });

  // Optional nav buttons
  if (el.prevBtn) el.prevBtn.addEventListener('click', () => step(-1));
  if (el.nextBtn) el.nextBtn.addEventListener('click', () => step(+1));

  // CSV load → render
  await loadFixturesCSV('./data/fixtures.csv');

  // Start loop
  animate();
}

function onResize() {
  const { clientWidth, clientHeight } = el.globeWrap;
  renderer.setSize(clientWidth, clientHeight);
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
  // update fxaa + bloom
  const res = new THREE.Vector2(
    1 / (clientWidth * renderer.getPixelRatio()),
    1 / (clientHeight * renderer.getPixelRatio())
  );
  composer.passes.forEach(p => {
    if (p instanceof THREE.ShaderPass && p.material?.uniforms?.resolution) {
      p.material.uniforms.resolution.value.copy(res);
    }
  });
  if (bloomPass?.setSize) {
    bloomPass.setSize(clientWidth, clientHeight);
  }
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  composer.render();
}

// ---------------------------
// CSV ingest & normalization
// ---------------------------
async function loadFixturesCSV(url) {
  const res = await fetch(`${url}?v=${Date.now()}`);
  const text = await res.text();
  const { data, errors } = Papa.parse(text, { header: true, skipEmptyLines: true });
  if (errors?.length) {
    console.warn('[CSV parse errors]', errors);
  }

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
        // Metrics (defaults):
        predicted_winner: row.predicted_winner || '',
        confidence_ftr: parseFloat(row.confidence_ftr || row.confidence || 0) || 0,
        xg_home: parseFloat(row.xg_home || 0) || 0,
        xg_away: parseFloat(row.xg_away || 0) || 0,
        ppg_home: parseFloat(row.ppg_home || 0) || 0,
        ppg_away: parseFloat(row.ppg_away || 0) || 0,
        over25_prob: parseFloat(row.over25_prob || 0) || 0,
        btts_prob: parseFloat(row.btts_prob || 0) || 0,
        key_players_shots: (row.key_players_shots || '').trim(),
        key_players_tackles: (row.key_players_tackles || '').trim(),
        key_players_bookings: (row.key_players_bookings || '').trim()
      };
    })
    .filter(f => Number.isFinite(f.latitude) && Number.isFinite(f.longitude));

  if (!fixtures.length) {
    el.fixtureTitle.textContent = 'No fixtures available';
    el.fixtureContext.textContent = 'Please check your data file.';
    return;
  }

  // Seed active flag
  fixtures.forEach(f => (f.__active = false));

  // Performance: for very large sets, reduce overhead
  const many = fixtures.length > 250;
  globe
    .pointsMerge(many)
    .pointResolution(many ? 4 : 8);

  // Bind to globe
  globe
    .pointLat('latitude')
    .pointLng('longitude')
    .pointsData(fixtures);

  // Ready to interact
  globe.addEventListener('ready', () => {
    globeReady = true;
    // Ensure we have at least one selection
    selectIndex(0, { fly: true, skipUpdate: false });

    // Create pulse ring (tangent to surface)
    createSelectionRing();
  });

  // If globe already in DOM, kick readiness after a tick
  setTimeout(() => {
    if (!globeReady) {
      // Not strictly required; safety check
    }
  }, 300);
}

// ---------------------------
// Selection & Navigation
// ---------------------------
function step(delta) {
  if (!fixtures.length) return;
  const next = (activeIdx + delta + fixtures.length) % fixtures.length;
  selectIndex(next, { fly: true });
}

function selectIndex(idx, opts = {}) {
  const { fly = false, skipUpdate = false } = opts;
  activeIdx = idx;
  const f = fixtures[activeIdx];
  selectedId = f?.fixture_id || null;

  fixtures.forEach(d => (d.__active = d.fixture_id === selectedId));
  // Update markers without full rebuild (re-apply accessors)
  globe
    .pointAltitude(() => SURFACE_EPS)
    .pointRadius(d => (d.__active ? RADIUS_ACTIVE : RADIUS_BASE))
    .pointColor(d => (d.__active ? COLORS.markerActive : COLORS.marker))
    .pointsData(fixtures); // trigger refresh

  if (fly) flyToFixture(f);
  if (!skipUpdate) renderPanel(f);
  updateSelectionRing(f);
}

function flyToFixture(f) {
  if (!f || !globe) return;
  try {
    globe.pointOfView(
      { lat: f.latitude, lng: f.longitude, altitude: CAMERA_ALT },
      800
    );
  } catch (e) {
    // Fallback: orbit controls
    // no-op
  }
}

// ---------------------------
// Selection ring (tangent halo)
// ---------------------------
function createSelectionRing() {
  const R = globe.getGlobeRadius ? globe.getGlobeRadius() : 100;
  const ringGeom = new THREE.RingGeometry(R * (1 + SURFACE_EPSlocal()), R * (1 + SURFACE_EPSlocal() + 0.006), 48);
  const ringMat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(COLORS.ring),
    transparent: true,
    opacity: 0.55,
    side: THREE.DoubleSide,
    depthWrite: false
  });
  pulseRing = new THREE.Mesh(ringGeom, ringMat);
  pulseRing.visible = false;
  globe.add(pulseRing);
  animateRing();
}

function SURFACE_EPSlocal() {
  // Slightly larger than SURFACE_EPS to sit just above point altitude
  return SURFACE_EPS * 1.2;
}

function updateSelectionRing(f) {
  if (!pulseRing || !f) return;
  const R = globe.getGlobeRadius ? globe.getGlobeRadius() : 100;
  const lat = THREE.MathUtils.degToRad(90 - f.latitude);
  const lon = THREE.MathUtils.degToRad(180 - f.longitude);

  const r = R * (1 + SURFACE_EPSlocal());
  const x = r * Math.sin(lat) * Math.cos(lon);
  const y = r * Math.cos(lat);
  const z = r * Math.sin(lat) * Math.sin(lon);

  // Update geometry to match world scale
  pulseRing.geometry.dispose();
  pulseRing.geometry = new THREE.RingGeometry(R * (1 + SURFACE_EPSlocal() - 0.0005), R * (1 + SURFACE_EPSlocal() + 0.006), 48);
  pulseRing.position.set(x, y, z);
  // Orient tangent to surface (normal aligned with radial vector)
  const outward = new THREE.Vector3(x, y, z).normalize();
  const target = outward.clone().multiplyScalar(R * 2.0); // look outward so +Z aligns radial
  pulseRing.lookAt(target);
  pulseRing.visible = true;
}

function animateRing() {
  if (!pulseRing) return;
  const T = 1800; // ms per pulse
  const t = (performance.now() % T) / T;
  const base = 0.35 + 0.25 * Math.sin(t * Math.PI * 2);
  pulseRing.material.opacity = 0.15 + 0.35 * Math.pow(Math.sin(t * Math.PI), 2);
  const R = globe.getGlobeRadius ? globe.getGlobeRadius() : 100;
  const inner = R * (1 + SURFACE_EPSlocal() + base * 0.001);
  const outer = inner + R * 0.007;
  pulseRing.geometry.dispose();
  pulseRing.geometry = new THREE.RingGeometry(inner, outer, 48);
  requestAnimationFrame(animateRing);
}

// ---------------------------
// Hover feedback
// ---------------------------
let hoverId = null;
function handleHover(d) {
  hoverId = d?.fixture_id || null;
  // make hovered (non-selected) pop a bit
  globe.pointRadius(pt => {
    if (pt.fixture_id === selectedId) return RADIUS_ACTIVE;
    if (hoverId && pt.fixture_id === hoverId) return RADIUS_BASE * 1.6;
    return RADIUS_BASE;
  });
}

// ---------------------------
// Panel rendering
// ---------------------------
function renderPanel(f) {
  if (!f) return;

  // Title + subtitle
  const formatDate = (iso) => {
    try {
      const d = new Date(iso);
      const date = d.toLocaleDateString(undefined, { weekday: 'short', day: '2-digit', month: 'short' });
      const time = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
      return `${date} · ${time} GMT`;
    } catch {
      return iso || '';
    }
  };

  el.fixtureTitle.textContent = `${f.home_team} vs ${f.away_team}`;
  const ctx = [f.competition, formatDate(f.date_utc), f.stadium && `${f.stadium} (${f.city || ''})`, f.country]
    .filter(Boolean)
    .join(' • ');
  el.fixtureContext.textContent = ctx;

  // Logos (fallback to initials)
  setBadge(el.homeBadge, f.home_logo_url, formatTeamInitials(f.home_team));
  setBadge(el.awayBadge, f.away_logo_url, formatTeamInitials(f.away_team));

  // Match intelligence
  clearNode(el.matchList);
  const mi = document.createElement('div');
  mi.className = 'mi-wrap';
  mi.innerHTML = `
    <div><strong>Full-time prediction:</strong> ${f.predicted_winner || '–'} ${f.confidence_ftr ? `(${pct(f.confidence_ftr)})` : ''}</div>
    <div><strong>xG edge:</strong> ${num(f.xg_home)} vs ${num(f.xg_away)}</div>
    <div><strong>Points momentum:</strong> ${num(f.ppg_home)} PPG • ${num(f.ppg_away)} PPG</div>
  `;
  el.matchList.appendChild(mi);

  // Player watchlist
  clearNode(el.watchlist);
  const watch = parseKV(f.key_players_shots).slice(0, 6);
  if (watch.length) {
    watch.forEach(w => {
      const li = document.createElement('div');
      li.className = 'row';
      li.textContent = `${w.k} ${w.v}`;
      el.watchlist.appendChild(li);
    });
  } else {
    const empty = document.createElement('div');
    empty.className = 'row';
    empty.textContent = 'No player highlights available.';
    el.watchlist.appendChild(empty);
  }

  // Market snapshot
  clearNode(el.market);
  el.market.innerHTML = `
    <div><strong>Over 2.5 goals:</strong> ${pct(f.over25_prob)}</div>
    <div><strong>Both teams to score:</strong> ${pct(f.btts_prob)}</div>
  `;

  // Keep right-hand list scroll in sync if needed (no-op by default)
}

function num(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n.toFixed(1) : '–';
}

// ---------------------------
// Selection buttons (optional)
// ---------------------------
function wireFixtureButtons() {
  if (!el.prevBtn || !el.nextBtn) return;
  el.prevBtn.addEventListener('click', () => step(-1));
  el.nextBtn.addEventListener('click', () => step(+1));
}

// ---------------------------
// Upload CTA (kept simple)
// ---------------------------
if (el.upload) {
  el.upload.addEventListener('change', () => {
    // TODO: integrate OCR / audit pipeline
    toast('Your slip was uploaded. Parsing…');
  });
}

// ---------------------------
// Toasts (simple UI helper)
// ---------------------------
function toast(msg, type = 'success') {
  const div = document.createElement('div');
  div.className = `og-toast ${type}`;
  div.textContent = msg;
  document.body.appendChild(div);
  setTimeout(() => {
    div.classList.add('show');
    setTimeout(() => {
      div.classList.remove('show');
      setTimeout(() => div.remove(), 300);
    }, 2500);
  }, 0);
}

// ---------------------------
// Keyboard & selection pairing
// ---------------------------
function findFixtureById(id) {
  return fixtures.find(f => f.fixture_id === id);
}

function createSelectorIndexFromId(id) {
  const idx = fixtures.findIndex(f => f.fixture_id === id);
  return Math.max(0, idx);
}

// -----------------------------------
// Init + public entrypoint
// -----------------------------------
init();

// -----------------------------------
// Export helper (optional for Codex)
// -----------------------------------
export {};
