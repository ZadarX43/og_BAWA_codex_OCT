// ===== Imports (ESM) =====
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Postprocessing
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass }     from 'three/examples/jsm/postprocessing/RenderPass.js';
import { ShaderPass }     from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { UnrealBloomPass }from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

// Shaders
import { FXAAShader }               from 'three/examples/jsm/shaders/FXAAShader.js';
import { CopyShader }               from 'three/examples/jsm/shaders/CopyShader.js';
import { LuminosityHighPassShader } from 'three/examples/jsm/shaders/LuminosityHighPassShader.js';

// ---------- Robust three-globe loader (ESM preferred, UMD fallback) ----------
async function importScriptUMD(src) {
  await new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.onload = () => resolve();
    s.onerror = (e) => reject(e);
    document.head.appendChild(s);
  });
  if (window.Globe) return window.Globe;
  throw new Error('UMD three-globe loaded but window.Globe was not defined');
}

async function loadThreeGlobe() {
  // 1) Try local ESM copies if you add them to the repo
  const localEsm = [
    './vendor/three-globe.module.js', // preferred official filename
    './vendor/three-globe.mjs'        // alternative if you saved it this way
  ];
  for (const p of localEsm) {
    try {
      const m = await import(p);
      console.info('[three-globe] using local ESM:', p);
      return m.default ?? m;
    } catch {}
  }

  // 2) Try esm.sh (ESM) across a few versions; bundle to avoid deep import churn; externalize three
  const esmVersions = ['2.30.1', '2.29.3', '2.28.0'];
  for (const v of esmVersions) {
    const url = `https://esm.sh/three-globe@${v}?bundle&external=three`;
    try {
      const m = await import(url);
      console.warn('[three-globe] using esm.sh ESM:', url);
      return m.default ?? m;
    } catch (e) {
      console.warn('[three-globe] esm.sh failed:', url, e);
    }
  }

  // 3) Fall back to UMD (local first, then CDNs)
  const umdCandidates = [
    './vendor/three-globe.min.js',
    'https://cdn.jsdelivr.net/npm/three-globe@2.29.3/dist/three-globe.min.js',
    'https://unpkg.com/three-globe@2.29.3/dist/three-globe.min.js'
  ];
  for (const url of umdCandidates) {
    try {
      const ctor = await importScriptUMD(url);
      console.warn('[three-globe] using UMD:', url);
      return ctor;
    } catch (e) {
      console.warn('[three-globe] UMD failed:', url, e);
    }
  }

  // If everything failed, show a visible error and abort
  const gc = document.getElementById('globe-container');
  if (gc) {
    gc.innerHTML = `<div class="globe-error">
      <strong>three-globe failed to load.</strong><br/>
      Add a local copy at <code>docs/vendor/three-globe.module.js</code> (ESM)
      or <code>docs/vendor/three-globe.min.js</code> (UMD).
    </div>`;
  }
  throw new Error('three-globe could not be loaded from any source');
}
const ThreeGlobeCtor = await loadThreeGlobe();

// PapaParse is UMD on window
const Papa = window.Papa;

// ====== DOM refs ======
const globeContainer = document.getElementById('globe-container');
const insightsContent = document.getElementById('insights-content');
const fixtureTitle = document.getElementById('fixture-title');
const fixtureContext = document.getElementById('fixture-context');
const matchIntelligenceList = document.getElementById('match-intelligence');
const playerWatchlist = document.getElementById('player-watchlist');
const marketSnapshot = document.getElementById('market-snapshot');
const deepDiveButton = document.getElementById('deep-dive-btn');
const homeBadge = document.getElementById('home-badge');
const awayBadge = document.getElementById('away-badge');
const uploadInput = document.getElementById('bet-upload');

// Quick dependency sanity
if (!Papa) {
  globeContainer.innerHTML = `<div class="globe-error">PapaParse failed to load.</div>`;
  throw new Error('PapaParse missing');
}

// ====== State ======
let fixtures = [];
let activeIndex = 0;

// ====== THREE basics ======
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
if ('outputColorSpace' in renderer) {
  renderer.outputColorSpace = THREE.SRGBColorSpace;
} else if ('outputEncoding' in renderer) {
  renderer.outputEncoding = THREE.sRGBEncoding;
}
renderer.setSize(globeContainer.clientWidth, globeContainer.clientHeight);

const scene = new THREE.Scene();
scene.add(new THREE.AmbientLight(0xffffff, 1.0));

const camera = new THREE.PerspectiveCamera(
  45,
  globeContainer.clientWidth / globeContainer.clientHeight,
  0.1,
  1000
);
camera.position.set(0, 0, 220);

globeContainer.innerHTML = '';
globeContainer.appendChild(renderer.domElement);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.enablePan = false;
controls.minDistance = 130;
controls.maxDistance = 320;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.45;

// ====== Globe + atmosphere ======
const globe = new ThreeGlobeCtor({ waitForGlobeReady: true })
  .showAtmosphere(true)
  .atmosphereAltitude(0.22)
  .atmosphereColor('#66e3d2')
  .globeImageUrl('https://unpkg.com/three-globe/example/img/earth-dark.jpg')
  .bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png')
  .pointLabel((d) => `${d.city || ''} • ${d.home_team} vs ${d.away_team}`) // tooltip
  .pointRadius((d) => d.__active ? 0.7 : 0.24); // enlarged active marker

scene.add(globe);

globe
  .pointAltitude((d) => (d.baseAltitude ?? 0.02) + (d.__active ? 0.085 : 0))
  .pointColor((d) => d.__active ? '#7df9c4' : (d.baseColor ?? '#56cfe1'))
  .pointsTransitionDuration?.(700);

globe
  .htmlElementsData([])
  .htmlElement((d) => createFixtureCallout(d.fixture))
  .htmlLat((d) => d.lat)
  .htmlLng((d) => d.lng)
  .htmlAltitude((d) => d.altitude ?? 0.04);

globe.htmlTransitionDuration?.(320);

// Useful helpers
function getGlobeRadius() {
  // find the globe mesh and read its bounding sphere
  const mesh = globe.children.find(o => o.type === 'Mesh' && o.geometry);
  if (mesh && mesh.geometry) {
    mesh.geometry.computeBoundingSphere();
    return mesh.geometry.boundingSphere.radius || 100;
  }
  return 100; // sane default
}

// Loading overlay until globe textures ready
const loaderDiv = document.createElement('div');
loaderDiv.className = 'globe-loading';
loaderDiv.textContent = 'Loading globe…';
globeContainer.appendChild(loaderDiv);
if (typeof globe.onGlobeReady === 'function') {
  globe.onGlobeReady(() => loaderDiv.remove());
} else {
  setTimeout(() => loaderDiv.remove(), 1500);
}

// ====== Starfield backdrop ======
const starGeom = new THREE.BufferGeometry();
const starCount = (window.devicePixelRatio > 2 || window.innerWidth < 480) ? 1200 : 2000;
const positions = new Float32Array(starCount * 3);
for (let i = 0; i < starCount; i++) {
  const r = 520 + Math.random() * 480;
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(Math.random() * 2 - 1);
  positions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
  positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
  positions[i * 3 + 2] = r * Math.cos(phi);
}
starGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
const starMat = new THREE.PointsMaterial({ size: 0.9, transparent: true, opacity: 0.6 });
const stars = new THREE.Points(starGeom, starMat);
scene.add(stars);

// ====== Postprocessing: EffectComposer + FXAA + Bloom ======
const composer = new EffectComposer(renderer);
const renderPass = new RenderPass(scene, camera);
composer.addPass(renderPass);

const fxaaPass = new ShaderPass(FXAAShader);
function updateFXAA() {
  const d = renderer.domElement;
  const px = renderer.getPixelRatio?.() ?? (window.devicePixelRatio || 1);
  fxaaPass.material.uniforms['resolution'].value.set(
    1 / (d.clientWidth * px),
    1 / (d.clientHeight * px)
  );
}
updateFXAA();
composer.addPass(fxaaPass);

// Bloom: lower strength/raise threshold so points/rings don’t square out
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(renderer.domElement.width, renderer.domElement.height),
  0.35,   // strength
  0.4,    // radius
  0.92    // threshold (higher => fewer blown highlights)
);
composer.addPass(bloomPass);

// ====== Render loop ======
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  composer.render();
}
animate();

// ====== Helpers ======
function formatDate(iso) {
  const date = new Date(iso);
  return new Intl.DateTimeFormat('en-GB', {
    weekday: 'short', day: 'numeric', month: 'short',
    hour: '2-digit', minute: '2-digit', timeZoneName: 'short',
  }).format(date);
}
function toPercent(value, dp = 0) {
  const v = Number(value);
  return Number.isFinite(v) ? `${(v * 100).toFixed(dp)}%` : '—';
}
function badgeLabel(name) {
  return (name || '')
    .split(/\s+/).filter(Boolean)
    .map((p)=>p[0]).join('').slice(0,3).toUpperCase();
}

function normalizeLogoUrl(url) {
  if (!url) return '';
  const trimmed = String(url).trim();
  if (!trimmed) return '';
  if (trimmed.startsWith('//')) return `https:${trimmed}`;
  if (/^https?:\/\//i.test(trimmed)) {
    if (typeof window !== 'undefined' && window.location?.protocol === 'https:' && /^http:\/\//i.test(trimmed)) {
      return trimmed.replace(/^http:\/\//i, 'https://');
    }
    return trimmed;
  }
  if (trimmed.startsWith('data:')) return trimmed;
  if (/^[./]/.test(trimmed)) return trimmed; // relative path inside docs/
  return '';
}

const badgeLoadTokens = new WeakMap();
function setBadge(el, url, fallbackText) {
  const token = {};
  badgeLoadTokens.set(el, token);
  el.classList.remove('has-logo');

  const normalizedUrl = normalizeLogoUrl(url);
  if (normalizedUrl) {
    el.innerHTML = '';
    const img = new Image();
    img.decoding = 'async';
    img.alt = '';
    img.loading = 'lazy';
    if (!normalizedUrl.startsWith('data:')) {
      img.crossOrigin = 'anonymous';
      img.referrerPolicy = 'no-referrer';
    }
    img.onload = () => {
      if (badgeLoadTokens.get(el) !== token) return;
      el.innerHTML = '';
      el.appendChild(img);
      el.classList.add('has-logo');
    };
    img.onerror = () => {
      if (badgeLoadTokens.get(el) !== token) return;
      el.innerHTML = '';
      el.textContent = fallbackText;
      el.classList.remove('has-logo');
    };
    img.src = normalizedUrl;
  } else {
    el.innerHTML = '';
    el.textContent = fallbackText;
  }
}

function formatShortDate(iso) {
  if (!iso) return 'Kick-off TBC';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return 'Kick-off TBC';
  return new Intl.DateTimeFormat('en-GB', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  }).format(date);
}

function fixtureLocationString(fixture) {
  const grouped = [fixture.city, fixture.country].filter(Boolean).join(', ');
  if (grouped) return grouped;
  if (fixture.stadium) return fixture.stadium;
  return 'Venue TBC';
}

function createFixtureCallout(fixture) {
  const anchor = document.createElement('div');
  anchor.className = 'fixture-callout-anchor';
  anchor.setAttribute('data-fixture-id', fixture.fixture_id || '');

  const wrapper = document.createElement('div');
  wrapper.className = 'fixture-callout';

  const createSide = (team, logoUrl, modifier) => {
    const side = document.createElement('div');
    side.className = `fixture-callout__side fixture-callout__side--${modifier}`;

    const badge = document.createElement('div');
    badge.className = 'fixture-callout__badge';
    badge.title = team || (modifier === 'home' ? 'Home team' : 'Away team');
    setBadge(badge, logoUrl, badgeLabel(team) || '—');

    const name = document.createElement('span');
    name.className = 'fixture-callout__team';
    name.textContent = team || (modifier === 'home' ? 'Home' : 'Away');
    name.title = team || '';

    side.append(badge, name);
    return side;
  };

  const header = document.createElement('div');
  header.className = 'fixture-callout__header';
  const homeSide = createSide(fixture.home_team, fixture.home_logo_url, 'home');
  const mid = document.createElement('div');
  mid.className = 'fixture-callout__header-mid';
  const vsChip = document.createElement('span');
  vsChip.className = 'fixture-callout__vs';
  vsChip.textContent = 'vs';
  mid.appendChild(vsChip);
  const awaySide = createSide(fixture.away_team, fixture.away_logo_url, 'away');
  header.append(homeSide, mid, awaySide);

  const meta = document.createElement('div');
  meta.className = 'fixture-callout__meta';
  const metaParts = [formatShortDate(fixture.date_utc), fixtureLocationString(fixture)]
    .filter(Boolean)
    .join(' • ');
  meta.textContent = metaParts;

  const prediction = document.createElement('div');
  prediction.className = 'fixture-callout__prediction';
  prediction.textContent = fixture.predicted_winner
    ? `Predicted: ${fixture.predicted_winner}`
    : 'Prediction pending';

  const confidenceValue = Number.isFinite(fixture.confidence_ftr)
    ? Math.max(0, Math.min(1, fixture.confidence_ftr))
    : 0;
  const confidence = document.createElement('div');
  confidence.className = 'fixture-callout__confidence';
  const dot = document.createElement('span');
  dot.className = 'fixture-callout__confidence-dot';
  const confidenceText = document.createElement('span');
  confidenceText.textContent = Number.isFinite(fixture.confidence_ftr)
    ? `${toPercent(confidenceValue, 0)} confidence edge`
    : 'Confidence pending';
  confidence.append(dot, confidenceText);

  const confidenceBar = document.createElement('div');
  confidenceBar.className = 'fixture-callout__confidence-bar';
  const confidenceFill = document.createElement('span');
  confidenceFill.className = 'fixture-callout__confidence-fill';
  confidenceFill.style.width = `${Math.max(8, Math.round(confidenceValue * 100))}%`;
  confidenceBar.appendChild(confidenceFill);

  wrapper.append(header, meta, prediction, confidence, confidenceBar);
  anchor.appendChild(wrapper);
  return anchor;
}
function flyTo(lat, lng, altitude = 1.8, ms = 900) {
  const pv = globe.pointOfView() || { lat: 0, lng: 0, altitude: 2 };
  const start = { lat: pv.lat || 0, lng: pv.lng || 0, altitude: pv.altitude || 2 };
  const end = { lat, lng, altitude };
  const t0 = performance.now();
  function tick(now) {
    const t = Math.min(1, (now - t0) / ms);
    const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    const lerp = (a,b)=> a + (b-a)*ease;
    globe.pointOfView({ lat: lerp(start.lat,end.lat), lng: lerp(start.lng,end.lng), altitude: lerp(start.altitude,end.altitude) });
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ====== CSV helpers ======
const num = (v) => { const n = Number(v); return Number.isFinite(n) ? n : undefined; };
const pickFirstString = (obj, keys) => {
  for (const key of keys) {
    const val = obj?.[key];
    if (typeof val === 'string' && val.trim()) return val.trim();
  }
  return '';
};
const stadiumLookup = {
  "Stadio Olimpico (Roma)": { lat: 41.9339, lng: 12.4545 },
  "Volksparkstadion (Hamburg)": { lat: 53.5870, lng: 9.8980 },
  "Stadion Feijenoord (Rotterdam)": { lat: 51.8939, lng: 4.5233 },
  "Parc des Princes (Paris)": { lat: 48.8414, lng: 2.2530 },
  "Stadio Giuseppe Meazza (Milano)": { lat: 45.4781, lng: 9.1240 },
  "Etihad Stadium (Manchester)": { lat: 53.4831, lng: -2.2004 },
  "Stadion Wankdorf (Bern)": { lat: 46.9630, lng: 7.4630 },
  "Estadi Olímpic Lluís Companys (Barcelona)": { lat: 41.3649, lng: 2.1516 },
  "Rams Global Stadium (İstanbul)": { lat: 41.1033, lng: 28.9913 },
  "Estadio Ramón Sánchez Pizjuán (Sevilla)": { lat: 37.3840, lng: -5.9700 },
  "Allianz Arena (München)": { lat: 48.2188, lng: 11.6247 },
  "Emirates Stadium (London)": { lat: 51.5550, lng: -0.1080 },
  "Estadio Santiago Bernabéu (Madrid)": { lat: 40.4531, lng: -3.6883 },
  "Estádio Municipal de Braga (Braga)": { lat: 41.5610, lng: -8.4270 },
  "Estádio do Sport Lisboa e Benfica (da Luz) (Lisboa)": { lat: 38.7527, lng: -9.1847 },
  "Reale Arena (Donostia-San Sebastián)": { lat: 43.3030, lng: -1.9730 }
};
function processPlayerField(field) {
  if (!field) return [];
  return field.split(';').map((entry) => {
    const [name, detail] = entry.split('|');
    return { name: (name || '').trim(), detail: (detail || '').trim() };
  });
}
function tidyStat(value, dp=1) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  return n.toFixed(dp);
}
function hydrateFixtures(rawFixtures) {
  return rawFixtures
    .filter((f) => f.fixture_id) // basic guard
    .map((f) => {
      const latCsv = num(f.latitude);
      const lngCsv = num(f.longitude);
      const fallback = stadiumLookup[f.stadium?.trim?.() || ""];
      const latitude  = latCsv ?? fallback?.lat;
      const longitude = lngCsv ?? fallback?.lng;

      // Normalize strings/numbers
      const cfRaw = num(f.confidence_ftr);
      const cf = Number.isFinite(cfRaw) ? cfRaw : undefined;
      const safeCf = Math.max(0, cf ?? 0);

      const homeLogo = normalizeLogoUrl(pickFirstString(f, [
        'home_logo_url', 'home_badge_url', 'home_logo', 'home_badge', 'homecrest'
      ]));
      const awayLogo = normalizeLogoUrl(pickFirstString(f, [
        'away_logo_url', 'away_badge_url', 'away_logo', 'away_badge', 'awaycrest'
      ]));

      return {
        ...f,
        latitude, longitude,
        lat: latitude,
        lng: longitude,
        xg_home: num(f.xg_home),
        xg_away: num(f.xg_away),
        ppg_home: num(f.ppg_home),
        ppg_away: num(f.ppg_away),
        confidence_ftr: cf,
        over25_prob: num(f.over25_prob),
        btts_prob: num(f.btts_prob),

        stadium: f.stadium?.trim?.() || "",
        city:    f.city?.trim?.()    || "",
        country: f.country?.trim?.() || "",

        home_logo_url: homeLogo,
        away_logo_url: awayLogo,

        key_players_shots:    processPlayerField(f.key_players_shots),
        key_players_bookings: processPlayerField(f.key_players_bookings),
        key_players_tackles:  processPlayerField(f.key_players_tackles),

        // keep markers low to the surface; add tiny lift for visibility
        baseAltitude: 0.018 + safeCf * 0.04,
        baseColor:
          safeCf > 0.7 ? '#65e3d3' :
          safeCf > 0.5 ? '#56cfe1' :
                     '#ffae8b',

        __active: false // selection state
      };
    })
    .filter((fx) => Number.isFinite(fx.latitude) && Number.isFinite(fx.longitude));
}

// ====== Highlight helpers (rings + active marker) ======
function updateHighlight() {
  fixtures.forEach((d, i) => { d.__active = i === activeIndex; });
  // refresh points so the accessor for radius re-evaluates
  globe.pointsData([...fixtures]);

  const active = fixtures[activeIndex];
  if (!active) {
    globe.ringsData([]);
    globe.htmlElementsData([]);
    return;
  }

  const altitude = (active.baseAltitude ?? 0.02);

  // a soft pulse ring anchored at ground
  globe
    .ringsData([
      {
        lat: active.latitude,
        lng: active.longitude,
        maxR: 2.05,           // in degrees
        propagationSpeed: 1.25, // deg/s
        repeatPeriod: 1400,  // ms
        altitude
      }
    ])
    .ringColor(() => 'rgba(102,227,210,0.85)')
    .ringAltitude((d) => d.altitude)
    .ringMaxRadius((d) => d.maxR)
    .ringPropagationSpeed((d) => d.propagationSpeed)
    .ringRepeatPeriod((d) => d.repeatPeriod);

  globe.htmlElementsData([
    {
      lat: active.latitude,
      lng: active.longitude,
      altitude: altitude + 0.06,
      fixture: active
    }
  ]);
}

// ====== Render a single fixture’s side panel ======
function renderFixture(index) {
  const fixture = fixtures[index];
  if (!fixture) return;

  activeIndex = index;

  // Top name + crests
  fixtureTitle.innerHTML = `
    <span class="fixture-title-text">
      ${fixture.home_team} <span class="vs">vs</span> ${fixture.away_team}
    </span>`;

  // Club logos (if present)
  setBadge(homeBadge, fixture.home_logo_url, badgeLabel(fixture.home_team));
  setBadge(awayBadge, fixture.away_logo_url, badgeLabel(fixture.away_team));

  // Context line
  const parts = [fixture.stadium?.trim?.(), fixture.city?.trim?.(), fixture.country?.trim?.()]
    .filter(Boolean).join(", ");
  const ctxTail = parts ? ` • ${parts}` : "";
  fixtureContext.textContent =
    `${fixture.competition} • ${formatDate(fixture.date_utc)}${ctxTail}`;

  // Match Intelligence
  matchIntelligenceList.innerHTML = '';
  const ft = `${fixture.predicted_winner || '—'} (${toPercent(fixture.confidence_ftr,0)})`;
  const xg = `${fixture.home_team} ${tidyStat(fixture.xg_home)} vs ${fixture.away_team} ${tidyStat(fixture.xg_away)}`;
  const pm = `${fixture.home_team} ${tidyStat(fixture.ppg_home)} PPG • ${fixture.away_team} ${tidyStat(fixture.ppg_away)} PPG`;
  [
    { label: 'Full-time prediction', value: ft },
    { label: 'xG edge',              value: xg },
    { label: 'Points momentum',      value: pm },
  ].forEach((item) => {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${item.label}:</strong> ${item.value}`;
    matchIntelligenceList.appendChild(li);
  });

  // Player watchlist
  playerWatchlist.innerHTML = '';
  fixture.key_players_shots.forEach((p) => {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${p.name}</strong> ${p.detail}`;
    playerWatchlist.appendChild(li);
  });

  // Market snapshot
  marketSnapshot.innerHTML = '';
  const marketItems = [
    { label: 'Over 2.5 goals', value: toPercent(fixture.over25_prob,0) },
    { label: 'Both teams to score', value: toPercent(fixture.btts_prob,0) },
  ];
  fixture.key_players_bookings.forEach((p) => marketItems.push({ label: `${p.name} booking risk`, value: p.detail }));
  fixture.key_players_tackles.forEach((p) => marketItems.push({ label: `${p.name} tackles`, value: p.detail }));
  marketItems.forEach((item) => {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${item.label}:</strong> ${item.value}`;
    marketSnapshot.appendChild(li);
  });

  // Deep dive
  deepDiveButton.onclick = () => {
    const summary =
      `Fixture: ${fixture.home_team} vs ${fixture.away_team}\n` +
      `Kick-off: ${formatDate(fixture.date_utc)}\n` +
      `Prediction: ${ft}\n` +
      `Over 2.5: ${toPercent(fixture.over25_prob)}\n` +
      `BTTS: ${toPercent(fixture.btts_prob)}`;
    alert(summary);
  };

  // Globe camera + highlight
  updateHighlight();
  flyTo(fixture.latitude, fixture.longitude, 2.1, 1000);
}

// ====== Focus fixture with smooth POV (and ensure selection state) ======
function focusFixture(index) {
  if (index < 0 || index >= fixtures.length) return;
  renderFixture(index);
}

// ====== Load fixtures and boot ======
const csvUrl = new URL('./data/fixtures.csv', window.location.href);
csvUrl.searchParams.set('v', Date.now().toString()); // cache-bust

Papa.parse(csvUrl.href, {
  download: true,
  header: true,
  // delimiter: "\t", // uncomment if your snapshot is TSV
  skipEmptyLines: true,
  dynamicTyping: false,
  complete: (results) => {
    fixtures = hydrateFixtures(results.data || []);

    if (!fixtures.length) {
      fixtureTitle.textContent = 'No fixtures found';
      fixtureContext.textContent = 'Check data/fixtures.csv format.';
      return;
    }

    // Prefer first EU fixture if present
    const eu = fixtures.findIndex(f =>
      ['England','Scotland','Wales','Northern Ireland','Ireland','Spain','Portugal','France','Germany','Italy','Netherlands','Belgium','Norway','Sweden','Denmark','Switzerland','Austria','Poland','Czech Republic','Slovakia','Slovenia','Croatia','Serbia','Greece','Turkey'].includes(f.country)
    );
    activeIndex = eu !== -1 ? eu : 0;

    // Bind to globe
    globe.pointsData(fixtures);

    // Initial panel + focus
    renderFixture(activeIndex);
  },
  error: (error) => {
    console.error('Failed to load fixtures:', error);
    fixtureTitle.textContent = 'Unable to load fixtures';
    fixtureContext.textContent = 'Check the data directory and reload the page.';
  },
});

// Click → focus that fixture
if (typeof globe.onPointClick === 'function') {
  globe.onPointClick((pt) => {
    const idx = fixtures.findIndex((f) => f.fixture_id === pt.fixture_id);
    if (idx !== -1) {
      focusFixture(idx);
    }
  });
}

// Keyboard: Left/Right to switch fixtures (and keep selection synced)
window.addEventListener('keydown', (event) => {
  if (!fixtures.length) return;
  if (event.key === 'ArrowRight') {
    activeIndex = (activeIndex + 1) % fixtures.length;
    focusFixture(activeIndex);
  }
  if (event.key === 'ArrowLeft') {
    activeIndex = (activeIndex - 1 + fixtures.length) % fixtures.length;
    focusFixture(activeIndex);
  }
});

// Resize handling
window.addEventListener('resize', () => {
  const w = globeContainer.clientWidth, h = globeContainer.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  updateFXAA();
});

// Upload placeholder
uploadInput?.addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  alert(
    `Bet slip uploaded: ${file.name}\n\nNext steps:\n` +
    `• OCR the slip to extract selections\n` +
    `• Run the BetChecker audit pipeline\n` +
    `• Generate OG Co-Pilot insights`
  );
  uploadInput.value = '';
});

/* ----------- SMALL CSS hook for crest images ----------- *
   Add to styles.css (already styled in your theme; included here for clarity)

   .badge img { width: 100%; height: 100%; object-fit: contain; display:block; }
   .badge.has-logo { background: rgba(255,255,255,.08); }
*/
