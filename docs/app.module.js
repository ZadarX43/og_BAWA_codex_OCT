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

// ----------------------------
// DOM refs
// ----------------------------
const el = {
  globeWrap:      document.getElementById('globe-container'),
  insights:       document.getElementById('insights-content'),
  fixtureTitle:   document.getElementById('fixture-title'),
  fixtureContext: document.getElementById('fixture-context'),
  matchList:      document.getElementById('match-intelligence'),
  watchlist:      document.getElementById('player-watchlist'),
  market:         document.getElementById('market-snapshot'),
  deepBtn:        document.getElementById('deep-dive-btn'),
  homeBadge:      document.getElementById('home-badge'),
  awayBadge:      document.getElementById('away-badge'),
  upload:         document.getElementById('bet-upload'),
  prevBtn:        document.getElementById('nav-prev'),
  nextBtn:        document.getElementById('nav-next')
};

if (!Papa) {
  throw new Error('PapaParse missing from window');
}

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
  // 1) Local ESM first
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
  // 2) esm.sh ESM (externalize three)
  for (const v of ['2.30.1','2.29.3','2.28.0']) {
    const url = `https://esm.sh/three-globe@${v}?bundle&external=three`;
    try {
      const m = await import(url);
      console.warn('[three-globe] using esm.sh ESM:', url);
      return m.default ?? m;
    } catch (e) {
      console.warn('[three-globe] esm.sh failed:', url, e);
    }
  }
  // 3) UMD
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
  if (el.globeWrap) {
    el.globeWrap.innerHTML = `<div class="globe-error">
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
  marker:        '#8CEFE5',
  markerActive:  '#CFFFFA',
  ring:          '#9EE7E3'
};

const SURFACE_EPS   = 0.018;
const RADIUS_BASE   = 0.22;
const RADIUS_ACTIVE = 0.65;
const CAMERA_ALT    = 2.1;

const BLOOM = { strength: 0.9, radius: 0.6, threshold: 0.75 };

// ----------------------------
// Utilities
// ----------------------------
const clamp01 = v => Math.max(0, Math.min(1, v));
const pct = n => `${Math.round(clamp01(n) * 100)}%`;

function clearNode(node) {
  if (!node) return;
  while (node.firstChild) node.removeChild(node.firstChild);
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
function setBadge(elm, url, fallbackText) {
  if (!elm) return;
  const token = {};
  badgeLoadTokens.set(elm, token);
  elm.classList.remove('has-logo');
  elm.innerHTML = '';

  const normalizedUrl = normalizeLogoUrl(url);
  if (normalizedUrl) {
    const img = new Image();
    img.decoding = 'async';
    img.loading = 'lazy';
    img.alt = fallbackText || '';
    if (!normalizedUrl.startsWith('data:')) {
      img.crossOrigin = 'anonymous';
      img.referrerPolicy = 'no-referrer';
    }
    img.onload = () => {
      if (badgeLoadTokens.get(elm) !== token) return;
      elm.innerHTML = '';
      elm.appendChild(img);
      elm.classList.add('has-logo');
    };
    img.onerror = () => {
      if (badgeLoadTokens.get(elm) !== token) return;
      elm.innerHTML = '';
      elm.textContent = fallbackText || '';
      elm.classList.remove('has-logo');
    };
    img.src = normalizedUrl;
  } else {
    elm.textContent = fallbackText || '';
  }
}

function initials(name = '') {
  const words = name.split(/\s+/).filter(Boolean);
  if (!words.length) return '';
  if (words.length === 1) return words[0].slice(0, 3).toUpperCase();
  return `${(words[0][0] || '').toUpperCase()}${(words[1]?.[0] || '').toUpperCase()}`;
}

function pick(row, keys) {
  for (const key of keys) {
    const val = row?.[key];
    if (typeof val === 'string' && val.trim()) return val.trim();
  }
  return '';
}

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
  return String(field)
    .split(';')
    .map(entry => {
      const [name, detail] = entry.split('|');
      return { name: (name || '').trim(), detail: (detail || '').trim() };
    })
    .filter(p => p.name || p.detail);
}

function tidyStat(value, dp = 1) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  return n.toFixed(dp);
}

function parseNumber(row, keys) {
  for (const key of keys) {
    const raw = row?.[key];
    if (raw === undefined || raw === null || raw === '') continue;
    const n = Number(raw);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

function formatDate(iso) {
  if (!iso) return 'Kick-off TBC';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return 'Kick-off TBC';
  const date = d.toLocaleDateString(undefined, {
    weekday: 'short', day: '2-digit', month: 'short'
  });
  const time = d.toLocaleTimeString(undefined, {
    hour: '2-digit', minute: '2-digit', timeZoneName: 'short'
  });
  return `${date} · ${time}`;
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
    setBadge(badge, logoUrl, initials(team) || '—');

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
    ? clamp01(fixture.confidence_ftr)
    : 0;
  const confidence = document.createElement('div');
  confidence.className = 'fixture-callout__confidence';
  const dot = document.createElement('span');
  dot.className = 'fixture-callout__confidence-dot';
  const confidenceText = document.createElement('span');
  confidenceText.textContent = Number.isFinite(fixture.confidence_ftr)
    ? `${pct(confidenceValue)} confidence edge`
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

// ----------------------------
// Scene init
// ----------------------------
async function init() {
  ThreeGlobeCtor = await loadThreeGlobe();

  scene = new THREE.Scene();

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  if ('outputColorSpace' in renderer) {
    renderer.outputColorSpace = THREE.SRGBColorSpace;
  } else if ('outputEncoding' in renderer) {
    renderer.outputEncoding = THREE.sRGBEncoding;
  }
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
  controls.autoRotate = false;
  controls.autoRotateSpeed = 0.6;
  controls.minDistance = 140;
  controls.maxDistance = 1200;

  scene.add(new THREE.AmbientLight(0xffffff, 0.9));

  // Starfield backdrop
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

  // --- Postprocessing ---
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));

  const fxaaPass = new ShaderPass(FXAAShader);
  const setFXAA = () => {
    const px = renderer.getPixelRatio?.() ?? (window.devicePixelRatio || 1);
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
    .pointAltitude(d => (d.baseAltitude ?? SURFACE_EPS) + (d.__active ? 0.08 : 0))
    .pointRadius(d => (d.__active ? RADIUS_ACTIVE : RADIUS_BASE))
    .pointColor(d => (d.__active ? COLORS.markerActive : (d.baseColor ?? COLORS.marker)))
    .pointResolution(12)
    .pointsMerge(true)
    .pointLabel(d => `${d.city ? `${d.city} • ` : ''}${d.home_team} vs ${d.away_team}`);

  globe
    .htmlElementsData([])
    .htmlElement(d => createFixtureCallout(d.fixture))
    .htmlLat(d => d.lat)
    .htmlLng(d => d.lng)
    .htmlAltitude(d => d.altitude ?? (SURFACE_EPS + 0.08));
  globe.htmlTransitionDuration?.(320);

  scene.add(globe);

  const loaderDiv = document.createElement('div');
  loaderDiv.className = 'globe-loading';
  loaderDiv.textContent = 'Loading globe…';
  el.globeWrap.appendChild(loaderDiv);
  if (typeof globe.onGlobeReady === 'function') {
    globe.onGlobeReady(() => loaderDiv.remove());
  } else {
    setTimeout(() => loaderDiv.remove(), 1500);
  }

  if (typeof globe.onPointHover === 'function') {
    globe.onPointHover(handleHover);
  }
  globe.onPointClick?.(pt => {
    if (!pt) return;
    const idx = fixtures.findIndex(f => f.fixture_id === pt.fixture_id);
    if (idx >= 0) selectIndex(idx, { fly: true });
  });

  window.addEventListener('resize', () => {
    const { clientWidth, clientHeight } = el.globeWrap;
    renderer.setSize(clientWidth, clientHeight);
    camera.aspect = clientWidth / clientHeight;
    camera.updateProjectionMatrix();
    const px = renderer.getPixelRatio?.() ?? (window.devicePixelRatio || 1);
    fxaaPass.material.uniforms.resolution.value.set(
      1 / (clientWidth * px),
      1 / (clientHeight * px)
    );
    bloomPass.setSize?.(clientWidth, clientHeight);
  });

  window.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight') { e.preventDefault(); step(+1); }
    if (e.key === 'ArrowLeft')  { e.preventDefault(); step(-1); }
  });

  el.prevBtn?.addEventListener('click', () => step(-1));
  el.nextBtn?.addEventListener('click', () => step(+1));

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
      showCsvError(`Could not load ${url} (HTTP ${response.status}). Make sure the file exists and the path is correct.`);
      return;
    }
    const text = await response.text();
    const { data, errors } = Papa.parse(text, { header: true, skipEmptyLines: true });
    if (errors?.length) console.warn('[CSV parse errors]', errors);

    fixtures = (data || [])
      .map(row => {
        const lat = parseNumber(row, ['latitude', 'lat', 'Latitude', 'lat_deg']);
        const lng = parseNumber(row, ['longitude', 'lon', 'lng', 'Longitude', 'long_deg']);
        const fallback = stadiumLookup[row.stadium?.trim?.() || ''];
        const latitude = Number.isFinite(lat) ? lat : fallback?.lat;
        const longitude = Number.isFinite(lng) ? lng : fallback?.lng;

        const confidenceRaw = parseNumber(row, ['confidence_ftr', 'confidence']);
        const confidence = Number.isFinite(confidenceRaw) ? clamp01(confidenceRaw) : undefined;
        const safeConfidence = confidence ?? 0;

        const homeLogo = normalizeLogoUrl(pick(row, [
          'home_badge_url', 'home_logo_url', 'home_logo', 'home_badge', 'homecrest'
        ]));
        const awayLogo = normalizeLogoUrl(pick(row, [
          'away_badge_url', 'away_logo_url', 'away_logo', 'away_badge', 'awaycrest'
        ]));

        const fixtureId = pick(row, ['fixture_id', 'id'])
          || `${row.home_team || row.Home || 'home'}-${row.away_team || row.Away || 'away'}-${row.date_utc || row.date || ''}`;

        return {
          fixture_id: fixtureId,
          home_team: (row.home_team || row.Home || '').trim(),
          away_team: (row.away_team || row.Away || '').trim(),
          home_logo_url: homeLogo,
          away_logo_url: awayLogo,
          date_utc: row.date_utc || row.date || '',
          competition: row.competition || row.league || '',
          stadium: row.stadium?.trim?.() || '',
          city: row.city?.trim?.() || '',
          country: row.country?.trim?.() || row.venue_country?.trim?.() || '',
          latitude,
          longitude,
          lat: latitude,
          lng: longitude,
          predicted_winner: row.predicted_winner || '',
          confidence_ftr: confidence,
          xg_home: parseNumber(row, ['xg_home']),
          xg_away: parseNumber(row, ['xg_away']),
          ppg_home: parseNumber(row, ['ppg_home']),
          ppg_away: parseNumber(row, ['ppg_away']),
          over25_prob: parseNumber(row, ['over25_prob']),
          btts_prob: parseNumber(row, ['btts_prob']),
          key_players_shots: processPlayerField(row.key_players_shots),
          key_players_tackles: processPlayerField(row.key_players_tackles),
          key_players_bookings: processPlayerField(row.key_players_bookings),
          baseAltitude: SURFACE_EPS + safeConfidence * 0.04,
          baseColor:
            safeConfidence > 0.7 ? '#65e3d3' :
            safeConfidence > 0.5 ? '#56cfe1' :
            '#ffae8b',
          __active: false
        };
      })
      .filter(f => Number.isFinite(f.latitude) && Number.isFinite(f.longitude));

    if (!fixtures.length) {
      showCsvError('No fixtures with valid latitude/longitude found.');
      return;
    }

    const many = fixtures.length > 250;
    globe.pointsMerge?.(many).pointResolution(12);
    globe
      .pointLat('latitude')
      .pointLng('longitude')
      .pointsData(fixtures);

    createSelectionRing();

    const preferred = fixtures.findIndex(f =>
      ['England','Scotland','Wales','Northern Ireland','Ireland','Spain','Portugal','France','Germany','Italy','Netherlands','Belgium','Norway','Sweden','Denmark','Switzerland','Austria','Poland','Czech Republic','Slovakia','Slovenia','Croatia','Serbia','Greece','Turkey'].includes(f.country)
    );
    activeIdx = preferred !== -1 ? preferred : 0;

    const boot = () => {
      selectIndex(activeIdx, { fly: true, immediate: true });
      globe.pointsTransitionDuration?.(650);
    };

    if (typeof globe.onGlobeReady === 'function') {
      globe.onGlobeReady(boot);
    } else {
      setTimeout(boot, 350);
    }
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
  if (!fixtures.length) return;
  const { fly = false } = opts;
  activeIdx = ((idx % fixtures.length) + fixtures.length) % fixtures.length;
  const fixture = fixtures[activeIdx];
  selectedId = fixture?.fixture_id || null;

  fixtures.forEach(d => {
    d.__active = d.fixture_id === selectedId;
  });
  globe.pointsData([...fixtures]);

  if (fixture) {
    updateHighlight(fixture);
    renderPanel(fixture);
    if (fly) flyToFixture(fixture, opts.immediate);
  } else {
    updateHighlight(null);
  }
}

function flyToFixture(fixture, immediate = false) {
  if (!fixture || !globe?.pointOfView) return;
  globe.pointOfView(
    { lat: fixture.latitude, lng: fixture.longitude, altitude: CAMERA_ALT },
    immediate ? 0 : 650
  );
}

// ----------------------------
// Selection halo aligned to surface
// ----------------------------
function getGlobeRadius() {
  if (globe?.getGlobeRadius) return globe.getGlobeRadius();
  try {
    const m = globe.children?.find(c => c.geometry?.parameters?.radius);
    return m?.geometry?.parameters?.radius || 100;
  } catch {
    return 100;
  }
}

function createSelectionRing() {
  if (pulseRing) return;
  const R = getGlobeRadius();
  const inner = R * (1 + SURFACE_EPS + 0.001);
  const outer = inner + R * 0.007;
  const ringGeom = new THREE.RingGeometry(inner, outer, 48);
  const ringMat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(COLORS.ring),
    transparent: true,
    opacity: 0.42,
    side: THREE.DoubleSide,
    depthWrite: false
  });
  pulseRing = new THREE.Mesh(ringGeom, ringMat);
  pulseRing.visible = false;
  scene.add(pulseRing);
  pulsePulse();
}

function updateSelectionRing(fixture) {
  if (!pulseRing) return;
  if (!fixture) {
    pulseRing.visible = false;
    return;
  }
  const R = getGlobeRadius();
  const latRad = THREE.MathUtils.degToRad(90 - fixture.latitude);
  const lonRad = THREE.MathUtils.degToRad(180 - fixture.longitude);
  const r = R * (1 + SURFACE_EPS + 0.001);
  const x = r * Math.sin(latRad) * Math.cos(lonRad);
  const y = r * Math.cos(latRad);
  const z = r * Math.sin(latRad) * Math.sin(lonRad);
  pulseRing.position.set(x, y, z);
  const outward = new THREE.Vector3(x, y, z).normalize();
  pulseRing.lookAt(outward.clone().multiplyScalar(R * 2));
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

function updateHighlight(fixture) {
  if (!fixture) {
    globe.ringsData([]);
    globe.htmlElementsData([]);
    updateSelectionRing(null);
    return;
  }

  const altitude = fixture.baseAltitude ?? SURFACE_EPS;
  globe
    .ringsData([
      {
        lat: fixture.latitude,
        lng: fixture.longitude,
        maxR: 2.05,
        propagationSpeed: 1.25,
        repeatPeriod: 1400,
        altitude
      }
    ])
    .ringColor(() => 'rgba(102,227,210,0.85)')
    .ringAltitude(d => d.altitude)
    .ringMaxRadius(d => d.maxR)
    .ringPropagationSpeed(d => d.propagationSpeed)
    .ringRepeatPeriod(d => d.repeatPeriod);

  globe.htmlElementsData([
    {
      lat: fixture.latitude,
      lng: fixture.longitude,
      altitude: altitude + 0.06,
      fixture
    }
  ]);

  updateSelectionRing(fixture);
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

  el.fixtureTitle.textContent = `${f.home_team} vs ${f.away_team}`;
  const parts = [f.competition, formatDate(f.date_utc), f.stadium && `${f.stadium}${f.city ? ` (${f.city})` : ''}`, f.country]
    .filter(Boolean);
  el.fixtureContext.textContent = parts.join(' • ');

  setBadge(el.homeBadge, f.home_logo_url, initials(f.home_team));
  setBadge(el.awayBadge, f.away_logo_url, initials(f.away_team));

  clearNode(el.matchList);
  const ft = `${f.predicted_winner || '—'}${Number.isFinite(f.confidence_ftr) ? ` (${pct(f.confidence_ftr)})` : ''}`;
  const xg = `${f.home_team} ${tidyStat(f.xg_home)} vs ${f.away_team} ${tidyStat(f.xg_away)}`;
  const pm = `${f.home_team} ${tidyStat(f.ppg_home)} PPG • ${f.away_team} ${tidyStat(f.ppg_away)} PPG`;
  [
    { label: 'Full-time prediction', value: ft },
    { label: 'xG edge', value: xg },
    { label: 'Points momentum', value: pm }
  ].forEach(item => {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${item.label}:</strong> ${item.value}`;
    el.matchList.appendChild(li);
  });

  clearNode(el.watchlist);
  if (f.key_players_shots.length) {
    f.key_players_shots.forEach(p => {
      const li = document.createElement('li');
      li.innerHTML = `<strong>${p.name}</strong> ${p.detail}`;
      el.watchlist.appendChild(li);
    });
  } else {
    const li = document.createElement('li');
    li.textContent = 'No player shot trends available.';
    el.watchlist.appendChild(li);
  }

  clearNode(el.market);
  const marketItems = [
    { label: 'Over 2.5 goals', value: pct(f.over25_prob ?? 0) },
    { label: 'Both teams to score', value: pct(f.btts_prob ?? 0) }
  ];
  f.key_players_bookings.forEach(p => marketItems.push({ label: `${p.name} booking risk`, value: p.detail }));
  f.key_players_tackles.forEach(p => marketItems.push({ label: `${p.name} tackles`, value: p.detail }));
  marketItems.forEach(item => {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${item.label}:</strong> ${item.value}`;
    el.market.appendChild(li);
  });

  if (el.deepBtn) {
    el.deepBtn.onclick = () => {
      alert(
        `Fixture: ${f.home_team} vs ${f.away_team}\n` +
        `Kick-off: ${formatDate(f.date_utc)}\n` +
        `Prediction: ${f.predicted_winner || '—'} ${Number.isFinite(f.confidence_ftr) ? `(${pct(f.confidence_ftr)})` : ''}\n` +
        `Over 2.5: ${pct(f.over25_prob ?? 0)} • BTTS: ${pct(f.btts_prob ?? 0)}`
      );
    };
  }
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
// Boot
// ----------------------------
init();
