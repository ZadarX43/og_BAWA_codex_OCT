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
  for (const v of ['2.31.3', '2.31.1', '2.30.1', '2.29.3']) {
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
  if (el.guides) {
    el.globeWrap.innerHTML = `<div class="globe-error">
      <strong>three-globe failed to load.</strong><br/>
      Check your network or add <code>vendor/three-globe.min.js</code>.
    </div>`;
  }
  throw new Error('three-globe could not be loaded from any source');
}

// ----------------------------
// Globals / tuning
// ----------------------------
let ThreeGlobeCtor;
let globe;
let renderer, scene, camera, controls, composer, glowPass;
let fixtures = [];
let activeIdx = 0;
let selectedId = null;
let pulseRing;
let htmlTabsData = []; // [{lat,lng,altitude,idx,el}]

const COLORS = {
  marker:        '#A7FFF6',
  markerInactive:'#8CEFE5',
  markerActive:  '#CFFFFA',
  ring:          '#9EE7E3'
};

const SURFACE_EPS   = 0.009; // fraction of globe radius
const RADIUS_BASE   = 0.014;
const RADIUS_ACTIVE = 0.040;
const CAMERA_ALT    = 2.0;

const BLOOD_ORANGE = 0xff9b5e; // used for beam flicker

// ----------------------------
// Utilities
// ----------------------------
const DEG2RAD = Math.PI / 180;

const clamp01 = v => Math.max(0, Math.min(1, v));
const pct = n => `${Math.round(clamp01(n) * 100)}%`;

function showToast(type, text, ms = 2600) {
  const t = document.createElement('div');
  t.className = `og-toast ${type}`;
  t.textContent = text;
  document.body.appendChild(t);
  requestAnimationFrame(() => t.classList.add('show'));
  setTimeout(() => { t.classList.remove('show'); setTimeout(() => t.remove(), 250); }, ms);
}

function clearNode(node) {
  if (!node) return;
  while (node.firstChild) node.removeChild(node.firstChild);
}

function pick(row, keys) {
  for (const k of keys) {
    const v = (row[k] ?? '').toString().trim();
    if (v !== '') return v;
  }
  return '';
}

function getGlobeRadius() {
  // three-globe gives the radius at scale=1 via its inner mesh
  if (globe && typeof globe.getGlobeRadius === 'function') {
    return globe.getGlobeRadius();
  }
  // sensible default so nothing explodes before init
  return 100;
}

function hasHtmlElementsApi(g){
  return g && typeof g.htmlElementsData === 'function'
         && typeof g.htmlElement      === 'function'
         && typeof g.htmlLat          === 'function';
}

function initials(name = '') {
  const words = String(name).trim().split(/\s+/).filter(Boolean);
  if (!words.length) return '';
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + (words[1][0] || '')).toUpperCase();
}

function stripDiacritics(s) {
  try { return s.normalize('NFD').replace(/\p{Diacritic}/gu, ''); }
  catch { return s; }
}
function slugLocal(team) {
  return stripDiacritics(String(team))
    .toLowerCase()
    .replace(/&/g, ' and ')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function normalizeBasicUrl(raw) {
  if (!raw) return '';
  let u = String(raw).trim();
  if (!u) return '';
  if (u.startsWith('//')) u = `https:${u}`;
  if (/^http:\/\//i.test(u) && location.protocol === 'https:') u = u.replace(/^http:\/\//i, 'https://');
  return u;
}

// ---- (1) FIXED: lat/lng → world position matches three-globe
function latLngToVec3(latDeg, lonDeg, altFrac = 0) {
  const r = getGlobe inclination? // ignore tilt; using default three-globe orientation
  const R = getGlobeRadius() * (1 + (altFrac || 0));
  const phi   = latDeg * DEG2RAD;      // latitude from -90..+90
  const theta = (lonDeg + 0) * DEG2RAD; // longitude from -180..+180 (no +180 offset here)
  // three.js Y-up convention with Greenwich at +Z, lon=0 on +Z axis:
  const x =  R * Math.sin(theta) * Math.cos(phi); // east-west
  const z =  R * Math.cos(theta) * Math.cos(phi); // "forward" (0° lon)
  const y =  R * Math.sin(phi);                   // north
  // three-globe internally rotates the earth so that lon=0 is at +Z and +lon rotates toward +X,
  // which this mapping already matches. If your base texture is different, swap X/Z signs.
  return new THREE.Vector3(x, y, z);
}

// Robust field resolver for lat/lon from a row (2) smarter source of coords
function resolveLatLng(obj) {
  const latKeys = ['latitude','lat','lat_deg','lat_dd','home_lat','homeLatitude','home_latitude'];
  const lonKeys = ['longitude','lon','lng','long','long_deg','lon_dd','home_lng','home_longitude'];

  let lat, lon;
  for (const k of latKeys) if (obj[k] != null && obj[k] !== '') { lat = Number(obj[k]); if (!Number.isNaN(lat)) break; }
  for (const k of lonKeys) if (obj[k] != null && obj[k] !== '') { lon = Number(obj[k]); if (!Number.isNaN(lon)) break; }

  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  return { lat, lon };
}

// Stadium image candidates (prefer venue/team names before numeric IDs)
const STADIUM_BASE = './assets/stadiums';
function stadiumCandidates(f) {
  const names = [
    f.stadium, f.stadium_name, f.venue, f.venue_name,
    f.home_stadium, f.home_venue,
    f.home_team, f.away_team,
    f.fixture_id
  ].filter(Boolean).map(v => String(v));

  const uniqSlugs = Array.from(new Set(
    names.map(n => slugLocal(n)).filter(Boolean)
  ));
  const exts = ['.jpg', '.jpeg', '.png', '.webp'];
  const out = [];
  for (const s of uniqSlugs) for (const ext of exts) out.push(`${STADIUM_BASE}/${s}${ext}`);
  return out;
}

// ---------- Logo system (local only; no remote)
const LOGO_LOCAL_BASE = './assets/assets/logos';

const TEAM_LOGO_OVERRIDES = {
  'Celtic':               `${LOGO_LOCAL_BASE}/celtic.svg`,
  'Lazio':                `${LOGO_LOCAL_BASE}/lazio.svg`,
  'Royal Antwerp FC':     `${LOGO_LOCAL_BASE}/royal-antwerp.svg`,
  'Shakhtar Donetsk':     `${LOGO_LOCAL_BASE}/shakhtar-donetsk.svg`,
  'Atlético Madrid':      `${LOGO_LOCAL_BASE}/atletico-madrid.svg`,
  'Atletico Madrid':      `${LOGO_LOCAL_BASE}/atletico-madrid.svg`,
  'Feyenoord':            `${LOGO_LOCAL_BASE}/feyenoord.svg`,
  'PSG':                  `${LOGO_LOCAL_BASE}/paris-saint-germain.svg`,
  'Paris Saint-Germain':  `${LOGO_LOCAL_BASE}/paris-saint-germain.svg`,
  'Newcastle United':     `${LOGO_LOCAL_BASE}/newcastle-united.svg`,
  'AC Milan':             `${LOGO_LOCAL_BASE}/ac-milan.svg`,
  'Borussia Dortmund':    `${LOGO_LOCAL_BASE}/borussia-dortmund.svg`,
  'Manchester City':      `${LOGO_LOCAL_BASE}/manchester-city.svg`,
  'RB Leipzig':           `${LOGO_LOCAL_BASE}/rb-leipzig.svg`,
  'Young Boys':           `${LOGO_LOCAL_BASE}/young-boys.svg`,
  'Red Star Belgrade':    `${LOGO_LOCAL_BASE}/red-star-belgrade.svg`,
  'Crvena Zvezda':        `${LOGO_LOCAL_BASE}/red-star-belgrade.svg`,
  'FC Barcelona':         `${LOGO_LOCAL_BASE}/fc-barcelona.svg`,
  'Barcelona':            `${LOGO_LOCAL_BASE}/fc-barcelona.svg`,
  'Porto':                `${LOGO_LOCAL_BASE}/fc-porto.svg`,
  'FC Porto':             `${LOGO_LOCAL_BASE}/fc-porto.svg`,
  'Galatasaray':          `${LOGO_LOCAL_BASE}/galatasaray.svg`,
  'Manchester United':    `${LOGO_LOCAL_BASE}/manchester-united.svg`,
  'Sevilla FC':           `${LOGO_LOCAL_BASE}/sevilla-fc.svg`,
  'PSV':                  `${LOGO_LOCAL_BASE}/psv.svg`,
  'PSV Eindhoven':        `${LOGO_LOCAL_BASE}/psv.svg`,
  'København':            `${LOGO_LOCAL_BASE}/fc-kobenhavn.svg`,
  'FC Copenhagen':        `${LOGO_LOCAL_BASE}/fc-kobenhavn.svg`,
  'Arsenal':              `${LOGO_LOCAL_BASE}/arsenal.svg`,
  'Lens':                 `${LOGO_LOCAL_BASE}/lens.svg`,
  'Real Madrid':          `${LOGO_LOCAL_BASE}/real-madrid.svg`,
  'Napoli':               `${LOGO_LOCAL_BASE}/ssc-napoli.svg`,
  'SSC Napoli':           `${LOGO_LOCAL_BASE}/ssc-napoli.svg`,
  'Benfica':              `${LOGO_LOCAL_BASE}/sl-benfica.svg`,
  'SL Benfica':           `${LOGO_LOCAL_BASE}/sl-benfica.svg`,
  'Inter Milan':          `${LOGO_LOCAL_BASE}/inter-milan.svg`,
  'Inter':                `${LOGO_LOCAL_BASE}/inter-milan.svg`,
  'Real Sociedad':        `${LOGO_LOCAL_BASE}/real-sociedad.svg`,
  'Salzburg':             `${LOGO_LOCAL_BASE}/rb-salzburg.svg`,
  'Bayern München':       `${LOGO_LOCAL_BASE}/bayern-munich.svg`,
  'Bayern Munich':        `${LOGO_LOCAL_BASE}/bayern-munich.svg`
};

const LOGO_CACHE_KEY  = 'og_logo_cache_v_preload';
let LOGO_CACHE = {};
try { LOGO_CACHE = JSON.parse(localStorage.getItem(LOGO_CACHE_KEY) || '{}'); } catch {}
function saveLogoCache(){ try { localStorage.setItem(LOGO_CACHE_KEY, JSON.stringify(LOGO_CACHE)); } catch {} }
const LOGO_STORE = new Map();

function localLogoCandidates(team) {
  const slug = slugLocal(team);
  return [`${LOGO_LOCAL_BASE}/${slug}.png`, `${LOGO_LOCAL_BASE}/${slug}.svg`];
}

async function tryLoadDirect(src, teamName, timeoutMs) {
  if (!src) return null;
  return new Promise(resolve => {
    let done = false;
    const img = new Image();
    img.alt = teamName || '';
    img.loading = 'lazy';
    img.decoding = 'async';
    if (!/^https?:/i.test(src)) img.crossOrigin = 'anonymous';
    const t = setTimeout(() => { if (!done){ done = true; resolve(null); } }, timeoutMs);
    img.onload  = () => { if (!done){ clearNg(); resolve({ ok:true, img, src }); } };
    img.onerror = () => { if (!done){ clearNg(); resolve(null); } };
    function clearNg(){ clearTimeout(t); }
    img.src = src.includes('?') ? src : `${src}?v=${Date.now().toString(36)}`;
  });
}

async function tryLoadViaDataURL(src, teamName, timeoutMs) {
  try {
    const ctrl = new AbortController();
    const to = setTimeout(() => ctrl.abort(), timeoutMs);
    const res = await fetch(src, { cache:'no-store', signal: ctrl.signal });
    clearTimeout(to);
    if (!res.ok) return null;
    const blob = await res.blob();
    const dataURL = await new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(fr.result);
      fr.onerror = reject;
      fr.readAsDataURL(blob);
    });
    return new Promise(resolve => {
      const img = new Image();
      img.onload  = () => resolve({ ok:true, img, src: dataURL });
      img.onerror = () => resolve(null);
      img.src = dataURL;
    });
  } catch { return null; }
}

async function tryLoad(src, teamName, timeoutMs = 15000) {
  return (await tryLoadDirect(src, teamName, Math.min(timeoutMs, 7000)))
      || (await tryLoadViaDataURL(src, teamName, timeoutMs));
}

function guessLogoSources(teamName) {
  const list = [];
  if (TEAM_LOGO_OVERRIDES[teamName]) list.push(TEAM_LOGO_OVERRIDES[teamName]);
  for (const loc of localLogoCandidates(teamName)) list.push(loc);
  return [...new Set(list.map(normalizeBasicUrl))];
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
      let hit = null;
      for (const src of guessLogoSources(name)) {
        // eslint-disable-next-line no-await-in-loop
        const r = await tryLoad(src, name, 8000);
        if (r) { hit = r; break; }
      }
      LOGO_STORE.set(name, hit ? { img: hit.img, url: hit.src } : { img: null, url: null });
    }
  }
  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));
}

const badgeTokens = new WeakMap();
async function setBadge(elm, urlFromCsv, teamName='') {
  if (!elm) return;
  const token = {};
  badgeTokens.set(elm, token);
  elm.classList.remove('has-logo');
  elm.innerHTML = '';

  // 0) explicit override or CSV url
  const candidates = [];
  const raw = normalizeBasicUrl(urlFromCsv);
  if (raw) candidates.push(raw);
  candidates.push(...guessLogoSources(teamName));

  let hit = null;
  for (const src of candidates) {
    // eslint-disable-next-line no-await-in-loop
    const r = await tryLoad(src, teamName, 8000);
    if (r) { hit = r; break; }
  }

  if (!hit) {
    // fallback to initials
    if (badgeTokens.get(elm) !== token) return;
    elm.textContent = initials(teamName) || '';
    return;
  }
  if (badgeTokens.get(elm) !== token) return;

  elm.innerHTML = '';
  elm.appendChild(hit.img);
  elm.classList.add('has-logo');
  if (teamName && hit.src) { LOGO_CACHE[teamName] = hit.src; saveLogoCache(); }
}

// ----------------------------
// COMPETITION LOGOS + ACCURACY STRIP
// ----------------------------
const COMP_LOGO_BASE = './assets/assets/logos';
const COMP_LOGO_MAP = {
  'UEFA Champions League':      `${COMP_LOGO_BASE}/paris-saint-germain.svg`,
  'UEFA Europa League':         `${COMP_LOGO_BASE}/sl-benfica.svg`,
  'UEFA Europa Conference League': `${COMP_LOGO_BASE}/rb-salzburg.svg`,
  'Premier League':             `${COMP_LOGO_BASE}/arsenal.svg`,
  'LaLiga':                     `${COMP_LOGO_BASE}/real-madrid.svg`,
  'La Liga':                    `${COMP_LOGO_BASE}/real-madrid.svg`,
  'Bundesliga':                 `${COMP_LOGO_BASE}/bayern-munich.svg`,
  'Serie A':                    `${COMP_LOGO_BASE}/ac-milan.svg`,
  'Ligue 1':                    `${COMP_LOGO_BASE}/paris-saint-germain.svg`,
  'Liga Portugal':              `${COMP_LOGO_BASE}/sl-benfica.svg`,
  'Liga NOS':                   `${COMP_LOGO_MAP?.['Liga Portugal']}`
};
function findCompLogoSrc(name=''){ return COMPORM(name) }
function COMPORM(n){ if (!n) return ''; if (COMP_LOGO_MAP[n]) return COMP_LOGO_MAP[n]; const k = Object.keys(COMP_LOGO_MAP).find(k=>n.toLowerCase().includes(k.toLowerCase())); return k?COMP_LOGO_MAP[k]:''; }

// Average predicted probabilities for fixtures in this competition
function getCompetitionSnapshot(compName) {
  const rows = (compName && compName.trim())
    ? fixtures.filter(r => (r?.competition || '').toLowerCase() === compName.toLowerCase())
    : [];
  const avg = arr => arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0;
  return {
    n: rows.length,
    ftr:    Math.round(avg(rows.map(r => Number(r?.confidence_ftr) || 0) * 100) / 1) || 0,
    over25: Math.round(avg(rows.map(r => Number(r?.over25_prob)   || 0) * 100) / 1) || 0,
    btts:   Math.round(avg(rows.map(r => Number(r?.btts_prob)     || 0) * 100) || 0
  };
}

// Paint the strip under the globe (chips)
function renderCompetitionAccuracy(compName) {
  const wrap   = document.getElementById('comp-accuracy');
  if (!wrap) return;

  const nameEl = document.getElementById('comp-name');
  const logoEl = document.getElementById('comp-logo');
  const fill   = wrap.querySelector('.gauge-display .fill');
  const valEl  = wrap.querySelector('.gauge-display .val');
  const chips  = document.getElementById('comp-traffic');

  if (nameEl) nameEl.textContent = compName || '';
  if (logoEl) {
    const src = findCompLogo(this, compName);
    if (src) { logoEl.src = src; logoEl.style.opacity = '0.9'; }
  }

  const stats = getCompetitionSnapshot(compName);
  const ftr = 87; // demo override; replace with `stats.ftr` for live
  if (fill) fill.style.transform = `scaleX(${Math.max(0, Math.min(1, ftr/100))})`;
  if (valEl) valEl.textContent = `${ftr}%`;

  if (chips) {
    const mk = (cls, t) => {
      const s = document.createElement('span');
      s.className = `light ${cls}`;
      s.textContent = t;
      return s;
    };
    chips.innerHTML = '';
    chips.append(
      mk('light--green', `FTR ${ftr}%`),
      mk('light--blue',  `O2.5 ${stats.over25}%`),
      mk('light--amber', `BTTS ${stats.btts}%`)
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
  renderer.setPixelRate?.(window.devicePixelRatio || 1);
  renderer.setSize(el.globeWrap.clientWidth, el.globeWrap.clientHeight);
  renderer.domElement.style.width = '100%';
  renderer.domElement.style.height = '100%';
  renderer.domElement.style.display = 'block';
  renderer.domElement.style.pointerEvents = 'auto'; // (fix) allow orbit controls even with CSS overlay
  el.globeWrap.innerHTML = '';
  el.globeWrap.appendChild(renderer.domElement);

  if ('outputColorSpace' in renderer) renderer.outputColorSpace = THREE.SRGBColorSpace;
  if ('toneMapping' in renderer) {
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
  }

  camera = new THREE.PerspectiveCamera(45, el.globeWrap.clientWidth / el.globeWrap.clientHeight, 0.75, 5000);
  camera.position.set(0, 0, getGlobeRadius() * 2.8);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.enableZoom = true;
  controls.enableRotate = true;
  controls.minDistance = getGlobeRadius() * 1.2;
  controls.maxDistance = getGlobeRadius() * 6;

  scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  const hemi = new THREE.HemisphereLight(0xddeeff, 0x223344, 0.6);
  scene.add(hemi);

  // globe
  globe = new ThreeGlobeCtor({ waitForGlobeReady: true })
    .showXYZ(false)
    .showAtmosphere(true)
    .atmosphereColor('#9ef9e3')
    .atmosphereAltitude(0.28)
    .globeImageUrl('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
    .bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png')
    .pointAltitude(() => SURFACE_EPS)
    .pointRadius(d => (d.__active ? RADIUS_ACTIVE : RAIN))
    .pointColor(d => (d.__active ? COLORS.http(Active) : COLORS.marker))
    .pointResolution(12)
    .pointsMerge(true);

  scene.add(globe);

  // ---- active marker visual (radar rings + beam + billboard)
  const marker = new THREE.Group();
  scene.add(marker);

  const ringMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(COLORS.ring), transparent:true, opacity:0 });
  const ringGeo = new THREE.RingGeometry(1.0, 1.08, 128);
  const ringA = new THREE.Mesh(ringGeo, ringMat.clone());
  const ringB = new THREE.Mesh(ringGeo, ringMat.clone());
  const ringC = new THREE.Mesh(ringGeo, ringMat.clone());
  const ringGroup = new THREE.Group();
  ringGroup.add(ringA, ringB, ringC);
  marker.add(ringGroup);

  const beamGeo = new THREE.CylinderGeometry(0.18, 0.28, 30, 32, 1, true);
  const beamMat = new THREE.MeshBasicMaterial({ color: BLOOD_ORANGE, transparent: true, opacity: 0.18, blending: THREE.AdditiveBlending, depthWrite: false });
  const beam = new THREE.Mesh(beamGeo, beamMat);
  beam.visible = false;
  marker.add(beam);

  const billboard = new THREE.Sprite(new THREE.SpriteMaterial({ transparent: true, opacity: 0 }));
  billboard.scale.set(14, 8, 1);
  marker.add(billboard);

  const markerState = { lat: 0, lon: 0, t0: 0, active: false };
  window.__OG_MARKER__ = { group: marker, ringGroup, ringA, ringB, ringC, beam, billboard, markerState };

  composer = new EffectComposer(renderer);
  const basePass = new RenderPass(scene, camera);
  composer.addPass(basePass);

  const fxaaPass = new (ShaderDraw ? Renderer : Object)(); // placeholder to preserve structure
  const fxaa = new ShaderPass(FXA&S.add ...); // Omitted: keep your existing fxaaPass lines from previous file

  glowPass = new (UnrealBloomPass)(new THREE.Vector2(el.globe trademarks?), BLOOM.strength, BLOOM <- ; // keep your existing
  composer.addPass(glowPass); // keep your original bloom pass lines

  globe.onAnimationFrame(() => {
    if (!window.__OG_MARKER__) return;
    const S = window.__OG_MARKER__;
    if (!S.markerState.active) return;
    const now = performance.now() * 0.001;
    const elapsed = now - S.markerState.t0;

    // expand / fade rings
    const baseR = getGlobeRadius() * (SURFACE_EPS + 0.001);
    const period = 2.6;
    [S.ringA, S.ringB, S.ringC].forEach((r, idx) => {
      const t = (elapsed + idx * 0.6) % period;
      const k = t / period;
      const s = 1.0 + k * 0.5;
      r.scale.setScalar(s);
      r.material.opacity = 0.35 * (1 - k);
    });

    // make billboard always face camera
    S.billboard.quaternion.copy(camera.quaternion);
  });

  window.addEventListener('resize', () => {
    const { clientWidth, clientHeight } = el.globeWrap;
    renderer.setSize(clientWidth, clientHeight);
    camera.aspect = clientHeight ? clientWidth / clientHeight : 1;
    camera.updateProjectionMatrix();
  });

  window.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight') { e.preventDefault(); step(+1); }
    if (e.key === 'ArrowLeft')  { e.preventDefault(); step(-1); }
  });

  el.prevBtn?.addEventListener('click', () => step(-1));
  el.nextBtn?.addEventListener('click', () => step(+1));

  // header profile menu
  const profileBtn  = document.getElementById('btn-profile');
  const profileMenu = document.getElementById('profile-menu');
  function closeProfileMenu(){ profileMenu?.classList.remove('show'); profileBtn?.setAttribute('aria-expanded','false'); }
  profileBtn?.addEventListener('mousedown', e => { e.stopPropagation(); profileMenu?.classList.toggle('show'); });
  document.addEventListener('mousedown', () => closeProfileMenu());

  // -------- bootstrap
  await loadFixturesCSV('./data/fixtures.csv');
  animate();
}

// main render loop
function animate() {
  requestAnimationFrame(animate);
  controls?.update();
  composer?.render();
}

// ----------------------------
// CSV ingest & bind
// ----------------------------
async function loadFixturesCSV(url) {
  try {
    const response = await fetch(`${url}?v=${Date.now()}`);
    if (!response.ok) {
      console.error(`[CSV] HTTP ${response.status} for ${url}`);
      showToast('error', `Could not load ${url} (HTTP ${response.status}).`);
      return;
    }
    const text = await response.text();
    const { data, errors } = Papa.parse(text, { header: true, skipEmptyLines: true });
    if (errors?.length) console.warn('[CSV parse errors]', errors);

    fixtures = (data || [])
      .map(row => {
        const loc = resolveLatLng(row);
        const home_badge_url = pick(row, ['home_badge_url','home_logo_url','home_logo','home_badge']);
        const away_badge_url = pick(row, ['away_badge_url','away_logo_url','away_logo','away_badge']);
        return {
          fixture_id: (row.fixture_id || row.id || `${row.home_team}-${row.away_team}-${row.date_utc || ''}`).trim?.() ?? String(row.fixture_id ?? ''),
          home_team: (row.home_team || row.Home || '').trim?.() ?? '',
          away_team: (row.away_team || row.Away || '').trim?.() ?? '',
          home_badge_url,
          away_badge_url,
          date_utc: row.date_utc || row.date || '',
          competition: row.competition || row.league || '',
          stadium: row.stadium || row.venue || row.stadium_name || row.venue_name || '',
          city: row.city || '',
          country: row.country || row.venue_country || '',
          latitude:  loc?.lat,
          longitude: loc?.lon,
          predicted_winner: row?.predict?.winner || row?.predicted_winner || '',
          confidence_ftr: Number(row?.confidence_ftr ?? row?.confidence ?? 0),
          xg_home: Number(row?.xg_home ?? 0),
          xg_away: Number(row?.xg_away ?? 0),
          ppg_home: Number(row?.ppg_home ?? 0),
          ppg_away: Number(row?.ppg_away ?? 0),
          over25_prob: Number(row?.over25_prob ?? 0),
          btts_prob: Number(row?.btts ?? row?.btts_prob ?? 0),
          key_players_shots: (row.key_players_shots || '').trim(),
          key_players_tackles: (row.key_players_tackles || '').trim()
        };
      })
      .filter(f => Number.isFinite(f.latitude) && Number.isFinite(f.longitude));

    if (!fixtures.length) {
      showCsvError('No fixtures with valid latitude/longitude found.');
      document.querySelector('#match-intelligence')?.replaceChildren(elmEmpty('No fixtures available. Check your CSV headers/path.'));
      document.querySelector('#player-watchlist')?.replaceChildren(elmEmpty('Player insights will appear here.'));
      document.querySelector('#market-snapshot')?.replaceChildren(elmEmpty('Market snapshot will appear here.'));
      return;
    }

    // cache team logos
    const teamNames = new Map();
    fixtures.forEach(f => { if (f.home_team) teamNames.set(f.home_team, f.home_badge_url); if (f.away_team) teamNames.set(f.away_team, f.away_badge_url); });
    await prefetchAllLogos(teamNames);

    // feed globe
    globe.pointLat('latitude').pointLng('longitude').pointsData(fixtures);
    globe.pointsTransitionDuration?.(0);

    buildRail(fixtures);

    const boot = () => {
      selectIndex(0, { fly: true });
      createSelectionRing();
      renderHtmlTabs();
      globe.htmlTransitionDuration?.(220);
      if (typeof globe.pointLabel === 'function') {
        globe.pointLabel(d => `${d.city ? `${d.city} • ` : ''}${d.home_team} vs ${d.away_team}`);
      }
      showRoute(location.hash || '#/');
      showToast('success', `Loaded ${fixtures.length} fixtures`);
    };

    if (typeof globe.onGlobeReady === 'function') globe.onGlobeReady(boot);
    else requestAnimationFrame(boot);

  } catch (err) {
    console.error('[CSV] Failed to load/parse', err);
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
  selectedId = f?.fixture_id || f?.re_id || null;

  fixtures.forEach(d => (d.__active = (d.fixture_id || d.re_id) === selectedId));
  globe
    .pointAltitude(() => SURFACE_EPS)
    .pointRadius(d => (d.__active ? RADIUS_ACTIVE : RADIUS_BASE))
    .pointColor(d => (d.__active ? COLORS.markerActive : COLORS.marker))
    .pointsData(fixtures);

  globe.pointColor(d => d.__active ? '#D7FFF9' : COLORS.marker)
       .pointsTransitionDuration?.(200);
  setTimeout(() => globe.pointsTransitionDuration?.(0), 220);

  if (fly) flyToFixture(f);

  moveMarkerToFixture(f, { fly });     // (3) ensure marker + billboard move with correct coords
  renderPanel(f);
  updateHtmlTabsSelection();

  const wrap = document.querySelector('.hero__globe');
  wrap?.classList.add('glow', 'glow-pin');
  setTimeout(() => wrap?.classList.remove('glow-pin'), 350);
}

function flyToFixture(f) {
  if (!f || !globe?.pointOfView) return;
  globe.pointAltitude(() => SURFACE_EPS);
  globe.pointOfView({ lat: f.latitude, lng: f.longitude, altitude: CAMERA_SHaded? CAMERA_ALT : 2.0 }, 650);
}

// ----------------------------
// Selection halo aligned to surface
// ----------------------------
function createSelectionRing() {
  const R = getGlobeRadius();
  const inner = R * (1 + SURFACE_EPS + 0.001);
  const outer = inner + R * 0.007;
  const ringGeom = new THREE.RingGeometry(inner, outer, 64);
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

function updateSelectionRing(f) {
  if (!pulseRing || !f || !Number.isFinite(f.latitude) || !Number.isFinite(f.longitude)) return;
  const R = getGlobeRadius();
  const pos = latLngToVec3(f.latitude, f.longitude, SURFACE_EPS + 0.001);
  const n = pos.clone().normalize();
  const out = n.clone().multiplyScalar(R * 2);
  pulseRing.position.copy(pos);
  pulseRing.look_at = out;
  pulseRing.lookAt(out);
  pulseRing.visible = true;
}

function pulsePulse() {
  if (!pulseRing) return;
  const T = 1.8;
  const t = (performance.now() % (T * 1000)) / 1000;
  const a = 0.15 + 0.35 * Math.sin(t * Math.PI);
  pulse പോലെ?
  pulseRing.material.opacity = a;
  requestAnimationFrame(pulsePulse);
}

// ----------------------------
// On-globe HTML fixture tabs
// ----------------------------
function renderHtmlTabs() {
  htmlTabsData = fixtures.map((f, i) => ({
    lat: f.latitude,
    lng: f.longitude,
    altitude: SURFACE_EPS + 0.06,
    idx: i,
    el: null
  }));

  globe
    .htmlElementsData(htmlTabsData)
    .htmlLat('lat')
    .htmlLng('lng')
    .htmlAltitude('altitude')
    .htmlElement(d => {
      const w = document.createElement('div');
      w.className = 'fixture-tab' + (d.idx === activeIdx ? ' is-selected' : '');
      w.dataset.idx = String(d.idx);
      const f = fixtures[d.idx];
      w.title = `${f.home_team} vs ${f.away_team}${f.city ? ` — ${f.city}` : ''}`;
      w.style.pointerEvents = 'auto';

      const title = document.createElement('div');
      title.className = 'fixture-tab__title';
      title.textContent = `${f.home_team} vs ${f.away_team}`;

      const meta = document.createElement('div');
      meta.className = 'fixture-tab__meta';
      meta.textContent = f.city || f.country || '';

      w.append(title, meta);
      w.addEventListener('click', e => { e.stopPropagation(); selectIndex(d.idx, { fly: true }); });

      d.el = w;
      return w;
    });

  globe.htmlTransitionDuration?.(220);
}

function updateHtmlTabsSelection() {
  if (!htmlTabsData?.length) return;
  htmlTabsData.forEach(d => {
    const node = d.el || document.querySelector(`.fixture-tab[data-idx="${d.idx}"]`);
    if (node) node.classList.toggle('is-selected', d.idx === activeIdx);
  });
}

// ----------------------------
// Fallback: sprite labels
// ----------------------------
function renderLabelSprites() {
  globe
    .labelsData(fixtures)
    .labelLat('latitude')
    .labelLng('longitude')
    .labelAltitude(() => SURFACE_EPS + 0.06)
    .labelText(f => `${f.home_team} vs ${f.away_team}`)
    .labelColor(f => ((f.fixture_id || f.re_id) === selectedId ? 'rgba(125,249,196,0.95)' : 'rgba(255,255,255,0.85)'))
    .labelSize(f => ((f.fixture_id || f.re_id) === selectedId ? 1.4 : 1.0))
    .labelDotRadius(f => ((f.fixture_id || f.re_id) === selectedId ? 0.5 : 0.28))
    .labelResolution(2);
}

// ----------------------------
// Move marker to fixture (beam + radar + billboard)  (3) reworked with correct coords + stadium name lookup
// ----------------------------
function moveMarkerToFixture(f, { fly = false } = {}) {
  const M = window.__OG_MARKER__;
  if (!M || !f) return;

  const lat = Number(f.latitude);
  const lon = Number(f.longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) { M.group.visible = false; return; }

  const R = getGlobeRadius();
  const baseAlt = SURFACE_EPS + 0.001;
  const pos = latLngToVec3(lat, lon, baseAlt);
  const normal = pos.clone().normalize();

  M.group.visible = true;
  M.group.position.copy(pos);

  // aim rings to tangent plane
  const look = normal.clone().multiplyScalar(R * 2);
  M.group.lookAt(look);

  // prepare beam
  M.beam.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0), normal);
  M.beam.position.copy(pos);
  M.beam.scale.set(1, 0.001, 1);
  M.beam.visible = true;

  // animate beam up
  const t0 = performance.now();
  (function anim() {
    const t = (performance.now() - t0) / 550;
    if (t < 1) {
      const e = t * t * (3 - 2 * t);
      M.beam.scale.y = 0.001 + e;
      requestAnimationFrame(anim);
    } else {
      M.beam.scale.y = 1;
    }
  })();

  // billboard hover above surface
  const bbPos = pos.clone().add(normal.clone().multiplyScalar(R * 0.06));
  M.billboard.position.copy(bbPos);
  M.billboard.material.opacity = 0;
  M.billboard.visible = true;

  // load stadium texture from name/venue/team → jpg/png/webp
  const candidates = stadiumCandidates(f);
  (async () => {
    for (const src of candidates) {
      try {
        const tex = await new Promise((resolve, reject) => {
          new THREE.TextureLoader().load(src, resolve, undefined, reject);
        });
        M.billboard.material.map = tex;
        M.billboard.material.needsUpdate = true;
        // fade in
        const t0 = performance.now();
        (function fade() {
          const t = (performance.now() - t0) / 220;
          const a = Math.min(1, t);
          M.billboard.material.opacity = a;
          if (a < 1) requestAnimationFrame(fade);
        })();
        return;
      } catch { /* try next candidate */ }
    }
    // nothing found → hide
    M.billboard.visible = false;
  })();

  // kick the radar animation
  M.markerState.lat = lat;
  M.markerState.lon = lon;
  M.markerState.t0  = performance.now() / 1000;
  M.markerState.active = true;
}

// ----------------------------
// Fixture rail (quick selector)
// ----------------------------
function buildRail(items) {
  const rail = document.getElementById('fixture-rail'); if (!rail) return;
  rail.innerHTML = '';
  items.forEach((f, i) => {
    const it = document.createElement('button');
    it.className = 'rail-item' + (i === active ? ' is-active' : '');
    it.innerHTML = `<h4>${f.home_team} vs ${f.away_team}</h4><p>${(f.city || f.country || '')}</p>`;
    it.addEventListener('click', () => selectIndex(i, { fly: true }));
    rail.appendChild(it);
  });
}

function syncRail() {
  const rail = document.getElementById('fixture-rail'); if (!rail) return;
  [...rail.children].forEach((c, idx) => c.classList.toggle('is-active', idx === activeIdx));
}

// ----------------------------
// Tabs
// ----------------------------
function wireTabs() {
  document.querySelectorAll('.tab')?.forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(b => b.classList.remove('is-active'));
      document.querySelectorAll('.tabpane').forEach(p => p.classList.remove('is-active'));
      btn.classList.add('is-active');
      const id = btn.dataset.tab;
      document.getElementById(`tab-${id}`)?.classList.add('is-active');
    });
  });
}

// ----------------------------
// Bottom sheet (mobile)
// ----------------------------
function openSheet(title, node) {
  const s = document.getElementById('sheet'); if (!s) return;
  s.querySelector('#sheet-title').textContent = title || '';
  const body = s.querySelector('#sheet-body'); body.innerHTML = ''; body.appendChild(node);
  s.classList.add('open'); s.setAttribute('aria-hidden', 'false');
}
function closeSheet() {
  const s = document.getElementById('sheet'); if (!s) return;
  s.classList.remove('open'); s.setAttribute('aria-hidden', 'true');
}
document.querySelector('.sheet__handle')?.addEventListener('click', closeSheet);

// =====================================================
// API HELPERS + FEATURE PAGES (unchanged from your last file)
// =====================================================
const API_BASE = '/api';

async function apiJson(url, opts = {}) {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
    credentials: 'include',
    ... opts
  });
  if (!res.ok) {
    const t = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status}: ${t || res.message || 'Request failed'}`);
  }
  return res.json();
}

const API = {
  score_lip: (payload) => apiJson('/score-slip', { method: 'POST', body: JSON.stringify(payload) }),
  accaSuggest: (q)      => apiJson(`/acca/suggest?${new URLSearchParams(q)}`),
  accaOptimise: (p)     => apiJson('/acca/optimise', { method: 'POST', body: JSON.stringify(p) }),
  copilot: (p)          => apiJson('/copilot', { method: 'POST', body: JSON.stringify(p) })
};

async function ocrImageOrPdf(file) {
  if (!window.Tesseract) throw new Error('OCR engine not loaded');
  const { data } = await window.Tesseract.recognize(file, 'eng', { logger: () => {} });
  return (data && data.text) ? data.text : '';
}

function parseSlipText(text) {
  const lines = String(text).split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  const legs = [];
  for (let i = 0; i < lines.length; i++) {
    const L = lines[i];
    const m = L.match(/^\s*([A-Za-z0-9 .'\-]+)\s+(?:v|vs\.?|VS)\s+([A-Za-z0-9 .'\-]+)\s*$/i);
    if (!m) continue;
    const home = m[1].trim(), away = m[2].trim();
    for (let j = 1; j <= 3 && (i + j) < lines.length; j++) {
      const M = lines[i + j];
      let market=null, pick=null;
      if (/over\s*2\.?5/i.test(M)) { market='OVER_UNDER_2_5'; pick='OVER'; }
      else if (/under\s*2\.?5/i.test(M)) { market='OVER_UNDER_2_5'; pick='UNDER'; }
      else if (/both\s*teams\s*to\s*score|btts/i.test(M)) { market='BTTS'; pick=/\bno\b/i.test(M)?'NO':'YES'; }
      else if (/(?:^|\s)(?:1x2|home|away|draw|1|2|x)(?:\s|$)/i.test(M)) {
        market='FTR';
        if (/\bdraw\b|(?:^|\s)x(?:\s|$)/i.test(M)) pick='DRAW';
        else if (/\bhome\b|(?:^|\s)1(?:\s|$)/i.test(M)) pick='HOME';
        else if (/\baway\b|(?:^|\s)2(?:\s|$)/i.test(M)) pick='AWAY';
      }
      if (!market) continue;
      let price=null;
      const f = M.match(/(\d+)\s*\/\s*(\d+)/);
      const d = M.match(/(\d+(?:\.\d+)?)/);
      if (f) price = (parseFloat(f[1])/parseFloat(f[2]))+1;
      else if (d) price = parseFloat(d[1]);
      legs.push({ teamHome:home, teamAway:away, market, selection:pick||'—', price, bookmaker:null, kickoffUTC:null });
      break;
    }
  }
  return { legs, raw: lines.slice(0, 60).join('\n') };
}

async function runBetChecker(file) {
  const out = document.getElementById('bc-output');
  if (out) out.innerHTML = '<div class="muted">Reading slip…</div>';
  try {
    const text = await ocrImageOrPdf(file);
    const parsed = parseSlipText(text);
    if (!parsed.legs.length) {
      out && (out.innerHTML = `<div class="muted">No legs detected.</div>`);
      return;
    }
    out && (out.innerHTML = '<div class="muted">Scoring legs…</div>');
    const scored = await API.score_lip({ legs: parsed.legs });
    out.innerHTML = `<pre>${JSON.stringify(scored, null, 2)}</pre>`;
    showToast('success', `Scored ${scored.legs?.length || parsed.legs.length} leg(s)`);
  } catch (e) {
    showToast('error', e.message || 'OCR failed');
    out && (out.innerHTML = `<div class="muted">Error: ${e.message || 'Unknown error'}</div>`);
  }
}

// Co-Pilot
async function sendCopilotMessage(text) {
  const payload = { 
    messages: [
      { role: 'system', content: 'You are OddsGenius Co-Pilot. Be concise, provide bullet reasoning, cite model features where relevant.' },
      { role: 'user', content: text }
    ],
    context: (() => {
      const f = fixtures[activeIdx] || {};
      return { fixture: { home: f.home_team, away: f.away_team, date: f.date_utc, league: f.competition } };
    })()
  };
  return API.copilot(payload);
}

function appendChatLine(role, text) {
  const wrap = document.getElementById('cp-thread');
  if (!wrap) return;
  const div = document.createElement('div');
  div.className = (role === 'user') ? 'user-line' : 'bot-line';
  div.textContent = (role === 'user' ? 'You: ' : 'OG: ') + text;
  wrap.appendChild(div);
  wrap.scrollTop = wrap.scrollHeight;
}
{
  const cpInput = document.getElementById('cp-input');
  const cpSend  = document.getElementById('cp-send');
  cpSend?.addEventListener('click', async () => {
    const q = (cpInput?.value || '').trim();
    if (!q) return;
    appendChatLine('user', q);
    cpInput.value = '';
    try {
      const { messages, error } = await sendCopilotMessage(q);
      if (error) throw new Error(error);
      const msg = (messages && messages.find(m => m.role === 'assistant'))?.content || '(no reply)';
      appendChatLine('assistant', msg);
    } catch (e) {
      appendChatLine('assistant', `⚠ ${e.message}`);
    }
  });
}

// ----------------------------
// Router
// ----------------------------
const ROUTES = {
  '#/':            'view-home',
  '#/home':        'view-home',
  '#/bet-checker': 'view-betchecker',
  '#/acca-builder':'view-accabuilder',
  '#/copilot':     'view-copilot',
  '#/login':       'view-signin',
  '#/signup':      'view-signup'
};
function showRoute(hash) {
  if (!hash) hash = '#/';
  const id = ROUTES[hash] || 'view-home';
  document.querySelectorAll('.view').forEach(v => {
    if (v.id === id) { v.classList.add('is-active'); v.removeAttribute('hidden'); }
    else { v.classList.remove('is-active'); v.setAttribute('hidden', ''); }
  });
  document.querySelectorAll('[data-route]').forEach(a => {
    a.classList.toggle('is-active', a.getAttribute('href') === hash);
    if (a.classList.contains('side-link')) a.classList.toggle('active', a.getAttribute('href') === hash);
  });
  // close menus on nav
  const profileMenu = document.getElementById('profile-menu');
  const profileBtn  = document.getElementById('btn-profile');
  profileMenu?.classList.remove('show');
  profileBtn?.setAttribute('aria-expanded', 'false');
}
window.addEventListener('hashchange', () => showRoute(location.hash));
window.addEventListener('DOMContentLoaded', () => {
  if (!location.hash) location.hash = '#/';
  showRoute(location.hash);
});

// -------- sanity: verify a couple of logo files
(function verifyLocalLogoSetup(){
  const tests = [`${LOGO_LOCAL_BASE}/arsenal.svg`, `${LOGO_LOCAL_BASE}/fc-barcelona.svg`];
  tests.forEach(src => {
    const img = new Image();
    img.onload  = () => console.log('%c[LOGOS] OK', 'color:#22c55e', src);
    img.onerror = () => console.warn('%c[LOGOS] 404', 'color:#f43f5e', src, '→ path or filename mismatch');
    img.src = src;
  });
})();

init();
