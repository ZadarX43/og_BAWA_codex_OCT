// app.module.js
// Odds Genius — Full UI (Globe + Filters + Panels + OCR + Acca + Co-Pilot)

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

// -----------------------------------------
// DOM refs
// -----------------------------------------
const el = {
  globeWrap:     document.getElementById('globe-container'),
  insights:      document.getElementById('insights-content'),
  fixtureTitle:  document.getElementById('fixture-title'),
  fixtureContext:document.getElementById('fixture-context'),
  matchList:     document.getElementById('match-intelligence'),
  watchlist:     document.getElementById('player-watchlist'),
  market:        document.getElementById('market-snapshot'),
  deepBtn:       document.getElementById('deep-dive-btn'),
  homeBadge:     document.getElementById('home-badge'),
  awayBadge:     document.getElementById('away-badge'),

  // Date & league filters
  dateToday:    document.querySelector('[data-range="today"]'),
  dateTomorrow: document.querySelector('[data-range="tomorrow"]'),
  dateWeekend:  document.querySelector('[data-range="weekend"]'),
  datePrev:     document.getElementById('cal-prev'),
  dateNext:     document.getElementById('cal-next'),
  dateA:        document.getElementById('date-day-a'),
  dateB:        document.getElementById('date-day-b'),
  leagueChips:  document.getElementById('league-chips'),

  // Competition strip
  compWrap:      document.getElementById('comp-accuracy'),
  compName:      document.getElementById('comp-name'),
  compLogo:      document.getElementById('comp-logo'),
  compTraffic:   document.getElementById('comp-traffic')
};

// -----------------------------------------
// Globals / tuning
// -----------------------------------------
let ThreeGlobeCtor;
let globe;
let renderer, scene, camera, controls, composer;
let fixtures = [];             // all
let visibleFixtures = [];      // filtered
let htmlTabsData = [];
let MARKER = null;             // custom marker group object

const SURFACE_EPS   = 0.009;
const RADIUS_BASE   = 0.014;
const RADIUS_ACTIVE = 0.040;
const CAMERA_ALT    = 2.0;
const BLOOM = { strength: 0.9, radius: 0.6, threshold: 0.75 };

const COLORS = {
  marker:         '#A7FFF6',
  markerInactive: '#8CEFE5',
  markerActive:   '#CFFFFA',
  ring:           '#9EE7E3'
};

// ---- UI demo “today”
const UI = {
  anchorISO: '2023-11-28', // fixed demo “today”
  offsetDays: 0,           // 0=today, 1=tomorrow, etc
  rangeDays: 1,            // 1 or 2 (weekend)
  league: 'ALL',
  leagues: []
};

// -----------------------------------------
// Utilities
// -----------------------------------------
const clamp01 = v => Math.max(0, Math.min(1, v));
const easeInOut = t => t*t*(3-2*t);

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
  for (const k of keys) { const v = (row[k] ?? '').toString().trim(); if (v) return v; }
  return '';
}

// ---- Dates
const MS_DAY = 24*60*60*1000;
function baseDate(){ return new Date(Date.parse(`${UI.anchorISO}T00:00:00Z`) + UI.offsetDays*MS_DAY); }
function datePlusDays(base, n){ return new Date(base.getTime() + n*MS_DAY); }
function fmtDay(d){ return String(d.getUTCDate()).padStart(2,'0'); }
function isoDay(d){ return d.toISOString().slice(0,10); }

// ---- THREE helpers
function getGlobeRadius(){
  if (globe?.getGlobeRadius) return globe.getGlobeRadius();
  const m = globe?.children?.find?.(c => c.geometry?.parameters?.radius);
  return m?.geometry?.parameters?.radius || 100;
}

// three-globe: phi = (90 - lat), theta = (180 - lon)
function latLngToUnit(latDeg, lonDeg){
  const phi   = THREE.MathUtils.degToRad(90 - latDeg);
  const theta = THREE.MathUtils.degToRad(180 - lonDeg);
  return new THREE.Vector3(
    Math.sin(phi) * Math.cos(theta),
    Math.cos(phi),
    Math.sin(phi) * Math.sin(theta)
  ).normalize();
}
function latLngToVec3(lat, lon, alt=0){
  const R = getGlobeRadius();
  const n = latLngToUnit(lat, lon);
  return n.clone().multiplyScalar(R*(1+alt));
}
function makeFallbackCanvasTexture(label='STADIUM'){
  const c = document.createElement('canvas');
  c.width = 512; c.height = 512;
  const g = c.getContext('2d');

  // soft gradient disc
  const r = 220, cx = 256, cy = 256;
  const grd = g.createRadialGradient(cx, cy, r*0.25, cx, cy, r);
  grd.addColorStop(0, 'rgba(255,255,255,0.95)');
  grd.addColorStop(1, 'rgba(125,249,196,0.20)');
  g.fillStyle = grd;
  g.beginPath(); g.arc(cx, cy, r, 0, Math.PI*2); g.fill();

  // label
  g.fillStyle = '#0b1f29';
  g.font = 'bold 46px Montserrat, system-ui, sans-serif';
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(label, cx, cy);

  const tex = new THREE.CanvasTexture(c);
  if ('colorSpace' in tex) tex.colorSpace = THREE.SRGBColorSpace;
  const maxAniso = renderer?.capabilities?.getMaxAnisotropy?.() || 1;
  tex.anisotropy = maxAniso;
  return tex;
}

// -----------------------------------------
// Logos (local-only) — club crests
// -----------------------------------------
const LOGO_LOCAL_BASE = './assets/assets/logos';

// Strip accents & punctuation for slugging (“København” -> “kobenhavn”)
function stripDiacritics(s = '') {
  try {
    return s.normalize('NFD').replace(/\p{Diacritic}/gu, '');
  } catch {
    return s.replace(/[\u0300-\u036f]/g, '');
  }
}

function slugLocal(name = '') {
  return stripDiacritics(String(name))
    .toLowerCase()
    .replace(/&/g, 'and')
    .replace(/[\u2019'’]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

// Explicit overrides mapped to your actual SVG filenames
const TEAM_LOGO_OVERRIDES = {
  // Accented / alt forms
  'Atlético Madrid'       : `${LOGO_LOCAL_BASE}/atletico-madrid.svg`,
  'Atletico Madrid'       : `${LOGO_LOCAL_BASE}/atletico-madrid.svg`,
  'Bayern München'        : `${LOGO_LOCAL_BASE}/bayern-munich.svg`,
  'Bayern Munich'         : `${LOGO_LOCAL_BASE}/bayern-munich.svg`,
  'Crvena Zvezda'         : `${LOGO_LOCAL_BASE}/red-star-belgrade.svg`,
  'Red Star Belgrade'     : `${LOGO_LOCAL_BASE}/red-star-belgrade.svg`,
  'København'             : `${LOGO_LOCAL_BASE}/fc-kobenhavn.svg`,
  'FC København'          : `${LOGO_LOCAL_BASE}/fc-kobenhavn.svg`,

  // Clubs exactly matching your files
  'AC Milan'              : `${LOGO_LOCAL_BASE}/ac-milan.svg`,
  'Arsenal'               : `${LOGO_LOCAL_BASE}/arsenal.svg`,
  'Barcelona'             : `${LOGO_LOCAL_BASE}/fc-barcelona.svg`,
  'FC Barcelona'          : `${LOGO_LOCAL_BASE}/fc-barcelona.svg`,
  'Borussia Dortmund'     : `${LOGO_LOCAL_BASE}/borussia-dortmund.svg`,
  'Celtic'                : `${LOGO_LOCAL_BASE}/celtic.svg`,
  'FC Porto'              : `${LOGO_LOCAL_BASE}/fc-porto.svg`,
  'Feyenoord'             : `${LOGO_LOCAL_BASE}/feyenoord.svg`,
  'Galatasaray'           : `${LOGO_LOCAL_BASE}/galatasaray.svg`,
  'Inter'                 : `${LOGO_LOCAL_BASE}/inter-milan.svg`,
  'Inter Milan'           : `${LOGO_LOCAL_BASE}/inter-milan.svg`,
  'Lazio'                 : `${LOGO_LOCAL_BASE}/lazio.svg`,
  'Lens'                  : `${LOGO_LOCAL_BASE}/lens.svg`,
  'Manchester City'       : `${LOGO_LOCAL_BASE}/manchester-city.svg`,
  'Man City'              : `${LOGO_LOCAL_BASE}/manchester-city.svg`,
  'Manchester United'     : `${LOGO_LOCAL_BASE}/manchester-united.svg`,
  'Man Utd'               : `${LOGO_LOCAL_BASE}/manchester-united.svg`,
  'Newcastle United'      : `${LOGO_LOCAL_BASE}/newcastle-united.svg`,
  'PSG'                   : `${LOGO_LOCAL_BASE}/paris-saint-germain.svg`,
  'Paris Saint-Germain'   : `${LOGO_LOCAL_BASE}/paris-saint-germain.svg`,
  'PSV'                   : `${LOGO_LOCAL_BASE}/psv.svg`,
  'RB Leipzig'            : `${LOGO_LOCAL_BASE}/rb-leipzig.svg`,
  'RB Salzburg'           : `${LOGO_LOCAL_BASE}/rb-salzburg.svg`,
  'Real Madrid'           : `${LOGO_LOCAL_BASE}/real-madrid.svg`,
  'Real Sociedad'         : `${LOGO_LOCAL_BASE}/real-sociedad.svg`,
  'Royal Antwerp'         : `${LOGO_LOCAL_BASE}/royal-antwerp.svg`,
  'Sevilla'               : `${LOGO_LOCAL_BASE}/sevilla-fc.svg`,
  'Sevilla FC'            : `${LOGO_LOCAL_BASE}/sevilla-fc.svg`,
  'Shakhtar Donetsk'      : `${LOGO_LOCAL_BASE}/shakhtar-donetsk.svg`,
  'SL Benfica'            : `${LOGO_LOCAL_BASE}/sl-benfica.svg`,
  'Sporting Braga'        : `${LOGO_LOCAL_BASE}/sporting-braga.svg`,
  'SSC Napoli'            : `${LOGO_LOCAL_BASE}/ssc-napoli.svg`,
  'Napoli'                : `${LOGO_LOCAL_BASE}/ssc-napoli.svg`,
  'Union Berlin'          : `${LOGO_LOCAL_BASE}/union-berlin.svg`,
  'Young Boys'            : `${LOGO_LOCAL_BASE}/young-boys.svg`,
};

// Generate candidate .svg paths based on the team name
function localLogoCandidates(teamName = '') {
  const name = String(teamName || '').trim();
  if (!name) return [];

  // 1) Exact override first
  const override = TEAM_LOGO_OVERRIDES[name];
  if (override) return [override];

  // 2) Slug-based guesses (for anything not in overrides)
  const s = slugLocal(name);  // e.g. "FC Barcelona" -> "fc-barcelona"
  if (!s) return [];

  const b = LOGO_LOCAL_BASE;
  return Array.from(new Set([
    `${b}/${s}.svg`,
    `${b}/${s.replace(/^fc-/, '')}.svg`,   // "fc-barcelona" -> "barcelona.svg"
    `${b}/${s.replace(/-fc$/, '')}.svg`,   // "barcelona-fc" -> "barcelona.svg"
  ]));
}

function initialsFor(name = '') {
  const p = String(name).trim().split(/\s+/);
  return p.length ? (p[0][0] + (p[1]?.[0] || '')).toUpperCase() : '';
}

// *** Simple badge loader – no races, loud logging ***
function setBadgeLocal(elm, _urlFromCsv, teamName = '') {
  if (!elm) return;

  // show initials while loading
  elm.classList.remove('has-logo');
  elm.innerHTML = '';
  elm.textContent = initialsFor(teamName);

  const candidates = localLogoCandidates(teamName);
  console.log('[badge] candidates', { teamName, candidates });

  if (!candidates.length) return;

  const src = candidates[0]; // for your known clubs, this should be the correct SVG
  const img = new Image();
  img.alt = teamName;

  img.onload = () => {
    console.log('[badge] success', teamName, '→', src);
    elm.innerHTML = '';
    elm.appendChild(img);
    elm.classList.add('has-logo');
  };

  img.onerror = () => {
    console.log('[badge] error', teamName, '→', src);
    // keep initials fallback
  };

  img.src = src;
}
// -----------------------------------------
// Competition badges (for comp strip under globe)
// -----------------------------------------

// If you later add real SVGs, map them here:
const COMP_LOGO_OVERRIDES = {
  // Example if you add these files:
  // 'UEFA Champions League': './assets/assets/logos/ucl.svg',
  // 'UEFA Europa League'   : './assets/assets/logos/uel.svg',
};

const COMP_BADGE_CACHE = new Map();

function makeCompetitionBadgeDataUrl(label = 'LEAGUE') {
  const c = document.createElement('canvas');
  const size = 256;
  c.width = size;
  c.height = size;
  const g = c.getContext('2d');

  g.clearRect(0, 0, size, size);

  // circular gradient badge
  const grd = g.createLinearGradient(0, 0, size, size);
  grd.addColorStop(0, '#0f766e');
  grd.addColorStop(1, '#22c55e');
  g.fillStyle = grd;

  const r = size * 0.42;
  g.beginPath();
  g.arc(size / 2, size / 2, r, 0, Math.PI * 2);
  g.fill();

  // league initials
  const parts = String(label || '').split(/\s+/).filter(Boolean);
  const initials = (
    (parts[0]?.[0] || 'L') +
    (parts[1]?.[0] || 'G')
  ).toUpperCase();

  g.fillStyle = '#e5f9ff';
  g.font = 'bold 72px Montserrat, system-ui, sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(initials, size / 2, size / 2);

  return c.toDataURL('image/png');
}

function setCompetitionLogo(leagueName) {
  if (!el.compLogo) return;

  const name = String(leagueName || '').trim();
  if (!name) {
    el.compLogo.removeAttribute('src');
    el.compLogo.style.visibility = 'hidden';
    return;
  }

  // 1) If we have a real logo mapped, use it
  const override = COMP_LOGO_OVERRIDES[name];
  if (override) {
    el.compLogo.src = override;
    el.compLogo.style.visibility = 'visible';
    return;
  }

  // 2) Otherwise generate/cache a fallback badge with initials
  let url = COMP_BADGE_CACHE.get(name);
  if (!url) {
    url = makeCompetitionBadgeDataUrl(name);
    COMP_BADGE_CACHE.set(name, url);
  }
  el.compLogo.src = url;
  el.compLogo.style.visibility = 'visible';
}

// -----------------------------------------
// Stadium billboard (local-only)
// -----------------------------------------
const STADIUM_BASE = './assets/stadiums';

// Only the files you said you uploaded:
const STADIUM_OVERRIDES = {
  'AC Milan':             'ac-milan.jpg',
  'Arsenal':              'arsenal.jpg',
  'Galatasaray':          'galatasaray.jpg',
  'Lazio':                'lazio.jpg',
  'Manchester City':      'man-city.jpg',
  'PSG':                  'psg.jpg',
  'Paris Saint-Germain':  'psg.jpg',
  'Real Madrid':          'real-madrid.jpg',
  'Sevilla FC':           'sevilla.jpg',
  'Shakhtar Donetsk':     'shakhtar-donetsk.jpg',
  'Young Boys':           'young-boys.jpg',
};

// Return 0 or 1 local path; no blind guesses that 404
function stadiumCandidates(f) {
  const name = String(f?.home_team || '').trim();
  const file = STADIUM_OVERRIDES[name];
  return file ? [`${STADIUM_BASE}/${file}`] : [];
}

// Texture queue/cache (unchanged)
const __TEX_CACHE = new Map();
const __TEX_WAIT  = new Map();
function loadTextureQueued(url){
  if (__TEX_CACHE.has(url)) return Promise.resolve(__TEX_CACHE.get(url));
  if (__TEX_WAIT.has(url))  return __TEX_WAIT.get(url);

  const p = new Promise((resolve, reject)=>{
    new THREE.TextureLoader().load(
      url,
      tex => { __TEX_CACHE.set(url, tex); __TEX_WAIT.delete(url); resolve(tex); },
      undefined,
      err => { __TEX_WAIT.delete(url); reject(err); }
    );
  });
  __TEX_WAIT.set(url, p);
  return p;
}

// -----------------------------------------
// Competition accuracy strip
// -----------------------------------------
const DEMO_FTR = 0.87; // fixed 87% per request
function getCompetitionSnapshot(league){
  const rows = league && league!=='ALL'
    ? fixtures.filter(f => (f.competition||'').toLowerCase() === league.toLowerCase())
    : fixtures.slice();

  const avg = a => a.length ? a.reduce((x,y)=>x+y,0)/a.length : 0;
  const pct = x => Math.round(x*100);
  return {
    n: rows.length,
    ftr:   pct(avg(rows.map(r => +r.confidence_ftr||0))),
    over25:pct(avg(rows.map(r => +r.over25_prob ||0))),
    btts:  pct(avg(rows.map(r => +r.btts_prob   ||0)))
  };
}
function renderCompetitionAccuracy(league){
  if (!el.compWrap) return;
  const stats = getCompetitionSnapshot(league);

  const name = league || '—';
  if (el.compName) el.compName.textContent = name;

  // NEW: set competition badge logo
  setCompetitionLogo(name);

  if (el.compTraffic){
    el.compTraffic.innerHTML = `
      <span class="light light--green">FTR ${Math.round(DEMO_FTR*100)}%</span>
      <span class="light light--blue">O2.5 ${stats.over25||0}%</span>
      <span class="light light--amber">BTTS ${stats.btts||0}%</span>`;
  }
}


// -----------------------------------------
// Date & League filter UI  [A: prev/next restored here]
// -----------------------------------------
function buildDateStrip(){
  const base = baseDate();
  const dayA = base;
  const dayB = datePlusDays(base,1);

  if (el.dateA) { el.dateA.textContent = fmtDay(dayA); el.dateA.dataset.iso = isoDay(dayA); }
  if (el.dateB) { el.dateB.textContent = fmtDay(dayB); el.dateB.dataset.iso = isoDay(dayB); }

  [el.dateToday, el.dateTomorrow, el.dateWeekend]
    .filter(Boolean).forEach(b => b.classList.remove('is-active'));

  if (UI.rangeDays===1 && UI.offsetDays===0 && el.dateToday)    el.dateToday.classList.add('is-active');
  if (UI.rangeDays===1 && UI.offsetDays===1 && el.dateTomorrow) el.dateTomorrow.classList.add('is-active');
  if (UI.rangeDays>=2 && el.dateWeekend)                         el.dateWeekend.classList.add('is-active');

  const t = document.getElementById('cal-title');
  if (t) {
    const mo = base.toLocaleString(undefined, { month: 'long', year: 'numeric', timeZone: 'UTC' });
    t.textContent = mo;
  }
}

function bindDateControls(){
  el.dateToday?.addEventListener('click', ()=>{
    UI.offsetDays = 0; UI.rangeDays = 1;
    buildDateStrip(); applyFiltersAndRender();
  });
  el.dateTomorrow?.addEventListener('click', ()=>{
    UI.offsetDays = 1; UI.rangeDays = 1;
    buildDateStrip(); applyFiltersAndRender();
  });
  el.dateWeekend?.addEventListener('click', ()=>{
    const b = baseDate(); const dow = b.getUTCDay(); const toSat = (6 - dow + 7) % 7;
    UI.offsetDays = toSat; UI.rangeDays = 2;
    buildDateStrip(); applyFiltersAndRender();
  });

  el.datePrev?.addEventListener('click', ()=>{
    UI.offsetDays -= UI.rangeDays; buildDateStrip(); applyFiltersAndRender();
  });
  el.dateNext?.addEventListener('click', ()=>{
    UI.offsetDays += UI.rangeDays; buildDateStrip(); applyFiltersAndRender();
  });

  // Prev/Next fixture (restored)
  document.getElementById('nav-prev')?.addEventListener('click', ()=>{
    if (!visibleFixtures.length) return;
    const cur = visibleFixtures.findIndex(f=>f.__active);
    const idx = (cur - 1 + visibleFixtures.length) % visibleFixtures.length;
    selectIndex(idx, { fly:true });
  });
  document.getElementById('nav-next')?.addEventListener('click', ()=>{
    if (!visibleFixtures.length) return;
    const cur = visibleFixtures.findIndex(f=>f.__active);
    const idx = (cur + 1) % visibleFixtures.length;
    selectIndex(idx, { fly:true });
  });

  el.dateA?.addEventListener('click', ()=>{
    const iso = el.dateA.dataset.iso; if (!iso) return;
    UI.offsetDays = Math.round((Date.parse(`${iso}T00:00:00Z`) - Date.parse(`${UI.anchorISO}T00:00:00Z`))/MS_DAY);
    UI.rangeDays = 1; buildDateStrip(); applyFiltersAndRender();
  });
  el.dateB?.addEventListener('click', ()=>{
    const iso = el.dateB.dataset.iso; if (!iso) return;
    UI.offsetDays = Math.round((Date.parse(`${iso}T00:00:00Z`) - Date.parse(`${UI.anchorISO}T00:00:00Z`))/MS_DAY);
    UI.rangeDays = 1; buildDateStrip(); applyFiltersAndRender();
  });
}

function buildLeagueChips(){
  if (!el.leagueChips) return;
  const uniq = Array.from(new Set(fixtures.map(f => f.competition).filter(Boolean))).sort();
  UI.leagues = ['ALL', ...uniq];
  el.leagueChips.innerHTML = '';
  for (const name of UI.leagues){
    const b = document.createElement('button');
    b.className = `chip${name===UI.league?' is-active':''}`;
    b.dataset.league = name;
    b.textContent = name;
    b.addEventListener('click', ()=>{
      document.querySelectorAll('#league-chips .chip').forEach(x=>x.classList.remove('is-active'));
      b.classList.add('is-active');
      UI.league = name; applyFiltersAndRender();
    });
    el.leagueChips.appendChild(b);
  }
}
function applyFiltersAndRender(){
  if (!fixtures.length) return;
  const start = baseDate();
  const end = datePlusDays(start, UI.rangeDays);
  visibleFixtures = fixtures.filter(f=>{
    const d = new Date(f.date_utc); if (isNaN(d)) return false;
    if (!(d>=start && d<end)) return false;
    if (UI.league !== 'ALL' && (f.competition||'').toLowerCase() !== UI.league.toLowerCase()) return false;
    return true;
  });

  const many = visibleFixtures.length > 250;
  globe.pointsMerge(many).pointResolution(12);
  globe.pointLat('latitude').pointLng('longitude').pointsData(visibleFixtures);

  buildRail(visibleFixtures);
  if (visibleFixtures.length) selectIndex(0, { fly:true });
  renderCompetitionAccuracy(UI.league==='ALL' ? (visibleFixtures[0]?.competition||'—') : UI.league);
}

// -----------------------------------------
// Three-Globe loader & scene init
// -----------------------------------------
async function loadThreeGlobe(){
  for (const v of ['2.31.3','2.31.1','2.30.1','2.29.3']){
    try { const m = await import(`https://esm.sh/three-globe@${v}?bundle&external=three`);
          console.warn('[three-globe] using esm.sh ESM:', v); return m.default ?? m; } catch {}
  }
  for (const url of [
    'https://cdn.jsdelivr.net/npm/three-globe@2.31.1/dist/three-globe.min.js',
    'https://unpkg.com/three-globe@2.31.1/dist/three-globe.min.js',
    './vendor/three-globe.min.js'
  ]){ try { const ctor = await import(url); console.warn('[three-globe] using UMD:', url); return ctor; } catch {} }
  throw new Error('three-globe failed to load');
}

async function init(){
  ThreeGlobeCtor = await loadThreeGlobe();

  scene = new THREE.Scene();
  renderer = new THREE.WebGLRenderer({ antialias:true, alpha:true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
  renderer.setSize(el.globeWrap.clientWidth, el.globeWrap.clientHeight);
  el.globeWrap.innerHTML = '';
  el.globeWrap.appendChild(renderer.domElement);
  if ('outputColorSpace' in renderer) renderer.outputColorSpace = THREE.SRGBColorSpace;

  camera = new THREE.PerspectiveCamera(45, el.globeWrap.clientWidth/el.globeWrap.clientHeight, 0.1, 5000);
  camera.position.set(0, 0, getGlobeRadius()*3);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.enablePan = false; controls.enableZoom = true;
  controls.autoRotate = false; controls.minDistance = getGlobeRadius()*1.2; controls.maxDistance = getGlobeRadius()*6;

  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const fxaa = new ShaderPass(FXAAShader);
  const setFXAA = ()=>{ const px=renderer.getPixelRatio();
    fxaa.material.uniforms['resolution'].value.set( 1/(el.globeWrap.clientWidth*px), 1/(el.globeWrap.clientHeight*px) ); };
  setFXAA(); composer.addPass(fxaa);
  const bloom = new UnrealBloomPass(new THREE.Vector2(el.globeWrap.clientWidth, el.globeWrap.clientHeight), BLOOM.strength, BLOOM.radius, BLOOM.threshold);
  composer.addPass(bloom);

  scene.add(new THREE.AmbientLight(0xffffff,0.9));
  const hemi = new THREE.HemisphereLight(0xddeeff, 0x223344, 0.6); scene.add(hemi);

  globe = new ThreeGlobeCtor({ waitForGlobeReady:true })
    .showAtmosphere(true).atmosphereColor('#9ef9e3').atmosphereAltitude(0.28)
    .globeImageUrl('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
    .bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png')
    .pointAltitude(()=>SURFACE_EPS)
    .pointRadius(d=>d.__active?RADIUS_ACTIVE:RADIUS_BASE)
    .pointColor(d=>d.__active?COLORS.markerActive:COLORS.marker)
    .pointsMerge(true);

  scene.add(globe);

  // custom marker group
  MARKER = createMarker();
  scene.add(MARKER.group);

  // hover & click
  if (typeof globe.onPointHover === 'function') globe.onPointHover(handleHover);
  globe.onPointClick?.(pt=>{
    if (!pt) return;
    const idx = visibleFixtures.findIndex(f=>f.latitude===pt.latitude && f.longitude===pt.longitude);
    if (idx>=0) selectIndex(idx, { fly:true });
  });

  // Resize
  window.addEventListener('resize', ()=>{
    const {clientWidth:w, clientHeight:h} = el.globeWrap;
    renderer.setSize(w,h); camera.aspect=w/h; camera.updateProjectionMatrix(); setFXAA();
  });

  bindTabs();
  bindDateControls();

  await loadFixturesCSV('./data/fixtures.csv');
  buildLeagueChips();
  buildDateStrip();
  applyFiltersAndRender();

  // loop
  (function loop(){
    requestAnimationFrame(loop);
    controls.update(); composer.render();
  })();
}
// -----------------------------------------
// CSV ingest
// -----------------------------------------
async function loadFixturesCSV(url){
  const res = await fetch(`${url}?v=${Date.now()}`); // cache-bust CSV only
  if (!res.ok){ showToast('error',`Could not load ${url} (HTTP ${res.status}).`); return; }
  const text = await res.text();
  const { data, errors } = Papa.parse(text, { header:true, skipEmptyLines:true });
  if (errors?.length) console.warn('[CSV parse errors]', errors);

  fixtures = (data||[])
    .map(row=>{
      const lat = parseFloat(row.latitude ?? row.lat ?? row.Latitude ?? row.lat_deg);
      const lon = parseFloat(row.longitude ?? row.lon ?? row.lng ?? row.Longitude);
      return {
        fixture_id:(row.fixture_id||row.id||`${row.home_team}-${row.away_team}-${row.date_utc||''}`).trim(),
        home_team:(row.home_team||row.Home||'').trim(),
        away_team:(row.away_team||row.Away||'').trim(),
        home_badge_url: pick(row,['home_badge_url','home_logo_url','home_logo','home_badge']),
        away_badge_url: pick(row,['away_badge_url','away_logo_url','away_logo','away_badge']),
        date_utc: row.date_utc || row.date || '',
        competition: row.competition || row.league || '',
        stadium: row.stadium || '',
        city: row.city || '',
        country: row.country || row.venue_country || '',
        latitude: Number.isFinite(lat)?lat:undefined,
        longitude: Number.isFinite(lon)?lon:undefined,
        predicted_winner: row.predicted_winner || '',
        confidence_ftr: +row.confidence_ftr || +row.confidence || 0,
        xg_home:+row.xg_home||0, xg_away:+row.xg_away||0,
        ppg_home:+row.ppg_home||0, ppg_away:+row.ppg_away||0,
        over25_prob:+row.over25_prob||0, btts_prob:+row.btts_prob||0,
        key_players_shots:(row.key_players_shots||'').trim(),
        key_players_tackles:(row.key_players_tackles||'').trim(),
        key_players_bookings:(row.key_players_bookings||'').trim(),
        __active:false
      };
    })
    .filter(f=>Number.isFinite(f.latitude)&&Number.isFinite(f.longitude));

  showToast('success', `Loaded ${fixtures.length} fixtures`);
}

// ----------------------------
// Selection, hover, rail, panel
// ----------------------------
function handleHover(pt){
  const hoverMatch = pt ? (p => p.latitude===pt.latitude && p.longitude===pt.longitude) : ()=>false;
  globe.pointRadius(p=>{
    if (p.__active) return RADIUS_ACTIVE;
    return hoverMatch(p) ? RADIUS_BASE*1.6 : RADIUS_BASE;
  });
}

function buildRail(items){
  const rail = document.getElementById('fixture-rail'); if (!rail) return;
  rail.innerHTML = '';
  items.forEach((f,i)=>{
    const it = document.createElement('button');
    it.className = `rail-item${i===0?' is-active':''}`;
    it.innerHTML = `<h4>${f.home_team} vs ${f.away_team}</h4><p>${f.city||f.country||''}</p>`;
    it.addEventListener('click', ()=>selectIndex(i,{fly:true}));
    rail.appendChild(it);
  });
}
function syncRail(activeIdx){
  const rail = document.getElementById('fixture-rail'); if (!rail) return;
  [...rail.children].forEach((c,idx)=>c.classList.toggle('is-active', idx===activeIdx));
}

function selectIndex(idx,{fly=false}={}){
  if (!visibleFixtures.length) return;
  const f = visibleFixtures[idx]; if (!f) return;
  visibleFixtures.forEach(it=>it.__active = (it===f));
  globe.pointColor(d=>d.__active?COLORS.markerActive:COLORS.marker)
       .pointRadius(d=>d.__active?RADIUS_ACTIVE:RADIUS_BASE)
       .pointsTransitionDuration?.(200);
  moveMarkerToFixture(f,{fly});
  renderPanel(f);
  syncRail(idx);
}

function elmEmpty(msg){ const d=document.createElement('div'); d.className='empty'; d.textContent=msg; return d; }

function renderPanel(f){
  if (!f) return;
  const fmt = iso=>{
    try{
      const d=new Date(iso);
      const date=d.toLocaleDateString(undefined,{weekday:'short', day:'2-digit', month:'short'});
      const time=d.toLocaleTimeString(undefined,{hour:'2-digit', minute:'2-digit'});
      return `${date} · ${time} GMT`;
    }catch{ return iso||''; }
  };
  el.fixtureTitle && (el.fixtureTitle.textContent = `${f.home_team} vs ${f.away_team}`);
  el.fixtureContext && (el.fixtureContext.textContent = [
    f.competition, fmt(f.date_utc), f.stadium && `${f.stadium} (${f.city||''})`, f.country
  ].filter(Boolean).join(' • '));

    // We’re local-only now; keep CSV URLs as optional future input
    setBadgeLocal(el.homeBadge,  f.home_badge_url || null, f.home_team);
    setBadgeLocal(el.awayBadge,  f.away_badge_url || null, f.away_team);


  if (el.matchList){
    clearNode(el.matchList);
    const mi = document.createElement('div');
    mi.innerHTML = `
      <div><strong>Full-time prediction:</strong> ${f.predicted_winner||'–'} ${f.confidence_ftr ? `(${Math.round(f.confidence_ftr*100)}%)` : ''}</div>
      <div><strong>xG edge:</strong> ${(f.xg_home||0).toFixed(1)} vs ${(f.xg_away||0).toFixed(1)}</div>
      <div><strong>Points momentum:</strong> ${(f.ppg_home||0).toFixed(1)} PPG • ${(f.ppg_away||0).toFixed(1)} PPG</div>`;
    el.matchList.appendChild(mi);
  }

  if (el.watchlist){
    clearNode(el.watchlist);
    const shots = (f.key_players_shots||'').split(';').map(s=>s.trim()).filter(Boolean).slice(0,6);
    if (shots.length){
      shots.forEach(s=>{ const row=document.createElement('div'); row.className='row'; row.textContent=s; el.watchlist.appendChild(row); });
    } else el.watchlist.appendChild(elmEmpty('No player highlights available.'));
  }

  if (el.market){
    el.market.innerHTML = `
      <div><strong>Over 2.5 goals:</strong> ${Math.round((f.over25_prob||0)*100)}%</div>
      <div><strong>Both teams to score:</strong> ${Math.round((f.btts_prob ||0)*100)}%</div>`;
  }

  renderCompetitionAccuracy(f.competition);
}

function bindTabs(){
  document.querySelectorAll('.tab')?.forEach(btn=>{
    btn.addEventListener('click',()=>{
      document.querySelectorAll('.tab').forEach(b=>b.classList.remove('is-active'));
      document.querySelectorAll('.tabpane').forEach(p=>p.classList.remove('is-active'));
      btn.classList.add('is-active');
      document.getElementById(`tab-${btn.dataset.tab}`)?.classList.add('is-active');
    });
  });
}

// ----------------------------
// Marker creation & movement  [PATCH B]
// ----------------------------
function createMarker(){
  const group = new THREE.Group();
  group.visible = false;

  const R = getGlobeRadius();


  // --- Radar: 4 concentric static rings (inner thickest, outer faintest) ---
  const radarRings = [];
  const baseInner = R * 0.02;   // start a bit chunkier so it reads clearly
  const baseWidth = R * 0.008;

  for (let i = 0; i < 4; i++) {
    const inner = baseInner + i * (baseWidth * 0.9);      // push each ring out
    const outer = inner + baseWidth * (1 - i * 0.2);      // shrink thickness ~20% each step

    const ringGeom = new THREE.RingGeometry(inner, outer, 64);
    const ringMat  = new THREE.MeshBasicMaterial({
      color: new THREE.Color(COLORS.ring),
      transparent: true,
      opacity: 0.35 * (1 - i * 0.18),   // inner brightest, outer faintest
      side: THREE.DoubleSide,
      depthWrite: false,
      depthTest: false                  // always on top of globe
    });

    const ring = new THREE.Mesh(ringGeom, ringMat);
    // Lie flat in local XZ (normal +Y)
    ring.rotation.x = Math.PI / 2;
    ring.renderOrder = 998;
    group.add(ring);
    radarRings.push(ring);
  }


  // Beam straight "up" from the radar (local +Y)
  const beamGeom = new THREE.CylinderGeometry(0.18, 0.28, 30, 24, 1, true);
  const beamMat  = new THREE.MeshBasicMaterial({
    color: 0x7df9c4,
    transparent: true,
    opacity: 0.0,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    depthTest: false
  });
  const beam = new THREE.Mesh(beamGeom, beamMat);
  beam.visible = false;
  beam.renderOrder = 998;
  group.add(beam);

  // Stadium pill panel – Plane so we can tilt it (Sprite is always face-on)
  const billboardGeom = new THREE.PlaneGeometry(24, 14);
  const billboardMat  = new THREE.MeshBasicMaterial({
    transparent: true,
    opacity: 0,
    depthTest: false,   // draw on top of globe
    depthWrite: false
  });
  const billboard = new THREE.Mesh(billboardGeom, billboardMat);
  billboard.renderOrder = 999;      // draw last
  group.add(billboard);

    return {
      group,
      radar: radarRings,            // array of rings now
      beam,
      billboard,
      state: { lat: 0, lon: 0, reqId: 0 },
      raf:   { travel: null, beam: null, fade: null, radar: null }
    };
  }



function cancelRAF(handle){
  if (handle && handle.id) cancelAnimationFrame(handle.id);
}
function makeRAF(){
  return {
    id: null,
    run(fn){
      cancelRAF(this);
      const loop = () => { fn(); this.id = requestAnimationFrame(loop); };
      this.id = requestAnimationFrame(loop);
    },
    cancel(){ cancelRAF(this); this.id = null; }
  };
}

// Great-circle slerp for unit vectors
function slerpUnitVec(fromN, toN, t) {
  const v0 = fromN.clone().normalize();
  const v1 = toN.clone().normalize();
  let dot = THREE.MathUtils.clamp(v0.dot(v1), -1, 1);

  if (dot > 0.9995) return v0.lerp(v1, t).normalize();
  if (dot < -0.9995) {
    const ortho = Math.abs(v0.x) < 0.9 ? new THREE.Vector3(1,0,0) : new THREE.Vector3(0,1,0);
    const axis  = new THREE.Vector3().crossVectors(v0, ortho).normalize();
    const q     = new THREE.Quaternion().setFromAxisAngle(axis, Math.PI * t);
    return v0.clone().applyQuaternion(q).normalize();
  }

  const angle = Math.acos(dot);
  const axis  = new THREE.Vector3().crossVectors(v0, v1).normalize();
  const q     = new THREE.Quaternion().setFromAxisAngle(axis, angle * t);
  return v0.clone().applyQuaternion(q).normalize();
}

function moveMarkerToFixture(f, { fly=false } = {}){
  if (!MARKER || !f) return;

  console.log('[marker] move to', f.home_team, f.city || f.country, f.latitude, f.longitude);
  const S   = MARKER;
  const lat = Number(f.latitude), lon = Number(f.longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) { S.group.visible = false; return; }

  S.state.reqId++;
  const myReq = S.state.reqId;
  const R     = getGlobeRadius();

  const toN   = latLngToUnit(lat, lon);
  const fromN = S.group.visible ? S.group.position.clone().normalize() : toN.clone();

  const angle = Math.acos(THREE.MathUtils.clamp(fromN.dot(toN), -1, 1));
  const distK = angle * R;
  const dur   = (fly && S.group.visible) ? THREE.MathUtils.clamp(distK * 2.0, 300, 900) : 0;
  const t0    = performance.now();

  S.raf.travel = S.raf.travel || makeRAF();
  S.raf.beam   = S.raf.beam   || makeRAF();
  S.raf.fade   = S.raf.fade   || makeRAF();
  S.raf.radar  = S.raf.radar  || makeRAF();

  S.raf.travel.cancel();
  S.raf.beam.cancel();
  S.raf.fade.cancel();
  S.raf.radar.cancel();


  S.group.visible = true;

  S.raf.travel.run(()=>{
    if (S.state.reqId !== myReq) { S.raf.travel.cancel(); return; }
    const t = dur ? Math.min(1, (performance.now() - t0) / dur) : 1;
    const k = easeInOut(t);

    const curN    = slerpUnitVec(fromN, toN, k);
    const worldPos= curN.clone().multiplyScalar(R * (1 + SURFACE_EPS));
    S.group.position.copy(worldPos);

    // Orient local +Y to surface normal
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,1,0), curN);
    S.group.quaternion.copy(q);

    if (t >= 1) {
      S.raf.travel.cancel();

      // Beam grow along local +Y
      S.beam.position.set(0, 0, 0);
      S.beam.quaternion.identity();
      S.beam.scale.set(1, 0.001, 1);
      S.beam.material.opacity = 0.0;
      S.beam.visible = true;

      const b0 = performance.now(), bd = 550;
      S.raf.beam.run(() => {
        if (S.state.reqId !== myReq) { S.raf.beam.cancel(); return; }
        const tb = Math.min(1, (performance.now() - b0) / bd);
        const e  = easeInOut(tb);
        S.beam.scale.y          = 0.001 + e;
        S.beam.material.opacity = 0.5 * e;   // brighter flicker
        if (tb >= 1) S.raf.beam.cancel();
      });

      // Billboard “info pill” – a bit above the radar and pushed out from the globe
      const PILL_ALT = R * 0.04;  // how far above the radar centre
      const PILL_OUT = R * 0.08;  // how far out away from the globe along local Z
      S.billboard.position.set(0, PILL_ALT, PILL_OUT);
      S.billboard.material.opacity = 0;
      S.billboard.visible = true;

      // Tilt panel ~25° so it’s not perfectly face-on (matches your 45° sketch feel)
      S.billboard.rotation.set(THREE.MathUtils.degToRad(-25), 0, 0);

      // Only hide if really far side of the globe
      if (curN.dot(camera.position.clone().normalize()) < -0.25) {
        S.billboard.visible = false;
      }


      // Load stadium texture (queued, cached)
      (async ()=>{
        for (const url of stadiumCandidates(f)) {
          try {
            const tex = await loadTextureQueued(url);
            if (S.state.reqId !== myReq) return;
            S.billboard.material.map = tex;
            S.billboard.material.needsUpdate = true;
      
            const fa0 = performance.now(), fad = 220;
            S.raf.fade.run(()=>{
              if (S.state.reqId !== myReq) { S.raf.fade.cancel(); return; }
              const ft = Math.min(1, (performance.now() - fa0) / fad);
              S.billboard.material.opacity = ft;
              if (ft >= 1) S.raf.fade.cancel();
            });
            return;
          } catch { /* try next */ }
        }
      
        // No file matched -> fallback disc with team/city
        if (S.state.reqId === myReq) {
          const name = (f.city || f.home_team || 'STADIUM').toUpperCase();
          const tex  = makeFallbackCanvasTexture(name);
          S.billboard.material.map = tex;
          S.billboard.material.needsUpdate = true;
          S.billboard.material.opacity = 1.0;
          S.billboard.visible = true;
        }
      })();
    }
  }); // <-- end S.raf.travel.run loop
} // <-- end moveMarkerToFixture

// =====================================================
// API HELPERS + FEATURE PAGES (BetChecker / Acca / Co-Pilot)
// =====================================================
const API_BASE = '/api';

async function apiJson(url, opts = {}) {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...(opts.headers||{}) },
    credentials: 'include',
    ...opts
  });
  if (!res.ok) { const t = await res.text().catch(()=> ''); throw new Error(`HTTP ${res.status}: ${t || res.statusText}`); }
  return res.json();
}

const API = {
  scoreSlip: (payload)=> apiJson('/score-slip', { method:'POST', body: JSON.stringify(payload) }),
  accaSuggest: (q)=>   apiJson(`/acca/suggest?${new URLSearchParams(q)}`),
  accaOptimise: (p)=>  apiJson('/acca/optimise', { method:'POST', body: JSON.stringify(p) }),
  copilot: (p)=>       apiJson('/copilot', { method:'POST', body: JSON.stringify(p) })
};

// ---------- Bet Checker ----------
async function ocrImageOrPdf(file) {
  if (!window.Tesseract) throw new Error('OCR engine not loaded');
  const { data } = await window.Tesseract.recognize(file, 'eng', { logger: () => {} });
  return (data && data.text) ? data.text : '';
}
function parseSlipText(text) {
  const lines = String(text).split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
  const legs = [];
  for (let i=0;i<lines.length;i++){
    const L = lines[i];
    const m = L.match(/^\s*([A-Za-z0-9 .'\-]+)\s+(?:v|vs\.?|VS)\s+([A-Za-z0-9 .'\-]+)\s*$/i);
    if (!m) continue;
    const home = m[1].trim(), away = m[2].trim();
    for (let j=1;j<=3 && (i+j)<lines.length;j++){
      const M = lines[i+j];
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
      const frac=M.match(/(\d+)\s*\/\s*(\d+)/); const dec=M.match(/(\d+(?:\.\d+)?)/);
      if (frac) price = (parseFloat(frac[1])/parseFloat(frac[2]))+1;
      else if (dec) price = parseFloat(dec[1]);
      legs.push({ teamHome:home, teamAway:away, market, selection:pick||'—', price, bookmaker:null, kickoffUTC:null });
      break;
    }
  }
  return { legs, raw: lines.slice(0,60).join('\n') };
}
async function runBetChecker(file) {
  const out = document.getElementById('bc-output');
  if (out) out.innerHTML = '<div class="muted">Reading slip…</div>';
  try {
    const text = await ocrImageOrPdf(file);
    const parsed = parseSlipText(text);
    if (!parsed.legs.length) {
      out && (out.innerHTML = `<div class="muted">No legs detected. <br/><pre style="white-space:pre-wrap">${parsed.raw || text.slice(0,400)}…</pre></div>`);
      return;
    }
    out && (out.innerHTML = '<div class="muted">Scoring legs…</div>');
    const scored = await API.scoreSlip({ legs: parsed.legs });

    const container = document.createElement('div');
    const sum = scored.summary || {};
    container.innerHTML = `
      <div class="insight-card"><h2>Summary</h2>
      <div>Implied EV: <strong>${(sum.evPct??0).toFixed(1)}%</strong> · Prob: <strong>${Math.round((sum.comboProb??0)*100)}%</strong> · Payout: <strong>${sum.payout ? '£'+sum.payout.toFixed(2) : '—'}</strong></div></div>`;
    (scored.legs||[]).forEach((lg,i)=>{
      const card=document.createElement('div'); card.className='insight-card';
      card.innerHTML = `<h2>Leg ${i+1}: ${lg.teamHome} vs ${lg.teamAway}</h2>
        <ul><li><strong>Market:</strong> ${lg.market} · <strong>Pick:</strong> ${lg.selection}</li>
        <li><strong>Model%:</strong> ${lg.prob?Math.round(lg.prob*100)+'%':'—'} · <strong>Fair:</strong> ${lg.fair?.toFixed?.(2)??'—'} · <strong>Book:</strong> ${lg.price?.toFixed?.(2)??'—'} · <strong>Edge:</strong> ${lg.edgePct!=null?lg.edgePct.toFixed(1)+'%':'—'}</li></ul>`;
      container.appendChild(card);
    });
    out.innerHTML=''; out.appendChild(container);
    showToast('success', `Scored ${scored.legs?.length || parsed.legs.length} leg(s)`);
  } catch (e) {
    showToast('error', e.message);
    out && (out.innerHTML = `<div class="muted">Error: ${e.message}</div>`);
  }
}
{
  const bcUpload = document.getElementById('bc-upload');
  bcUpload?.addEventListener('change', (e)=>{
    const f = e.target.files?.[0];
    if (f) runBetChecker(f);
    e.target.value = '';
  });
}

// ---------- Acca Builder ----------
async function runAccaSuggest() {
  const league = (document.getElementById('ab-league')?.value || 'ALL');
  const market = (document.getElementById('ab-market')?.value || 'ou25');
  const legs   = (document.getElementById('ab-legs')?.value || '4').split(' ')[0];

  const grid = document.getElementById('ab-grid');
  if (!grid) return;
  grid.innerHTML = '<div class="muted">Loading candidates…</div>';

  try {
    const data = await API.accaSuggest({
      market, league, from: new Date().toISOString().slice(0,10), to: '', limit: 50
    });
    grid.innerHTML = '';
    const chosen = [];
    (data.items || []).slice(0, 30).forEach((m, i)=>{
      const card = document.createElement('div');
      card.className = 'insight-card';
      const prob = Math.round((m.prob||0)*100);
      const edge = m.edgePct != null ? `${m.edgePct.toFixed(1)}%` : '—';
      card.innerHTML = `
        <h2>${m.home} vs ${m.away}</h2>
        <ul>
          <li><strong>Market:</strong> ${m.market} · <strong>Pick:</strong> ${m.pick || '—'}</li>
          <li><strong>%:</strong> ${prob}% · <strong>Fair:</strong> ${m.fair?.toFixed?.(2)??'—'} · <strong>Price:</strong> ${m.price?.toFixed?.(2)??'—'} · <strong>Edge:</strong> ${edge}</li>
        </ul>
        <button class="cta" data-add="${i}">Add leg</button>`;
      grid.appendChild(card);
    });

    grid.querySelectorAll('button[data-add]').forEach(btn=>{
      btn.addEventListener('click', ()=>{
        const idx = +btn.getAttribute('data-add');
        chosen.push(data.items[idx]);
        btn.textContent = 'Added ✓'; btn.disabled = true;
        if (chosen.length === Number(legs)) {
          optimiseAcca(chosen, market);
        }
      });
    });
  } catch (e) {
    grid.innerHTML = `<div class="muted">Suggest failed: ${e.message}</div>`;
  }
}
async function optimiseAcca(chosen, market) {
  showToast('info', `Optimising ${chosen.length}-fold…`);
  const grid = document.getElementById('ab-grid');
  try {
    const res = await API.accaOptimise({ market, legs: chosen, strategy: { objective:'max_ev', stake:10 }});
    grid.innerHTML = '';
    const sum = document.createElement('div');
    sum.className = 'insight-card';
    sum.innerHTML = `<h2>Optimised Acca</h2>
      <div><strong>EV:</strong> ${(res.evPct??0).toFixed(1)}% · <strong>Prob:</strong> ${Math.round((res.comboProb??0)*100)}% · <strong>Payout:</strong> £${(res.payout??0).toFixed(2)}</div>`;
    grid.appendChild(sum);

    (res.legs||chosen).forEach((lg, i)=>{
      const c = document.createElement('div');
      c.className = 'insight-card';
      c.innerHTML = `<h2>Leg ${i+1}: ${lg.home} vs ${lg.away}</h2>
        <div class="muted">${lg.market} • ${lg.pick} • ${Math.round((lg.prob||0)*100)}% @ ${lg.price?.toFixed?.(2)??'—'}</div>`;
      grid.appendChild(c);
    });
    showToast('success', 'Acca ready');
  } catch (e) {
    grid.innerHTML = `<div class="muted">Optimiser failed: ${e.message}</div>`;
  }
}
document.getElementById('ab-build')?.addEventListener('click', ()=> runAccaSuggest());

// ---------- OG Co-Pilot ----------
async function sendCopilotMessage(text) {
  const payload = { 
    messages: [
      { role:'system', content: 'You are OddsGenius Co-Pilot. Be concise, provide bullet reasoning, cite model features where relevant.' },
      { role:'user', content: text }
    ],
    context: (function(){
      const f = visibleFixtures[0];
      if (!f) return null;
      return { fixture: { home:f.home_team, away:f.away_team, date:f.date_utc, league:f.competition } };
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
  const cpThread= document.getElementById('cp-thread');

  cpSend?.addEventListener('click', async ()=>{
    const q = (cpInput?.value || '').trim();
    if (!q) return;
    appendChatLine('user', q);
    cpInput.value = '';
    try {
      const { messages, error } = await sendCopilotMessage(q);
      if (error) throw new Error(error);
      const msg = (messages && messages.find(m=>m.role==='assistant'))?.content || '(no reply)';
      appendChatLine('assistant', msg);
    } catch (e) {
      appendChatLine('assistant', `⚠ ${e.message}`);
    }
  });
}

// ----------------------------
// Router + Boot
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
    else { v.classList.remove('is-active'); v.setAttribute('hidden',''); }
  });
  document.querySelectorAll('[data-route]').forEach(a=>{
    a.classList.toggle('is-active', a.getAttribute('href') === hash);
    if (a.classList.contains('side-link')) a.classList.toggle('active', a.getAttribute('href') === hash);
  });
  const profileMenu = document.getElementById('profile-menu');
  const profileBtn  = document.getElementById('btn-profile');
  profileMenu?.classList.remove('show'); profileBtn?.setAttribute('aria-expanded','false');
  const drawer  = document.getElementById('side-drawer');
  const scrim   = document.getElementById('scrim');
  drawer?.classList.remove('show'); scrim?.classList.remove('show'); drawer?.setAttribute('aria-hidden','true');
}
window.addEventListener('hashchange', ()=> showRoute(location.hash));
window.addEventListener('DOMContentLoaded', ()=>{
  if (!location.hash) location.hash = '#/'; showRoute(location.hash);
  init().catch(err=>{ console.error(err); showToast('error', String(err)); });
});

// ----------------------------
// Quick self-test for two known files
// ----------------------------
(function verifyLocalLogoSetup(){
  const tests = [
    './assets/assets/logos/arsenal.svg',
    './assets/assets/logos/fc-barcelona.svg'
  ];
  tests.forEach(src=>{
    const img = new Image();
    img.onload  = () => console.log('%c[LOGOS] OK', 'color:#22c55e', src);
    img.onerror = () => console.warn('%c[LOGOS] 404', 'color:#f43f5e', src, '→ path or filename mismatch');
    img.src = src;
  });
})();
