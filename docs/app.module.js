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
let currentFixture = null;
let isHomeActive = true;       // controls whether globe render loop runs

// Demo auth + portfolio
let currentUser = null;        // { email, role }
let savedAccas  = [];          // [{ id, userEmail, name, status, stake, legs, ... }]

const STORAGE_KEYS = {
  user:  'og_user',
  accas: 'og_saved_accas'
};

// Acca Builder state
let abCurrentMarket  = 'all';  // default: show all markets in middle column
let abCartLegs       = [];     // legs currently in the "Your Acca" cart
let currentAccaLegs  = [];     // snapshot of last rendered cart (for saving)
const abFixtureById  = new Map();

const SURFACE_EPS   = 0.010;  // slightly higher so dots sit above the texture
const RADIUS_BASE   = 0.018;  // bigger = easier to see non-active fixtures
const RADIUS_ACTIVE = 0.040;  // keep the active one noticeably larger
const CAMERA_ALT    = 2.0;
const BLOOM = { strength: 0.9, radius: 0.6, threshold: 0.75 };

const COLORS = {
  marker:         'rgba(125,249,196,0.45)',  // soft teal glow for all fixtures
  markerInactive: 'rgba(125,249,196,0.15)',  // (kept for future use if you want)
  markerActive:   '#FFFFFF',                 // bright white for active
  ring:           '#9EE7E3'          'rgba(255, 194, 112, 0.9)'
};

// Small manual tweak to align fixture positions on the globe.
// Adjust lonBias / latBias until Europe-based fixtures sit over Europe.
const MAP_OFFSET = {
  latBias: 0,   // leave N/S alone for now
  lonBias: 95   // shift everything ~15° further east (was 40)
};

// Raycaster for clickable pill sprite
const raycaster = new THREE.Raycaster();
const pointer   = new THREE.Vector2();

// ---- UI demo “today”
const UI = {
  anchorISO: '2023-11-28',
  offsetDays: 0,
  rangeDays: 1,
  league: 'ALL',
  leagues: []
};

// -----------------------------------------
// Utilities
// -----------------------------------------
const clamp01   = v => Math.max(0, Math.min(1, v));
const easeInOut = t => t*t*(3-2*t);

function showToast(type, text, ms=2600){
  const t = document.createElement('div');
  t.className = `og-toast ${type}`;
  t.textContent = text;
  document.body.appendChild(t);
  requestAnimationFrame(()=>t.classList.add('show'));
  setTimeout(()=>{
    t.classList.remove('show');
    setTimeout(()=>t.remove(),250);
  }, ms);
}

function clearNode(node) {
  if (!node) return;
  while (node.firstChild) node.removeChild(node.firstChild);
}

function pick(row, keys) {
  for (const k of keys) {
    const v = (row[k] ?? '').toString().trim();
    if (v) return v;
  }
  return '';
}

// ---- Session & portfolio persistence (demo: localStorage) ----
function loadSessionFromStorage() {
  try {
    const rawUser  = window.localStorage.getItem(STORAGE_KEYS.user);
    const rawAccas = window.localStorage.getItem(STORAGE_KEYS.accas);
    currentUser = rawUser  ? JSON.parse(rawUser)  : null;
    savedAccas  = rawAccas ? JSON.parse(rawAccas) : [];
  } catch (e) {
    console.warn('Failed to parse stored session', e);
    currentUser = null;
    savedAccas = [];
  }
}

function persistSession() {
  try {
    if (currentUser) {
      window.localStorage.setItem(STORAGE_KEYS.user, JSON.stringify(currentUser));
    } else {
      window.localStorage.removeItem(STORAGE_KEYS.user);
    }
    window.localStorage.setItem(STORAGE_KEYS.accas, JSON.stringify(savedAccas || []));
  } catch (e) {
    console.warn('Failed to persist session', e);
  }
}

function updateAuthUI() {
  const profileBtn  = document.getElementById('btn-profile');
  const profileMenu = document.getElementById('profile-menu');
  if (!profileBtn) return;

  const avatar = profileBtn.querySelector('.avatar');
  const items  = profileMenu?.querySelectorAll('.profile-item') || [];

  if (currentUser) {
    if (avatar) {
      const initials = (currentUser.email || '?')
        .split('@')[0]
        .slice(0, 2)
        .toUpperCase();
      avatar.textContent = initials;
    }
    items.forEach(el => {
      const href = el.getAttribute('href');
      if (href === '#/login' || href === '#/signup') {
        el.style.display = 'none';
      }
    });
    const logoutBtn = document.querySelector('[data-action="logout"]');
    if (logoutBtn) logoutBtn.style.display = 'block';
  } else {
    if (avatar) avatar.textContent = 'OG';
    items.forEach(el => {
      const href = el.getAttribute('href');
      if (href === '#/login' || href === '#/signup') {
        el.style.display = '';
      }
    });
    const logoutBtn = document.querySelector('[data-action="logout"]');
    if (logoutBtn) logoutBtn.style.display = 'none';
  }
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

  const r = 220, cx = 256, cy = 256;
  const grd = g.createRadialGradient(cx, cy, r*0.25, cx, cy, r);
  grd.addColorStop(0, 'rgba(255,255,255,0.95)');
  grd.addColorStop(1, 'rgba(125,249,196,0.20)');
  g.fillStyle = grd;
  g.beginPath(); g.arc(cx, cy, r, 0, Math.PI*2); g.fill();

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

function makeStadiumPillTexture(f, img) {
  const w = 1024;
  const h = 512;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const g = c.getContext('2d');

  g.clearRect(0, 0, w, h);

  // Background pill
  const padX = 32;
  const padY = 32;
  const pillX = padX;
  const pillY = padY;
  const pillW = w - padX * 2;
  const pillH = h - padY * 2;
  const radius = 80;

  const grd = g.createLinearGradient(pillX, pillY, pillX + pillW, pillY + pillH);
  grd.addColorStop(0, '#0d2a32');
  grd.addColorStop(0.5, '#0f4a40');
  grd.addColorStop(1, '#0a344a');

  g.fillStyle = grd;
  g.beginPath();
  g.moveTo(pillX + radius, pillY);
  g.lineTo(pillX + pillW - radius, pillY);
  g.quadraticCurveTo(pillX + pillW, pillY, pillX + pillW, pillY + radius);
  g.lineTo(pillX + pillW, pillY + pillH - radius);
  g.quadraticCurveTo(pillX + pillW, pillY + pillH, pillX + pillW - radius, pillY + pillH);
  g.lineTo(pillX + radius, pillY + pillH);
  g.quadraticCurveTo(pillX, pillY + pillH, pillX, pillY + pillH - radius);
  g.lineTo(pillX, pillY + radius);
  g.quadraticCurveTo(pillX, pillY, pillX + radius, pillY);
  g.closePath();
  g.fill();

  // Stadium circle on the left
  const cx = pillX + 170;
  const cy = pillY + pillH / 2;
  const cr = 110;

  if (img && img.width && img.height) {
    g.save();
    g.beginPath();
    g.arc(cx, cy, cr, 0, Math.PI * 2);
    g.closePath();
    g.clip();

    const iw = img.width;
    const ih = img.height;
    const scale = Math.max((cr * 2) / iw, (cr * 2) / ih);
    const drawW = iw * scale;
    const drawH = ih * scale;
    const dx = cx - drawW / 2;
    const dy = cy - drawH / 2;
    g.drawImage(img, dx, dy, drawW, drawH);
    g.restore();
  }

  // White circle border
  g.strokeStyle = 'rgba(255,255,255,0.9)';
  g.lineWidth = 8;
  g.beginPath();
  g.arc(cx, cy, cr, 0, Math.PI * 2);
  g.stroke();

  // Text: fixture + time + city/country
  const home = f?.home_team || 'Home';
  const away = f?.away_team || 'Away';
  const line1 = `${home} vs ${away}`;

  let timeStr = '';
  try {
    if (f?.date_utc) {
      const d = new Date(f.date_utc);
      timeStr = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    }
  } catch {}

  const placeParts = [];
  if (f?.city)    placeParts.push(f.city);
  if (f?.country) placeParts.push(f.country);
  const place = placeParts.join(', ');
  const line2 = [timeStr, place].filter(Boolean).join(' – ');

  const textX = cx + cr + 60;
  const textY = cy - 18;

  g.fillStyle = 'rgba(229,246,255,0.96)';
  g.font = 'bold 48px Montserrat, system-ui, sans-serif';
  g.textAlign = 'left';
  g.textBaseline = 'middle';
  g.fillText(line1, textX, textY);

  g.fillStyle = 'rgba(198,221,239,0.96)';
  g.font = '32px Montserrat, system-ui, sans-serif';
  g.fillText(line2, textX, textY + 52);

  const tex = new THREE.CanvasTexture(c);
  if ('colorSpace' in tex) tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = renderer?.capabilities?.getMaxAnisotropy?.() || 1;
  return tex;
}

function makeStadiumPillTextureFromImage(img) {
  if (!img) return null;

  const w = 512;
  const h = 256;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const g = c.getContext('2d');

  g.clearRect(0, 0, w, h);

  const radius = 56;
  const x = 16;
  const y = 16;
  const width = w - 32;
  const height = h - 32;

  const grd = g.createLinearGradient(x, y, x + width, y + height);
  grd.addColorStop(0, '#0d2a32');
  grd.addColorStop(0.5, '#0f4a40');
  grd.addColorStop(1, '#0a344a');

  g.fillStyle = grd;
  g.beginPath();
  g.moveTo(x + radius, y);
  g.lineTo(x + width - radius, y);
  g.quadraticCurveTo(x + width, y, x + width, y + radius);
  g.lineTo(x + width, y + height - radius);
  g.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  g.lineTo(x + radius, y + height);
  g.quadraticCurveTo(x, y + height, x, y + height - radius);
  g.lineTo(x, y + radius);
  g.quadraticCurveTo(x, y, x + radius, y);
  g.closePath();
  g.fill();

  const cx = x + 110;
  const cy = y + height / 2;
  const cr = 80;

  g.save();
  g.beginPath();
  g.arc(cx, cy, cr, 0, Math.PI * 2);
  g.closePath();
  g.clip();

  const iw = img.width;
  const ih = img.height;
  if (iw && ih) {
    const scale = Math.max((cr * 2) / iw, (cr * 2) / ih);
    const drawW = iw * scale;
    const drawH = ih * scale;
    const dx = cx - drawW / 2;
    const dy = cy - drawH / 2;
    g.drawImage(img, dx, dy, drawW, drawH);
  }
  g.restore();

  g.strokeStyle = 'rgba(255,255,255,0.85)';
  g.lineWidth = 6;
  g.beginPath();
  g.arc(cx, cy, cr, 0, Math.PI * 2);
  g.stroke();

  const tex = new THREE.CanvasTexture(c);
  if ('colorSpace' in tex) tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = renderer?.capabilities?.getMaxAnisotropy?.() || 1;
  return tex;
}

// -----------------------------------------
// Logos (local-only) — club crests
// -----------------------------------------
const LOGO_LOCAL_BASE = './assets/assets/logos';

function stripDiacritics(s = '') {
  try { return s.normalize('NFD').replace(/\p{Diacritic}/gu, ''); }
  catch { return s.replace(/[\u0300-\u036f]/g, ''); }
}

function slugLocal(name = '') {
  return stripDiacritics(String(name))
    .toLowerCase()
    .replace(/&/g, 'and')
    .replace(/[\u2019'’]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

// explicit overrides mapped to actual filenames
const TEAM_LOGO_OVERRIDES = {
  'Atlético Madrid'       : `${LOGO_LOCAL_BASE}/atletico-madrid.svg`,
  'Atletico Madrid'       : `${LOGO_LOCAL_BASE}/atletico-madrid.svg`,
  'Bayern München'        : `${LOGO_LOCAL_BASE}/bayern-munich.svg`,
  'Bayern Munich'         : `${LOGO_LOCAL_BASE}/bayern-munich.svg`,
  'Crvena Zvezda'         : `${LOGO_LOCAL_BASE}/red-star-belgrade.svg`,
  'Red Star Belgrade'     : `${LOGO_LOCAL_BASE}/red-star-belgrade.svg`,
  'København'             : `${LOGO_LOCAL_BASE}/fc-kobenhavn.svg`,
  'FC København'          : `${LOGO_LOCAL_BASE}/fc-kobenhavn.svg`,
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

function localLogoCandidates(teamName = '') {
  const name = String(teamName || '').trim();
  if (!name) return [];
  const override = TEAM_LOGO_OVERRIDES[name];
  if (override) return [override];

  const s = slugLocal(name);
  if (!s) return [];
  const b = LOGO_LOCAL_BASE;
  return Array.from(new Set([
    `${b}/${s}.svg`,
    `${b}/${s.replace(/^fc-/, '')}.svg`,
    `${b}/${s.replace(/-fc$/, '')}.svg`,
  ]));
}

function initialsFor(name = '') {
  const p = String(name).trim().split(/\s+/);
  return p.length ? (p[0][0] + (p[1]?.[0] || '')).toUpperCase() : '';
}

// *** Simple badge loader – no races, loud logging ***
function setBadgeLocal(elm, _urlFromCsv, teamName = '') {
  if (!elm) return;

  elm.classList.remove('has-logo');
  elm.innerHTML = '';
  elm.textContent = initialsFor(teamName);

  const candidates = localLogoCandidates(teamName);
  console.log('[badge] candidates', { teamName, candidates });

  if (!candidates.length) return;

  const src = candidates[0];
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
  };
  img.src = src;
}

// -----------------------------------------
// Competition badges (for comp strip under globe)
// -----------------------------------------
const COMP_LOGO_OVERRIDES = {};
const COMP_BADGE_CACHE    = new Map();

function makeCompetitionBadgeDataUrl(label = 'LEAGUE') {
  const c = document.createElement('canvas');
  const size = 256;
  c.width = size;
  c.height = size;
  const g = c.getContext('2d');

  g.clearRect(0, 0, size, size);
  const grd = g.createLinearGradient(0, 0, size, size);
  grd.addColorStop(0, '#0f766e');
  grd.addColorStop(1, '#22c55e');
  g.fillStyle = grd;

  const r = size * 0.42;
  g.beginPath();
  g.arc(size/2, size/2, r, 0, Math.PI * 2);
  g.fill();

  const parts = String(label || '').split(/\s+/).filter(Boolean);
  const initials = (
    (parts[0]?.[0] || 'L') +
    (parts[1]?.[0] || 'G')
  ).toUpperCase();

  g.fillStyle = '#e5f9ff';
  g.font = 'bold 72px Montserrat, system-ui, sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(initials, size/2, size/2);

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
  const override = COMP_LOGO_OVERRIDES[name];
  if (override) {
    el.compLogo.src = override;
    el.compLogo.style.visibility = 'visible';
    return;
  }
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

function stadiumCandidates(f) {
  const name = String(f?.home_team || '').trim();
  const file = STADIUM_OVERRIDES[name];
  return file ? [`${STADIUM_BASE}/${file}`] : [];
}

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
const DEMO_FTR = 0.87;
function getCompetitionSnapshot(league){
  const rows = league && league!=='ALL'
    ? fixtures.filter(f => (f.competition||'').toLowerCase() === league.toLowerCase())
    : fixtures.slice();
  const avg = a => a.length ? a.reduce((x,y)=>x+y,0)/a.length : 0;
  const pct = x => Math.round(x*100);
  return {
    n: rows.length,
    ftr:    pct(avg(rows.map(r => +r.confidence_ftr||0))),
    over25: pct(avg(rows.map(r => +r.over25_prob   ||0))),
    btts:   pct(avg(rows.map(r => +r.btts_prob     ||0)))
  };
}
function renderCompetitionAccuracy(league){
  if (!el.compWrap) return;
  const stats = getCompetitionSnapshot(league);
  const name = league || '—';
  if (el.compName) el.compName.textContent = name;
  setCompetitionLogo(name);
  if (el.compTraffic){
    el.compTraffic.innerHTML = `
      <span class="light light--green">FTR ${Math.round(DEMO_FTR*100)}%</span>
      <span class="light light--blue">O2.5 ${stats.over25||0}%</span>
      <span class="light light--amber">BTTS ${stats.btts||0}%</span>`;
  }
}

// -----------------------------------------
// Date & League filter UI
// -----------------------------------------

// Navigate to previous/next fixture in the current visibleFixtures window
function goToPrevFixture() {
  if (!visibleFixtures.length) return;
  const cur = visibleFixtures.findIndex(f => f.__active);
  const idx = cur <= 0 ? visibleFixtures.length - 1 : cur - 1;
  selectIndex(idx, { fly: true });
}

function goToNextFixture() {
  if (!visibleFixtures.length) return;
  const cur = visibleFixtures.findIndex(f => f.__active);
  const idx = (cur + 1) % visibleFixtures.length;
  selectIndex(idx, { fly: true });
}

function buildDateStrip() {
  const base = baseDate();
  const dayA = base;
  const dayB = datePlusDays(base, 1);

  if (el.dateA) {
    el.dateA.textContent = fmtDay(dayA);
    el.dateA.dataset.iso = isoDay(dayA);
  }
  if (el.dateB) {
    el.dateB.textContent = fmtDay(dayB);
    el.dateB.dataset.iso = isoDay(dayB);
  }

  [el.dateToday, el.dateTomorrow, el.dateWeekend]
    .filter(Boolean)
    .forEach(b => b.classList.remove('is-active'));

  if (UI.rangeDays === 1 && UI.offsetDays === 0 && el.dateToday) {
    el.dateToday.classList.add('is-active');
  }
  if (UI.rangeDays === 1 && UI.offsetDays === 1 && el.dateTomorrow) {
    el.dateTomorrow.classList.add('is-active');
  }
  if (UI.rangeDays >= 2 && el.dateWeekend) {
    el.dateWeekend.classList.add('is-active');
  }

  const t = document.getElementById('cal-title');
  if (t) {
    const mo = base.toLocaleString(undefined, {
      month: 'long',
      year: 'numeric',
      timeZone: 'UTC'
    });
    t.textContent = mo;
  }
}

function bindDateControls() {
  // Quick range buttons
  el.dateToday?.addEventListener('click', () => {
    UI.offsetDays = 0;
    UI.rangeDays  = 1;
    buildDateStrip();
    applyFiltersAndRender();
  });

  el.dateTomorrow?.addEventListener('click', () => {
    UI.offsetDays = 1;
    UI.rangeDays  = 1;
    buildDateStrip();
    applyFiltersAndRender();
  });

  el.dateWeekend?.addEventListener('click', () => {
    const b   = baseDate();
    const dow = b.getUTCDay();
    const toSat = (6 - dow + 7) % 7;
    UI.offsetDays = toSat;
    UI.rangeDays  = 2;
    buildDateStrip();
    applyFiltersAndRender();
  });

  // Month navigation
  el.datePrev?.addEventListener('click', () => {
    UI.offsetDays -= UI.rangeDays;
    buildDateStrip();
    applyFiltersAndRender();
  });

  el.dateNext?.addEventListener('click', () => {
    UI.offsetDays += UI.rangeDays;
    buildDateStrip();
    applyFiltersAndRender();
  });

  // Direct date cells (day A/B)
  el.dateA?.addEventListener('click', () => {
    const iso = el.dateA.dataset.iso;
    if (!iso) return;
    UI.offsetDays = Math.round(
      (Date.parse(`${iso}T00:00:00Z`) - Date.parse(`${UI.anchorISO}T00:00:00Z`)) / MS_DAY
    );
    UI.rangeDays = 1;
    buildDateStrip();
    applyFiltersAndRender();
  });

  el.dateB?.addEventListener('click', () => {
    const iso = el.dateB.dataset.iso;
    if (!iso) return;
    UI.offsetDays = Math.round(
      (Date.parse(`${iso}T00:00:00Z`) - Date.parse(`${UI.anchorISO}T00:00:00Z`)) / MS_DAY
    );
    UI.rangeDays = 1;
    buildDateStrip();
    applyFiltersAndRender();
  });

  // Prev/Next fixture buttons under the globe
  const navPrev = document.getElementById('nav-prev');
  const navNext = document.getElementById('nav-next');

  if (navPrev) {
    navPrev.addEventListener('click', (ev) => {
      ev.preventDefault();
      goToPrevFixture();
    });
  }

  if (navNext) {
    navNext.addEventListener('click', (ev) => {
      ev.preventDefault();
      goToNextFixture();
    });
  }

  // Keyboard: left/right arrows also cycle fixtures when home view is active
  window.addEventListener('keydown', (ev) => {
    if (!isHomeActive) return;
    if (ev.key === 'ArrowLeft') {
      ev.preventDefault();
      goToPrevFixture();
    } else if (ev.key === 'ArrowRight') {
      ev.preventDefault();
      goToNextFixture();
    }
  });
}

function buildLeagueChips() {
  if (!el.leagueChips) return;
  const uniq = Array.from(new Set(fixtures.map(f => f.competition).filter(Boolean))).sort();
  UI.leagues = ['ALL', ...uniq];
  el.leagueChips.innerHTML = '';
  for (const name of UI.leagues) {
    const b = document.createElement('button');
    b.className = `chip${name === UI.league ? ' is-active' : ''}`;
    b.dataset.league = name;
    b.textContent = name;
    b.addEventListener('click', () => {
      document
        .querySelectorAll('#league-chips .chip')
        .forEach(x => x.classList.remove('is-active'));
      b.classList.add('is-active');
      UI.league = name;
      applyFiltersAndRender();
    });
    el.leagueChips.appendChild(b);
  }
}
function centerCameraOnVisibleFixtures() {
  if (!visibleFixtures.length || !camera || !controls || !globe) return;

  const sum = new THREE.Vector3(0, 0, 0);
  let count = 0;

  for (const f of visibleFixtures) {
    const lat = Number(f.latitude);
    const lon = Number(f.longitude);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;

    const n = latLngToUnit(lat, lon); // unit vector from globe centre
    sum.add(n);
    count++;
  }

  if (!count) return;

  const avg = sum.multiplyScalar(1 / count).normalize();
  const R   = getGlobeRadius();

  // Target: just above the surface at the cluster centre
  const target = avg.clone().multiplyScalar(R * (1 + SURFACE_EPS));

  // Choose a camera distance that’s closer than the old R*3, but not *too* close
  const minDist     = R * 1.2;
  const maxDist     = R * 1.8;
  const currentDist = camera.position.length() || (R * 2.0);
  const dist        = Math.min(maxDist, Math.max(minDist, currentDist));

  const camPos = avg.clone().multiplyScalar(dist);

  camera.position.copy(camPos);
  controls.target.copy(target);
  controls.update();
}


function applyFiltersAndRender() {
  if (!fixtures.length) return;

  const start = baseDate();
  const end   = datePlusDays(start, UI.rangeDays);

  visibleFixtures = fixtures.filter(f => {
    const d = new Date(f.date_utc);
    if (isNaN(d)) return false;
    if (!(d >= start && d < end)) return false;
    if (UI.league !== 'ALL' &&
        (f.competition || '').toLowerCase() !== UI.league.toLowerCase()) return false;
    return true;
  });

  // Auto-center camera on the cluster of visible fixtures
  centerCameraOnVisibleFixtures();

  const many = visibleFixtures.length > 250;
  globe
    .pointsMerge(many)
    .pointResolution(12)
    .pointLat('latitude')
    .pointLng('longitude')
    .pointsData(visibleFixtures);

  buildRail(visibleFixtures);

  if (visibleFixtures.length) {
    // This will also refresh point colours / altitudes + stadium pill
    selectIndex(0, { fly: true });
  }

  renderCompetitionAccuracy(
    UI.league === 'ALL' ? (visibleFixtures[0]?.competition || '—') : UI.league
  );
}


// -----------------------------------------
// Acca Builder helpers (UI acca builder, not API acca)
// -----------------------------------------
function accaLegKey(fixtureId, marketKey, pickId) {
  return `${fixtureId}__${marketKey}__${pickId}`;
}

function initAccaFromFixtures(){
  const fixtureSelect = document.getElementById('ab-fixture-select');
  const leagueSelect  = document.getElementById('ab-league');
  const marketNav     = document.getElementById('ab-market-nav');
  if (!fixtureSelect || !leagueSelect || !marketNav) return;

  abFixtureById.clear();

  const leagues = Array.from(new Set(fixtures.map(f => f.competition).filter(Boolean))).sort();
  leagueSelect.innerHTML = '<option value="ALL">All Leagues</option>';
  leagues.forEach(name=>{
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    leagueSelect.appendChild(opt);
  });

  function refreshFixtureOptions(){
    const leagueFilter = leagueSelect.value || 'ALL';
    fixtureSelect.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = 'Choose from upcoming matches…';
    fixtureSelect.appendChild(placeholder);

    const filtered = fixtures
      .filter(f => leagueFilter === 'ALL' || (f.competition || '') === leagueFilter)
      .slice()
      .sort((a,b) => new Date(a.date_utc) - new Date(b.date_utc));

    filtered.forEach(f=>{
      const opt = document.createElement('option');
      opt.value = f.fixture_id;
      const d = f.date_utc ? new Date(f.date_utc) : null;
      const timeStr = d && !isNaN(d) ? d.toLocaleTimeString(undefined,{hour:'2-digit', minute:'2-digit'}) : '';
      opt.textContent = `${f.home_team} vs ${f.away_team}${timeStr ? ' • ' + timeStr : ''}`;
      fixtureSelect.appendChild(opt);
      abFixtureById.set(f.fixture_id, f);
    });
  }

  refreshFixtureOptions();

  leagueSelect.addEventListener('change', ()=>{
    refreshFixtureOptions();
    setAccaFixture(null);
  });

  fixtureSelect.addEventListener('change', ()=>{
    const id = fixtureSelect.value;
    const f  = abFixtureById.get(id) || null;
    setAccaFixture(f);
  });

  marketNav.querySelectorAll('.market-chip').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      marketNav.querySelectorAll('.market-chip').forEach(b=>b.classList.remove('is-active'));
      btn.classList.add('is-active');
      abCurrentMarket = btn.dataset.market || 'ftr';
      refreshAccaPicks();
    });
  });

  setAccaFixture(null);
  refreshAccaPicks();
}

function setAccaFixture(f){
  currentFixture = f || null;

  const titleEl   = document.getElementById('ab-fixture-title');
  const metaEl    = document.getElementById('ab-fixture-meta');
  const crestHome = document.getElementById('ab-crest-home');
  const crestAway = document.getElementById('ab-crest-away');

  if (!f){
    if (titleEl) titleEl.textContent = 'Select a fixture';
    if (metaEl)  metaEl.textContent  = '';
    if (crestHome){ crestHome.innerHTML = ''; crestHome.textContent = ''; }
    if (crestAway){ crestAway.innerHTML = ''; crestAway.textContent = ''; }
    refreshAccaPicks();
    return;
  }

  if (titleEl) titleEl.textContent = `${f.home_team} vs ${f.away_team}`;
  if (metaEl){
    const parts = [];
    if (f.competition) parts.push(f.competition);
    if (f.date_utc){
      try{
        const d = new Date(f.date_utc);
        const date = d.toLocaleDateString(undefined,{weekday:'short', day:'2-digit', month:'short'});
        const time = d.toLocaleTimeString(undefined,{hour:'2-digit', minute:'2-digit'});
        parts.push(`${date} · ${time} GMT`);
      }catch{}
    }
    if (f.stadium) parts.push(f.stadium + (f.city ? `, ${f.city}` : ''));
    metaEl.textContent = parts.join(' • ');
  }

  if (crestHome) setBadgeLocal(crestHome, null, f.home_team);
  if (crestAway) setBadgeLocal(crestAway, null, f.away_team);

  refreshAccaPicks();
}

function marketLabelFromKey(key){
  switch(key){
    case 'all':        return 'All markets';
    case 'ftr':        return 'Full Time Result';
    case 'goals_main': return 'Match Goals (Over/Under)';
    case 'btts':       return 'Both Teams To Score';
    default:           return 'Selected market';
  }
}

function refreshAccaPicks(){
  const listEl    = document.getElementById('ab-picks-list');
  const emptyEl   = document.getElementById('ab-picks-empty');
  const summaryEl = document.getElementById('ab-picks-summary');
  if (!listEl || !emptyEl || !summaryEl) return;

  listEl.innerHTML = '';

  if (!currentFixture){
    emptyEl.hidden = false;
    emptyEl.textContent = 'Select a fixture to see model picks.';
    summaryEl.textContent = '';
    return;
  }

  const f = currentFixture;
  const picks = [];

  const wantAll  = (abCurrentMarket === 'all');
  const wantFTR  = (abCurrentMarket === 'ftr'        || wantAll);
  const wantGOAL = (abCurrentMarket === 'goals_main' || wantAll);
  const wantBTTS = (abCurrentMarket === 'btts'       || wantAll);

  if (wantFTR){
    picks.push(
      { id: 'HOME', marketKey: 'ftr', label: `${f.home_team} to Win`, prob: 0.62, fair: 1.61, price: 1.80, edge: 6.5 },
      { id: 'DRAW', marketKey: 'ftr', label: 'Draw',                   prob: 0.22, fair: 4.55, price: 4.75, edge: 1.8 },
      { id: 'AWAY', marketKey: 'ftr', label: `${f.away_team} to Win`,  prob: 0.16, fair: 6.25, price: 6.50, edge: 1.2 }
    );
  }
  if (wantGOAL){
    picks.push(
      { id: 'O25', marketKey: 'goals_main', label: 'Over 2.5 Goals',  prob: 0.71, fair: 1.41, price: 1.65, edge: 6.4 },
      { id: 'U25', marketKey: 'goals_main', label: 'Under 2.5 Goals', prob: 0.29, fair: 3.45, price: 3.60, edge: 1.3 }
    );
  }
  if (wantBTTS){
    picks.push(
      { id: 'BTTS_Y', marketKey: 'btts', label: 'Both Teams To Score – Yes', prob: 0.65, fair: 1.54, price: 1.75, edge: 4.5 },
      { id: 'BTTS_N', marketKey: 'btts', label: 'Both Teams To Score – No',  prob: 0.35, fair: 2.85, price: 3.10, edge: 3.2 }
    );
  }

  if (!picks.length){
    emptyEl.hidden = false;
    emptyEl.textContent = 'This market is not wired yet. Try Full Time Result or Match Goals.';
    summaryEl.textContent = '';
    return;
  }

  emptyEl.hidden = true;
  summaryEl.textContent = `Showing ${picks.length} picks for ${marketLabelFromKey(abCurrentMarket)}.`;

 picks.forEach((pick, idx) => {
  const marketKey = pick.marketKey || abCurrentMarket;
  const key       = accaLegKey(currentFixture.fixture_id, marketKey, pick.id);
  const isInCart  = abCartLegs.some(l => l.key === key);

  const card = document.createElement('article');
  card.className = 'pick-card';
  card.dataset.pickId = pick.id;

  const main = document.createElement('div');
  main.className = 'pick-main';

  // NEW: tiny leg pill at the top of each card
  const legPill = document.createElement('span');
  legPill.className = 'pick-leg-pill';
  legPill.textContent = `Leg ${idx + 1} of ${picks.length}`;

  const h3 = document.createElement('h3');
  h3.className = 'pick-title';
  h3.textContent = pick.label;

  const sub = document.createElement('p');
  sub.className = 'pick-subtitle';
  const probPct   = Math.round(pick.prob * 100);
  const fairText  = pick.fair  != null ? pick.fair.toFixed(2)  : '–';
  const priceText = pick.price != null ? pick.price.toFixed(2) : '–';
  sub.textContent = `Model ${probPct}% • Fair ${fairText} • Price ${priceText}`;

  main.appendChild(legPill);
  main.appendChild(h3);
  main.appendChild(sub);

  const meta = document.createElement('div');
  meta.className = 'pick-meta';

  const badge = document.createElement('div');
  badge.className = 'pick-badge' + (pick.edge != null && pick.edge >= 0 ? ' pick-badge--positive' : '');
  if (pick.edge != null){
    const edgeTxt = pick.edge.toFixed(1);
    badge.textContent = `EV ${pick.edge >= 0 ? '+' : ''}${edgeTxt}%`;
  } else {
    badge.textContent = 'Model pick';
  }

  const btn = document.createElement('button');
  btn.className = 'pick-add-btn';

  if (isInCart) {
    btn.classList.add('pick-add-btn--active');
    btn.textContent = 'Remove';
  } else {
    btn.textContent = '+ Add';
  }

  btn.addEventListener('click', () => addLegToAcca(pick));

  meta.appendChild(badge);
  meta.appendChild(btn);

  card.appendChild(main);
  card.appendChild(meta);
  listEl.appendChild(card);
});
}

function addLegToAcca(pick){
  if (!currentFixture) return;

  const marketKey = pick.marketKey || abCurrentMarket;
  const key       = accaLegKey(currentFixture.fixture_id, marketKey, pick.id);

  const existingIdx = abCartLegs.findIndex(l => l.key === key);
  if (existingIdx >= 0){
    abCartLegs.splice(existingIdx, 1);
  } else {
    abCartLegs.push({
      key,
      fixture_id: currentFixture.fixture_id,
      fixture_label: `${currentFixture.home_team} vs ${currentFixture.away_team}`,
      market: marketKey,
      label: pick.label,
      prob: pick.prob,
      fair: pick.fair,
      price: pick.price,
      edge: pick.edge
    });
  }
  renderAccaCart();
  refreshAccaPicks();
}

function removeLegFromAcca(index){
  abCartLegs.splice(index, 1);
  renderAccaCart();
  refreshAccaPicks();
}

function renderAccaCart(){
  const listEl   = document.getElementById('ab-cart-legs');
  const emptyEl  = document.getElementById('ab-cart-empty');
  const sumEl    = document.getElementById('ab-cart-summary');
  const priceEl  = document.getElementById('ab-cart-price');
  const probEl   = document.getElementById('ab-cart-prob');
  const evEl     = document.getElementById('ab-cart-ev');
  const btnCopy  = document.getElementById('ab-copy-slip');
  const btnCheck = document.getElementById('ab-send-to-checker');
  const saveBtn  = document.getElementById('acc-save');
  const saveHint = document.getElementById('acc-save-hint');

  if (!listEl || !emptyEl || !sumEl) return;

  listEl.innerHTML = '';

  if (!abCartLegs.length){
    emptyEl.hidden = false;
    sumEl.hidden   = true;
    if (priceEl) priceEl.textContent = '–';
    if (probEl)  probEl.textContent  = '–';
    if (evEl)    evEl.textContent    = '–';
    if (btnCopy)  btnCopy.disabled  = true;
    if (btnCheck) btnCheck.disabled = true;
    if (saveBtn)  saveBtn.disabled  = true;
    if (saveHint) saveHint.textContent = 'Build an acca to save it to your portfolio.';
    currentAccaLegs = [];
    return;
  }

  emptyEl.hidden = true;
  sumEl.hidden   = false;
  if (saveBtn)  saveBtn.disabled  = false;
  if (saveHint) saveHint.textContent = currentUser
    ? 'Click “Save to portfolio” to store this acca.'
    : 'Sign in to save this acca to your portfolio.';

  abCartLegs.forEach((leg, idx)=>{
    const li = document.createElement('li');
    li.className = 'acca-leg';

    const main = document.createElement('div');
    main.className = 'acca-leg-main';
    const label = document.createElement('div');
    label.className = 'acca-leg-label';
    label.textContent = `${leg.fixture_label} – ${leg.label}`;
    const sub = document.createElement('div');
    sub.className = 'acca-leg-sub muted';
    const probPct = leg.prob != null ? Math.round(leg.prob*100) + '%' : '–';
    const fairTxt  = leg.fair  != null ? leg.fair.toFixed(2)  : '–';
    const priceTxt = leg.price != null ? leg.price.toFixed(2) : '–';
    const edgeTxt  = leg.edge  != null ? (leg.edge >=0 ? '+' : '') + leg.edge.toFixed(1) + '%' : '–';
    sub.textContent = `Model ${probPct} • Fair ${fairTxt} • Price ${priceTxt} • EV ${edgeTxt}`;
    main.appendChild(label);
    main.appendChild(sub);

    const btn = document.createElement('button');
    btn.className = 'acca-leg-remove';
    btn.textContent = '✕';
    btn.addEventListener('click', ()=> removeLegFromAcca(idx));

    li.appendChild(main);
    li.appendChild(btn);
    listEl.appendChild(li);
  });

  // Combined stats (demo)
  let comboProb = 1;
  let avgEdge   = 0;
  let comboPrice = 1;
  let countWithEdge = 0;
  abCartLegs.forEach(leg=>{
    if (leg.prob  != null) comboProb  *= leg.prob;
    if (leg.price != null) comboPrice *= leg.price;
    if (leg.edge  != null){ avgEdge += leg.edge; countWithEdge++; }
  });
  if (countWithEdge) avgEdge /= countWithEdge;

  if (priceEl) priceEl.textContent = comboPrice && isFinite(comboPrice) ? comboPrice.toFixed(2) : '–';
  if (probEl)  probEl.textContent  = comboProb && isFinite(comboProb) ? Math.round(comboProb*100) + '%' : '–';
  if (evEl)    evEl.textContent    = countWithEdge ? (avgEdge >=0 ? '+' : '') + avgEdge.toFixed(1) + '%' : '–';

  if (btnCopy){
    btnCopy.disabled = false;
    btnCopy.onclick = ()=> copyAccaToClipboard();
  }
  if (btnCheck){
    btnCheck.disabled = false;
    btnCheck.onclick = ()=> {
      showToast('info','Bet Checker integration coming soon.');
    };
  }

  // keep snapshot for portfolio save
  currentAccaLegs = abCartLegs.map(l => ({ ...l }));
}

function copyAccaToClipboard(){
  if (!abCartLegs.length) return;
  const lines = abCartLegs.map(leg=>{
    const priceTxt = leg.price != null ? '@ ' + leg.price.toFixed(2) : '';
    return `${leg.fixture_label} – ${leg.label} ${priceTxt}`;
  });
  const text = lines.join('\n');
  if (navigator.clipboard && navigator.clipboard.writeText){
    navigator.clipboard.writeText(text).then(()=>{
      showToast('success','Acca copied to clipboard');
    }).catch(()=>{
      showToast('error','Could not copy acca');
    });
  } else {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); showToast('success','Acca copied to clipboard'); }
    catch { showToast('error','Could not copy acca'); }
    ta.remove();
  }
}

// ----------------------------
// Portfolio stats helpers
// ----------------------------
function computePortfolioStats() {
  const userEmail = currentUser?.email || null;
  const userAccas = savedAccas.filter(a =>
    !userEmail || !a.userEmail || a.userEmail === userEmail
  );

  if (!userAccas.length) {
    return {
      count: 0,
      totalStake: 0,
      totalReturn: 0,
      netProfit: 0,
      wins: 0,
      losses: 0,
      strikeRate: 0
    };
  }

  let totalStake  = 0;
  let totalReturn = 0;
  let wins        = 0;
  let losses      = 0;

  userAccas.forEach(acc => {
    const stake = Number(acc.stake) || 0;
    totalStake += stake;

    const combinedPrice = acc.legs?.reduce((prod, leg) => {
      return prod * (leg.price != null ? leg.price : 1);
    }, 1) || 1;

    const returnAmount = stake * combinedPrice;

    if (acc.status === 'won') {
      wins += 1;
      totalReturn += returnAmount;
    } else if (acc.status === 'lost') {
      losses += 1;
    }
  });

  const netProfit  = totalReturn - totalStake;
  const strikeRate = wins + losses > 0 ? (wins / (wins + losses)) * 100 : 0;

  return {
    count: userAccas.length,
    totalStake,
    totalReturn,
    netProfit,
    wins,
    losses,
    strikeRate
  };
}

function renderPortfolioStats() {
  const elStats = document.getElementById('portfolio-stats');
  if (!elStats) return;

  const stats = computePortfolioStats();

  if (!stats.count) {
    elStats.innerHTML = `
      <span class="muted">
        No saved accas yet. Build one in the Acca Builder and click “Save to portfolio”.
      </span>`;
    return;
  }

  const fmtMoney = v => '£' + v.toFixed(2);
  const fmtPct   = v => v.toFixed(1) + '%';

  elStats.innerHTML = `
    <div class="portfolio-stats__item">
      <span class="portfolio-stats__label">Accas:</span>
      <span class="portfolio-stats__value">${stats.count}</span>
    </div>
    <div class="portfolio-stats__item">
      <span class="portfolio-stats__label">Staked:</span>
      <span class="portfolio-stats__value">${fmtMoney(stats.totalStake)}</span>
    </div>
    <div class="portfolio-stats__item">
      <span class="portfolio-stats__label">Returned:</span>
      <span class="portfolio-stats__value">${fmtMoney(stats.totalReturn)}</span>
    </div>
    <div class="portfolio-stats__item">
      <span class="portfolio-stats__label">Net P/L:</span>
      <span class="portfolio-stats__value">${fmtMoney(stats.netProfit)}</span>
    </div>
    <div class="portfolio-stats__item">
      <span class="portfolio-stats__label">Strike rate:</span>
      <span class="portfolio-stats__value">${fmtPct(stats.strikeRate)}</span>
    </div>
  `;
}

// ----------------------------
// Portfolio (saved accas)
// ----------------------------
function getCurrentAccaSnapshot() {
  if (!currentFixture || !abCartLegs || !abCartLegs.length) return null;

  return {
    id: `acc_${Date.now()}`,
    userEmail: currentUser?.email || null,
    name: `${currentFixture.home_team} vs ${currentFixture.away_team}`,
    createdAt: new Date().toISOString(),
    fixture: {
      home_team:   currentFixture.home_team,
      away_team:   currentFixture.away_team,
      date_utc:    currentFixture.date_utc,
      competition: currentFixture.competition,
      stadium:     currentFixture.stadium,
      city:        currentFixture.city,
      country:     currentFixture.country,
    },
    legs: abCartLegs.map(l => ({ ...l })),
    status: 'pending',
    stake: 0
  };
}

function saveCurrentAccaToPortfolio() {
  if (!currentUser) {
    showToast('error', 'Sign in to save your acca');
    return;
  }
  const snapshot = getCurrentAccaSnapshot();
  if (!snapshot) {
    showToast('error', 'No acca to save yet');
    return;
  }
  savedAccas.push(snapshot);
  persistSession();
  showToast('success', 'Acca saved to your portfolio');
  renderPortfolio();
  renderPortfolioStats();
}

function renderPortfolio() {
  const container = document.getElementById('portfolio-list');
  if (!container) return;

  container.innerHTML = '';

  const userAccas = savedAccas.filter(
    a => !currentUser || !a.userEmail || a.userEmail === currentUser.email
  );

  if (!userAccas.length) {
    container.innerHTML = `
      <p class="muted">
        No saved accas yet. Build one in the Acca Builder and click “Save to portfolio”.
      </p>`;
    return;
  }

  userAccas
    .slice()
    .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
    .forEach(acc => {
      const when  = new Date(acc.createdAt).toLocaleString();
      const legs  = Array.isArray(acc.legs) ? acc.legs : [];
      const stake = typeof acc.stake === 'number' && !Number.isNaN(acc.stake)
        ? acc.stake.toFixed(2)
        : '0.00';
      const status = acc.status || 'pending';

      const locParts = [];
      if (acc.fixture?.city)    locParts.push(acc.fixture.city);
      if (acc.fixture?.country) locParts.push(acc.fixture.country);
      const locText = locParts.join(', ');

      const card = document.createElement('div');
      card.className = 'insight-card portfolio-card';
      card.innerHTML = `
      <header class="portfolio-card__header">
        <div>
          <h2>${acc.name}</h2>
          <p class="muted">
            ${when}
            ${acc.fixture?.competition ? ' • ' + acc.fixture.competition : ''}
            ${locText ? ' • ' + locText : ''}
          </p>
          <p class="muted">${legs.length} leg(s)</p>
        </div>
      </header>
    
      ${
        legs.length
          ? `
            <ul class="portfolio-leg-list">
              ${legs.map(l => {
                const probPct = l.prob  != null ? Math.round(l.prob * 100) + '%' : '–';
                const fairTxt  = l.fair  != null ? l.fair.toFixed(2)          : '–';
                const priceTxt = l.price != null ? l.price.toFixed(2)         : '–';
                const edgeTxt  = l.edge  != null
                  ? (l.edge >= 0 ? '+' : '') + l.edge.toFixed(1) + '%'
                  : '–';
                const label    = l.label || '';
                const fixLabel = l.fixture_label ||
                  `${acc.fixture.home_team} vs ${acc.fixture.away_team}`;
                return `
                  <li class="portfolio-leg">
                    <div class="portfolio-leg__label">${fixLabel} – ${label}</div>
                    <div class="portfolio-leg__meta muted">
                      Model ${probPct}
                      • Fair ${fairTxt}
                      • Price ${priceTxt}
                      • EV ${edgeTxt}
                    </div>
                  </li>
                `;
              }).join('')}
            </ul>
          `
          : '<p class="muted">No legs saved on this slip.</p>'
      }
    
      <div class="portfolio-meta-row">
        <div class="portfolio-meta-col">
          <span class="control-label">Status</span>
          <span class="portfolio-status" data-status="${acc.id}">
            ${status}
          </span>
        </div>
        <label class="control control--compact portfolio-meta-col">
          <span class="control-label">Stake (£)</span>
          <input
            type="number"
            min="0"
            step="0.01"
            class="portfolio-stake-input"
            data-stake="${acc.id}"
            value="${stake}"
          />
        </label>
      </div>
    
      <div class="portfolio-actions">
        <div class="portfolio-actions-left">
          <button class="cta" data-load-acca="${acc.id}">
            Open in builder
          </button>
          <button class="cta cta--secondary" data-status-win="${acc.id}">
            Mark as won
          </button>
          <button class="cta cta--secondary" data-status-loss="${acc.id}">
            Mark as lost
          </button>
        </div>
        <div class="portfolio-actions-right">
          <button class="cta cta--ghost" data-status-reset="${acc.id}">
            Reset
          </button>
          <button
            class="cta cta--ghost cta--danger"
            type="button"
            data-delete-acca="${acc.id}"
            aria-label="Delete saved acca"
          >
            ✕
          </button>
        </div>
      </div>
    `;

      container.appendChild(card);
    });

  // Wire: open in builder (fixture + legs, with clear confirm)
  container.querySelectorAll('button[data-load-acca]').forEach(btn => {
    btn.addEventListener('click', () => {
      const id  = btn.getAttribute('data-load-acca');
      const acc = savedAccas.find(a => a.id === id);
      if (!acc) return;

      if (abCartLegs.length && acc.legs && acc.legs.length) {
        const ok = window.confirm('Picks already in Builder – clear current selections?');
        if (!ok) return;
      }

      const match = fixtures.find(
        f =>
          f.home_team === acc.fixture.home_team &&
          f.away_team === acc.fixture.away_team &&
          f.date_utc  === acc.fixture.date_utc
      );
      if (!match) {
        showToast('error', 'Original fixture not found in current data set');
        return;
      }

      setAccaFixture(match);

      abCartLegs = Array.isArray(acc.legs) ? acc.legs.map(l => ({ ...l })) : [];
      renderAccaCart();
      refreshAccaPicks();

      window.location.hash = '#/acca-builder';
      showToast('info', 'Loaded saved acca into builder');
    });
  });

  // Wire: stake input change
  container.querySelectorAll('.portfolio-stake-input').forEach(input => {
    input.addEventListener('change', () => {
      const id   = input.getAttribute('data-stake');
      const acc  = savedAccas.find(a => a.id === id);
      if (!acc) return;
      const val  = parseFloat(input.value);
      acc.stake  = !Number.isNaN(val) && val >= 0 ? val : 0;
      persistSession();
      renderPortfolioStats();
    });
  });

  // Wire: mark won / lost / reset
  container.querySelectorAll('[data-status-win]').forEach(btn => {
    btn.addEventListener('click', () => {
      const id  = btn.getAttribute('data-status-win');
      const acc = savedAccas.find(a => a.id === id);
      if (!acc) return;
      acc.status = 'won';
      persistSession();
      renderPortfolio();
      renderPortfolioStats();
    });
  });
  container.querySelectorAll('[data-status-loss]').forEach(btn => {
    btn.addEventListener('click', () => {
      const id  = btn.getAttribute('data-status-loss');
      const acc = savedAccas.find(a => a.id === id);
      if (!acc) return;
      acc.status = 'lost';
      persistSession();
      renderPortfolio();
      renderPortfolioStats();
    });
  });
  container.querySelectorAll('[data-status-reset]').forEach(btn => {
    btn.addEventListener('click', () => {
      const id  = btn.getAttribute('data-status-reset');
      const acc = savedAccas.find(a => a.id === id);
      if (!acc) return;
      acc.status = 'pending';
      acc.stake  = 0;
      persistSession();
      renderPortfolio();
      renderPortfolioStats();
    });
  });

  // Wire: delete with confirm
  container.querySelectorAll('[data-delete-acca]').forEach(btn => {
    btn.addEventListener('click', () => {
      const id  = btn.getAttribute('data-delete-acca');
      const acc = savedAccas.find(a => a.id === id);
      if (!acc) return;

      const ok = window.confirm('Delete slip? Are you sure?');
      if (!ok) return;

      savedAccas = savedAccas.filter(a => a.id !== id);
      persistSession();
      renderPortfolio();
      renderPortfolioStats();
      showToast('info', 'Slip deleted from portfolio');
    });
  });
}

// -----------------------------------------
// Three-Globe loader & scene init
// -----------------------------------------
// Helper: load UMD script and resolve when window.ThreeGlobe is ready
function loadThreeGlobeUMD(url) {
  return new Promise((resolve, reject) => {
    // Make sure the UMD build sees THREE on the global object
    if (!window.THREE) {
      window.THREE = THREE; // the ES-module import from the top of this file
    }

    // Already loaded?
    if (window.ThreeGlobe) {
      return resolve(window.ThreeGlobe);
    }

    const s = document.createElement('script');
    s.src = url;
    s.async = true;

    s.onload = () => {
      if (window.ThreeGlobe) {
        resolve(window.ThreeGlobe);
      } else {
        reject(new Error('ThreeGlobe global not found after ' + url));
      }
    };

    s.onerror = () => {
      reject(new Error('Failed to load three-globe from ' + url));
    };

    document.head.appendChild(s);
  });
}


async function loadThreeGlobe() {
  // Try public CDNs, then your local vendored copy.
  const candidates = [
    'https://cdn.jsdelivr.net/npm/three-globe@2.31.1/dist/three-globe.min.js',
    'https://unpkg.com/three-globe@2.31.1/dist/three-globe.min.js',
    './vendor/three-globe.min.js'
  ];

  for (const url of candidates) {
    try {
      const ctor = await loadThreeGlobeUMD(url);
      console.warn('[three-globe] using UMD build from:', url);
      return ctor;
    } catch (err) {
      console.warn('[three-globe] failed to load from', url, err);
    }
  }

  throw new Error('three-globe failed to load');
}


async function init(){
  loadSessionFromStorage();
  updateAuthUI();

  ThreeGlobeCtor = await loadThreeGlobe();

  scene = new THREE.Scene();
  renderer = new THREE.WebGLRenderer({ antialias:true, alpha:true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
  renderer.setSize(el.globeWrap.clientWidth, el.globeWrap.clientHeight);
  el.globeWrap.innerHTML = '';
  el.globeWrap.appendChild(renderer.domElement);
  if ('outputColorSpace' in renderer) renderer.outputColorSpace = THREE.SRGBColorSpace;

  camera = new THREE.PerspectiveCamera(
    45,
    el.globeWrap.clientWidth/el.globeWrap.clientHeight,
    0.1,
    5000
  );
  camera.position.set(0, 0, getGlobeRadius()*3);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.enableZoom = true;
  controls.autoRotate = false;
  controls.minDistance = getGlobeRadius()*1.2;
  controls.maxDistance = getGlobeRadius()*6;

  // --- Postprocessing: FXAA + Bloom ---
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));

  const fxaa = new ShaderPass(FXAAShader);
  const setFXAA = () => {
    const px = renderer.getPixelRatio();
    fxaa.material.uniforms['resolution'].value.set(
      1 / (el.globeWrap.clientWidth * px),
      1 / (el.globeWrap.clientHeight * px)
    );
  };
  setFXAA();
  composer.addPass(fxaa);

  const bloom = new UnrealBloomPass(
    new THREE.Vector2(el.globeWrap.clientWidth, el.globeWrap.clientHeight),
    BLOOM.strength,
    BLOOM.radius,
    BLOOM.threshold
  );
  composer.addPass(bloom);

  scene.add(new THREE.AmbientLight(0xffffff,0.9));
  const hemi = new THREE.HemisphereLight(0xddeeff, 0x223344, 0.6);
  scene.add(hemi);

  globe = new ThreeGlobeCtor({ waitForGlobeReady:true })
    .showAtmosphere(true).atmosphereColor('#9ef9e3').atmosphereAltitude(0.28)
    .globeImageUrl('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
    .bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png')
    // Tiny cyan dots for all fixtures, slightly higher + warmer for the active one
    .pointAltitude(d => d.__active ? SURFACE_EPS * 1.7 : SURFACE_EPS)
    .pointRadius(d => d.__active ? RADIUS_ACTIVE : RADIUS_BASE)
    .pointColor(d => d.__active ? COLORS.markerActive : COLORS.marker)
    .pointsMerge(true);


  scene.add(globe);

  // custom marker group
  MARKER = createMarker();
  scene.add(MARKER.group);

  // click handler for pill sprite
  renderer.domElement.addEventListener('click', onCanvasClick);

  // hover & click on globe points
  if (typeof globe.onPointHover === 'function') globe.onPointHover(handleHover);
  globe.onPointClick?.(pt=>{
    if (!pt) return;
    const idx = visibleFixtures.findIndex(f=>f.latitude===pt.latitude && f.longitude===pt.longitude);
    if (idx>=0) selectIndex(idx, { fly:true });
  });

  // --- Resize: reuse setFXAA defined above ---
  window.addEventListener('resize', () => {
    const {clientWidth:w, clientHeight:h} = el.globeWrap;
    renderer.setSize(w,h);
    camera.aspect = w/h;
    camera.updateProjectionMatrix();
    setFXAA();
  });

  bindHeaderNav();
  bindTabs();
  bindDateControls();
  bindSheet();

  // deep-dive opens sheet for current fixture
  el.deepBtn?.addEventListener('click', () => {
    if (currentFixture) openSheetForFixture(currentFixture);
  });

  await loadFixturesCSV('./data/fixtures.csv');
  buildLeagueChips();
  buildDateStrip();
  applyFiltersAndRender();
  initAccaFromFixtures();

  // Save to portfolio button
  const accSaveBtn   = document.getElementById('acc-save');
  const accSaveHint  = document.getElementById('acc-save-hint');
  if (accSaveBtn) {
    accSaveBtn.addEventListener('click', () => {
      saveCurrentAccaToPortfolio();
      if (accSaveHint) {
        accSaveHint.textContent = currentUser
          ? 'Acca saved. View it in your Portfolio tab.'
          : 'Sign in to save accas permanently.';
      }
    });
  }

  // loop
  (function loop(){
    requestAnimationFrame(loop);
    if (!isHomeActive) return;
    controls.update();
    composer.render();
  })();
}


// -----------------------------------------
// CSV ingest
// -----------------------------------------
async function loadFixturesCSV(url){
  const res = await fetch(`${url}?v=${Date.now()}`); // cache-bust CSV only
  if (!res.ok){
    showToast('error',`Could not load ${url} (HTTP ${res.status}).`);
    return;
  }
  const text = await res.text();
  const { data, errors } = Papa.parse(text, { header:true, skipEmptyLines:true });
  if (errors?.length) console.warn('[CSV parse errors]', errors);

  fixtures = (data||[])
  .map(row=>{
    // --- existing parse, but we keep raw values separate ---
    const rawLat = parseFloat(row.latitude ?? row.lat ?? row.Latitude ?? row.lat_deg);
    const rawLon = parseFloat(row.longitude ?? row.lon ?? row.lng ?? row.Longitude);

    // Apply global offset (demo alignment tweak)
    let lat = Number.isFinite(rawLat) ? rawLat + MAP_OFFSET.latBias : NaN;
    let lon = Number.isFinite(rawLon) ? rawLon + MAP_OFFSET.lonBias : NaN;

    // Normalise longitude into [-180, 180] so Three-Globe stays happy
    if (Number.isFinite(lon)) {
      if (lon > 180) lon -= 360;
      if (lon < -180) lon += 360;
    }

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
      latitude:  Number.isFinite(lat) ? lat : undefined,
      longitude: Number.isFinite(lon) ? lon : undefined,
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

function selectIndex(idx, { fly = false } = {}) {
  if (!visibleFixtures.length) return;
  const f = visibleFixtures[idx];
  if (!f) return;

  currentFixture = f;

  // Mark which fixture is active
  visibleFixtures.forEach(it => {
    it.__active = (it === f);
  });

  // Re-apply point styling so active dot “pops”
  globe
    .pointColor(d => d.__active ? COLORS.markerActive : COLORS.marker)
    .pointRadius(d => d.__active ? RADIUS_ACTIVE : RADIUS_BASE)
    .pointAltitude(d => d.__active ? SURFACE_EPS * 1.7 : SURFACE_EPS)
    .pointsTransitionDuration?.(220);

  moveMarkerToFixture(f, { fly });
  renderPanel(f);
  syncRail(idx);
}


function elmEmpty(msg){
  const d=document.createElement('div');
  d.className='empty';
  d.textContent=msg;
  return d;
}

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
      shots.forEach(s=>{
        const row=document.createElement('div');
        row.className='row';
        row.textContent=s;
        el.watchlist.appendChild(row);
      });
    } else el.watchlist.appendChild(elmEmpty('No player highlights available.'));
  }

  if (el.market){
    el.market.innerHTML = `
      <div><strong>Over 2.5 goals:</strong> ${Math.round((f.over25_prob||0)*100)}%</div>
      <div><strong>Both teams to score:</strong> ${Math.round((f.btts_prob ||0)*100)}%</div>`;
  }

  renderCompetitionAccuracy(f.competition);
}

// ----------------------------
// Header nav bindings (profile menu + side drawer)
// ----------------------------
function bindHeaderNav() {
  const profileBtn  = document.getElementById('btn-profile');
  const profileMenu = document.getElementById('profile-menu');

  if (profileBtn && profileMenu) {
    profileBtn.addEventListener('click', () => {
      const isOpen = profileMenu.classList.toggle('show');
      profileBtn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
      profileMenu.setAttribute('aria-hidden', isOpen ? 'false' : 'true');
    });
  }

  const btnMenu   = document.getElementById('btn-menu');
  const sideDrawer= document.getElementById('side-drawer');
  const scrim     = document.getElementById('scrim');
  const btnClose  = document.getElementById('btn-close-drawer');

  if (btnMenu && sideDrawer && scrim) {
    const openDrawer = () => {
      sideDrawer.classList.add('show');
      scrim.classList.add('show');
      sideDrawer.setAttribute('aria-hidden', 'false');
    };
    const closeDrawer = () => {
      sideDrawer.classList.remove('show');
      scrim.classList.remove('show');
      sideDrawer.setAttribute('aria-hidden', 'true');
    };

    btnMenu.addEventListener('click', openDrawer);
    btnClose?.addEventListener('click', closeDrawer);
    scrim.addEventListener('click', closeDrawer);
  }
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

function openSheetForFixture(f) {
  const sheet = document.getElementById('sheet');
  if (!sheet || !f) return;

  sheet.classList.add('open');
  sheet.setAttribute('aria-hidden', 'false');

  const titleEl = document.getElementById('sheet-title');
  const bodyEl  = document.getElementById('sheet-body');

  if (titleEl) {
    titleEl.textContent = `${f.home_team} vs ${f.away_team}`;
  }
  if (bodyEl) {
    const fmtTime = (iso) => {
      try {
        const d = new Date(iso);
        const date = d.toLocaleDateString(undefined,{weekday:'short', day:'2-digit', month:'short'});
        const time = d.toLocaleTimeString(undefined,{hour:'2-digit', minute:'2-digit'});
        return `${date} · ${time}`;
      } catch { return iso || ''; }
    };

    bodyEl.innerHTML = `
      <p><strong>Competition:</strong> ${f.competition || '—'}</p>
      <p><strong>Kick-off:</strong> ${fmtTime(f.date_utc)}</p>
      <p><strong>Stadium:</strong> ${f.stadium || '—'}</p>
      <p><strong>City:</strong> ${f.city || '—'}</p>
      <p><strong>Country:</strong> ${f.country || '—'}</p>
      <p><strong>Prediction:</strong> ${f.predicted_winner || '—'} ${f.confidence_ftr ? `(${Math.round(f.confidence_ftr*100)}%)` : ''}</p>
      <p><strong>xG:</strong> ${(f.xg_home||0).toFixed(1)} vs ${(f.xg_away||0).toFixed(1)}</p>
      <p><strong>PPG:</strong> ${(f.ppg_home||0).toFixed(1)} vs ${(f.ppg_away||0).toFixed(1)}</p>
    `;
  }
}

function closeSheet() {
  const sheet = document.getElementById('sheet');
  if (!sheet) return;
  sheet.classList.remove('open');
  sheet.setAttribute('aria-hidden', 'true');
}

function bindSheet() {
  const sheet  = document.getElementById('sheet');
  const handle = sheet?.querySelector('.sheet__handle');
  handle?.addEventListener('click', closeSheet);
}

// ----------------------------
// Marker creation & movement
// ----------------------------
function createMarker(){
  const group = new THREE.Group();
  group.visible = false;

  const R = getGlobeRadius();

  const radarRings = [];
  const baseInner = R * 0.02;
  const baseWidth = R * 0.008;

  for (let i = 0; i < 4; i++) {
    const inner = baseInner + i * (baseWidth * 0.9);
    const outer = inner + baseWidth * (1 - i * 0.2);
    const baseAlpha = 0.35 * (1 - i * 0.18);

    const ringGeom = new THREE.RingGeometry(inner, outer, 64);
    const ringMat  = new THREE.MeshBasicMaterial({
      color: new THREE.Color(COLORS.ring),
      transparent: true,
      opacity: baseAlpha,
      side: THREE.DoubleSide,
      depthWrite: false,
      depthTest: false
    });
    const ring = new THREE.Mesh(ringGeom, ringMat);
    ring.rotation.x = Math.PI / 2;
    ring.renderOrder = 998;
    ring.userData.baseAlpha = baseAlpha;
    group.add(ring);
    radarRings.push(ring);
  }

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

  const billboardMat = new THREE.SpriteMaterial({
    transparent: true,
    opacity: 0,
    depthTest: false,
    depthWrite: false
  });
  const billboard = new THREE.Sprite(billboardMat);
  const baseScaleX = 15;
  const baseScaleY = 7.5;
  billboard.scale.set(baseScaleX, baseScaleY, 1);
  billboard.userData.baseScale = { x: baseScaleX, y: baseScaleY };
  billboard.renderOrder = 999;
  group.add(billboard);

  return {
    group,
    radar: radarRings,
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

function moveMarkerToFixture(f, { fly = false } = {}) {
  if (!MARKER || !f) return;

  console.log('[marker] move to', f.home_team, f.city || f.country, f.latitude, f.longitude);
  const S   = MARKER;
  const lat = Number(f.latitude);
  const lon = Number(f.longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    S.group.visible = false;
    return;
  }

  S.state.reqId++;
  const myReq = S.state.reqId;
  const R     = getGlobeRadius();

  const toN   = latLngToUnit(lat, lon);
  const fromN = S.group.visible ? S.group.position.clone().normalize() : toN.clone();

  const angle = Math.acos(THREE.MathUtils.clamp(fromN.dot(toN), -1, 1));
  const distK = angle * R;
  const dur   = (fly && S.group.visible)
    ? THREE.MathUtils.clamp(distK * 2.0, 300, 900)
    : 0;
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

  S.raf.travel.run(() => {
    if (S.state.reqId !== myReq) { S.raf.travel.cancel(); return; }

    const t = dur ? Math.min(1, (performance.now() - t0) / dur) : 1;
    const k = easeInOut(t);

    const curN     = slerpUnitVec(fromN, toN, k);
    const worldPos = curN.clone().multiplyScalar(R * (1 + SURFACE_EPS));
    S.group.position.copy(worldPos);

    const up = curN.clone().normalize();
    const toCam = camera.position.clone().sub(worldPos).normalize();
    let forward = toCam.clone().sub(up.clone().multiplyScalar(toCam.dot(up)));
    if (forward.lengthSq() < 1e-6) forward = new THREE.Vector3(0,0,1);
    else forward.normalize();
    const right = new THREE.Vector3().crossVectors(forward, up).normalize();
    forward.crossVectors(up, right).normalize();

    const basis = new THREE.Matrix4();
    basis.makeBasis(right, up, forward);
    S.group.quaternion.setFromRotationMatrix(basis);

    if (t >= 1) {
      S.raf.travel.cancel();

      // Beam animation
      S.beam.position.set(0, 0, 0);
      S.beam.quaternion.identity();
      S.beam.scale.set(1, 0.001, 1);
      S.beam.material.opacity = 0.0;
      S.beam.visible = true;

      const b0 = performance.now();
      const bd = 550;
      S.raf.beam.run(() => {
        if (S.state.reqId !== myReq) { S.raf.beam.cancel(); return; }
        const tb = Math.min(1, (performance.now() - b0) / bd);
        const e  = easeInOut(tb);
        S.beam.scale.y          = 0.001 + e;
        S.beam.material.opacity = 0.5 * e;
        if (tb >= 1) S.raf.beam.cancel();
      });

      // Radar breathing
      const rings      = Array.isArray(S.radar) ? S.radar : (S.radar ? [S.radar] : []);
      const radarStart = performance.now();
      rings.forEach(r => { if (r) r.visible = true; });

      S.raf.radar.run(() => {
        if (S.state.reqId !== myReq) {
          S.raf.radar.cancel();
          return;
        }
        const now   = performance.now();
        const tWave = (now - radarStart) / 800;
        rings.forEach((ring, idx) => {
          if (!ring) return;
          const base  = ring.userData?.baseAlpha ?? 0.25;
          const phase = tWave + idx * 0.8;
          const sinVal= Math.sin(phase);
          const wave  = 0.2 + 0.8 * Math.max(0, sinVal);
          ring.material.opacity = base * wave;
        });
      });

      // Pill positioning & texture
      const PILL_ALT = R * 0.05;
      const PILL_OUT = R * 0.03;
      S.billboard.position.set(0, PILL_ALT, PILL_OUT);
      S.billboard.material.rotation = 0;

      const basePillTex = makeStadiumPillTexture(f, null);
      S.billboard.material.map = basePillTex;
      S.billboard.material.needsUpdate = true;
      S.billboard.material.opacity = 1.0;
      S.billboard.visible = true;

      if (curN.dot(camera.position.clone().normalize()) < -0.25) {
        S.billboard.visible = false;
      }

      (async () => {
        if (S.state.reqId !== myReq) return;
        let stadiumImage = null;
        for (const url of stadiumCandidates(f)) {
          try {
            const tex = await loadTextureQueued(url);
            if (S.state.reqId !== myReq) return;
            stadiumImage = tex.image || null;
            break;
          } catch {}
        }
        if (!stadiumImage || S.state.reqId !== myReq) return;
        const pillTex = makeStadiumPillTexture(f, stadiumImage);
        S.billboard.material.map = pillTex;
        S.billboard.material.needsUpdate = true;
        S.billboard.visible = true;
      })();
    }
  });
}

function onCanvasClick(event) {
  if (!MARKER || !MARKER.billboard || !currentFixture || !renderer || !camera) return;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObject(MARKER.billboard, false);
  if (hits.length > 0) {
    openSheetForFixture(currentFixture);
  }
}

// =====================================================
// API HELPERS + FEATURE PAGES (BetChecker / Acca / Co-Pilot)
// =====================================================
const API_BASE = '/api';

async function apiJson(url, opts = {}) {
  const res = await fetch(`${API_BASE}${url}`, {
    ...opts,
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
    credentials: 'include'
  });
  if (!res.ok) {
    const t = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status}: ${t || res.statusText}`);
  }
  return res.json();
}

const API = {
  scoreSlip:   (payload) => apiJson('/score-slip', { method: 'POST', body: JSON.stringify(payload) }),
  accaSuggest: (q)       => apiJson(`/acca/suggest?${new URLSearchParams(q)}`),
  accaOptimise:(p)       => apiJson('/acca/optimise', { method: 'POST', body: JSON.stringify(p) }),
  copilot:     (p)       => apiJson('/copilot', { method: 'POST', body: JSON.stringify(p) })
};

// ---------- Bet Checker ----------

// ✅ Correct OCR helper (this is what runBetChecker calls)
async function ocrImageOrPdf(file) {
  if (!window.Tesseract) throw new Error('OCR engine not loaded');
  const { data } = await window.Tesseract.recognize(file, 'eng', { logger: () => {} });
  return (data && data.text) ? data.text : '';
}

// --- Helpers for BetChecker parsers ---
function fracToDecimal(fracStr) {
  const m = String(fracStr).trim().match(/^(\d+)\s*\/\s*(\d+)$/);
  if (!m) return null;
  const num = parseFloat(m[1]);
  const den = parseFloat(m[2]);
  if (!den) return null;
  return 1 + num / den;
}

// Generic “TeamA v TeamB” parser – works for classic match slips
function parseGenericSlip(text) {
  const lines = String(text).split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  const legs = [];

  for (let i = 0; i < lines.length; i++) {
    const L = lines[i];
    // “TeamA v TeamB” / “TeamA vs TeamB”
    const m = L.match(/^\s*([A-Za-z0-9 .'-]+)\s+(?:V|VS\.?)\s+([A-Za-z0-9 .'-]+)\s*$/i);
    if (!m) continue;

    const home = m[1].trim();
    const away = m[2].trim();

    // Look ahead a few lines for the market/price
    for (let j = 1; j <= 3 && (i + j) < lines.length; j++) {
      const M = lines[i + j];
      let market = null;
      let pick   = null;

      if (/OVER\s*2\.?5/i.test(M))      { market = 'OVER_UNDER_2_5'; pick = 'OVER'; }
      else if (/UNDER\s*2\.?5/i.test(M)){ market = 'OVER_UNDER_2_5'; pick = 'UNDER'; }
      else if (/BOTH\s*TEAMS\s*TO\s*SCORE|BTTS/i.test(M)) {
        market = 'BTTS';
        pick   = /\bNO\b/i.test(M) ? 'NO' : 'YES';
      } else if (/(?:^|\s)(?:1X2|HOME|AWAY|DRAW|1|2|X)(?:\s|$)/i.test(M)) {
        market = 'FTR';
        if      (/\bDRAW\b|(?:^|\s)X(?:\s|$)/i.test(M)) pick = 'DRAW';
        else if (/\bHOME\b|(?:^|\s)1(?:\s|$)/i.test(M)) pick = 'HOME';
        else if (/\bAWAY\b|(?:^|\s)2(?:\s|$)/i.test(M)) pick = 'AWAY';
      }

      if (!market) continue;

      let price = null;
      const frac = M.match(/(\d+)\s*\/\s*(\d+)/);
      const dec  = M.match(/(\d+(?:\.\d+)?)/);
      if (frac)      price = (parseFloat(frac[1]) / parseFloat(frac[2])) + 1;
      else if (dec)  price = parseFloat(dec[1]);

      legs.push({
        teamHome:   home,
        teamAway:   away,
        market,
        selection:  pick || '—',
        price,
        bookmaker:  null,
        kickoffUTC: null
      });
      break; // stop after first market line for this fixture
    }
  }

  return { legs, raw: lines.slice(0, 60).join('\n') };
}

// --- Betfred-specific parser (team total 1.5+ etc) ---
function parseBetfred(text) {
  const lines = String(text)
    .split(/\r?\n/)
    .map(s => s.trim())
    .filter(Boolean);

  const legs = [];

  // Pattern: "Spain total OVER 1.5- 2/7"
  const reTeamTotal = /^(.+?)\s+TOTAL\s+OVER\s+([0-9.]+)\s*-\s*([0-9]+\/[0-9]+)/i;
  // Pattern: "Over 1.5-1/9" (no team name)
  const reBareOver  = /^OVER\s+([0-9.]+)\s*-\s*([0-9]+\/[0-9]+)/i;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    let m = reTeamTotal.exec(line);
    if (m) {
      const team    = m[1].trim();
      const goalVal = m[2];
      const frac    = m[3];
      const price   = fracToDecimal(frac);

      legs.push({
        teamHome:      team,
        teamAway:      '',
        market:        'TEAM_GOALS_OVER',
        selection:     `OVER_${goalVal}`,
        price,
        bookmaker:     'BETFRED',
        kickoffUTC:    null
      });
      continue;
    }

    m = reBareOver.exec(line);
    if (m) {
      const goalVal = m[1];
      const frac    = m[2];
      const price   = fracToDecimal(frac);

      legs.push({
        teamHome:      'TOTAL_GOALS',
        teamAway:      '',
        market:        'GOALS_OVER',
        selection:     `OVER_${goalVal}`,
        price,
        bookmaker:     'BETFRED',
        kickoffUTC:    null
      });
    }
  }

  return { legs, raw: lines.slice(0, 60).join('\n') };
}

// --- Bookie detection (MVP: just enough for routing) ---
function detectBookie(text) {
  const upper = String(text).toUpperCase();
  if (upper.includes('BET365'))   return 'BET365';
  if (upper.includes('PADDY'))    return 'PADDY';
  if (upper.includes('CORAL'))    return 'CORAL';
  if (upper.includes('WILLIAM'))  return 'WILLIAM_HILL';
  if (upper.includes('BETFRED'))  return 'BETFRED';
  return 'GENERIC';
}

// --- Bet365 parser (goals, team goals, BTTS, cards, simple corners) ---
function parseBet365(text) {
  const raw   = String(text);
  const lines = raw.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  const legs  = [];

  // Try to detect any "TeamA v TeamB" line as a global fixture fallback
  let fixtureHome = '';
  let fixtureAway = '';
  for (const L of lines) {
    const m = L.match(/^(.+?)\s+(?:v|vs\.?)\s+(.+?)$/i);
    if (m) {
      fixtureHome = m[1].trim();
      fixtureAway = m[2].trim();
      break;
    }
  }

  const cleanTeamName = (name) => {
    let t = String(name || '').trim();
    // strip obvious bullets / symbols
    t = t.replace(/^[^A-Za-z0-9]+/, '').trim();
    // strip leading "o " bullet
    if (t.toLowerCase().startsWith('o ')) t = t.slice(2).trim();
    // strip stray bullets again
    t = t.replace(/^[•\-–]+\s*/, '');
    return t;
  };

  // Regexes — allow leading bullets/icons before "Over"
  const reTeamOverGoals =
    /^[^A-Za-z0-9]*(.+?)\s+Over\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+\/[0-9]+)/i; // "Celta Vigo Over 0.5 2/7"
  const reOverGoals =
    /^[^A-Za-z0-9]*Over\s+([0-9]+(?:\.[0-9]+)?)\s+Goals(?:\s+in\s+the\s+Match|\s+in\s+Match|\s+in\s+90\s+Minutes)?/i;
  const reOverBare =
    /^[^A-Za-z0-9]*Over\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9]+\/[0-9]+)/i; // "© Over3.0 1/4"
  const reOverCards =
    /^[^A-Za-z0-9]*Over\s+(\d+)\s+Cards\b/i;
  const reBTTS =
    /(Both\s+Teams\s+To\s+Score|BTTS)/i;

  const isSelectionLine = (s) =>
    reTeamOverGoals.test(s) ||
    reOverGoals.test(s) ||
    reOverBare.test(s) ||
    reOverCards.test(s) ||
    reBTTS.test(s);

  const findNearbyPrice = (startIdx) => {
    // Search this line + next 2 for a fractional or decimal price
    for (let j = 0; j <= 2 && (startIdx + j) < lines.length; j++) {
      const cand = lines[startIdx + j];

      // fractional like "2/7"
      const fracMatch = cand.match(/(\d+)\s*\/\s*(\d+)/);
      if (fracMatch) {
        const dec = fracToDecimal(fracMatch[0]);
        if (dec != null) return dec;
      }

      // decimal like "1.80"
      const decMatch = cand.match(/\b(\d+(?:\.\d+)?)\b/);
      if (decMatch) {
        const v = parseFloat(decMatch[1]);
        if (v >= 1.01 && v <= 1000) return v;
      }
    }
    return null;
  };

  // Look forward from a selection line to find the next two team names
  const findFixtureAfter = (startIdx) => {
    let team1 = '';
    let team2 = '';

    for (let k = 1; k <= 8 && (startIdx + k) < lines.length; k++) {
      const txt = lines[startIdx + k];

      // Stop if we hit another obvious selection line and we haven't found anything yet
      if (isSelectionLine(txt) && !team1 && !team2) break;
      // Stop once we’ve found two team names
      if (team1 && team2) break;

      // Extract leading wordy chunk as candidate team name
      const m = txt.match(/^[^A-Za-z0-9]*([A-Za-z][A-Za-z .']+)/);
      if (!m) continue;

      let candidate = cleanTeamName(m[1]);

      // Filter out obvious non-team stuff (days, generic labels)
      if (/^(mon|tue|wed|thu|fri|sat|sun)\b/i.test(candidate)) continue;
      if (/^(bet builder|cash out|stake|to return)/i.test(candidate)) continue;

      if (!team1) {
        team1 = candidate;
      } else if (!team2 && candidate.toLowerCase() !== team1.toLowerCase()) {
        team2 = candidate;
        break;
      }
    }

    if (!team1 && !team2) {
      return { home: fixtureHome || '', away: fixtureAway || '' };
    }
    return {
      home: team1 || fixtureHome || '',
      away: team2 || fixtureAway || ''
    };
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // --- Team goals: "Celta Vigo Over 0.5 2/7" ---
    let m = line.match(reTeamOverGoals);
    if (m) {
      let team    = cleanTeamName(m[1]);
      const goalStr = m[2];
      const frac    = m[3];
      const price   = fracToDecimal(frac);

      const fx = findFixtureAfter(i);
      const fixture_label =
        fx.home && fx.away ? `${fx.home} vs ${fx.away}` : '';

      legs.push({
        teamHome:   team,          // emphasise the team whose goals we're backing
        teamAway:   '',
        fixture_label,
        market:     'TEAM_GOALS_OVER',
        selection:  `OVER_${goalStr}`,
        price,
        bookmaker:  'BET365',
        kickoffUTC: null
      });
      continue;
    }

    m = line.match(reOverCards);
    if (m) {
      const cards = m[1];
      const price = findNearbyPrice(i);
      const fx    = findFixtureAfter(i);
      const fixture_label =
        fx.home && fx.away ? `${fx.home} vs ${fx.away}` : '';
    
      legs.push({
        teamHome:   fx.home,
        teamAway:   fx.away,
        fixture_label,
        market:     'CARDS_OVER',
        selection:  `OVER_${cards}`,
        price,
        bookmaker:  'BET365',
        kickoffUTC: null
      });
      continue;
    }


    m = line.match(reOverGoals);
    if (m) {
      const goalStr = m[1];
      const price   = findNearbyPrice(i);
      const fx      = findFixtureAfter(i);
      const fixture_label =
        fx.home && fx.away ? `${fx.home} vs ${fx.away}` : '';
    
      let market    = 'GOALS_OVER';
      let selection = `OVER_${goalStr}`;
      if (goalStr === '2.5' || goalStr === '2.50') {
        market    = 'OVER_UNDER_2_5';
        selection = 'OVER';
      }
    
      legs.push({
        teamHome:   fx.home || 'TOTAL_GOALS',
        teamAway:   fx.away || '',
        fixture_label,
        market,
        selection,
        price,
        bookmaker:  'BET365',
        kickoffUTC: null
      });
      continue;
    }


    m = line.match(reOverBare);
    if (m) {
      const goalStr = m[1];
      const frac    = m[2];
      const price   = fracToDecimal(frac);
      const fx      = findFixtureAfter(i);
      const fixture_label =
        fx.home && fx.away ? `${fx.home} vs ${fx.away}` : '';
    
      legs.push({
        teamHome:   fx.home,
        teamAway:   fx.away,
        fixture_label,
        market:     'CORNERS_OVER',
        selection:  `OVER_${goalStr}`,
        price,
        bookmaker:  'BET365',
        kickoffUTC: null
      });
      continue;
    }
      if (reBTTS.test(line)) {
        const fx = findFixtureAfter(i);
        const fixture_label =
          fx.home && fx.away ? `${fx.home} vs ${fx.away}` : '';
      
        const search = [lines[i], lines[i + 1] || ''].join(' ');
        let selection = 'YES';
        if (/\bNO\b/i.test(search)) selection = 'NO';
      
        const price = findNearbyPrice(i);
      
        legs.push({
          teamHome:   fx.home,
          teamAway:   fx.away,
          fixture_label,
          market:     'BTTS',
          selection,
          price,
          bookmaker:  'BET365',
          kickoffUTC: null
        });
        continue;
    }
  }

  // If nothing Bet365-specific matched, fall back to generic parsing
  if (!legs.length) {
    return parseGenericSlip(text);
  }

  return { legs, raw: lines.slice(0, 60).join('\n') };
}


// --- Dispatcher: choose parser based on bookie / format ---
function parseSlipText(text) {
  const bookie = detectBookie(text);

  // Betfred totals first – these patterns are very specific
  if (bookie === 'BETFRED') {
    const bf = parseBetfred(text);
    if (bf.legs && bf.legs.length) return bf;
  }

  if (bookie === 'BET365') {
    const b365 = parseBet365(text);
    if (b365.legs && b365.legs.length) return b365;
  }

  if (bookie === 'PADDY') {
    const paddy = parsePaddyPower(text);
    if (paddy.legs && paddy.legs.length) return paddy;
  }

  // Coral / William Hill can also fall back to generic for now
  return parseGenericSlip(text);
}
// --- Slip meta extraction (Report 1: header + stake/return) ---
function extractSlipMeta(text) {
  const lines = String(text).split(/\r?\n/).map(s => s.trim()).filter(Boolean);

  // Bookmaker from text
  const bookmaker = detectBookie(text) || 'UNKNOWN';

  // Created time line (e.g. "Thu 06 Nov 16:13 bet365")
  const createdLine = lines.find(l => /\b\d{2}:\d{2}\b/.test(l) && /bet365/i.test(l)) || '';
  const createdAt   = createdLine || null;

  // Stake / To Return
  let stake   = null;
  let ret     = null;
  let currency = '£';

  for (let i = 0; i < lines.length; i++) {
    const L = lines[i];
    if (/stake/i.test(L) && /return/i.test(L)) {
      const next = lines[i + 1] || '';
      // e.g. "£9.00 £18.80"
      const m = next.match(/([£€$])\s*([\d.,]+)\s+([£€$])\s*([\d.,]+)/);
      if (m) {
        currency = m[1];
        stake    = parseFloat(m[2].replace(/,/g, '')) || null;
        ret      = parseFloat(m[4].replace(/,/g, '')) || null;
      }
      break;
    }
  }

  // Bet type: look for "Bet Builder", "Single", "Accumulator" etc.
  const betTypeLine = lines.find(l => /bet builder|bet\s+builder|\bacca\b|single|accumulator/i.test(l)) || '';
  const betType = betTypeLine || null;

  return { bookmaker, createdAt, stake, ret, currency, betType };
}

async function runBetChecker(file) {
  const out = document.getElementById('bc-output');
  if (out) out.innerHTML = '<div class="muted">Reading slip…</div>';

  try {
    const text = await ocrImageOrPdf(file);

    // Log OCR so we can keep refining parsers
    console.log('[BetChecker OCR raw text]\n', text);

    // REPORT 1: slip meta from raw text
    const meta = extractSlipMeta(text);

    // REPORT 2: structured legs (markets / selections / prices)
    const parsed = parseSlipText(text);
    console.log('[BetChecker parsed legs]', parsed.legs);

    if (!parsed.legs.length) {
      if (out) {
        out.innerHTML =
          `<div class="muted">No legs detected. <br/><pre style="white-space:pre-wrap">${parsed.raw || text.slice(0, 400)}…</pre></div>`;
      }
      return;
    }

    let scored;
    if (out) out.innerHTML = '<div class="muted">Scoring legs…</div>';

    try {
      // Real backend path (will 405 on GitHub Pages)
      scored = await API.scoreSlip({ legs: parsed.legs });
    } catch (err) {
      console.warn('[BetChecker] scoreSlip failed, falling back to parsed legs only', err);
      showToast('error', 'Scoring API unavailable – showing parsed legs only.');

      scored = {
        summary: null,
        legs: parsed.legs.map(l => ({
          ...l,
          prob: null,
          fair: null,
          edgePct: null
        }))
      };
    }

    const container = document.createElement('div');

    // --------- Card 1: Slip Overview (Report 1) ---------
    const stakeText = meta.stake != null ? `${meta.currency}${meta.stake.toFixed(2)}` : '—';
    const retText   = meta.ret   != null ? `${meta.currency}${meta.ret.toFixed(2)}`   : '—';

    const overview = document.createElement('div');
    overview.className = 'insight-card';
    overview.innerHTML = `
      <h2>Slip Overview</h2>
      <ul>
        <li><strong>Bookmaker:</strong> ${meta.bookmaker}</li>
        <li><strong>Bet type:</strong> ${meta.betType || 'Bet Builder'}</li>
        <li><strong>Created:</strong> ${meta.createdAt || '—'}</li>
        <li><strong>Stake:</strong> ${stakeText}</li>
        <li><strong>To return:</strong> ${retText}</li>
      </ul>`;
    container.appendChild(overview);

    // --------- Card 2: Model Summary (will be mostly zero in demo) ---------
    const sum = scored.summary || {};
    const summaryCard = document.createElement('div');
    summaryCard.className = 'insight-card';
    summaryCard.innerHTML = `
      <h2>Model Summary</h2>
      <div>
        Implied EV: <strong>${(sum.evPct ?? 0).toFixed(1)}%</strong>
        · Prob: <strong>${Math.round((sum.comboProb ?? 0) * 100)}%</strong>
        · Payout: <strong>${sum.payout ? '£' + sum.payout.toFixed(2) : '—'}</strong>
      </div>`;
    container.appendChild(summaryCard);

    // --------- Cards 3+: each leg ---------
    (scored.legs || []).forEach((lg, i) => {
      const card = document.createElement('div');
      card.className = 'insight-card';

      const probPct = lg.prob != null ? Math.round(lg.prob * 100) + '%' : '—';
      const fairTxt  = lg.fair  != null ? lg.fair.toFixed(2)  : '—';
      const priceTxt = lg.price != null ? lg.price.toFixed(2) : '—';
      const edgeTxt  = lg.edgePct != null
        ? (lg.edgePct >= 0 ? '+' : '') + lg.edgePct.toFixed(1) + '%'
        : '—';

      // Stable, friendly title for the leg
      let leftLabel = 'Selection';
      if (lg.fixture_label) {
        leftLabel = lg.fixture_label;
      } else if (lg.teamHome && lg.teamAway) {
        leftLabel = `${lg.teamHome} vs ${lg.teamAway}`;
      } else if (lg.market === 'TEAM_GOALS_OVER' && lg.teamHome) {
        leftLabel = `${lg.teamHome} Team Goals`;
      } else if (lg.market === 'CARDS_OVER') {
        leftLabel = 'Total Cards';
      } else if (lg.market === 'GOALS_OVER') {
        leftLabel = 'Total Goals';
      } else if (lg.market === 'OVER_UNDER_2_5') {
        leftLabel = 'Match Goals (2.5 line)';
      } else if (lg.market === 'BTTS') {
        leftLabel = 'Both Teams To Score';
      } else if (lg.market === 'CORNERS_OVER') {
        leftLabel = 'Total Corners';
      } else if (lg.teamHome) {
        leftLabel = lg.teamHome;
      }

      card.innerHTML = `
        <h2>Leg ${i + 1}: ${leftLabel}</h2>
        <ul>
          <li><strong>Market:</strong> ${lg.market || '—'} · <strong>Pick:</strong> ${lg.selection || '—'}</li>
          <li><strong>Model%:</strong> ${probPct}
              · <strong>Fair:</strong> ${fairTxt}
              · <strong>Book:</strong> ${priceTxt}
              · <strong>Edge:</strong> ${edgeTxt}</li>
        </ul>`;
      container.appendChild(card);
    });

    if (out) {
      out.innerHTML = '';
      out.appendChild(container);
    }

    showToast('success', `Parsed ${scored.legs?.length || parsed.legs.length} leg(s)`);
  } catch (e) {
    console.error(e);
    showToast('error', e.message);
    const outEl = document.getElementById('bc-output');
    if (outEl) {
      outEl.innerHTML = `<div class="muted">Error: ${e.message}</div>`;
    }
  }
}

// Wire up BetChecker inputs (BetChecker view + floating home bar)
{
  // BetChecker view
  const bcUpload = document.getElementById('bc-upload');
  bcUpload?.addEventListener('change', (e) => {
    const f = e.target.files?.[0];
    if (f) runBetChecker(f);
    e.target.value = '';
  });

  // Floating upload on Home
  const homeUpload = document.getElementById('bet-upload');
  homeUpload?.addEventListener('change', (e) => {
    const f = e.target.files?.[0];
    if (f) {
      // Show results in the BetChecker view
      window.location.hash = '#/bet-checker';
      runBetChecker(f);
    }
    e.target.value = '';
  });
}



// ---------- Acca Builder via API (legacy view) ----------
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
  if (!grid) return;
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
// Demo Auth: Sign up / Sign in / Sign out
// ----------------------------

function handleSignup() {
  const emailInput = document.getElementById('su-email');
  const passInput  = document.getElementById('su-pass');
  if (!emailInput || !passInput) return;

  const email = (emailInput.value || '').trim();
  const password = (passInput.value || '').trim();

  if (!email || !password) {
    showToast('error', 'Please enter email and password');
    return;
  }

  if (email !== 'admin' || password !== 'admin') {
    showToast('error', 'Demo mode: use admin / admin');
    return;
  }

  currentUser = { email: 'admin', role: 'admin' };
  persistSession();
  updateAuthUI();
  showToast('success', 'Admin account ready');
  window.location.hash = '#/portfolio';
}

function handleLogin() {
  const emailInput = document.getElementById('li-email');
  const passInput  = document.getElementById('li-pass');
  if (!emailInput || !passInput) return;

  const email = (emailInput.value || '').trim();
  const password = (passInput.value || '').trim();

  if (!email || !password) {
    showToast('error', 'Enter email & password');
    return;
  }

  if (email !== 'admin' || password !== 'admin') {
    showToast('error', 'Demo mode: use admin / admin');
    return;
  }

  currentUser = { email: 'admin', role: 'admin' };
  persistSession();
  updateAuthUI();
  showToast('success', 'Signed in as admin');
  window.location.hash = '#/portfolio';
}

function handleLogout() {
  currentUser = null;
  persistSession();
  updateAuthUI();
  showToast('info', 'Signed out');
  window.location.hash = '#/login';
}

// Hook up auth forms (demo)
{
  const suBtn    = document.getElementById('su-submit');
  const liBtn    = document.getElementById('li-submit');
  const logoutEl = document.querySelector('[data-action="logout"]');

  if (suBtn) suBtn.addEventListener('click', (e) => {
    e.preventDefault();
    handleSignup();
  });

  if (liBtn) liBtn.addEventListener('click', (e) => {
    e.preventDefault();
    handleLogin();
  });

  if (logoutEl) logoutEl.addEventListener('click', (e) => {
    e.preventDefault();
    handleLogout();
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
  '#/signup':      'view-signup',
  '#/portfolio':   'view-portfolio'
};

function showRoute(hash) {
  if (!hash) hash = '#/';

  // Guarded routes: require login
  const guardedRoutes = ['#/acca-builder', '#/portfolio'];
  if (!currentUser && guardedRoutes.includes(hash)) {
    showToast('error', 'Please sign in to use Acca Builder and Portfolio');
    window.location.hash = '#/login';
    return;
  }

  const id = ROUTES[hash] || 'view-home';

  // Toggle view visibility
  document.querySelectorAll('.view').forEach(v => {
    if (v.id === id) {
      v.classList.add('is-active');
      v.removeAttribute('hidden');
    } else {
      v.classList.remove('is-active');
      v.setAttribute('hidden','');
    }
  });

  // Update nav state
  document.querySelectorAll('[data-route]').forEach(a=>{
    a.classList.toggle('is-active', a.getAttribute('href') === hash);
    if (a.classList.contains('side-link')) {
      a.classList.toggle('active', a.getAttribute('href') === hash);
    }
  });

  // Close menus/drawers
  const profileMenu = document.getElementById('profile-menu');
  const profileBtn  = document.getElementById('btn-profile');
  profileMenu?.classList.remove('show');
  profileBtn?.setAttribute('aria-expanded','false');
  const drawer  = document.getElementById('side-drawer');
  const scrim   = document.getElementById('scrim');
  drawer?.classList.remove('show');
  scrim?.classList.remove('show');
  drawer?.setAttribute('aria-hidden','true');

  // Control globe render loop
  isHomeActive = (id === 'view-home');

  // Render portfolio view on demand
  if (id === 'view-portfolio') {
    renderPortfolio();
    renderPortfolioStats();
  }
}

window.addEventListener('hashchange', ()=> showRoute(location.hash));
window.addEventListener('DOMContentLoaded', ()=>{
  if (!location.hash) location.hash = '#/';
  showRoute(location.hash);
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
