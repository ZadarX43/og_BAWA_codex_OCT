// app.module.js
// Odds Genius — Globe Fixtures UI (ESM + local vendor imports)

import * as THREE from 'three';
import { OrbitControls }   from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer }  from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass }      from 'three/examples/jsm/postprocessing/RenderPass.js';
import { ShaderPass }      from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

// PapaParse (UMD) is loaded in index.html
const Papa = window.Papa;
if (!Papa) throw new Error('PapaParse missing from window');

// ----------------------------
// DOM refs
// ----------------------------
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

  // Date/league filter UI
  dateToday:     document.getElementById('date-pill-today'),
  dateTomorrow:  document.getElementById('date-pill-tomorrow'),
  dateWeekend:   document.getElementById('date-pill-weekend'),
  datePrev:      document.getElementById('date-prev'),
  dateNext:      document.getElementById('date-next'),
  dateA:         document.getElementById('date-day-a'),
  dateB:         document.getElementById('date-day-b'),
  leagueChips:   document.getElementById('league-chips'),

  // Competition strip
  compWrap:   document.getElementById('comp-accuracy'),
  compName:   document.getElementById('comp-name'),
  compLogo:   document.getElementById('comp-logo'),
  compTraffic:document.getElementById('comp-traffic')
};

// ----------------------------
// Globals / tuning
// ----------------------------
let ThreeGlobeCtor;
let globe;
let renderer, scene, camera, controls, composer, bloomPass;
let fixtures = [];          // all fixtures from CSV
let visibleFixtures = [];   // filtered subset for current UI
let htmlTabsData = [];      // on-globe UI tabs
let marker;                 // custom marker object

const COLORS = {
  marker:         '#A7FFF6',
  markerInactive: '#8CEFE5',
  markerActive:   '#CFFFFA',
  ring:           '#9EE7E3'
};

const SURFACE_EPS   = 0.009;  // a tiny lift off the sphere
const RADIUS_BASE   = 0.014;
const RADIUS_ACTIVE = 0.040;
const CAMERA_ALT    = 2.0;
const BLO_STR       = 0.9;
const BLO_RAD       = 0.6;
const BLO_THRESH    = 0.75;

// ---- UI State (date/league)
// For demo, pin “today” to 2023-11-28 to match your screenshot
const UI = {
  anchorISO: '2023-11-28',        // “today”
  offsetDays: 0,                  // ± days from anchor
  rangeDays: 1,                   // 1 (today) or >1 (e.g., weekend)
  league: 'ALL',
  leagues: []                     // filled from CSV (unique competitions)
};

// Helpers to compute base date range
const MS_DAY = 24*60*60*1000;
function baseDate() {
  const d = new Date(`${UI.anchorISO}T00:00:00Z`);
  return new Date(d.getTime() + UI.offsetDays*MS_DAY);
}
function datePlusDays(base, n) {
  const t = new Date(base);
  t.setUTCDate(t.getUTCDate() + n);
  return t;
}
function fmtDay(d) {
  // show just day-of-month number -> "28"
  return String(d.getUTCDate()).padStart(2,'0');
}
function isoDay(d) {
  return d.toISOString().slice(0,10);
}

// ----------------------------
// Utilities
// ----------------------------
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

function getGlobeRadius() {
  if (globe?.getGlobeRadius) return globe.getGlobeRadius();
  const m = globe?.children?.find?.(c => c.geometry?.parameters?.radius);
  return m?.geometry?.parameters?.radius || 100;
}

// Geographic math
function latLngToVec3(lat, lon, alt = 0) {
  const R = getGlobe().R;
  const phi = THREE.MathUtils.degToRad(90 - lat);
  const theta = THREE.rd || THREE.MathUtils.degToRad(lon);
  const n = new THREE.Vector3(
    Math.sin(phi)*Math.cos(theta),
    Math.cos(phi),
    Math.sin(phi)*Math.sin(theta)
  );
  return n.clone().multiplyScalar(R * (1 + alt));
}
function latLngToUnit(lat, lon) {
  const phi = THREE.MathUtils.degToRad(90 - lat);
  const theta = THREE.MathUtils.degToRad(lon);
  return new THREE.Vector3(
    Math.sin( phi ) * Math.cos( theta ),
    Math.cos( phi ),
    Math.sin( phi ) * Math.sin( theta )
  ).normalize();
}
function getGlobe() {
  return { R: getGlobeRadius() };
}

// ----- Shared texture cache/loader
const STADIUM_TEX_CACHE = new Map();
const TEXLOADER = new THREE.TextureLoader();
function loadStadiumTex(url) {
  if (STADIUM_TEX_CACHE.has(url)) return Promise.resolve(STADIUM_TEX_CACHE.get(url));
  return new Promise((res, rej) => {
    TEXLOADER.load(
      url,
      tex => { STADiUM_TEX_CACHE.set(url, tex); res(tex); },
      undefined,
      err => rej(err)
    );
  });
}

// ----------------------------
// Logo system (local only; no remote)
// ----------------------------
const LOGO_LOCAL_BASE = './assets/assets/logos';

const TEAM_LOGO_OVERRIDES = {
  'Celtic':              `${LOGO_LOCAL_BASE}/celtic.svg`,
  'Lazio':               `${LOGO_LOCAL_BASE}/lazio.svg`,
  'Royal Antwerp FC':    `${LOGO_LOCAL_BASE}/royal-antwerp.svg`,
  'Shakhtar Donetsk':    `${LOGO_LOCAL_BASE}/shakhtar-donetsk.svg`,
  'Atlético Madrid':     `${LOGO_LOCAL_BASE}/atletico-madrid.svg`,
  'Atletico Madrid':     `${LOGO_LOCAL_BASE}/atletico-madrid.svg`,
  'Feyenoord':           `${LOGO_LOCAL_BASE}/feyenoord.svg`,
  'PSG':                 `${LOGO_LOCAL_BASE}/paris-saint-germain.svg`,
  'Paris Saint-Germain': `${LOGO_LOCAL_BASE}/paris-saint-germain.svg`,
  'Newcastle United':    `${LOGO_LOCAL_BASE}/newcastle-united.svg`,
  'AC Milan':            `${LOGO_LOCAL_BASE}/ac-milan.svg`,
  'Borussia Dortmund':   `${LOGO_LOCAL_BASE}/borovski...` // placeholder if not present
};

function slugLocal(team) {
  return String(team || '')
    .toLowerCase()
    .replace(/&/g,'and')
    .replace(/[\u2019'’]/g,'')
    .replace(/[^a-z0-9]+/g,'-')
    .replace(/^-+|-+$/g,'');
}

function localLogoCandidates(team) {
  const slug = slugLocal(team);
  return [
    `${LOGO_LOCAL_BASE}/${slug}.png`,
    `${LOGO_LOCAL_BASE}/${slug}.svg`,
  ];
}

function guessLogoSources(teamName = '') {
  const list = [];
  if (TEAM_LOGO_OVERRIDES[teamName]) list.push( TEAM_LOGO_OVERRIDES[teamName] );
  for (const url of localLogoCandidates(teamName)) list.push( url );
  return [...new Set( list )];
}

// Async safety for per–badge loads
const BADGE_CACHE = new Map();
const BADGE_LOADER = new THREE.TextureLoader();

function setBadge(elm, urlFromCsv, teamName='') {
  if (!elm) return;
  const reqId = ( elm.__reqId = (elm.__reqId || 0) + 1 );

  // Show initials immediately
  elm.classList.remove('has-logo');
  elm.textContent = teamName ? initials(teamName) : '';

  // Set size and visibility from CSS only; let img replace content later
  const srcs = [ urlFromCsv, ...guessLogoSources(teamName) ].filter(Boolean);

  (async () => {
    for (const src of srcs) {
      // Use the browser to load the image; fallback chain
      try {
        const img = await new Promise((resolve, reject) => {
          const i = new Image();
          i.crossOrigin = 'anonymous';
          i.onload = () => resolve(i);
          i.onerror = reject;
          i.src = src.includes('?') ? `${src}&v=${Date.now().toString(36)}` : `${src}?v=${Date.now().toString(36)}`;
        });
        // If a newer request was started, abandon this result
        if (elm.__reqId !== reqId) return;

        clearNode( elm );
        elm.appendChild( img );
        elm.classList.add( 'has-logo' );
        BADGE_CACHE.set( teamName, img );
        return;
      } catch {
        // try next fallback
      }
    }
    // No luck → keep initials only
    if (elm.__reqId === reqId) {
      elm.classList.remove('has-logo');
    }
  })();
}

function initials( name = '' ) {
  const parts = name.trim().split(/\s+/);
  if (!parts.length) return '';
  if (parts.length === 1) return parts[0].slice(0,2).toUpperCase();
  return (parts[0][0] + (parts[1]?.[0]||'')).toUpperCase();
}

// ----------------------------
// Stadium image URL candidates (local only)
// ----------------------------
const STADIUM_BASE = './assets/assets/leagues'; // (league logos) — for comp strip
const TEAM_STADIUM_BASE = './assets/assets/logos'; // your team crests (used in panel)
function stadiumCandidates( f ) {
  const slug = slugLocal( f.home_team || '' );
  const explicit = ( f.home_badge_url || f.away_badge_url || '' ).trim(); // keep interface; may be empty
  const guesses = [
    explicit || '',
    `${TEAM_STADIUM_BASE}/${slug}.png`,
    `${TEAM_STADIUM_BASE}/${slug}.svg`
  ].filter(Boolean);
  return guesses;
}

// ----------------------------
// Competition stats & strip
// ----------------------------

// Demo baseline FTR accuracy
const DEMO_FIXED_FTR = 0.87; // 87%
const ACC = {
  'UEFA Champions League': { FTR: 0.85, OU25: 0.78, BTTS: 0.74 },
  'Premier League':        { FTR: 0.82, OU25: 0.76, BTTS: 0.71 },
  'LaLiga':                { FTR: 0.81, OU25: 0.74, BTTS: 0.68 }
};

function leagueAcc( league, key = 'FTR' ) {
  const row = ACC[ league ];
  if (!row) return 0;
  return Math.max(0, Math.min(1, row[key] ?? 0));
}

function getCompetitionSnapshot( league ) {
  const rows = (league && league !== 'ALL')
    ? fixtures.filter(f => (f.competition || '').toLowerCase() === league.toLowerCase())
    : fixtures.slice();

  if (!rows.length) return { n: 0, ftr: 0, over25: 0, btts: 0 };

  const avg = arr => arr.reduce((a,b)=>a+b,0) / arr.length;
  const toPct = x => Math.round( x * 100 );
  const fx = rows.map( r => +r.confidence_ftr || 0 );
  const o25 = rows.map( r => +r.over25_prob   || 0 );
  const btt = rows.map( r => +r.btts_prob     || 0 );
  return { n: rows.length, ftr: toPct(avg(fx)), over25: toPct(avg(o25)), btts: toPct(avg(btt)) };
}

function renderCompetitionAccuracy( league ) {
  if (!el.compWrap) return;
  const name = league || '—';
  const ftr = Math.round( DEMO_FIXED_FTR * 100 ); // fixed per your request
  const stats = getCompetitionSnapshot( league );

  if (el.compName) el.compName.textContent = name || '—';

  // logo
  if (el.compLogo) {
    const compSlugMap = {
      'UEFA Champions League': 'uefa-champions-league',
      'Premier League':        'premier-league',
      'LaLiga':                'la-liga',
      'Bundesliga':            'bundesliga',
      'Serie A':               'serie-a',
      'Ligue 1':               'ligue-1',
      'Liga Portugal':         'liga-portugal',
      'MLS':                   'usa-mls'
    };
    const slug = compSlugMap[ league ] || '';
    if (slug) {
      el.compLogo.src = `${STADIUM_BASE}/${slug}.svg`;
      el.compLogo.style.display = 'inline-block';
      el.compLogo.alt = `${league} logo`;
    } else {
      el.compLogo.removeAttribute('src');
      el.compLogo.style.display = 'none';
    }
  }

  // The three chips
  if (el.compTraffic) {
    el.compTraffic.innerHTML = `
      <span class="light light--green">FTR ${ftr}%</span>
      <span class="light light--blue">O2.5 ${stats.over25 || 0}%</span>
      <span class="light light--amber">BTTS ${stats.btts || 0}%</span>
    `;
  }
}

// ----------------------------
// Build / render filters (dates + leagues)
// ----------------------------
function buildDateStrip() {
  if (!el.dateToday || !el.dateA || !el.dateB) return;
  const base = baseDate();
  const dayA = base;
  const dayB = datePlusDays( base, 1 );

  el.dateA.textContent = fmtDay( dayA );
  el.dateA.dataset.iso = isoDay( dayA );

  el.dateB.textContent = fmtDay( dayB );
  el.dateB.dataset.iso = isoDay( dayB );

  // Reset pill styles
  for (const p of [el.dateToday, el.dateTomorrow, el.dateWeekend]) {
    if (!p) continue;
    p.classList.remove('is-active');
  }

  // Apply active style for the current preset
  if (UI.rangeDays === 1 && UI.offsetDays === 0 && el.dateToday) el.dateToday.classList.add('is-active');
  else if (UI.rangeDays === 1 && UI.offsetDays === 1 && el.dateTomorrow) el.dateTomorrow.classList.add('is-active');
  else if (UI.rangeDays >= 2 && el.dateWeekend) el.dateWeekend.classList.add('is-active');
}

function bindDateControls() {
  if (!el.dateToday) return;

  el.dateToday.addEventListener('click', () => {
    UI.offsetDays = 0;
    UI.rangeDays = 1;
    buildDateStrip();
    applyFiltersAndRender();
  });

  el.dateTomorrow.addEventListener('click', () => {
    UI.offsetDays = 1;
    UI.rangeDays = 1;
    buildDateStrip();
    applyFiltersAndRender();
  });

  el.dateWeekend.addEventListener('click', () => {
    // weekend = Saturday + Sunday around anchor
    const base = baseDate();
    const dow  = base.getUTCDay(); // 0..6 (Sun=0)
    const deltaToSat = (6 - dow + 7) % 7; // days to Saturday
    UI.offsetDays = deltaToSat;
    UI.rangeDays  = 2;
    buildDateStrip();
    applyFiltersAndRender();
  });

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

  // Clicking day cells sets base day explicitly (single-day view)
  el.dateA?.addEventListener('click', () => {
    const iso = el.dateA.dataset.iso;
    if (iso) {
      UI.offsetDays = Math.floor((new Date(iso) - new Date(`${UI.anchorISO}T00:00:00Z`)) / MS_DAY);
      UI.rangeDays  = 1;
      buildDateStrip();
      applyFiltersAndRender();
    }
  });
  el.dateB?.addEventListener('click', () => {
    const iso = el.dateB.dataset.iso;
    if (iso) {
      UI.offsetDays = Math.floor((new Date(iso) - new Date(`${UI.anchorISO}T00:00:00Z`)) / MS_DAY);
      UI.rangeDays  = 1;
      buildDateStrip();
      applyFiltersAndRender();
    }
  });
}

function buildLeagueChips() {
  if (!el.leagueChips) return;
  const uniques = Array.from( new Set( fixtures.map( f => f.competition ).filter(Boolean) ) ).sort();
  UI.leagues = [ 'ALL', ...uniques ];
  el.leagueChips.innerHTML = '';
  UI.leagues.forEach( name => {
    const b = document.createElement( 'button' );
    b.className = `chip${ name === UI.league ? ' is-active' : '' }`;
    b.dataset.league = name;
    b.textContent = name;
    b.addEventListener( 'click', () => {
      document.querySelectorAll('#league-chips .chip').forEach( c => c.classList.remove('is-active') );
      b.classList.add( 'is-active' );
      UI.league = name;
      applyFiltersAndRender();
    } );
    el.leagueChips.appendChild( b );
  } );
}

// ----------------------------
// Filtering + render wiring
// ----------------------------
function applyFiltersAndRender() {
  if (!fixtures.length) return;

  const start = baseDate();
  const end   = datePlusDays( start, UI.rangeDays ); // half-open [start, end)

  // filter by date window + league
  visible = fixtures.filter( f => {
    const d = new Date( f.date_utc );
    if (isNaN( d )) return false;
    if (!(d >= start && d < end)) return false;
    if (UI.age) { /* reserved */ }
    if (UI.league && UI.league !== 'ALL') {
      if ((f.competition || '').toLowerCase() !== UI.league.toLowerCase()) return false;
    }
    return true;
  } );

  visibleFixtures = visible.length ? visible : [];

  // Update globe points
  const many = visibleFixtures.length > 250;
  globe.pointsMerge( many ).pointResolution( 12 );
  globe.pointLat( 'latitude' )
       .pointLng( 'longitude' )
       .pointsData( visibleFixtures );

  // Rebuild rail for visible fixtures
  buildRail( visibleFixtures );

  // Select the first visible fixture if available
  if ( visibleFixtures.length ) {
    selectIndex( 0, { fly: true } );
  } else {
    // Clear panel & strip
    if (el.compName) el.compName.textContent = '—';
    if (el.compTraffic) el.compTraffic.innerHTML = '';
    if (el.matchList) clearNode( el.matchList );
    if (el.watchlist) clearNode( el.watchlist );
    if (el.market)    clearNode( el.market );
  }
}

// ----------------------------
// Scene init
// ----------------------------
async function loadThreeGlobe() {
  for ( const v of [ '2.31.3', '2.31.1', '2.30.1', '2.29.3' ] ) {
    const url = `https://esm.sh/three-globe@${v}?bundle&external=three`;
    try {
      const m = await import( url );
      console.warn( '[three-globe] using esm.sh ESM:', url );
      return m.default ?? m;
    } catch {}
  }
  for ( const url of [
    'https://cdn.jsdelivr.net/npm/three-globe@2.31.1/dist/three-globe.min.js',
    'https://unpkg.com/three-globe@2.31.1/dist/three-globe.min.js',
    './vendor/three-globe.min.js'
  ] ) {
    try {
      const ctor = await import( url );
      console.warn( '[three-globe] using UMD:', url );
      return ctor;
    } catch {}
  }
  throw new Error( 'Failed to load three-globe' );
}

async function init() {
  ThreeGlobeCtor = await loadThreeGlobe();

  scene = new THREE.Scene();
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio( Math.min( window.devicePixelRatio || 1, 2 ) );
  renderer.setSize( el.globeWrap.clientWidth, el.globeWrap.clientHeight );
  el.globeWrap.innerHTML = '';
  el.globeWrap.appendChild( renderer.domElement );

  camera = new THREE.PerspectiveCamera( 45, el.globeWrap.clientWidth / el.globeWrap.clientHeight, 0.1, 5000 );
  camera.position.set( 0, 0, getGlobe().R * 3.0 );

  controls = new OrbitControls( camera, renderer.domElement );
  controls.enableDamping = true;
  controls.enablePan     = false;
  controls.enableZoom    = true;
  controls.autoRotate    = false;
  controls.autoRotateSpeed = 0.6;
  controls.minDistance   = getGlobe().R * 1.2;
  controls.maxDistance   = getGlobe().R * 6;

  scene.add( new THREE.AmbientLight( 0xffffff, 0.9 ) );
  const hemi = new THREE.HemisphereLight( 0xddeeff, 0x223344, 0.6 );
  scene.add( hemi );

  composer = new EffectComposer( renderer );
  composer.addPass( new RenderPass( scene, camera ) );
  bloomPass = new  (UnrealBloomFan || UnrealBloomPass)(
    new THREE.Vector3( el.globeWrap.clientWidth, el.globeWrap.clientHeight ),
    BLO_STR, BLO_RAD, BLO_THRESH
  );
  composer.addPass( bloomPass );

  globe = new ThreeGlobeCtor({ waitForGlobeReady: true })
    .showAtmosphere( true )
    .atmosphereColor( '#9ef9e3' )
    .atmosphereAltitude( 0.28 )
    .globeImageUrl( 'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg' )
    .bumpImageUrl( 'https://unpkg.com/three-globe/example/img/earth-topology.png' )
    .pointAltitude( () => SURFACE_EPS )
    .pointRadius( d => ( d.__active ? RADIUS_ACTIVE : RADIUS_BASE ) )
    .pointColor( d => ( d.__active ? COLORS.markerActive : COLORS.marker ) )
    .pointsMerge( true );

  scene.add( globe );

  if ( typeof globe.onPointHover === 'function' ) {
    globe.onPointHover( handleHover );
  }

  if ( typeof globe.pointOfView === 'function' ) {
    globe.onPointClick?.( pt => {
      if ( !pt ) return;
      const i = visibleFixtures.findIndex( f => f && ( f.latitude === pt.latitude && f.longitude === pt.longitude ) );
      if ( i >= 0 ) selectIndex( i, { fly: true } );
    } );
  }

  window.addEventListener( 'resize', () => {
    const { clientWidth, clientHeight } = el.globeWrap;
    renderer.setSize( clientWidth, clientHeight );
    camera.aspect = clientWidth / clientHeight;
    camera.updateProjectionMatrix();
    bloomPass.setSize?.( clientWidth, clientHeight );
  } );

  // Build marker (group + beam + billboard), store as window.__OG_MARKER
  const group = new THREE.Group(); scene.add( group );
  const radarGroup = new THREE.Group(); group.add( radarGroup );

  const ringGeom = new THREE.RingGeometry( getGlobe().R * ( 1 + SURFACE_EPS + 0.001 ), getGlobe().R * ( 1 + SURFACE_EPS + 0.008 ), 48 );
  const ringMat  = new THREE.MeshBasicMaterial({ color: new THREE.Color( COLORS.ring ), side: THREE.DoubleSide, transparent: true, opacity: 0.42 });
  const ringMesh = new THREE.Mesh( ringGeom, ringMat );
  radarGroup.add( ringMesh );

  const beamGeom = new THREE.CylinderGeometry( 1, 1, 1, 24, 1, true );
  const beamMat  = new THREE.MeshBasicMaterial({ color: new THREE.Color( COLORS.ring ), transparent: true, opacity: 0.0 });
  const beam     = new THREE.Mesh( beamGeom, beamMat );
  beam.renderOrder = 10;
  beam.visible = false;
  group.add( beam );

  const billboard = new THREE.Sprite( new THREE.SpriteMaterial({ transparent: true, opacity: 0 }) );
  billboard.scale.set( 14, 8, 1 );
  group.add( billboard );

  marker = window.__OG_MARKER = {
    group, radarGroup, beam, billboard,
    markerState: { lat: 0, lon: 0, t0: 0, active: false, reqId: 0 },
    __raf: { travel: null, beam: null, fade: null }
  };

  // Wire tabs
  bindTabs();
  // Wire filter controls
  bindDateControls();

  // Load CSV
  await loadFixturesCSV( './data/fixtures.csv' );

  // Build league chips from data
  buildLeagueChips();

  // Initial date strip + filter
  buildDateStrip();

  // Apply initial filter + render
  applyFiltersAndRender();

  // Render loop
  ( function loop() {
    requestAnimationFrame( loop );
    controls.update();
    composer.render();
  } )();
}

// ----------------------------
// CSV ingest & bind
// ----------------------------
async function loadFixturesCSV( url ) {
  const response = await fetch( `${url}?v=${Date.now()}` );
  if ( !response.ok ) {
    showToast( 'error', `Could not load ${url} (HTTP ${response.status}).` );
    return;
  }
  const text = await response.text();
  const { data, errors } = Papa.parse( text, { header: true, skipEmptyLines: true } );
  if ( errors?.length ) console.warn( '[CSV parse errors]', errors );

  fixtures = ( data || [] )
    .map( row => {
      const lat = parseFloat( row.latitude  ?? row.lat ?? row.Latitude ?? row.lat_deg );
      const lon = parseFloat( row.longitude ?? row.lng ?? row.Longitude );
      return {
        fixture_id:       ( row.fixture_id || row.id || `${row.home_team}-${row.away_team}-${ row.date_utc || '' }` ).trim(),
        home_team:        ( row.home_team  || row.Home || '' ).trim(),
        away_team:        ( row.away_team  || row.Away || '' ).trim(),
        home_badge_url:   pick( row, [ 'home_badge_url', 'home_logo_url', 'home_logo', 'home_badge' ] ),
        away_badge_url:   pick( row, [ 'away_badge_url', 'away_logo_url', 'away_logo', 'away_badge' ] ),
        date_utc:         row.date_utc || row.date || '',
        competition:      row.competition || row.league || '',
        stadium:          row.stadium || '',
        city:             row.city || '',
        country:          row.country || row.venue_country || '',
        latitude:         Number.isFinite( lat ) ? lat : undefined,
        longitude:        Number.isFinite( lon ) ? lon : undefined,
        confidence_ftr:   + ( row.confidence_ftr || row.confidence || 0 ),
        xg_home:          + ( row.xg_home || 0 ),
        xg_away:          + ( row.xg_away || 0 ),
        ppg_home:         + ( row.ppg_home || 0 ),
        ppg_away:         + ( row.ppg_away || 0 ),
        over25_prob:      + ( row.over25_prob || 0 ),
        btts_prob:        + ( row.btts_prob || 0 )
      };
    } )
    .filter( f => Number.isFinite( f.latitude ) && Number.isFinite( f.longitude ) );

  showToast( 'success', `Loaded ${ fixtures.length } fixtures` );
}

// ----------------------------
// Selection + panel + tabs
// ----------------------------
function handleHover( pt ) {
  const selId = marker?.markerState?.selId;
  const id = visibleFixtures.find( f => f.latitude === pt?.latitude && f.longitude === pt?.longitude )?.fixture_id;
  globe.pointRadius( p => {
    const active = ( p.fixture_id === selId );
    const hover  = ( pt && p.latitude === pt?.latitude && p.longitude === pt?.longitude );
    if (active) return RADIUS_ACTIVE;
    return hover ? RHP( RADIUS_BASE ) : RADIUS_BASE;
  } );
}
function RHP( r ) { return r * 1.6; }

function selectIndex( idx, { fly=false } = {} ) {
  if ( !visibleFixtures.length ) return;
  const f = visibleFixtures[ idx ];
  if ( !f ) return;

  const selId = f.fixture_id;
  marker.markerState.selId = selId;

  // flip active flags for point sizes/colors
  visibleFixtures.forEach( it => { it.__active = ( it.fixture_id === selId ); } );
  globe.pointColor( p => ( p.__active ? COLORS.markerActive : COLORS.marker ) )
       .pointRadius( p => ( p.__active ? RADIUS_ACTIVE : RADIUS_BASE ) )
       .pointsTransitionDuration?.( 200 );

  // Move marker & update panel
  moveMarkerToFixture( f, { fly } );
  renderPanel( f );

  const gw = document.querySelector( '.hero__globe' );
  gw?.classList.add( 'glow', 'glow-pin' );
  setTimeout( () => gw?.classList.remove( 'glow-pin' ), 350 );
}

function renderPanel( f ) {
  if (!f) return;

  const fmt = iso => {
    try {
      const d = new Date( iso );
      const date = d.toLocaleDateString( undefined, { weekday:'short', day:'2-digit', month:'short' } );
      const time = d.toLocaleTimeString( undefined, { hour:'2-digit', minute:'2-digit' } );
      return `${ date }  ·  ${ time } GMT`;
    } catch { return iso || ''; }
  };

  if ( el.fixtureTitle ) el.fixtureTitle.textContent = `${ f.home_team } vs ${ f.away_team }`;
  if ( el.fixtureContext ) {
    el.fixtureContext.textContent = [
      f.competition,
      fmt( f.date_utc ),
      f.stadium && `${ f.stadium } (${ f.city || '' })`,
      f.country
    ].filter(Boolean).join('  •  ');
  }

  setBadge( el.homeBadge, f.home_badge_url, f.home_team );
  setBadge( el.awayBadge, f.away_badge_url, f.away_team );

  // Match Intelligence
  if ( el.matchList ) {
    clearNode( el.matchList );
    const div = document.createElement( 'div' );
    div.innerHTML = `
      <div><strong title="Predicted winner">Full-time prediction:</strong> ${ f.predicted_winner || '–' } ${ f.confidence_ftr ? `(${ Math.round( f.confidence_ftr * 100 ) }%)` : '' }</div>
      <div><strong title="Expected Goals">xG edge:</strong> ${ ( f.xg_home ?? 0 ).toFixed( 1 ) } vs ${ ( f.xg_away ?? 0 ).toFixed( 1 ) }</div>
      <div><strong title="Points Per Game">Points momentum:</strong> ${ ( f.ppg_home ?? 0 ).toFixed( 1 ) } PPG · ${ ( f.ppg_away ?? 0 ).toFixed( 1 ) } PPG</div>
    `;
    el.matchList.appendChild( div );
  }

  // Player Watchlist (shots)
  if ( el.watchlist ) {
    clearNode( el.watchlist );
    const shots = ( f.key_players_shipment || f.key_players_shots || '' )
      .split( ';' )
      .map( s => s.trim() )
      .filter( Boolean )
      .slice( 0, 6 );
    if ( shots.length ) {
      for ( const s of shots ) {
        const row = document.createElement( 'div' );
        row.className = 'row';
        row.textContent = s;
        el.watchlist.appendChild( row );
      }
    } else {
      el.watchlist.appendChild( elmEmpty( 'No player highlights available.' ) );
    }
  }

  // Market snapshot
  if ( el.market ) {
    el.market.innerHTML = `
      <div><strong>Over 2.5 goals:</strong> ${ Math.round( ( f.over25_prob || 0 ) * 100 ) }%</div>
      <div><strong>Both teams to score:</strong> ${ Math.round( ( f.btts_prob  || 0 ) * 100 ) }%</div>
    `;
  }

  // Update the competition strip based on this fixture
  renderCompetitionAccuracy( f.competition );
}

function elmEmpty( msg ) {
  const d = document.createElement( 'div' );
  d.className = 'empty';
  d.textContent = msg;
  return d;
}

function bindTabs() {
  document.querySelectorAll( '.tab' )?.forEach( btn => {
    btn.addEventListener( 'click', () => {
      document.querySelectorAll( '.tab' ).forEach( b => b.classList.remove( 'is-active' ) );
      document.querySelectorAll( '.tabpane' ).forEach( p => p.classList.remove( 'is-active' ) );
      btn.classList.add( 'is-active' );
      document.getElementById( `tab-${ btn.dataset.tab }` )?.classList.add( 'is-active' );
    } );
  } );
}

// ----------------------------
// Build / sync fixture rail
// ----------------------------
function buildRail( items ) {
  const rail = document.getElementById( 'fixture-rail' );
  if ( !rail ) return;
  rail.innerHTML = '';

  items.forEach( ( f, i ) => {
    const it = document.createElement( 'button' );
    it.className = `rail-item${ i === 0 ? ' is-active' : '' }`;
    it.innerHTML = `<h4>${ f.home_team } vs ${ f.away_team }</h4><p>${ f.city || f.country || '' }</p>`;
    it.addEventListener( 'click', () => selectIndex( i, { fly: true } ) );
    rail.appendChild( it );
  } );
}
function syncRail() {
    const rail = document.getElementById('fixture-rail'); if (!rail) return;
    [...rail.children].forEach( (el, i) => {
      el.classList.toggle( 'is-active', i === ( visibleFixtures.indexOf( visibleFixtures.find( f => f.fixture_id === marker?.markerState?.selId ) ) ) );
    } );
}

// ----------------------------
// Marker creation + movement (no THREE.Quaternion.slerp, precise placement)
// ----------------------------
function moveMarkerToFixture( f, { fly = false } = {} ) {
  const S = marker; if ( !S || !f ) return;

  const lat = Number( f.latitude ), lon = Number( f.longitude );
  if ( !Number.isFinite( lat ) || !Number.isFinite( lon ) ) { S.group.visible = false; return; }

  // Token for race-free async updates
  S.markerState.reqId = ( S.markerState.reqId || 0 ) + 1;
  const myReq = S.markerState.reqId;

  // update state
  S.markerState.lat = lat;
  S.markerState.lon = lon;
  S.markerState.t0  = performance.now() * 1e-3;
  S.markerState.active = true;

  // Setup RAF controllers
  S.__raf = S.__raf || { travel: null, beam: null, fade: null };
  S.__raf.travel  = S.__raf.travel  || makeTrader();
  S.__raf.beam    = S.__raf.beam    || makeTrader();
  S.__raf.fade    = S.__raf.fade    || makeTrader();
  S.__raf.travel.cancel();
  S.__raf.beam.cancel();
  S.__raf.fade.cancel();

  const R = getGlobeRadius();

  // Normals for great-circle interpolation
  const fromN = S.group.position.lengthSq() > 0
    ? S.group.position.clone().normalize()
    : latLngToUnit( lat, lon ); // first placement uses target as start
  const toN   = latLngToUnit( lat, lon );

  const isFirst   = ( S.group.visible !== true );
  const travelDur = ( fly && !isFirst ) ? 700 : 0;

  S.group.visible = true;

  // Travel tween
  const t0 = performance.now();
  function stepTravel() {
    const t = travelDur ? Math.min( 1, ( performance.now() - t0 ) / travelDur ) : 1;
    const k = easeInOut( t );

    const curN = fromN.clone().lerp( toN, k ).normalize();
    const worldPos = curN.clone().multiplyScalar( R * ( 1 + SURFACE_EPS ) );
    S.group.position.copy( worldPos );

    // Orient +Y along world “up” (surface normal)
    const up = new THREE.Vector3( 0, 1, 0 );
    const lookMatrix = new THREE.Matrix4().lookAt( new THREE.Vector3().copy( worldPos ), new THREE.Vector3( 0, 0, 0 ), up );
    S.group.quaternion.setFromRotationMatrix( lookMatrix );

    if ( t >= 1 ) {
      S.__raf.travel.cancel();
      startBeamAndCard();
    }
  }
  S.__raf.travel.run( stepTravel );

  function startBeamAndCard() {
    // Beam grow from surface
    S.beam.position.set( 0, 0, 0 );
    S.beam.quaternion.identity();
    S.beam.scale.set( 1, 0.001, 1 );
    S.beam.material.opacity = 0.0;
    S.beam.visible = true;

    const b0 = performance.now();
    const bd = 550;
    function stepBeam() {
      const tb = Math.min( 1, ( performance.now() - b0 ) / bd );
      const e  = tb * tb * ( 3 - 2 * tb );
      S.beam.scale.y = 0.001 + e;
      S.beam.material.opacity = 0.4 * e;
      if ( tb >= 1 ) S.__raf.beam.cancel();
    }
    S.__raf.beam.run( stepBeam );

    // Place billboard a bit above local +Y
    S.billboard.position.set( 0, R * 0.06, 0 );
    S.billboard.material.opacity = 0.0;
    S.billboard.visible = true;

    // Hide if behind globe relative to camera
    const camDir = camera.position.clone().normalize();
    const n = S.group.position.clone().normalize();
    if ( n.dot( camDir ) < -0.05 ) {
      S.billboard.visible = false;
    }

    // Load stadium texture safely
    ( async () => {
      for ( const url of stadiumCandidates( f ) ) {
        try {
          const tex = await new Promise( ( res, rej ) => {
            TEXLOADER.load(
              url.includes('?') ? `${url}&v=${Date.now().toString(36)}` : `${url}?v=${Date.now().toString(36)}`,
              t => res( t ),
              undefined,
              rej
            );
          } );
          if ( S.markerState.reqId !== myReq ) return;
          S.billboard.material.map = tex;
          S.billboard.material.needsUpdate = true;

          const fa0 = performance.now();
          const fad = 220;
          function stepFade() {
            const ft = Math.min( 1, ( performance.now() - fa0 ) / fad );
            S.billboard.material.opacity = ft;
            if ( ft >= 1 ) S.__raf.fade.cancel();
          }
          S.__raf.fade.run( stepFade );

          return;
        } catch {}
      }
      if ( S.markerState.reqId === myReq ) {
        S.billboard.visible = false;
      }
    } )();
  }
}

// ----------------------------
// Tabs + Misc
// ----------------------------
function elmEmpty( msg ) {
  const d = document.createElement( 'div' );
  d.className = 'empty';
  d.textContent = msg;
  return d;
}

function bindTabs() {
  document.querySelectorAll( '.tab' )?.forEach( btn => {
    btn.addEventListener( 'click', () => {
      document.querySelectorAll( '.tab' ).forEach( x => x.classList.remove( 'is-active' ) );
      document.querySelectorAll( '.tabpane' ).forEach( x => x.classList.remove( 'is-active' ) );
      btn.classList.add( 'is-active' );
      const id = btn.dataset.tab;
      document.getElementById( `tab-${ id }` )?.classList.add( 'is-active' );
    } );
  } );
}

// ----------------------------
// Minimal client-side router (unchanged)
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

function showRoute( hash ) {
  if ( !hash ) hash = '#/';
  const id = ROUTES[ hash ] || 'view-home';
  document.querySelectorAll( '.view' ).forEach( v => {
    if ( v.id === id ) { v.classList.add( 'is-active' ); v.removeAttribute( 'hidden' ); }
    else              { v.classList.remove( 'is-active' ); v.setAttribute( 'hidden', '' ); }
  } );
  document.querySelectorAll( '[data-route]' ).forEach( a => {
    a.classList.toggle( 'is-active', a.getAttribute( 'href' ) === hash );
    if ( a.classList.contains( 'side-link' ) ) a.classList.toggle( 'active', a.getAttribute( 'href' ) === hash );
  } );
  // close menus on navigation
  const profileMenu = document.getElementById('profile-menu');
  const profileBtn  = document.getElementById('btn-profile');
  profileMenu?.classList.remove('show');
  profileBtn?.setAttribute('aria-expanded','false');
  const drawer  = document.getElementById('side-drawer');
  const scrim   = document.getElementById('scrim');
  drawer?.classList.remove('show');
  scrim?.classList.remove('show');
}

window.addEventListener( 'hashchange', () => showRoute( location.hash ) );
window.addEventListener( 'DOMContentLoaded', () => {
  if (!location.hash) location.hash = '#/';
  showRoute( location.hash );
  init().catch( err => { console.error( err ); showToast( 'error', String( err ) ); } );
} );

// ----------------------------
// Helper to retrieve globe radius consistently
// ----------------------------
function getGlobeRadius() {
  if ( globe?.getGlobeRadius ) return globe.getGlobeRadius();
  const m = globe?.children?.find?.( c => c.geometry?.parameters?.radius );
  return m?.geometry?.parameters?.radius || 100;
}
