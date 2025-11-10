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
  // 1) Try local ESM copies if present
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

  // 2) esm.sh (bundle; externalize three)
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

  // 3) UMD fallback (local → CDNs)
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

// Nav chevrons
const navLeft = document.createElement('button');
const navRight = document.createElement('button');
Object.assign(navLeft.style,  {position:'absolute',left:'10px',top:'50%',transform:'translateY(-50%)',zIndex:2,background:'rgba(255,255,255,0.08)',border:'1px solid rgba(255,255,255,0.1)',borderRadius:'50%',width:'42px',height:'42px',color:'#a7ffea',backdropFilter:'blur(6px)',cursor:'pointer'});
Object.assign(navRight.style, {position:'absolute',right:'10px',top:'50%',transform:'translateY(-50%)',zIndex:2,background:'rgba(255,255,255,0.08)',border:'1px solid rgba(255,255,255,0.1)',borderRadius:'50%',width:'42px',height:'42px',color:'#a7ffea',backdropFilter:'blur(6px)',cursor:'pointer'});
navLeft.innerHTML  = '&#10094;'; // ‹
navRight.innerHTML = '&#10095;'; // ›
globeContainer.style.position = 'relative';
globeContainer.appendChild(navLeft);
globeContainer.appendChild(navRight);

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
controls.enablePan = true;
controls.minDistance = 120;
controls.maxDistance = 320;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;

// ====== Globe + atmosphere ======
const globe = new ThreeGlobeCtor({ waitForGlobeReady: true })
  .showAtmosphere(true)
  .atmosphereAltitude(0.22)
  .atmosphereColor('#66e3d2')
  .globeImageUrl('https://unpkg.com/three-globe/example/img/earth-dark.jpg')
  .bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png')
  .pointAltitude('pointAltitude')
  .pointColor('pointColor'); // labels set later

scene.add(globe);

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
  const r = 500 + Math.random() * 500;
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(Math.random() * 2 - 1);
  positions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
  positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
  positions[i * 3 + 2] = r * Math.cos(phi);
}
starGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
const starMat = new THREE.PointsMaterial({ size: 0.9, transparent: true, opacity: 0.8 });
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

const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(renderer.domElement.width, renderer.domElement.height),
  (window.devicePixelRatio > 2 || window.innerWidth < 480) ? 0.4 : 0.6,
  0.4,
  0.85
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
function toPercent(value) {
  if (value == null || isNaN(value)) return '–';
  return `${Math.round(Number(value) * 100)}%`;
}
function td(num, dp = 1) {
  if (num == null || isNaN(num)) return '–';
  return Number(num).toFixed(dp);
}
function badgeLabel(name) {
  return name.split(' ').map((p)=>p[0]).join('').slice(0,3).toUpperCase();
}
function setBadge(el, teamName, logoUrl) {
  if (logoUrl) {
    el.style.backgroundImage = `url(${logoUrl})`;
    el.style.backgroundSize = 'cover';
    el.style.backgroundPosition = 'center';
    el.textContent = '';
    el.classList.add('badge--img');
  } else {
    el.style.backgroundImage = '';
    el.textContent = badgeLabel(teamName);
    el.classList.remove('badge--img');
  }
}
function pickLogoUrl(fx, side /* 'home' | 'away' */) {
  const candidates = [
    'logo','logo_url','badge','badge_url','crest','crest_url','emblem','emblem_url'
  ];
  for (const c of candidates) {
    const k = `${side}_${c}`;
    if (fx[k] && /^https?:\/\//i.test(String(fx[k]))) return String(fx[k]);
  }
  // tolerate HomeLogoUrl etc.
  for (const c of candidates) {
    const k = `${side}${c.startsWith('_') ? '' : '_'}${c}`; 
    if (fx[k] && /^https?:\/\//i.test(String(fx[k]))) return String(fx[k]);
  }
  return undefined;
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
function hydrateFixtures(rawFixtures) {
  return rawFixtures
    .filter((f) => f.fixture_id)
    .map((f) => {
      const latCsv = num(f.latitude);
      const lngCsv = num(f.longitude);
      const fallback = stadiumLookup[f.stadium?.trim?.() || ""];
      const latitude = latCsv ?? fallback?.lat;
      const longitude = lngCsv ?? fallback?.lng;
      const cf = num(f.confidence_ftr) ?? 0;
      return {
        ...f,
        latitude, longitude,
        xg_home: num(f.xg_home),
        xg_away: num(f.xg_away),
        ppg_home: num(f.ppg_home),
        ppg_away: num(f.ppg_away),
        confidence_ftr: cf,
        over25_prob: num(f.over25_prob),
        btts_prob: num(f.btts_prob),
        stadium: f.stadium?.trim?.() || "",
        city: f.city?.trim?.() || "",
        country: f.country?.trim?.() || "",
        key_players_shots: processPlayerField(f.key_players_shots),
        key_players_bookings: processPlayerField(f.key_players_bookings),
        key_players_tackles: processPlayerField(f.key_players_tackles),
        pointAltitude: 0.18 + cf * 0.12,
        pointColor: cf > 0.7 ? '#64d863' : (cf > 0.5 ? '#00bcd4' : '#ff8a65'),
      };
    })
    .filter((fx) => Number.isFinite(fx.latitude) && Number.isFinite(fx.longitude));
}

// ====== Globe labels / clicks ======
function attachGlobeOverlays() {
  if (typeof globe.labelsData === 'function') {
    globe
      .labelsData(fixtures)
      .labelLat('latitude')
      .labelLng('longitude')
      .labelAltitude(d => (d.pointAltitude ?? 0.2) + 0.02)
      .labelDotRadius(() => 0.35)
      .labelColor(() => '#a7ffea')
      .labelSize(() => 1.6)
      .labelText(d =>
        [d.city, `${d.home_team} vs ${d.away_team}`].filter(Boolean).join(' • ')
      );

    if (typeof globe.onLabelClick === 'function') {
      globe.onLabelClick(pt => {
        const idx = fixtures.findIndex(f => f.fixture_id === pt.fixture_id);
        if (idx !== -1) {
          activeIndex = idx;
          focusFixture(idx);
        }
      });
    }
  }

  if (typeof globe.onPointClick === 'function') {
    globe.onPointClick(pt => {
      const idx = fixtures.findIndex(f => f.fixture_id === pt.fixture_id);
      if (idx !== -1) {
        activeIndex = idx;
        focusFixture(idx);
      }
    });
  }
}

// ====== Render a single fixture’s side panel ======
function renderFixture(index) {
  const fixture = fixtures[index];
  if (!fixture) return;

  activeIndex = index;
  fixtureTitle.textContent = `${fixture.home_team} vs ${fixture.away_team}`;

  const parts = [fixture.stadium?.trim?.(), fixture.city?.trim?.(), fixture.country?.trim?.()]
    .filter(Boolean).join(', ');
  const ctxTail = parts ? ` • ${parts}` : '';
  fixtureContext.textContent = `${fixture.competition} • ${formatDate(fixture.date_utc)}${ctxTail}`;

  // badges with logo if present
  const homeLogo = pickLogoUrl(fixture, 'home');
  const awayLogo = pickLogoUrl(fixture, 'away');
  setBadge(homeBadge, fixture.home_team, homeLogo);
  setBadge(awayBadge, fixture.away_team, awayLogo);

  // Match Intelligence
  matchIntelligenceList.innerHTML = '';
  [
    { label: 'Full-time prediction', value: `${fixture.predicted_winner || '—'} (${toPercent(fixture.confidence_ftr)})` },
    { label: 'xG edge', value: `${fixture.home_team} ${td(fixture.xg_home)} vs ${fixture.away_team} ${td(fixture.xg_away)}` },
    { label: 'Points momentum', value: `${fixture.home_team} ${td(fixture.ppg_home)} PPG • ${fixture.away_team} ${td(fixture.ppg_away)} PPG` },
  ].forEach((item) => {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${item.label}:</strong> ${item.value}`;
    matchIntelligenceList.appendChild(li);
  });

  // Player watchlist
  playerWatchlist.innerHTML = '';
  fixture.key_players_shots.forEach((p) => {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${p.name}</strong> ${p.detail || ''}`;
    playerWatchlist.appendChild(li);
  });

  // Market snapshot
  marketSnapshot.innerHTML = '';
  const marketItems = [
    { label: 'Over 2.5 goals', value: toPercent(fixture.over25_prob) },
    { label: 'Both teams to score', value: toPercent(fixture.btts_prob) },
  ];
  fixture.key_players_bookings.forEach((p) => marketItems.push({ label: `${p.name} booking risk`, value: p.detail || '' }));
  fixture.key_players_tackles.forEach((p) => marketItems.push({ label: `${p.name} tackles`, value: p.detail || '' }));
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
      `Prediction: ${fixture.predicted_winner || '—'} (${toPercent(fixture.confidence_ftr)})\n` +
      `Over 2.5: ${toPercent(fixture.over25_prob)}\n` +
      `BTTS: ${toPercent(fixture.btts_prob)}`;
    alert(summary);
  };
}

// ====== Focus fixture with smooth POV ======
function focusFixture(index) {
  const fixture = fixtures[index];
  if (!fixture) return;
  renderFixture(index);
  flyTo(fixture.latitude, fixture.longitude, 2.0, 1000);
}

// ====== Navigation helpers ======
function goPrev() {
  if (!fixtures.length) return;
  activeIndex = (activeIndex - 1 + fixtures.length) % fixtures.length;
  focusFixture(activeIndex);
}
function goNext() {
  if (!fixtures.length) return;
  activeIndex = (activeIndex + 1) % fixtures.length;
  focusFixture(activeIndex);
}
navLeft.onclick = goPrev;
navRight.onclick = goNext;
// keyboard arrows
window.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft')  goPrev();
  if (e.key === 'ArrowRight') goNext();
});

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
    const eu = fixtures.findIndex(f =>
      ['England','Scotland','Wales','Northern Ireland','Ireland','Spain','Portugal','France','Germany','Italy','Netherlands','Belgium','Norway','Sweden','Denmark','Switzerland','Austria','Poland','Czech Republic','Slovakia','Slovenia','Croatia','Serbia','Greece','Turkey'].includes(f.country)
    );
    activeIndex = eu !== -1 ? eu : 0;

    // push points and labels
    globe.pointsData(fixtures);
    // attach text labels + click handlers if supported
    if (typeof globe.pointLabel === 'function') {
      globe.pointLabel((d) => `${d.home_team} vs ${d.away_team}`);
    }
    attachGlobeOverlays();

    focusFixture(activeIndex);
  },
  error: (error) => {
    console.error('Failed to load fixtures:', error);
    fixtureTitle.textContent = 'Unable to load fixtures';
    fixtureContext.textContent = 'Check the data directory and reload the page.';
  },
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
