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

// three-globe as ESM (LOCAL file)
import ThreeGlobe from './vendor/three-globe.mjs';

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
controls.enablePan = true;
controls.minDistance = 120;
controls.maxDistance = 320;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;

// ====== Globe + atmosphere ======
const globe = new ThreeGlobe({ waitForGlobeReady: true })
  .showAtmosphere(true)
  .atmosphereAltitude(0.22)
  .atmosphereColor('#66e3d2')
  .globeImageUrl('https://unpkg.com/three-globe/example/img/earth-dark.jpg')
  .bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png')
  .pointAltitude('pointAltitude')
  .pointColor('pointColor');
// .pointLabel((d) => `${d.home_team} vs ${d.away_team}`); // not in this build

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
  return `${Math.round(Number(value) * 100)}%`;
}
function badgeLabel(name) {
  return name.split(' ').map((p)=>p[0]).join('').slice(0,3).toUpperCase();
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

// ====== Render a single fixture’s side panel ======
function renderFixture(index) {
  const fixture = fixtures[index];
  if (!fixture) return;

  activeIndex = index;
  fixtureTitle.textContent = `${fixture.home_team} vs ${fixture.away_team}`;

  const parts = [fixture.stadium?.trim?.(), fixture.city?.trim?.(), fixture.country?.trim?.()]
    .filter(Boolean).join(", ");
  const ctxTail = parts ? ` • ${parts}` : "";
  fixtureContext.textContent = `${fixture.competition} • ${formatDate(fixture.date_utc)}${ctxTail}`;

  homeBadge.textContent = badgeLabel(fixture.home_team);
  awayBadge.textContent = badgeLabel(fixture.away_team);

  matchIntelligenceList.innerHTML = '';
  [
    { label: 'Full-time prediction', value: `${fixture.predicted_winner} (${toPercent(fixture.confidence_ftr)})` },
    { label: 'xG edge', value: `${fixture.home_team} ${fixture.xg_home?.toFixed?.(1)} vs ${fixture.away_team} ${fixture.xg_away?.toFixed?.(1)}` },
    { label: 'Points momentum', value: `${fixture.home_team} ${fixture.ppg_home?.toFixed?.(1)} PPG • ${fixture.away_team} ${fixture.ppg_away?.toFixed?.(1)} PPG` },
  ].forEach((item) => {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${item.label}:</strong> ${item.value}`;
    matchIntelligenceList.appendChild(li);
  });

  playerWatchlist.innerHTML = '';
  fixture.key_players_shots.forEach((p) => {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${p.name}</strong> ${p.detail}`;
    playerWatchlist.appendChild(li);
  });

  marketSnapshot.innerHTML = '';
  const marketItems = [
    { label: 'Over 2.5 goals', value: toPercent(fixture.over25_prob) },
    { label: 'Both teams to score', value: toPercent(fixture.btts_prob) },
  ];
  fixture.key_players_bookings.forEach((p) => marketItems.push({ label: `${p.name} booking risk`, value: p.detail }));
  fixture.key_players_tackles.forEach((p) => marketItems.push({ label: `${p.name} tackles`, value: p.detail }));
  marketItems.forEach((item) => {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${item.label}:</strong> ${item.value}`;
    marketSnapshot.appendChild(li);
  });

  deepDiveButton.onclick = () => {
    const summary =
      `Fixture: ${fixture.home_team} vs ${fixture.away_team}\n` +
      `Kick-off: ${formatDate(fixture.date_utc)}\n` +
      `Prediction: ${fixture.predicted_winner} (${toPercent(fixture.confidence_ftr)})\n` +
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

// ====== Load fixtures and boot ======
const csvUrl = new URL('./data/fixtures.csv', window.location.href);
csvUrl.searchParams.set('v', Date.now().toString()); // cache-bust

Papa.parse(csvUrl.href, {
  download: true,
  header: true,
  // If your snapshot is TSV, uncomment next line:
  // delimiter: "\t",
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
    globe.pointsData(fixtures);
    focusFixture(activeIndex);
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
      activeIndex = idx;
      focusFixture(idx);
    }
  });
}

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
