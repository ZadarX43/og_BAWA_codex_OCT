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
  const localEsm = ['./vendor/three-globe.module.js','./vendor/three-globe.mjs'];
  for (const p of localEsm) { try { const m = await import(p); console.info('[three-globe] using local ESM:', p); return m.default ?? m; } catch {} }
  const esmVersions = ['2.30.1','2.29.3','2.28.0'];
  for (const v of esmVersions) {
    const url = `https://esm.sh/three-globe@${v}?bundle&external=three`;
    try { const m = await import(url); console.warn('[three-globe] using esm.sh:', url); return m.default ?? m; }
    catch (e) { console.warn('[three-globe] esm.sh failed:', url, e); }
  }
  const umdCandidates = [
    './vendor/three-globe.min.js',
    'https://cdn.jsdelivr.net/npm/three-globe@2.29.3/dist/three-globe.min.js',
    'https://unpkg.com/three-globe@2.29.3/dist/three-globe.min.js'
  ];
  for (const url of umdCandidates) {
    try { const ctor = await importScriptUMD(url); console.warn('[three-globe] using UMD:', url); return ctor; }
    catch (e) { console.warn('[three-globe] UMD failed:', url, e); }
  }
  const gc = document.getElementById('globe-container');
  if (gc) {
    gc.innerHTML = `<div class="globe-error">
      <strong>three-globe failed to load.</strong><br/>
      Add local <code>docs/vendor/three-globe.module.js</code> (ESM) or <code>three-globe.min.js</code> (UMD).
    </div>`;
  }
  throw new Error('three-globe could not be loaded');
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

// Venue/marker model
let venues = [];                  // [{id, lat, lng, city, country, fixtures:[idx], instanceId}]
const venueByKey = new Map();     // key=lat|lng string → venue
let selectedVenueIdx = 0;

// Textures cache for logos
const logoCache = new Map(); // url => THREE.Texture
const textureLoader = new THREE.TextureLoader();

// ====== THREE basics ======
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
if ('outputColorSpace' in renderer) renderer.outputColorSpace = THREE.SRGBColorSpace;
else if ('outputEncoding' in renderer) renderer.outputEncoding = THREE.sRGBEncoding;
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
  .bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png');

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

// ===== Helpers: globe radius and lat/lng → xyz =====
function computeGlobeRadius() {
  // Try three-globe hint(s)
  if (typeof globe.getGlobeRadius === 'function') return globe.getGlobeRadius();
  // try locate mesh with geometry sphere
  let r = 100; // default
  globe.traverse((obj) => {
    if (obj.isMesh && obj.geometry) {
      obj.geometry.computeBoundingSphere?.();
      const br = obj.geometry.boundingSphere?.radius;
      if (br && br > 10) r = br;
    }
  });
  return r;
}
const GLOBE_R = computeGlobeRadius();
const EPS_GROUND = 0.003; // sits on surface

function llToVec3(lat, lng, alt = 0) {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lng + 180) * (Math.PI / 180);
  const r = GLOBE_R * (1 + alt);
  const x = -r * Math.sin(phi) * Math.cos(theta);
  const z =  r * Math.sin(phi) * Math.sin(theta);
  const y =  r * Math.cos(phi);
  return new THREE.Vector3(x, y, z);
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

// ====== Instanced venue markers + glow ring + logo sprites ======
const MARKER_COUNT_MAX = 5000;
const markerGeom = new THREE.SphereGeometry(GLOBE_R * 0.012, 8, 8);
const markerMat  = new THREE.MeshBasicMaterial({ color: 0x8fffe7, transparent: true, opacity: 0.9 });
const markerMesh = new THREE.InstancedMesh(markerGeom, markerMat, MARKER_COUNT_MAX);
markerMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
scene.add(markerMesh);

const selectionRing = new THREE.Mesh(
  new THREE.RingGeometry(GLOBE_R * 0.018, GLOBE_R * 0.021, 32),
  new THREE.MeshBasicMaterial({ color: 0x66e3d2, transparent: true, opacity: 0.8, side: THREE.DoubleSide })
);
selectionRing.visible = false;
scene.add(selectionRing);

const spriteMaterialTemplate = new THREE.SpriteMaterial({ depthWrite: false, transparent: true, opacity: 1 });
const homeLogoSprite = new THREE.Sprite(spriteMaterialTemplate.clone());
const awayLogoSprite = new THREE.Sprite(spriteMaterialTemplate.clone());
homeLogoSprite.visible = false; awayLogoSprite.visible = false;
homeLogoSprite.scale.set(GLOBE_R*0.05, GLOBE_R*0.05, 1);
awayLogoSprite.scale.set(GLOBE_R*0.05, GLOBE_R*0.05, 1);
scene.add(homeLogoSprite, awayLogoSprite);

// picking
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

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
  const pv = globe.pointOfView?.() || { lat: 0, lng: 0, altitude: 2 };
  const start = { lat: pv.lat || 0, lng: pv.lng || 0, altitude: pv.altitude || 2 };
  const end = { lat, lng, altitude };
  const t0 = performance.now();
  function tick(now) {
    const t = Math.min(1, (now - t0) / ms);
    const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    const lerp = (a,b)=> a + (b-a)*ease;
    globe.pointOfView?.({ lat: lerp(start.lat,end.lat), lng: lerp(start.lng,end.lng), altitude: lerp(start.altitude,end.altitude) });
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

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
        venue_city: f.venue_city || f.city || '',
        home_logo_url: f.home_logo_url || '',
        away_logo_url: f.away_logo_url || '',
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
      };
    })
    .filter((fx) => Number.isFinite(fx.latitude) && Number.isFinite(fx.longitude));
}

// ===== Venue model & markers =====
function venueKey(lat, lng) { return `${lat.toFixed(4)}|${lng.toFixed(4)}`; }
function buildVenues() {
  venues = [];
  venueByKey.clear();
  fixtures.forEach((f, idx) => {
    const key = venueKey(f.latitude, f.longitude);
    let v = venueByKey.get(key);
    if (!v) {
      v = { id: key, lat: f.latitude, lng: f.longitude, city: f.venue_city || f.city || '', country: f.country || '', fixtures: [], instanceId: -1 };
      venueByKey.set(key, v);
      venues.push(v);
    }
    v.fixtures.push(idx);
  });
}
function updateInstancedMarkers() {
  const dummy = new THREE.Object3D();
  const n = Math.min(venues.length, MARKER_COUNT_MAX);
  for (let i = 0; i < n; i++) {
    const v = venues[i];
    const pos = llToVec3(v.lat, v.lng, EPS_GROUND);
    dummy.position.copy(pos);
    dummy.lookAt(new THREE.Vector3(0,0,0));
    dummy.updateMatrix();
    markerMesh.setMatrixAt(i, dummy.matrix);
    v.instanceId = i;
  }
  markerMesh.count = n;
  markerMesh.instanceMatrix.needsUpdate = true;
}

// Highlight ring + logo sprites for selected venue
function placeSelectionVisuals(v) {
  if (!v) { selectionRing.visible = false; homeLogoSprite.visible = false; awayLogoSprite.visible = false; return; }
  const pos = llToVec3(v.lat, v.lng, EPS_GROUND);
  selectionRing.visible = true;
  selectionRing.position.copy(pos);
  selectionRing.lookAt(new THREE.Vector3(0,0,0)); // face camera by normal
  // sprites just above
  const up = pos.clone().normalize().multiplyScalar(GLOBE_R*0.04);
  const right = new THREE.Vector3().crossVectors(pos, new THREE.Vector3(0,1,0)).normalize().multiplyScalar(GLOBE_R*0.05);
  homeLogoSprite.position.copy(pos).add(up).sub(right);
  awayLogoSprite.position.copy(pos).add(up).add(right);
  homeLogoSprite.visible = true; awayLogoSprite.visible = true;
}

// ====== UI binding / side panel ======
function logoOrInitial(el, url, fallbackInitials) {
  if (!url) { el.style.backgroundImage = ''; el.textContent = fallbackInitials; return; }
  if (logoCache.has(url)) {
    const tex = logoCache.get(url);
    el.style.backgroundImage = `url(${tex.image.src})`;
    el.textContent = '';
    return;
  }
  textureLoader.load(url,
    (tex) => { logoCache.set(url, tex); el.style.backgroundImage = `url(${tex.image.src})`; el.textContent = ''; },
    undefined,
    () => { el.style.backgroundImage = ''; el.textContent = fallbackInitials; }
  );
}

// ====== Render a single fixture’s side panel ======
function renderFixture(index) {
  const fixture = fixtures[index];
  if (!fixture) return;

  activeIndex = index;
  fixtureTitle.innerHTML = `${fixture.home_team} <span style="opacity:.6">vs</span> ${fixture.away_team}`;

  // Logos into round badges (via CSS background-image)
  homeBadge.style.backgroundSize = 'cover';
  awayBadge.style.backgroundSize = 'cover';
  logoOrInitial(homeBadge, fixture.home_logo_url, badgeLabel(fixture.home_team));
  logoOrInitial(awayBadge, fixture.away_logo_url, badgeLabel(fixture.away_team));

  const parts = [fixture.stadium?.trim?.(), fixture.city?.trim?.(), fixture.country?.trim?.()]
    .filter(Boolean).join(", ");
  const ctxTail = parts ? ` • ${parts}` : "";
  fixtureContext.textContent = `${fixture.competition} • ${formatDate(fixture.date_utc)}${ctxTail}`;

  // Intelligence block (tidy)
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

  // Watchlist
  playerWatchlist.innerHTML = '';
  fixture.key_players_shots.forEach((p) => {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${p.name}</strong> ${p.detail}`;
    playerWatchlist.appendChild(li);
  });

  // Market
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

  // ensure selected venue visuals pair
  const vKey = venueKey(fixture.latitude, fixture.longitude);
  const vIdx = venues.findIndex(v => v.id === vKey);
  if (vIdx !== -1) {
    selectedVenueIdx = vIdx;
    setSelectedVenue(selectedVenueIdx, /*fly*/true);
  }

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

// ====== Selection handling (venues) ======
function setSelectedVenue(idx, fly=true) {
  if (!venues.length) return;
  selectedVenueIdx = (idx + venues.length) % venues.length;
  const v = venues[selectedVenueIdx];

  // scale selected instance; reset others cheaply (only diff)
  const dummy = new THREE.Object3D();
  for (let i=0;i<markerMesh.count;i++){
    markerMesh.getMatrixAt(i, dummy.matrix);
    dummy.matrix.decompose(dummy.position, dummy.quaternion, dummy.scale);
    const targetS = (i===v.instanceId) ? 1.6 : 1.0;
    if (Math.abs(dummy.scale.x-targetS) > 1e-3){
      dummy.scale.setScalar(targetS);
      dummy.updateMatrix();
      markerMesh.setMatrixAt(i, dummy.matrix);
    }
  }
  markerMesh.instanceMatrix.needsUpdate = true;

  placeSelectionVisuals(v);

  // fly to venue
  if (fly) flyTo(v.lat, v.lng, 2.0, 900);

  // choose top fixture at this venue (closest in time or first)
  const fixtureIdx = v.fixtures[0];
  if (fixtureIdx !== undefined && fixtureIdx !== activeIndex) {
    renderFixture(fixtureIdx);
  }
}

// ====== Load fixtures and boot ======
const csvUrl = new URL('./data/fixtures.csv', window.location.href);
csvUrl.searchParams.set('v', Date.now().toString()); // cache-bust

Papa.parse(csvUrl.href, {
  download: true,
  header: true,
  // delimiter: "\t", // uncomment for TSV
  skipEmptyLines: true,
  dynamicTyping: false,
  complete: (results) => {
    fixtures = hydrateFixtures(results.data || []);
    if (!fixtures.length) {
      fixtureTitle.textContent = 'No fixtures found';
      fixtureContext.textContent = 'Check data/fixtures.csv format.';
      return;
    }

    // Build venues & markers
    buildVenues();
    updateInstancedMarkers();

    // Also feed three-globe points (ground-level) for hover label fallback
    globe
      .pointsData(fixtures.map(f => ({
        ...f,
        pointAltitude: EPS_GROUND,
        pointColor: '#8fffe7'
      })))
      .pointAltitude('pointAltitude')
      .pointColor('pointColor');

    if (typeof globe.pointLabel === 'function') {
      globe.pointLabel((d) => `${d.home_team} vs ${d.away_team}`);
    }

    // default selection to first EU venue if present
    const eu = fixtures.findIndex(f =>
      ['England','Scotland','Wales','Northern Ireland','Ireland','Spain','Portugal','France','Germany','Italy','Netherlands','Belgium','Norway','Sweden','Denmark','Switzerland','Austria','Poland','Czech Republic','Slovakia','Slovenia','Croatia','Serbia','Greece','Turkey'].includes(f.country)
    );
    activeIndex = eu !== -1 ? eu : 0;
    renderFixture(activeIndex);
  },
  error: (error) => {
    console.error('Failed to load fixtures:', error);
    fixtureTitle.textContent = 'Unable to load fixtures';
    fixtureContext.textContent = 'Check the data directory and reload the page.';
  },
});

// ====== Keyboard Left/Right → venue selection ======
window.addEventListener('keydown', (e) => {
  if (!venues.length) return;
  if (e.key === 'ArrowRight') {
    setSelectedVenue(selectedVenueIdx + 1, true);
  } else if (e.key === 'ArrowLeft') {
    setSelectedVenue(selectedVenueIdx - 1, true);
  }
});

// ====== Click pick → select venue ======
renderer.domElement.addEventListener('pointerdown', (event) => {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObject(markerMesh, true);
  if (intersects.length) {
    // find closest instanceId
    const id = intersects[0].instanceId ?? 0;
    const idx = venues.findIndex(v => v.instanceId === id);
    if (idx !== -1) setSelectedVenue(idx, true);
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
