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

// --- Helpers for parsing + stadium fallback ---

// Safe number parse
const num = (v) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
};

// Stadium -> lat/lng fallback (approximate coords)
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

// ====== Error helper ======
function showDependencyError(message) {
  if (!globeContainer) {
    console.error(message);
    return;
  }
  globeContainer.innerHTML = '';
  const errorBox = document.createElement('div');
  errorBox.className = 'globe-error';
  errorBox.innerHTML = `
    <h2>Globe assets unavailable</h2>
    <p>${message}</p>
    <p class="globe-error__hint">
      Check your internet connection or vendor the libraries locally in <code>docs/vendor/</code>, then update
      the <code>&lt;script&gt;</code> tags in <code>index.html</code>.
    </p>
  `;
  globeContainer.appendChild(errorBox);
}

// ====== Dependency checks (local vendor versions) ======
const dependencyChecks = [
  { name: 'three.js',            ref: window.THREE,                              url: 'vendor/three.min.js' },
  { name: 'OrbitControls',       ref: window.THREE && window.THREE.OrbitControls, url: 'vendor/OrbitControls.js' },
  { name: 'ThreeGlobe/Globe',    ref: window.ThreeGlobe || window.Globe,         url: 'vendor/three-globe.min.js' },
  { name: 'PapaParse',           ref: window.Papa,                                url: 'vendor/papaparse.min.js' },
];

const missingDeps = dependencyChecks.filter((d) => !d.ref);
if (missingDeps.length) {
  const advice = missingDeps.map((d) => `• ${d.name} from ${d.url}`).join('<br/>');
  showDependencyError(`The following scripts failed to load:<br/>${advice}`);
  throw new Error('Required globe dependencies did not load.');
}

// ====== State ======
let fixtures = [];
let activeIndex = 0;

// ====== THREE basics ======
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(globeContainer.clientWidth, globeContainer.clientHeight);
renderer.outputEncoding = THREE.sRGBEncoding;

const scene = new THREE.Scene();
scene.add(new THREE.AmbientLight(0xffffff, 1));

const camera = new THREE.PerspectiveCamera(
  45,
  globeContainer.clientWidth / globeContainer.clientHeight,
  0.1,
  500
);
camera.position.set(0, 0, 220);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 120;
controls.maxDistance = 320;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.6;

globeContainer.innerHTML = '';
globeContainer.appendChild(renderer.domElement);

// ====== Globe (supports either UMD name) + nicer atmosphere ======
const GlobeCtor = window.ThreeGlobe || window.Globe;
const globe = new GlobeCtor({ waitForGlobeReady: true })
  .showAtmosphere(true)
  .atmosphereAltitude(0.22)
  .atmosphereColor('#66e3d2')
  .globeImageUrl('https://unpkg.com/three-globe/example/img/earth-dark.jpg')
  .bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png')
  .pointAltitude('pointAltitude')
  .pointColor('pointColor')
  .pointLabel((d) => `${d.home_team} vs ${d.away_team}`);

scene.add(globe);

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
const composer = new THREE.EffectComposer(renderer);
const renderPass = new THREE.RenderPass(scene, camera);
composer.addPass(renderPass);

const fxaaPass = new THREE.ShaderPass(THREE.FXAAShader);
fxaaPass.material.uniforms['resolution'].value.set(
  1 / renderer.domElement.width,
  1 / renderer.domElement.height
);
composer.addPass(fxaaPass);

const bloomPass = new THREE.UnrealBloomPass(
  new THREE.Vector2(renderer.domElement.width, renderer.domElement.height),
  (window.devicePixelRatio > 2 || window.innerWidth < 480) ? 0.4 : 0.6, // strength
  0.4,  // radius
  0.85  // threshold
);
composer.addPass(bloomPass);

// ====== Render loop (composer) ======
function animate() {
  controls.update();
  composer.render(); // replaces renderer.render(scene, camera)
  requestAnimationFrame(animate);
}
animate();

// ====== Helpers ======
function formatDate(iso) {
  const date = new Date(iso);
  return new Intl.DateTimeFormat('en-GB', {
    weekday: 'short',
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    timeZoneName: 'short',
  }).format(date);
}

function toPercent(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

function badgeLabel(name) {
  return name
    .split(' ')
    .map((part) => part[0])
    .join('')
    .slice(0, 3)
    .toUpperCase();
}

// Smooth camera fly-to
function flyTo(lat, lng, altitude = 1.8, ms = 900) {
  const pv = globe.pointOfView() || { lat: 0, lng: 0, altitude: 2 };
  const start = { lat: pv.lat || 0, lng: pv.lng || 0, altitude: pv.altitude || 2 };
  const end = { lat, lng, altitude };
  const t0 = performance.now();

  function tick(now) {
    const t = Math.min(1, (now - t0) / ms);
    const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; // easeInOutCubic
    const lerp = (a, b) => a + (b - a) * ease;
    globe.pointOfView({
      lat: lerp(start.lat, end.lat),
      lng: lerp(start.lng, end.lng),
      altitude: lerp(start.altitude, end.altitude)
    });
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ====== Render a single fixture’s side panel ======
function renderFixture(index) {
  const fixture = fixtures[index];
  if (!fixture) return;

  activeIndex = index;
  fixtureTitle.textContent = `${fixture.home_team} vs ${fixture.away_team}`;

  const parts = [
    fixture.stadium?.trim?.(),
    fixture.city?.trim?.(),
    fixture.country?.trim?.()
  ].filter(Boolean).join(", ");
  const ctxTail = parts ? ` • ${parts}` : "";
  fixtureContext.textContent =
    `${fixture.competition} • ${formatDate(fixture.date_utc)}${ctxTail}`;

  homeBadge.textContent = badgeLabel(fixture.home_team);
  awayBadge.textContent = badgeLabel(fixture.away_team);

  // Match Intelligence
  matchIntelligenceList.innerHTML = '';
  const intelligenceItems = [
    { label: 'Full-time prediction', value: `${fixture.predicted_winner} (${toPercent(fixture.confidence_ftr)})` },
    { label: 'xG edge', value: `${fixture.home_team} ${fixture.xg_home?.toFixed?.(1)} vs ${fixture.away_team} ${fixture.xg_away?.toFixed?.(1)}` },
    { label: 'Points momentum', value: `${fixture.home_team} ${fixture.ppg_home?.toFixed?.(1)} PPG • ${fixture.away_team} ${fixture.ppg_away?.toFixed?.(1)} PPG` },
  ];
  intelligenceItems.forEach((item) => {
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

  // Deep dive (placeholder)
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

// ====== CSV helpers ======
function processPlayerField(field) {
  if (!field) return [];
  return field.split(';').map((entry) => {
    const [name, detail] = entry.split('|');
    return { name: (name || '').trim(), detail: (detail || '').trim() };
  });
}

function hydrateFixtures(rawFixtures) {
  return rawFixtures
    .filter((f) => f.fixture_id) // basic guard
    .map((f) => {
      // Prefer CSV lat/lng if present, else fallback via stadium name
      const latCsv = num(f.latitude);
      const lngCsv = num(f.longitude);
      const fallback = stadiumLookup[f.stadium?.trim?.() || ""];

      const latitude = latCsv ?? fallback?.lat;
      const longitude = lngCsv ?? fallback?.lng;

      const cf = num(f.confidence_ftr) || 0;

      return {
        ...f,
        latitude,
        longitude,
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
        pointColor:
          cf > 0.7 ? '#64d863' :
          cf > 0.5 ? '#00bcd4' :
                     '#ff8a65',
      };
    })
    // ensure we can plot on globe
    .filter((fx) => Number.isFinite(fx.latitude) && Number.isFinite(fx.longitude));
}

// ====== Focus fixture with smooth POV ======
function focusFixture(index) {
  const fixture = fixtures[index];
  if (!fixture) return;
  renderFixture(index);
  flyTo(fixture.latitude, fixture.longitude, 2.0, 1000);
}

// ====== Load fixtures TSV and boot ======
// Build cache-busted URL so GH Pages doesn’t serve stale CSV/TSV
const csvUrl = new URL('data/fixtures.csv', window.location.href);
csvUrl.searchParams.set('v', Date.now().toString());

Papa.parse(csvUrl.href, {
  download: true,
  header: true,
  delimiter: "\t",       // tab-delimited snapshot
  skipEmptyLines: true,  // ignore blank/trailing lines
  dynamicTyping: false,
  complete: (results) => {
    fixtures = hydrateFixtures(results.data);

    // Default to first UK/EU fixture if present
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

// Point click -> focus that fixture
if (typeof globe.onPointClick === 'function') {
  globe.onPointClick((pt) => {
    const idx = fixtures.findIndex((f) => f.fixture_id === pt.fixture_id);
    if (idx !== -1) {
      activeIndex = idx;
      focusFixture(idx);
    }
  });
}

// Resize handling (renderer + composer + FXAA)
window.addEventListener('resize', () => {
  const w = globeContainer.clientWidth, h = globeContainer.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  fxaaPass.material.uniforms['resolution'].value.set(1 / w, 1 / h);
});

// Keyboard nav
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

// Upload placeholder
uploadInput.addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  const message =
    `Bet slip uploaded: ${file.name}\n\nNext steps:\n` +
    `• OCR the slip to extract selections\n` +
    `• Run the BetChecker audit pipeline\n` +
    `• Generate OG Co-Pilot insights`;
  alert(message);
  uploadInput.value = '';
});
