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
      Check your internet connection or download the libraries locally and update the
      <code>&lt;script&gt;</code> tags in <code>index.html</code>.
    </p>
  `;
  globeContainer.appendChild(errorBox);
}

const dependencyChecks = [
  { name: 'three.js', ref: window.THREE, url: 'https://unpkg.com/three@0.160.0/build/three.min.js' },
  {
    name: 'OrbitControls',
    ref: window.THREE?.OrbitControls,
    url: 'https://unpkg.com/three@0.160.0/examples/js/controls/OrbitControls.js',
  },
  { name: 'ThreeGlobe', ref: window.ThreeGlobe, url: 'https://unpkg.com/three-globe@2.34.0/build/three-globe.min.js' },
  { name: 'PapaParse', ref: window.Papa, url: 'https://unpkg.com/papaparse@5.4.1/papaparse.min.js' },
];

const missingDeps = dependencyChecks.filter((dep) => !dep.ref);

if (missingDeps.length > 0) {
  const advice = missingDeps
    .map((dep) => `• ${dep.name} from ${dep.url}`)
    .join('<br />');
  showDependencyError(`The following scripts failed to load:<br />${advice}`);
  throw new Error('Required globe dependencies did not load.');
}

let fixtures = [];
let activeIndex = 0;

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

const globe = new ThreeGlobe({ waitForGlobeReady: true })
  .globeImageUrl('//unpkg.com/three-globe/example/img/earth-night.jpg')
  .bumpImageUrl('//unpkg.com/three-globe/example/img/earth-topology.png')
  .pointAltitude('pointAltitude')
  .pointColor('pointColor')
  .pointLabel((d) => `${d.home_team} vs ${d.away_team}`);

scene.add(globe);

const atmosphere = new THREE.Mesh(
  new THREE.SphereGeometry(globe.getGlobeRadius() * 1.02, 75, 75),
  new THREE.MeshBasicMaterial({
    color: 0x7df9c4,
    transparent: true,
    opacity: 0.08,
  })
);
scene.add(atmosphere);

const halo = new THREE.Mesh(
  new THREE.SphereGeometry(globe.getGlobeRadius() * 1.12, 50, 50),
  new THREE.MeshBasicMaterial({
    color: 0x0093c7,
    transparent: true,
    opacity: 0.06,
  })
);
scene.add(halo);

globeContainer.innerHTML = '';
globeContainer.appendChild(renderer.domElement);

function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

animate();

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
  return `${Math.round(value * 100)}%`;
}

function badgeLabel(name) {
  return name
    .split(' ')
    .map((part) => part[0])
    .join('')
    .slice(0, 3)
    .toUpperCase();
}

function renderFixture(index) {
  const fixture = fixtures[index];
  if (!fixture) return;

  activeIndex = index;
  fixtureTitle.textContent = `${fixture.home_team} vs ${fixture.away_team}`;
  fixtureContext.textContent = `${fixture.competition} • ${formatDate(fixture.date_utc)} • ${fixture.stadium}, ${fixture.city}, ${fixture.country}`;

  homeBadge.textContent = badgeLabel(fixture.home_team);
  awayBadge.textContent = badgeLabel(fixture.away_team);

  matchIntelligenceList.innerHTML = '';
  const intelligenceItems = [
    {
      label: 'Full-time prediction',
      value: `${fixture.predicted_winner} (${toPercent(fixture.confidence_ftr)})`,
    },
    {
      label: 'xG edge',
      value: `${fixture.home_team} ${fixture.xg_home.toFixed(1)} vs ${fixture.away_team} ${fixture.xg_away.toFixed(1)}`,
    },
    {
      label: 'Points momentum',
      value: `${fixture.home_team} ${fixture.ppg_home.toFixed(1)} PPG • ${fixture.away_team} ${fixture.ppg_away.toFixed(1)} PPG`,
    },
  ];

  intelligenceItems.forEach((item) => {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${item.label}:</strong> ${item.value}`;
    matchIntelligenceList.appendChild(li);
  });

  playerWatchlist.innerHTML = '';
  fixture.key_players_shots.forEach((player) => {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${player.name}</strong> ${player.detail}`;
    playerWatchlist.appendChild(li);
  });

  marketSnapshot.innerHTML = '';
  const marketItems = [
    { label: 'Over 2.5 goals', value: toPercent(fixture.over25_prob) },
    { label: 'Both teams to score', value: toPercent(fixture.btts_prob) },
  ];

  fixture.key_players_bookings.forEach((player) => {
    marketItems.push({ label: `${player.name} booking risk`, value: player.detail });
  });

  fixture.key_players_tackles.forEach((player) => {
    marketItems.push({ label: `${player.name} tackles`, value: player.detail });
  });

  marketItems.forEach((item) => {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${item.label}:</strong> ${item.value}`;
    marketSnapshot.appendChild(li);
  });

  deepDiveButton.onclick = () => {
    const summary = `Fixture: ${fixture.home_team} vs ${fixture.away_team}\nKick-off: ${formatDate(fixture.date_utc)}\nPrediction: ${fixture.predicted_winner} (${toPercent(
      fixture.confidence_ftr
    )})\nOver 2.5: ${toPercent(fixture.over25_prob)}\nBTTS: ${toPercent(fixture.btts_prob)}`;
    alert(summary);
  };
}

function processPlayerField(field) {
  if (!field) return [];
  return field.split(';').map((entry) => {
    const [name, detail] = entry.split('|');
    return { name: name.trim(), detail: (detail || '').trim() };
  });
}

function hydrateFixtures(rawFixtures) {
  return rawFixtures
    .filter((fixture) => fixture.fixture_id)
    .map((fixture) => ({
      ...fixture,
      latitude: Number(fixture.latitude),
      longitude: Number(fixture.longitude),
      xg_home: Number(fixture.xg_home),
      xg_away: Number(fixture.xg_away),
      ppg_home: Number(fixture.ppg_home),
      ppg_away: Number(fixture.ppg_away),
      confidence_ftr: Number(fixture.confidence_ftr),
      over25_prob: Number(fixture.over25_prob),
      btts_prob: Number(fixture.btts_prob),
      key_players_shots: processPlayerField(fixture.key_players_shots),
      key_players_bookings: processPlayerField(fixture.key_players_bookings),
      key_players_tackles: processPlayerField(fixture.key_players_tackles),
      pointAltitude: 0.18 + Number(fixture.confidence_ftr) * 0.12,
      pointColor:
        Number(fixture.confidence_ftr) > 0.7
          ? '#64d863'
          : Number(fixture.confidence_ftr) > 0.5
          ? '#00bcd4'
          : '#ff8a65',
    }));
}

function focusFixture(index) {
  const fixture = fixtures[index];
  if (!fixture) return;

  const target = globe
    .pointOfView({ lat: fixture.latitude, lng: fixture.longitude, altitude: 2.2 }, 1200)
    .then(() => renderFixture(index));

  return target;
}

Papa.parse('data/fixtures.csv', {
  download: true,
  header: true,
  dynamicTyping: false,
  complete: (results) => {
    fixtures = hydrateFixtures(results.data);
    globe.pointsData(fixtures);
    renderFixture(0);
    focusFixture(0);
  },
  error: (error) => {
    console.error('Failed to load fixtures:', error);
    fixtureTitle.textContent = 'Unable to load fixtures';
    fixtureContext.textContent = 'Check the data directory and reload the page.';
  },
});

globe.onPointClick((point) => {
  const index = fixtures.findIndex((fixture) => fixture.fixture_id === point.fixture_id);
  if (index !== -1) {
    activeIndex = index;
    focusFixture(index);
  }
});

window.addEventListener('resize', () => {
  const { clientWidth, clientHeight } = globeContainer;
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(clientWidth, clientHeight);
});

window.addEventListener('keydown', (event) => {
  if (fixtures.length === 0) return;

  if (event.key === 'ArrowRight') {
    activeIndex = (activeIndex + 1) % fixtures.length;
    focusFixture(activeIndex);
  }
  if (event.key === 'ArrowLeft') {
    activeIndex = (activeIndex - 1 + fixtures.length) % fixtures.length;
    focusFixture(activeIndex);
  }
});

const uploadInput = document.getElementById('bet-upload');
uploadInput.addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (!file) return;

  const message = `Bet slip uploaded: ${file.name}\n\nNext steps:\n• OCR the slip to extract selections\n• Run the BetChecker audit pipeline\n• Generate OG Co Pilot insights`; // placeholder for pipeline hookup
  alert(message);
  uploadInput.value = '';
});
