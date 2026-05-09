(function () {
  const DATA_ROOT = "./public/data";
  const PREMIUM_TOKEN_STORAGE_KEY = "og_premium_token";
  const app = document.getElementById("app");
  const page = document.body.dataset.page || "home";
  const query = new URLSearchParams(window.location.search);
  const premiumDemoMode = query.get("demo") === "1";
  const debugMode = query.get("debug") === "1";
  const accountIntent = query.get("intent") || "";
  const checkoutState = query.get("checkout") || "";
  const authState = query.get("auth") || "";
  const selectedFixtureKey = query.get("fixture") || "";
  const runtimeConfig = window.OG_CONFIG || {};
  const workerApiBase = String(runtimeConfig.WORKER_API_BASE || "").replace(/\/+$/, "");
  const checkoutPlaceholderHref = "./account.html?intent=checkout";

  const state = {
    summary: null,
    publicPredictions: [],
    premiumPredictions: [],
    securePremiumPredictions: [],
    fixtureIntelligence: [],
    weeklyResults: null,
    runtime: {
      workerApiBase,
      premiumToken: null,
      premiumFetchError: "",
      premiumSourceLabel: "",
      premiumGeneratedAt: "",
      premiumSubscriberCustomerId: "",
      checkoutMessage: "",
      accountMessage: "",
      authMessage: "",
      preferencesMessage: "",
      telegramMessage: "",
      fixtureAlertMessage: "",
      alertsMessage: "",
      sessionAuthenticated: false,
      sessionEntitled: false,
      sessionStatus: "",
      sessionAuthMode: "",
      sessionCustomerId: "",
      sessionSubscriptionId: "",
      accountState: null,
      accountAlerts: [],
      accountStateError: "",
      dashboardClassFilter: "ALL",
      dashboardReasonFilter: "ALL",
      telegramLinkCode: "",
      telegramLinkExpiresAt: "",
      telegramBotUsername: "",
      telegramDeepLinkUrl: "",
    },
  };

  const fetchJson = async (path) => {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load ${path}`);
    }
    return response.json();
  };

  const fetchOptionalJson = async (path) => {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return response.json();
  };

  const escapeHtml = (value) =>
    String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const readStoredPremiumToken = () => {
    try {
      return window.localStorage.getItem(PREMIUM_TOKEN_STORAGE_KEY) || "";
    } catch {
      return "";
    }
  };

  const writeStoredPremiumToken = (token) => {
    try {
      if (token) {
        window.localStorage.setItem(PREMIUM_TOKEN_STORAGE_KEY, token);
      } else {
        window.localStorage.removeItem(PREMIUM_TOKEN_STORAGE_KEY);
      }
    } catch {
      return;
    }
  };

  const workerConfigured = () => Boolean(state.runtime.workerApiBase);
  const premiumTokenPresent = () => Boolean(state.runtime.premiumToken);

  const workerApiUrl = (path) => {
    if (!workerConfigured()) {
      return "";
    }
    return new URL(path, `${state.runtime.workerApiBase}/`).toString();
  };

  const fetchWorkerJson = async (path, options = {}) => {
    const headers = new Headers(options.headers || {});
    headers.set("accept", "application/json");
    if (options.body && !headers.has("content-type")) {
      headers.set("content-type", "application/json");
    }
    if (options.withToken && state.runtime.premiumToken) {
      headers.set("authorization", `Bearer ${state.runtime.premiumToken}`);
    }
    const response = await fetch(workerApiUrl(path), {
      method: options.method || "GET",
      headers,
      body: options.body ? JSON.stringify(options.body) : undefined,
      credentials: "include",
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    return { response, payload };
  };

  const syncActiveNav = () => {
    const pageHrefMap = {
      home: "./index.html",
      dashboard: "./dashboard.html",
      fixture: "./dashboard.html",
      predictions: "./predictions.html",
      premium: "./premium.html",
      results: "./results.html",
      pricing: "./pricing.html",
      methodology: "./methodology.html",
      account: "./account.html",
    };
    const currentHref = pageHrefMap[page] || "";
    document.querySelectorAll(".nav a").forEach((anchor) => {
      const isActive = anchor.getAttribute("href") === currentHref;
      anchor.classList.toggle("is-active", isActive);
    });
  };

  const statPanel = (label, value, note = "") => `
    <article class="panel">
      <span class="muted">${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      ${note ? `<span>${escapeHtml(note)}</span>` : ""}
    </article>
  `;

  const renderNotice = (message, tone = "default") =>
    message ? `<div class="notice notice-${escapeHtml(tone)}">${escapeHtml(message)}</div>` : "";

  const formatDateTime = (value) => {
    if (!value) {
      return "";
    }
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
      return String(value);
    }
    return parsed.toLocaleString("en-GB", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const formatTelegramIdentity = (telegramLink) => {
    if (!telegramLink) {
      return "";
    }
    const username = String(telegramLink.telegram_username || "").trim();
    if (username) {
      return `@${username}`;
    }
    const chatId = String(telegramLink.telegram_chat_id || "").trim();
    if (chatId) {
      return `Linked chat • ${chatId.slice(-4)}`;
    }
    return "Linked account";
  };

  const joinPreferenceList = (value) => (Array.isArray(value) ? value.join(", ") : "");

  const normalizePreferenceText = (value) =>
    String(value || "")
      .normalize("NFKD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase()
      .replace(/&/g, " and ")
      .replace(/[^a-z0-9]+/g, " ")
      .trim()
      .replace(/\s+/g, " ");

  const parsePreferenceList = (value) => {
    if (Array.isArray(value)) {
      return value.map((entry) => String(entry || "").trim()).filter(Boolean);
    }
    return String(value || "")
      .split(",")
      .map((entry) => entry.trim())
      .filter(Boolean);
  };

  const fixturePreferenceLabel = (row) => `${row.home_team} v ${row.away_team}`;
  const fixtureDetailHref = (row) => `./fixture.html?fixture=${encodeURIComponent(String(row.fixture_key || ""))}`;

  const marketFamilyLabel = (value) => {
    const key = String(value || "").toUpperCase();
    if (key === "FTR") return "FTR";
    if (key === "BTTS") return "BTTS";
    if (key === "OU25") return "OU25";
    return key || "INTEL";
  };

  const publishClassRank = (value) => {
    const key = String(value || "").toUpperCase();
    if (key === "DEPLOY") return 4;
    if (key === "OBSERVE") return 3;
    if (key === "CONTEXT") return 2;
    if (key === "MONITOR") return 1;
    return 0;
  };

  const formatKickoffLabel = (value) => {
    if (!value) {
      return "Upcoming fixture";
    }
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
      return String(value);
    }
    return parsed.toLocaleString("en-GB", {
      weekday: "short",
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getFollowedIntelligenceMatches = (accountState, fixtures) => {
    const prefs = accountState?.notification_preferences || null;
    if (!prefs || !Array.isArray(fixtures) || !fixtures.length) {
      return [];
    }

    const teams = parsePreferenceList(prefs.favourite_teams).map(normalizePreferenceText);
    const leagues = parsePreferenceList(prefs.favourite_leagues).map(normalizePreferenceText);
    const markets = parsePreferenceList(prefs.favourite_markets).map((entry) => normalizePreferenceText(entry).replace(/\s+/g, ""));
    const followedFixtures = parsePreferenceList(prefs.followed_fixtures).map(normalizePreferenceText);

    return fixtures
      .map((row) => {
        const reasons = [];
        const rowHome = normalizePreferenceText(row.home_team);
        const rowAway = normalizePreferenceText(row.away_team);
        const rowLeague = normalizePreferenceText(row.league);
        const rowFixture = normalizePreferenceText(fixturePreferenceLabel(row));
        const rowMarket = normalizePreferenceText(marketFamilyLabel(row.signal_summary?.market_family)).replace(/\s+/g, "");

        if (teams.some((entry) => entry && (rowHome.includes(entry) || rowAway.includes(entry) || entry.includes(rowHome) || entry.includes(rowAway)))) {
          reasons.push("followed team");
        }
        if (leagues.some((entry) => entry && (rowLeague.includes(entry) || entry.includes(rowLeague)))) {
          reasons.push("followed league");
        }
        if (markets.some((entry) => entry && rowMarket === entry)) {
          reasons.push("followed market");
        }
        if (followedFixtures.some((entry) => entry && (rowFixture.includes(entry) || entry.includes(rowFixture)))) {
          reasons.push("followed fixture");
        }

        if (!reasons.length) {
          return null;
        }

        return {
          row,
          reasons,
          score: reasons.length * 10 + publishClassRank(row.publish_class),
        };
      })
      .filter(Boolean)
      .sort((left, right) => {
        if (right.score !== left.score) {
          return right.score - left.score;
        }
        const leftTime = Date.parse(left.row.kickoff_time || "") || Number.MAX_SAFE_INTEGER;
        const rightTime = Date.parse(right.row.kickoff_time || "") || Number.MAX_SAFE_INTEGER;
        return leftTime - rightTime;
      })
      .slice(0, 8);
  };

  const intelligenceCard = (entry, telegramEnabled) => {
    const row = entry.row;
    const reasons = entry.reasons.join(" • ");
    const publishClass = String(row.publish_class || row.fixture_class || "MONITOR").toUpperCase();
    const headline =
      row.signal_summary?.headline ||
      row.signal_summary?.summary_text ||
      "This fixture is being monitored through the intelligence layer.";
    const notes = Array.isArray(row.context_summary?.notes) ? row.context_summary.notes.slice(0, 3) : [];
    const notificationPriority = row.follow_relevance?.notification_priority || "normal";
    return `
      <article class="panel intelligence-card intelligence-card-${publishClass.toLowerCase()}">
        <div class="intelligence-card-head">
          <span class="pill">${escapeHtml(publishClass)}</span>
          <span class="muted">${escapeHtml(formatKickoffLabel(row.kickoff_time))}</span>
        </div>
        <div class="intelligence-card-fixture">
          <span class="muted fixture-league">
            ${safeLogoUrl(row.league_logo_url || row.league_flag_url) ? `<img class="league-badge" src="${escapeHtml(safeLogoUrl(row.league_logo_url || row.league_flag_url))}" alt="" loading="lazy" decoding="async" onerror="this.remove()" />` : ""}
            <span>${escapeHtml(row.league)}</span>
          </span>
          <strong class="fixture-teamline">
            ${badgeMarkup(row.home_team_logo_url, row.home_team)}
            <span class="team-name">${escapeHtml(row.home_team)}</span>
            <span class="versus">vs</span>
            ${badgeMarkup(row.away_team_logo_url, row.away_team)}
            <span class="team-name">${escapeHtml(row.away_team)}</span>
          </strong>
        </div>
        <div class="intelligence-meta">
          <span class="chip">${escapeHtml(marketFamilyLabel(row.signal_summary?.market_family))}</span>
          <span class="chip">${escapeHtml(reasons)}</span>
          <span class="chip">${telegramEnabled ? `Telegram ${escapeHtml(notificationPriority)}` : "Website preview"}</span>
        </div>
        <p class="intelligence-headline">${escapeHtml(headline)}</p>
        ${
          notes.length
            ? `<ul class="feature-list compact-list">${notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>`
            : `<p class="muted">No extra context note is published for this fixture yet.</p>`
        }
        <div class="cta-row">
          <a class="ghost-button" href="${fixtureDetailHref(row)}">Open fixture intelligence</a>
        </div>
      </article>
    `;
  };

  const telegramAlertPreview = (entry) => {
    const row = entry.row;
    const publishClass = String(row.publish_class || row.fixture_class || "MONITOR").toUpperCase();
    const headline =
      row.signal_summary?.headline ||
      row.signal_summary?.summary_text ||
      "Intelligence update available.";
    const firstNote = Array.isArray(row.context_summary?.notes) && row.context_summary.notes.length ? row.context_summary.notes[0] : "";
    const lines = [`${publishClass} | ${row.league}`, `${row.home_team} vs ${row.away_team}`, headline];
    if (firstNote) {
      lines.push(firstNote);
    }
    return lines.join("\n");
  };

  const notificationAlertCard = (alert) => {
    let payload = {};
    try {
      payload = JSON.parse(alert.payload_json || "{}");
    } catch {
      payload = {};
    }
    const fixture = payload.fixture || null;
    const reasons = Array.isArray(payload.reasons) ? payload.reasons : [];
    const relevanceTier = String(payload.relevance_tier || alert.notification_priority || "normal");
    const autoGate = String(payload.auto_gate || "").replace(/_/g, " ");
    const status = String(alert.status || "queued").toUpperCase();
    return `
      <article class="panel intelligence-card intelligence-card-${escapeHtml(String(alert.publish_class || "monitor").toLowerCase())}">
        <div class="intelligence-card-head">
          <span class="pill">${escapeHtml(status)}</span>
          <span class="muted">${escapeHtml(formatKickoffLabel(alert.scheduled_for))}</span>
        </div>
        <strong class="fixture-teamline">
          <span class="team-name">${escapeHtml(alert.fixture_label || "Followed fixture")}</span>
        </strong>
        <div class="intelligence-meta">
          <span class="chip">${escapeHtml(String(alert.publish_class || "MONITOR").toUpperCase())}</span>
          <span class="chip">${escapeHtml(alert.market_family || "INTEL")}</span>
          <span class="chip">${escapeHtml(relevanceTier)}</span>
          ${reasons.length ? `<span class="chip">${escapeHtml(reasons.join(" • "))}</span>` : ""}
        </div>
        <p class="intelligence-headline">${escapeHtml(
          fixture?.signal_summary?.headline ||
            fixture?.signal_summary?.summary_text ||
            `${alert.alert_kind || "follow alert"} is queued from the published intelligence feed.`
        )}</p>
        ${autoGate ? `<p class="muted">Auto route: ${escapeHtml(autoGate)}</p>` : ""}
        ${
          alert.delivered_at
            ? `<p class="muted">Delivered ${escapeHtml(formatDateTime(alert.delivered_at))}</p>`
            : alert.last_error
              ? `<p class="muted">Last error: ${escapeHtml(alert.last_error)}</p>`
              : `<p class="muted">Scheduled ${escapeHtml(formatDateTime(alert.scheduled_for))}</p>`
        }
        ${
          fixture?.fixture_key
            ? `<div class="cta-row"><a class="ghost-button" href="${fixtureDetailHref(fixture)}">Open fixture intelligence</a></div>`
            : ""
        }
      </article>
    `;
  };

  const filteredFollowedIntelligence = (entries) => {
    const classFilter = String(state.runtime.dashboardClassFilter || "ALL").toUpperCase();
    const reasonFilter = String(state.runtime.dashboardReasonFilter || "ALL").toLowerCase();
    return entries.filter((entry) => {
      const publishClass = String(entry.row.publish_class || entry.row.fixture_class || "MONITOR").toUpperCase();
      if (classFilter !== "ALL" && publishClass !== classFilter) {
        return false;
      }
      if (reasonFilter !== "all" && !entry.reasons.includes(reasonFilter.replace(/_/g, " "))) {
        return false;
      }
      return true;
    });
  };

  const dashboardFixtureCard = (entry, telegramEnabled) => {
    const row = entry.row;
    const publishClass = String(row.publish_class || row.fixture_class || "MONITOR").toUpperCase();
    const notes = Array.isArray(row.context_summary?.notes) ? row.context_summary.notes.slice(0, 3) : [];
    const odds = row.odds_summary || {};
    return `
      <article class="panel dashboard-card dashboard-card-${publishClass.toLowerCase()}">
        <div class="dashboard-card-top">
          <div>
            <div class="intelligence-card-head">
              <span class="pill">${escapeHtml(publishClass)}</span>
              <span class="chip">${escapeHtml(marketFamilyLabel(row.signal_summary?.market_family))}</span>
              <span class="chip">${escapeHtml(entry.reasons.join(" • "))}</span>
            </div>
            <strong class="fixture-teamline dashboard-teamline">
              ${badgeMarkup(row.home_team_logo_url, row.home_team)}
              <span class="team-name">${escapeHtml(row.home_team)}</span>
              <span class="versus">vs</span>
              ${badgeMarkup(row.away_team_logo_url, row.away_team)}
              <span class="team-name">${escapeHtml(row.away_team)}</span>
            </strong>
            <p class="muted">${escapeHtml(row.league)} • ${escapeHtml(formatKickoffLabel(row.kickoff_time))}</p>
          </div>
          <div class="dashboard-card-priority">
            <span class="metric-label">Telegram route</span>
            <span class="metric-value dashboard-route">${telegramEnabled ? escapeHtml(row.follow_relevance?.notification_priority || "ready") : "Website"}</span>
          </div>
        </div>
        <p class="intelligence-headline">${escapeHtml(row.signal_summary?.headline || row.signal_summary?.summary_text || "Monitoring update published.")}</p>
        <div class="prediction-meta-grid dashboard-odds-grid">
          <div class="signal-cell">
            <span class="signal-label">1X2</span>
            <span class="signal-value">${escapeHtml(
              odds.home_win_odds && odds.draw_odds && odds.away_win_odds
                ? `${odds.home_win_odds} / ${odds.draw_odds} / ${odds.away_win_odds}`
                : "N/A"
            )}</span>
          </div>
          <div class="signal-cell">
            <span class="signal-label">OU25</span>
            <span class="signal-value">${escapeHtml(
              odds.over25_odds && odds.under25_odds ? `${odds.over25_odds} / ${odds.under25_odds}` : "N/A"
            )}</span>
          </div>
          <div class="signal-cell">
            <span class="signal-label">BTTS</span>
            <span class="signal-value">${escapeHtml(
              odds.btts_yes_odds && odds.btts_no_odds ? `${odds.btts_yes_odds} / ${odds.btts_no_odds}` : "N/A"
            )}</span>
          </div>
        </div>
        ${
          notes.length
            ? `<ul class="feature-list compact-list">${notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>`
            : `<p class="muted">No additional context notes have been published for this fixture yet.</p>`
        }
        <div class="dashboard-telegram-preview">
          <span class="metric-label">Telegram alert preview</span>
          <pre>${escapeHtml(telegramAlertPreview(entry))}</pre>
        </div>
        <div class="cta-row">
          <a class="button" href="${fixtureDetailHref(row)}">Open fixture view</a>
          <button class="ghost-button" type="button" data-action="telegram-fixture-alert" data-fixture-key="${escapeHtml(String(row.fixture_key || ""))}">Send to Telegram</button>
        </div>
      </article>
    `;
  };

  const formatProbability = (value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return "N/A";
    }
    return `${Math.round(numeric * 100)}%`;
  };

  const edgeLabel = (row) => row.value_edge_display || row.value_edge || "N/A";
  const confidenceLabel = (row) => row.display_confidence || row.model_prob_display || formatProbability(row.model_prob);
  const tierClass = (tier) => (String(tier || "").toUpperCase() === "STANDARD" ? "standard" : "elite");

  const safeLogoUrl = (value) => {
    const raw = String(value || "").trim();
    if (!raw) {
      return "";
    }
    try {
      const url = new URL(raw, window.location.href);
      const isApiSports = url.protocol === "https:" && url.hostname === "media.api-sports.io";
      const isLocalAsset = url.origin === window.location.origin && url.pathname.includes("/assets/");
      return isApiSports || isLocalAsset ? url.toString() : "";
    } catch {
      return "";
    }
  };

  const teamInitials = (name) => {
    const words = String(name || "")
      .replace(/[^A-Za-z0-9\s-]/g, " ")
      .split(/\s+/)
      .filter(Boolean);
    if (!words.length) {
      return "FC";
    }
    return words
      .slice(0, 2)
      .map((word) => word[0])
      .join("")
      .toUpperCase();
  };

  const badgeMarkup = (url, name, className = "team-badge") => {
    const safeUrl = safeLogoUrl(url);
    const label = escapeHtml(name || "Team");
    const initials = escapeHtml(teamInitials(name));
    return `
      <span class="${className}" aria-hidden="true">
        <span class="badge-fallback">${initials}</span>
        ${safeUrl ? `<img src="${escapeHtml(safeUrl)}" alt="" loading="lazy" decoding="async" onerror="this.remove()" />` : ""}
      </span>
      <span class="sr-only">${label}</span>
    `;
  };

  const fixtureTeamsMarkup = (row, compact = false) => {
    const leagueBadge = safeLogoUrl(row.league_logo_url || row.league_flag_url);
    return `
      <div class="teams fixture-teams ${compact ? "fixture-teams-compact" : ""}">
        <span class="muted fixture-league">
          ${
            leagueBadge
              ? `<img class="league-badge" src="${escapeHtml(leagueBadge)}" alt="" loading="lazy" decoding="async" onerror="this.remove()" />`
              : ""
          }
          <span>${escapeHtml(row.league)} • ${escapeHtml(row.kickoff_time)}</span>
        </span>
        <strong class="fixture-teamline">
          ${badgeMarkup(row.home_team_logo_url, row.home_team)}
          <span class="team-name">${escapeHtml(row.home_team)}</span>
          <span class="versus">vs</span>
          ${badgeMarkup(row.away_team_logo_url, row.away_team)}
          <span class="team-name">${escapeHtml(row.away_team)}</span>
        </strong>
      </div>
    `;
  };

  const proofTile = (label, value, note = "") => `
    <article class="proof-tile">
      <span class="metric-label">${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      ${note ? `<span class="muted">${escapeHtml(note)}</span>` : ""}
    </article>
  `;

  const compactPercent = (value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return "Pending";
    }
    return `${(numeric * 100).toFixed(2)}%`;
  };

  const resultsStatusTone = (value) => {
    if (value == null || !Number.isFinite(Number(value))) {
      return "neutral";
    }
    if (Number(value) >= 0.8) {
      return "good";
    }
    if (Number(value) >= 0.55) {
      return "warn";
    }
    return "bad";
  };

  const buildProofProfile = (weekly) => {
    const total = Math.max(Number(weekly.total_picks) || 0, 1);
    const points = [{ label: "Window open", y: 0.14 }];
    let running = 0;

    (weekly.by_market || []).forEach((item) => {
      running += Number(item.total_picks) || 0;
      const coverage = running / total;
      const settledShare = (Number(item.settled_picks) || 0) / total;
      const hit = Number.isFinite(Number(item.hit_rate)) ? Number(item.hit_rate) : 0.55;
      const value = Math.min(0.9, 0.16 + coverage * 0.42 + settledShare * 0.22 + hit * 0.18);
      points.push({ label: item.market, y: value });
    });

    const overall = Number.isFinite(Number(weekly.overall_hit_rate)) ? Number(weekly.overall_hit_rate) : 0.5;
    const finalValue = Math.min(
      0.94,
      0.2 + ((Number(weekly.settled_picks) || 0) / total) * 0.3 + ((Number(weekly.pending_picks) || 0) / total) * 0.06 + overall * 0.34
    );
    points.push({ label: "Published proof", y: finalValue });
    return points;
  };

  const renderProofCurve = (weekly) => {
    const points = buildProofProfile(weekly);
    const width = 860;
    const height = 300;
    const xStep = width / Math.max(points.length - 1, 1);
    const coords = points.map((point, index) => ({
      x: index * xStep,
      y: height - point.y * height,
      label: point.label,
    }));

    const linePath = coords.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(" ");
    const areaPath = `${linePath} L ${width} ${height} L 0 ${height} Z`;
    const xLabels = coords
      .map(
        (point) => `
          <span>${escapeHtml(point.label)}</span>
        `
      )
      .join("");

    return `
      <article class="proof-graph-panel">
        <div class="proof-graph-head">
          <div>
            <p class="hero-kicker">Proof profile</p>
            <h2>Current graded window curve</h2>
            <p class="section-copy">
              Derived from the current published window using cumulative settled share and hit-rate checkpoints.
              It is a proof profile for this release window, not a fabricated back-history chart.
            </p>
          </div>
          <div class="chart-legend">
            <span><i class="legend-dot legend-dot-live"></i> AI proof profile</span>
            <span><i class="legend-dot"></i> Window baseline</span>
          </div>
        </div>
        <div class="proof-chart-shell">
          <svg viewBox="0 0 ${width} ${height}" class="proof-chart" aria-hidden="true" preserveAspectRatio="none">
            <defs>
              <linearGradient id="proofGlow" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#31c7ff"></stop>
                <stop offset="100%" stop-color="#4edea3"></stop>
              </linearGradient>
              <linearGradient id="proofFill" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="rgba(78, 222, 163, 0.35)"></stop>
                <stop offset="100%" stop-color="rgba(78, 222, 163, 0.03)"></stop>
              </linearGradient>
            </defs>
            <line x1="0" y1="${height - height * 0.26}" x2="${width}" y2="${height - height * 0.26}" class="chart-baseline"></line>
            <path d="${areaPath}" fill="url(#proofFill)"></path>
            <path d="${linePath}" class="chart-line"></path>
            ${coords
              .map(
                (point) => `
                  <circle cx="${point.x.toFixed(2)}" cy="${point.y.toFixed(2)}" r="5" class="chart-point"></circle>
                `
              )
              .join("")}
          </svg>
          <div class="chart-x-labels">${xLabels}</div>
        </div>
      </article>
    `;
  };

  const predictionCard = (row, locked) => {
    const shortlist = Array.isArray(row.correct_score_shortlist) ? row.correct_score_shortlist : [];
    const edge = locked ? row.value_edge_display || edgeLabel(row) : edgeLabel(row);
    return `
      <article class="card prediction-card">
        <div class="prediction-top">
          ${fixtureTeamsMarkup(row)}
          <div class="pill-row">
            <span class="market-badge">${escapeHtml(row.market)}</span>
            <span class="confidence-badge ${tierClass(row.confidence_tier)}">${escapeHtml(row.confidence_tier)}</span>
          </div>
        </div>
        <div class="prediction-core">
          <div class="prediction-call">
            <div>
              <span class="signal-label">Pick</span>
              <strong class="prediction-pick">${escapeHtml(row.pick)}</strong>
            </div>
            <div class="prediction-edge">
              <span class="signal-label">Edge</span>
              <span class="prediction-edge-chip">${escapeHtml(`EV ${edge}`)}</span>
            </div>
          </div>
          <div class="prediction-meta-grid">
            <div class="signal-cell">
              <span class="signal-label">Odds</span>
              <span class="signal-value">${escapeHtml(row.bookie_od ?? "N/A")}</span>
            </div>
            <div class="signal-cell">
              <span class="signal-label">${locked ? "Confidence" : "Model"}</span>
              <span class="signal-value">${escapeHtml(locked ? confidenceLabel(row) : formatProbability(row.model_prob))}</span>
            </div>
            <div class="signal-cell">
              <span class="signal-label">Tier</span>
              <span class="signal-value">${escapeHtml(row.confidence_tier || "N/A")}</span>
            </div>
          </div>
        </div>
        <div class="prediction-footer">
          ${
            locked
              ? `<span class="premium-lock">Free board view</span>`
              : `<span class="value-badge">Deployable edge cleared</span>`
          }
          <span class="pill">${escapeHtml(confidenceLabel(row))}</span>
        </div>
        <p class="muted">${escapeHtml(
          row.short_reason || row.human_reason || "Cleared value-edge threshold vs market price."
        )}</p>
        ${
          !locked && shortlist.length
            ? `<div class="detail-row"><span class="muted">Correct-score support</span><span>${shortlist
                .map((item) => `${escapeHtml(item.scoreline)} (${escapeHtml(item.probability)})`)
                .join(" · ")}</span></div>`
            : ""
        }
      </article>
    `;
  };

  const boardTable = (rows, premium) => {
    if (!rows.length) {
      return `<div class="empty-state">No predictions are available for this board yet.</div>`;
    }
    return `
      <div class="table-shell">
        <table>
          <thead>
            <tr>
              <th>Fixture</th>
              <th>Market</th>
              <th>Pick</th>
              <th>Tier</th>
              <th>${premium ? "Model" : "Confidence"}</th>
              <th>Odds</th>
              <th>${premium ? "Reason" : "Edge"}</th>
            </tr>
          </thead>
          <tbody>
            ${rows
              .map(
                (row) => `
                  <tr>
                    <td>
                      ${fixtureTeamsMarkup(row, true)}
                    </td>
                    <td>${escapeHtml(row.market)}</td>
                    <td>${escapeHtml(row.pick)}</td>
                    <td><span class="confidence-badge">${escapeHtml(row.confidence_tier)}</span></td>
                    <td>${escapeHtml(
                      premium ? row.model_prob ?? "N/A" : row.display_confidence || row.model_prob_display || "N/A"
                    )}</td>
                    <td>${escapeHtml(row.bookie_od ?? "N/A")}</td>
                    <td>${escapeHtml(
                      premium ? row.human_reason || "Qualified premium play." : row.value_edge_display || "N/A"
                    )}</td>
                  </tr>
                `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  };

  const lockedPreviewCards = () => {
    const seedRows = state.publicPredictions.slice(0, 3);
    const rows = seedRows.length
      ? seedRows
      : [
          {
            league: "Premium board",
            kickoff_time: "Subscriber view",
            home_team: "Locked fixture",
            away_team: "Upgrade required",
            market: "ELITE",
            confidence_tier: "Premium",
          },
          {
            league: "Value edge board",
            kickoff_time: "Subscriber view",
            home_team: "Locked fixture",
            away_team: "Upgrade required",
            market: "STANDARD",
            confidence_tier: "Premium",
          },
        ];

    return rows
      .map(
        (row, index) => `
          <article class="card locked-card">
            <div class="prediction-top">
              <div class="teams">
                <span class="muted">${escapeHtml(row.league)} • ${escapeHtml(row.kickoff_time)}</span>
                <strong>${escapeHtml(row.home_team)} vs ${escapeHtml(row.away_team)}</strong>
              </div>
              <div class="pill-row">
                <span class="market-badge">${escapeHtml(row.market || "Premium")}</span>
                <span class="confidence-badge ${tierClass(row.confidence_tier)}">${escapeHtml(row.confidence_tier || "Locked")}</span>
              </div>
            </div>
            <div class="locked-copy">
              <span class="premium-lock">Locked while subscribed</span>
              <h3>Premium card ${index + 1}</h3>
              <p>
                Unlock full pick detail, sharper edge context, shortlist support, and the complete premium board.
              </p>
            </div>
            <div class="blur-stack" aria-hidden="true">
              <span class="blur-line blur-line-short"></span>
              <span class="blur-line"></span>
              <span class="blur-line blur-line-tiny"></span>
            </div>
          </article>
        `
      )
      .join("");
  };

  const workerStatusCopy = () => {
    if (!workerConfigured()) {
      return "Worker not configured";
    }
    if (state.runtime.sessionEntitled) {
      return "Verified premium access";
    }
    if (state.runtime.sessionAuthenticated) {
      return "Signed in";
    }
    return premiumTokenPresent() ? "Token detected" : "Verification required";
  };

  const checkoutCta = () =>
    `<a class="button" data-action="worker-checkout" href="./account.html?intent=checkout">${
      workerConfigured() ? "Unlock founding membership" : "Open checkout placeholder"
    }</a>`;

  const homeView = () => `
    <section class="hero">
      <div class="hero-main">
        <div class="hero-copy-stack">
          <p class="hero-kicker">Prediction intelligence system</p>
          <h1>Identifying when the market is wrong.</h1>
          <p>
            Validated across 139 rolling walk-forward windows, Odds Genius identifies high-conviction football
            markets using league-calibrated model probability, bookmaker value, and goal-shape intelligence.
          </p>
          <div class="pill-row">
            <span class="stat-chip">28 competitions analysed</span>
            <span class="stat-chip">Historical walk-forward validation</span>
            <span class="stat-chip">Selective deployment only</span>
          </div>
          <div class="hero-actions">
            <a class="button" href="./predictions.html">View live board</a>
            <a class="ghost-button" href="./results.html">See proof</a>
            <a class="ghost-button" href="./premium.html">Unlock premium</a>
          </div>
          <p class="footer-note">Historical walk-forward validation. Not a guarantee of future results.</p>
        </div>
        <div class="proof-command">
          <div class="section-head home-proof-head">
            <div>
              <h2>ELITE / PREMIUM system performance</h2>
              <p class="section-copy">Benchmark-safe proof across the current production intelligence stack.</p>
            </div>
            <span class="pill">139 rolling windows</span>
          </div>
          <div class="proof-strip proof-strip-home">
            ${proofTile("Over 2.5 calibrated", "95.35%", "3,828 historical rows")}
            ${proofTile("BTTS calibrated", "93.55%", "3,382 historical rows")}
            ${proofTile("Premium value-edge ROI", "+53.9%", "15,203 historical picks")}
            ${proofTile("Home Team Over 1.5 premium", "93.24%", "1,643 graded rows")}
            ${proofTile("Competitions analysed", "28", "3-year research estate")}
            ${proofTile("Value edge system", "83.31%", "Premium historical hit rate")}
          </div>
        </div>
      </div>
      <aside class="hero-side">
        <article class="sample-board deployment-stack">
          <div class="sample-board-head">
            <div>
              <span class="metric-label">System state</span>
              <strong>Live board, proof layer, premium gate</strong>
            </div>
            <span class="pill">Founder access live</span>
          </div>
          <div class="system-stack">
            <article class="system-row system-row-founder">
              <span class="metric-label">OG Founder</span>
              <strong>£20/mo for life while active</strong>
              <p class="muted">First 250 users. Early access to new markets and selected future systems.</p>
              <a class="button button-small" href="./pricing.html">Claim founder pricing</a>
            </article>
            <article class="system-row">
              <span class="metric-label">Production core</span>
              <strong>FTR • BTTS • Over 2.5 • Value edge overlays</strong>
              <p class="muted">Dominant team over 1.5 goals is emerging as the cleanest next commercial lane.</p>
            </article>
            <article class="system-row">
              <span class="metric-label">Weekly production</span>
              <strong>65+ picks every week, year-round</strong>
              <p class="muted">Lowest reported weekly accuracy 87%. Average 92% across the current stack.</p>
            </article>
            <article class="system-row system-row-soon">
              <span class="metric-label">Coming soon</span>
              <strong>Player events</strong>
              <p class="muted">Shots, tackles, fouls, and bookings as the next controlled intelligence layer.</p>
            </article>
          </div>
        </article>
      </aside>
    </section>

    <section class="section split">
      <div>
        <div class="section-head">
          <div>
            <h2>Live public board</h2>
            <p class="section-copy">
              A compact look at the current free board. Premium members unlock the full deployable set, richer
              reasons, shortlist support, and Worker-protected access.
            </p>
          </div>
          <a class="ghost-button" href="./predictions.html">Open full board</a>
        </div>
        <div class="card-grid">
          ${state.publicPredictions.slice(0, 3).map((row) => predictionCard(row, true)).join("")}
        </div>
      </div>
      <article class="panel">
        <h3>Why this wins</h3>
        <ul class="method-list">
          <li>Most betting products predict every match. Odds Genius does the opposite.</li>
          <li>It scans broadly, filters aggressively, and only deploys when independent systems agree that the price is wrong and the signal is stable enough to act on.</li>
          <li>The edge is not just prediction accuracy. The edge is knowing when not to bet.</li>
        </ul>
      </article>
    </section>

    <section class="section split">
      <article class="panel">
        <h3>The decision layer is the moat.</h3>
        <ul class="method-list">
          <li>Core models estimate probability.</li>
          <li>Poisson goal mass checks match structure.</li>
          <li>Value edge compares model price against bookmaker price.</li>
          <li>Volatility and fragility filters suppress dangerous signals.</li>
          <li>When the systems align, the pick deploys. When they conflict, it stays out.</li>
        </ul>
      </article>
      <article class="panel">
        <h3>Why it stands apart</h3>
        <ul class="method-list">
          <li>Typical public tipster products struggle to sustain even 50–60% with lower volume and weaker proof discipline.</li>
          <li>Odds Genius combines a flagship selective layer with a weekly production engine delivering 65+ picks at scale.</li>
          <li>Consolidated proof includes FTR, OU2.5, BTTS, premium value edge, acca formatting, and correct score support.</li>
          <li>If the model doesn't beat the price, it doesn't deploy.</li>
        </ul>
      </article>
    </section>
  `;

  const predictionsView = () => `
    <section class="section">
      <div class="hero-main board-layout">
        <div class="board-toolbar">
          <div class="board-hero-copy">
            <p class="hero-kicker">Predictions Board</p>
            <h1>Live public picks.</h1>
          <p class="section-copy">
            Fast-scan board from the latest validated deploy. Premium carries the full signal density.
          </p>
          <p class="section-copy">All picks are derived from model probability vs bookmaker implied probability.</p>
        </div>
          <div class="pill-row">
            <span class="pill pill-elite">${state.summary.public_predictions_count} free picks</span>
            <span class="pill">${escapeHtml(sourceWindowLabel())}</span>
          </div>
        </div>
        <div class="proof-strip">
          ${proofTile("Source rows", state.summary.source_rows_read)}
          ${proofTile("Public cards", state.summary.public_predictions_count)}
          ${proofTile("Premium cards", state.summary.premium_predictions_count)}
          ${proofTile("Proof page", "Weekly results", "Settled outcomes published separately")}
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Current board</h2>
          <p class="section-copy">
            Denser live cards with clearer hierarchy across market, pick, odds, confidence, and edge.
          </p>
        </div>
        <a class="ghost-button" href="./premium.html">See premium unlock</a>
      </div>
      <div class="card-grid">
        ${state.publicPredictions.map((row) => predictionCard(row, true)).join("")}
      </div>
    </section>
  `;

  const premiumView = () => {
    if (premiumDemoMode && state.premiumPredictions.length) {
      return `
        <section class="section split">
          <article class="hero-main">
            <p class="hero-kicker">Premium Demo Mode</p>
            <h1>Internal premium preview.</h1>
            <p>
              Demo mode is enabled for internal product review, so the exported premium board is rendered below.
              Customer-facing access should continue to rely on the Worker route.
            </p>
            <div class="cta-row">
              <a class="button" href="./pricing.html">See founding plan</a>
              <a class="ghost-button" href="./premium.html">Return to locked view</a>
            </div>
          </article>
          <aside class="hero-side">
            <div class="metric">
              <span class="metric-label">Rendered premium cards</span>
              <span class="metric-value">${state.summary.premium_predictions_count}</span>
            </div>
            <div class="metric">
              <span class="metric-label">Mode</span>
              <span class="metric-value">Demo</span>
            </div>
          </aside>
        </section>
        <section class="section">
          <div class="section-head">
            <div>
              <h2>Premium board preview</h2>
              <p class="section-copy">
                Internal-only rendering of the premium board for product review.
              </p>
            </div>
          </div>
          ${boardTable(state.premiumPredictions, true)}
        </section>
      `;
    }

    const secureBoardReady = state.securePremiumPredictions.length > 0;
    const workerMessage =
      state.runtime.premiumFetchError ||
      (!state.runtime.sessionAuthenticated && !premiumTokenPresent()
        ? "Verify your email or sign in to unlock the premium board."
        : "");

    return `
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Premium Board</p>
          <h1>${secureBoardReady ? "Protected premium access is live." : "The strongest board stays locked by default."}</h1>
          <p>
            ${
              secureBoardReady
                ? "This board is being served through the Worker after verified session or premium-access checks."
                : state.runtime.sessionAuthenticated
                  ? "Your session is recognized. Premium unlock completes automatically once verified entitlement is active."
                  : "Premium unlocks the complete deployable board — not just more picks, but the deeper pricing intelligence behind them. Built for bettors who care about value, stability, and proof."
            }
          </p>
          <div class="cta-row">
            ${
              state.runtime.sessionAuthenticated
                ? `<a class="button" href="./account.html">Manage access</a>`
                : `<a class="button" data-action="worker-checkout" href="./account.html?intent=checkout">Unlock founding membership — £20/month</a>`
            }
            <a class="ghost-button" href="${state.runtime.sessionAuthenticated ? "./account.html" : "./pricing.html"}">${
              state.runtime.sessionAuthenticated ? "Open account" : "See pricing"
            }</a>
          </div>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Access state</span>
            <span class="metric-value">${escapeHtml(workerStatusCopy())}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Premium cards</span>
            <span class="metric-value">${secureBoardReady ? state.securePremiumPredictions.length : state.summary.premium_predictions_count}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Status</span>
            <span class="metric-value">${secureBoardReady ? "Unlocked" : "Locked"}</span>
          </div>
        </aside>
      </section>

      ${
        secureBoardReady
          ? `
            <section class="section">
              <div class="stats-grid">
                ${statPanel("Protected cards", state.securePremiumPredictions.length)}
                ${statPanel("Subscriber", state.runtime.premiumSubscriberCustomerId || "Verified")}
                ${statPanel("Worker source", state.runtime.premiumSourceLabel || "Configured")}
                ${statPanel("Published", state.runtime.premiumGeneratedAt ? state.runtime.premiumGeneratedAt.slice(0, 10) : "Current board")}
              </div>
            </section>
            <section class="section">
              ${boardTable(state.securePremiumPredictions, true)}
            </section>
          `
          : `
            <section class="section">
              ${renderNotice(
                workerMessage ||
                  (workerConfigured()
                    ? "Verified premium access is required before the full board is shown."
                    : "Worker premium access is not configured yet."),
                workerConfigured() ? "warning" : "default"
              )}
            </section>
            <section class="section">
              <div class="locked-grid">
                ${lockedPreviewCards()}
              </div>
            </section>
          `
      }

      <section class="section split">
        <article class="panel">
          <h3>What you actually get</h3>
          <ul class="feature-list">
            <li>Full deployable board.</li>
            <li>ELITE and STANDARD picks.</li>
            <li>Value edge vs bookmaker implied probability.</li>
            <li>Dominant team-goals angles where the proof is strongest.</li>
            <li>Correct score shortlist support.</li>
            <li>Acca safety signals.</li>
            <li>Stability and fragility context.</li>
            <li>Worker-protected subscriber access.</li>
          </ul>
        </article>
        <article class="panel">
          <h3>Premium value tier performance</h3>
          <ul class="feature-list">
            <li>15,203 signals.</li>
            <li>83.31% hit rate.</li>
            <li>+53.90% ROI.</li>
            <li>139 historical walk-forward windows.</li>
            <li>Historical validation only. Not a guarantee.</li>
          </ul>
        </article>
      </section>
      <section class="section">
        <div class="notice">Public shows the signal. Premium shows where the edge actually lives. Early pricing. Locked while subscribed.</div>
      </section>
    `;
  };

  const resultsView = () => {
    const weekly = state.weeklyResults;
    if (!weekly) {
      return `
        <section class="section split">
          <article class="hero-main">
            <p class="hero-kicker">Results & Proof</p>
            <h1>Proof layer waiting on settled grading.</h1>
            <p>
              This page will show settled proof once weekly results are available. The structure is live, but the
              current window has not been published here yet.
            </p>
          </article>
          <aside class="hero-side">
            <div class="metric">
              <span class="metric-label">Current source rows</span>
              <span class="metric-value">${state.summary.source_rows_read}</span>
            </div>
            <div class="metric">
              <span class="metric-label">Publish timestamp</span>
              <span class="metric-value">${escapeHtml(state.summary.generated_at.slice(0, 10))}</span>
            </div>
          </aside>
        </section>
        <section class="section">
          <div class="notice">
            Run grade_weekend_results.py after settled outcomes are available to publish weekly proof.
          </div>
        </section>
      `;
    }

    const marketCards = weekly.by_market
      .map(
        (item) => `
          <article class="panel market-proof-card market-proof-card--${resultsStatusTone(item.hit_rate)}">
            <span class="muted">${escapeHtml(item.market)}</span>
            <strong>${escapeHtml(item.hit_rate == null ? "Pending" : `${Math.round(item.hit_rate * 100)}%`)}</strong>
            <span>${escapeHtml(`${item.settled_picks}/${item.total_picks} settled`)}</span>
          </article>
        `
      )
      .join("");

    const tierCards = weekly.by_tier
      .map(
        (item) => `
          <article class="panel tier-panel tier-${String(item.tier || "").toLowerCase()}">
            <span class="muted">${escapeHtml(item.tier)}</span>
            <strong>${escapeHtml(item.hit_rate == null ? "Pending" : `${Math.round(item.hit_rate * 100)}%`)}</strong>
            <span>${escapeHtml(`${item.settled_picks}/${item.total_picks} settled`)}</span>
          </article>
        `
      )
      .join("");

    const featuredList = (rows, emptyLabel) =>
      rows.length
        ? rows
            .map(
              (row) => `
                <li>
                  <strong>${escapeHtml(row.home_team)} vs ${escapeHtml(row.away_team)}</strong><br />
                  <span class="muted">${escapeHtml(row.league)} • ${escapeHtml(row.market)} • ${escapeHtml(row.pick)} • ${escapeHtml(row.confidence_tier)}</span>
                </li>
              `
            )
            .join("")
        : `<li>${escapeHtml(emptyLabel)}</li>`;

    return `
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Weekly Proof</p>
          <h1>Settled board proof.</h1>
          <p>
            Public-safe weekly proof generated from scored deploy outputs. Use this page to evaluate settled
            performance, not hype.
          </p>
          <p class="section-copy">Results are published independently from predictions.</p>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Window</span>
            <span class="metric-value">${escapeHtml(`${weekly.period_start} → ${weekly.period_end}`)}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Overall hit rate</span>
            <span class="metric-value">${escapeHtml(weekly.overall_hit_rate == null ? "Pending" : compactPercent(weekly.overall_hit_rate))}</span>
            <span class="muted">Tracked across ${weekly.total_picks} picks (${weekly.settled_picks} settled so far)</span>
          </div>
          <div class="metric">
            <span class="metric-label">Settled picks</span>
            <span class="metric-value">${escapeHtml(`${weekly.settled_picks}/${weekly.total_picks}`)}</span>
          </div>
        </aside>
      </section>

      <section class="section">
        <div class="results-highlight results-highlight--four">
          ${statPanel("Total picks", weekly.total_picks, `${weekly.period_start} → ${weekly.period_end}`)}
          ${statPanel("Settled picks", weekly.settled_picks, `${weekly.pending_picks} still live`)}
          ${statPanel("Pending picks", weekly.pending_picks, "Awaiting graded resolution")}
          ${statPanel("Hit rate", weekly.overall_hit_rate == null ? "Pending" : compactPercent(weekly.overall_hit_rate), weekly.generated_at.slice(0, 10))}
        </div>
      </section>

      <section class="section">
        ${renderProofCurve(weekly)}
      </section>

      <section class="section">
        <div class="section-head">
          <div>
            <h2>By market</h2>
            <p class="section-copy">Public-safe hit-rate summary for deployable markets in the graded window.</p>
          </div>
        </div>
        <div class="stats-grid">${marketCards}</div>
      </section>

      <section class="section">
        <div class="section-head">
          <div>
            <h2>By tier</h2>
            <p class="section-copy">Deployable board performance split by live confidence tier.</p>
          </div>
        </div>
        <div class="stats-grid">${tierCards}</div>
      </section>

      <section class="section split">
        <article class="panel featured-proof-panel">
          <h3>Featured wins</h3>
          <ul class="feature-list">${featuredList(weekly.featured_wins, "No settled wins surfaced yet.")}</ul>
        </article>
        <article class="panel">
          <h3>Featured misses</h3>
          <ul class="feature-list">${featuredList(weekly.featured_misses, "No settled misses surfaced yet.")}</ul>
        </article>
      </section>

      <section class="section">
        <div class="notice">
          ${(weekly.notes || []).map((note) => escapeHtml(note)).join("<br />") || "No additional notes."}
        </div>
      </section>
    `;
  };

  const pricingView = () => `
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Pricing</p>
          <h1>Choose your intelligence level.</h1>
          <p>
            Start free. Secure founder pricing before the main Pro ladder expands.
          </p>
          <div class="pill-row">
            <span class="stat-chip">Worker-protected access</span>
            <span class="stat-chip">First 250 founders</span>
          </div>
          <div class="cta-row">
            ${checkoutCta().replace("Unlock founding membership", "Secure founder access")}
            <a class="ghost-button" href="./premium.html">Preview locked premium</a>
          </div>
        </article>
      <aside class="hero-side">
        <div class="metric">
          <span class="metric-label">Plan</span>
          <span class="metric-value">OG Founder</span>
        </div>
        <div class="metric">
          <span class="metric-label">Founder pricing</span>
          <span class="metric-value">£20/mo for life while active</span>
        </div>
        <div class="metric">
          <span class="metric-label">Cohort cap</span>
          <span class="metric-value">250 users</span>
        </div>
      </aside>
    </section>

    <section class="section">
      ${renderNotice(state.runtime.checkoutMessage, state.runtime.checkoutMessage ? "warning" : "default")}
      <div class="pricing-grid">
        <article class="card pricing-card pricing-card-free">
          <span class="pricing-tag">Free</span>
          <div class="pricing-price">£0</div>
          <p class="pricing-subcopy">Public preview of the board, proof, and methodology layer.</p>
          <ul class="feature-list">
            <li>Limited public board.</li>
            <li>Rounded confidence.</li>
            <li>Rounded edge display.</li>
            <li>Public proof access.</li>
            <li>No shortlist or premium context.</li>
          </ul>
        </article>
        <article class="card pricing-card featured pricing-card-premium">
          <span class="pricing-ribbon">First 250 users</span>
          <span class="pricing-tag">OG Founder</span>
          <div class="pricing-price">£20<span class="pricing-price-note">/month</span></div>
          <p class="pricing-subcopy">Fixed founder pricing while active. Full deployable board access, founder-only upside, and early access to selected advanced systems.</p>
          <ul class="feature-list">
            <li>Full deployable board.</li>
            <li>ELITE and STANDARD signals.</li>
            <li>Full value-edge layer.</li>
            <li>Correct score shortlist support.</li>
            <li>Acca safety indicators.</li>
            <li>Protected Worker-backed access.</li>
            <li>Early access to new markets.</li>
            <li>Early access to selected future systems.</li>
          </ul>
          <div class="notice founder-guardrail">
            £20/month for life while active. Non-transferable. Founder pricing ends after the first 250 users.
          </div>
          <div class="cta-row">
            ${checkoutCta().replace("Unlock founding membership", "Secure founder access")}
          </div>
        </article>
      </div>
      <div class="pricing-band">
        <article class="pricing-band-card">
          <span class="metric-label">Founder advantage</span>
          <strong>Grandfathered pricing while active</strong>
        </article>
        <article class="pricing-band-card">
          <span class="metric-label">Expansion path</span>
          <strong>Selected future systems and new markets unlocked earlier</strong>
        </article>
      </div>
      <div class="section-head pricing-matrix-head">
        <div>
          <h2>Technical matrix comparison</h2>
          <p class="section-copy">Useful for quickly seeing what the free board proves versus what founder access actually unlocks.</p>
        </div>
      </div>
      <div class="table-shell pricing-matrix-shell">
        <table class="pricing-matrix">
          <thead>
            <tr>
              <th>Feature detail</th>
              <th>Free tier</th>
              <th>OG Founder</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Board scope</td>
              <td>Limited public board</td>
              <td>Full deployable board</td>
            </tr>
            <tr>
              <td>Signal depth</td>
              <td>Rounded confidence and rounded edge display</td>
              <td>Full value-edge layer and premium explanation</td>
            </tr>
            <tr>
              <td>Shortlist support</td>
              <td>No shortlist or premium context</td>
              <td>Correct score shortlist support</td>
            </tr>
            <tr>
              <td>Slip support</td>
              <td>Public board only</td>
              <td>Acca safety indicators</td>
            </tr>
            <tr>
              <td>Protection layer</td>
              <td>Static public access</td>
              <td>Protected access through live Worker entitlement</td>
            </tr>
            <tr>
              <td>Founder upside</td>
              <td>None</td>
              <td>Early markets and selected advanced-system access</td>
            </tr>
          </tbody>
        </table>
      </div>
      <section class="pricing-visual-note">
        <article class="pricing-visual-card">
          <span class="metric-label">Founder access</span>
          <h2>Founder pricing now. Pro ladder later.</h2>
          <p class="section-copy">
            OG Founder sits above the free board and below the future Pro ladder. It gives the first cohort more than a normal entry tier without promising unlimited access to every future product line.
          </p>
        </article>
      </section>
      <section class="section">
        <article class="panel">
          <h3>Advanced plans coming later</h3>
          <p class="muted">
            Pro tiers will introduce deeper diagnostics, richer controls, and expanded workflow tooling.
          </p>
        </article>
      </section>
      <p class="footer-note">
        ${
          workerConfigured()
            ? "The founder CTA now routes to the live Worker checkout flow."
            : "Static-only mode cannot provide secure subscriber enforcement."
        }
      </p>
      <p class="footer-note">If you are serious about betting, you do not need more picks. You need better pricing.</p>
    </section>
  `;

  const methodologyView = () => `
    <section class="section split">
      <article class="hero-main">
        <p class="hero-kicker">Methodology</p>
        <h1>Built on generated outputs, not frontend guesswork.</h1>
        <p>
          Odds Genius does not re-run model logic in the browser. The website displays approved exports from the
          live deployment engine.
        </p>
        <p class="section-copy">
          The live shell reflects historical walk-forward validation across 139 rolling windows and a wider
          28-competition research estate, while only benchmark-safe production markets are presented as commercial proof.
        </p>
      </article>
      <aside class="hero-side">
        <div class="metric">
          <span class="metric-label">Public schema fields</span>
          <span class="metric-value">${state.summary.public_fields.length}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Premium schema fields</span>
          <span class="metric-value">${state.summary.premium_fields.length}</span>
        </div>
      </aside>
    </section>
    <section class="section">
      <div class="card-grid">
        <article class="panel">
          <h3>Pipeline boundary</h3>
          <p class="muted">Built on generated outputs, not frontend guesswork. Ingest, enrichment, routing, and slip logic remain in Python.</p>
        </article>
        <article class="panel">
          <h3>Decision stack</h3>
          <p class="muted">The system combines independent modelling approaches and only publishes signals that pass probability, structure, value, goal-shape, and stability checks.</p>
        </article>
        <article class="panel">
          <h3>Final line</h3>
          <p class="muted">Odds Genius is not built to predict every game. It is built to identify when the market is wrong.</p>
        </article>
      </div>
    </section>
  `;

  const accountView = () => {
    const checkoutNotice =
      checkoutState === "success"
        ? "Your membership is almost ready. Verify your email to unlock premium board access on this device."
        : checkoutState === "cancelled"
          ? "Checkout was cancelled. No premium access was granted."
          : "";
    const signedIn = state.runtime.sessionAuthenticated;
    const entitled = state.runtime.sessionEntitled;
    const accountHeadline = entitled
      ? "Premium access unlocked."
      : checkoutState === "success"
        ? "Your membership is almost ready."
        : "Verify your email to unlock premium access.";
    const accountCopy = entitled
      ? "This device is verified for your membership. Open the premium board or return to proof while subscription management catches up."
      : "Use the same email you used for checkout. Once verified, premium unlocks automatically on this device.";
    const accountState = state.runtime.accountState;
    const telegramLink = accountState?.telegram_link || null;
    const notificationPreferences = accountState?.notification_preferences || null;
    const telegramLinked = telegramLink?.link_status === "linked";
    const telegramReady = Boolean(state.runtime.telegramLinkCode);
    const subscriptionStatus = accountState?.subscription?.subscription_status || (entitled ? "active" : "pending");
    const displayEmail = accountState?.user?.email || "";
    const followedIntelligence = getFollowedIntelligenceMatches(accountState, state.fixtureIntelligence);
    const followedSignalsConfigured = Boolean(
      parsePreferenceList(notificationPreferences?.favourite_teams).length ||
        parsePreferenceList(notificationPreferences?.favourite_leagues).length ||
        parsePreferenceList(notificationPreferences?.favourite_markets).length ||
        parsePreferenceList(notificationPreferences?.followed_fixtures).length
    );

    return `
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Account</p>
          <h1>${accountHeadline}</h1>
          <p>${accountCopy}</p>
          <div class="cta-row">
            <a class="button" href="${entitled ? "./premium.html" : "./pricing.html"}">${
              entitled ? "Open premium board" : "See founding plan"
            }</a>
            ${signedIn ? `<a class="ghost-button" href="./dashboard.html">Open dashboard</a>` : ""}
            <a class="ghost-button" href="${entitled ? "./results.html" : "./premium.html"}">${
              entitled ? "Go to results" : "Open premium page"
            }</a>
            ${signedIn ? `<button class="ghost-button" type="button" data-action="auth-logout">Log out</button>` : ""}
          </div>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Worker</span>
            <span class="metric-value">${workerConfigured() ? "Configured" : "Placeholder"}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Identity</span>
            <span class="metric-value">${signedIn ? "Verified" : "Not verified"}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Membership</span>
            <span class="metric-value">${entitled ? "Premium active" : "Premium pending"}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Telegram</span>
            <span class="metric-value">${telegramLinked ? "Linked" : signedIn ? "Available" : "Locked"}</span>
          </div>
        </aside>
      </section>

      <section class="section">
        ${renderNotice(checkoutNotice, checkoutState === "success" ? "success" : "warning")}
        ${renderNotice(state.runtime.authMessage, state.runtime.authMessage ? "warning" : "default")}
        ${renderNotice(state.runtime.telegramMessage, state.runtime.telegramMessage ? "success" : "default")}
        ${renderNotice(state.runtime.preferencesMessage, state.runtime.preferencesMessage ? "success" : "default")}
        ${debugMode ? renderNotice(state.runtime.accountMessage, state.runtime.accountMessage ? "warning" : "default") : ""}
      </section>

      ${
        !signedIn
          ? `
            <section class="section split">
              <article class="panel">
                <h3>Verify your email</h3>
                <p class="muted">
                  Use the same email you used for checkout. If the address is eligible, a sign-in link will be sent.
                </p>
                <form id="magic-link-form" class="stack-form">
                  <label class="field-label" for="magic-link-email">Email</label>
                  <input id="magic-link-email" name="email" class="text-input" type="email" placeholder="you@example.com" autocomplete="email" />
                  <div class="cta-row">
                    <button class="button" type="submit">Send sign-in link</button>
                  </div>
                </form>
              </article>
              <article class="panel">
                <h3>${accountIntent === "checkout" ? "Checkout handoff" : "Access status"}</h3>
                <ul class="feature-list">
                  <li>Checkout confirms billing, not identity.</li>
                  <li>Webhook-backed subscriber state remains the authority for premium access.</li>
                  <li>Email verification unlocks this device when your active membership is confirmed.</li>
                </ul>
              </article>
            </section>
          `
          : debugMode
          ? `
            <section class="section split">
              <article class="panel">
                <h3>Developer/test token flow</h3>
                <p class="muted">
                  Internal only. This exists for controlled Worker verification before magic-link or authenticated
                  session issuance is finished.
                </p>
                <form id="premium-token-form" class="stack-form">
                  <label class="field-label" for="premium-token-input">Premium token</label>
                  <textarea id="premium-token-input" name="premium_token" class="text-input" rows="6" placeholder="Paste developer/test premium token here">${escapeHtml(
                    state.runtime.premiumToken || ""
                  )}</textarea>
                  <div class="cta-row">
                    <button class="button" type="submit">Save token</button>
                    <button class="ghost-button" type="button" data-action="clear-premium-token">Clear token</button>
                  </div>
                </form>
              </article>
              <article class="panel">
                <h3>${accountIntent === "checkout" ? "Checkout handoff" : "Subscription management placeholder"}</h3>
                <ul class="feature-list">
                  <li>Worker checkout route is used when WORKER_API_BASE is configured.</li>
                  <li>Webhook-backed subscriber state remains the authority for premium access.</li>
                  <li>Customer Portal and real sign-in still need production wiring later.</li>
                </ul>
              </article>
            </section>
          `
          : `
            <section class="section split">
              <article class="panel">
                <h3>Account state</h3>
                <ul class="feature-list">
                  <li>Verified email: ${displayEmail ? escapeHtml(displayEmail) : "Signed in"}</li>
                  <li>Membership status: ${escapeHtml(subscriptionStatus)}</li>
                  <li>D1 profile state: ${accountState ? "Active" : state.runtime.accountStateError ? "Unavailable" : "Pending"}</li>
                </ul>
              </article>
              <article class="panel">
                <h3>Telegram premium access</h3>
                <p class="muted">
                  Link Telegram when you want premium comms, elite deployment alerts, and future acca drops beyond the website shell.
                </p>
                ${
                  telegramLinked
                    ? `
                      <ul class="feature-list">
                        <li>Telegram linked: ${escapeHtml(formatTelegramIdentity(telegramLink) || "Linked account")}</li>
                        <li>Linked at: ${escapeHtml(formatDateTime(telegramLink.linked_at) || "Recently linked")}</li>
                        <li>Telegram alerts: ${notificationPreferences?.telegram_enabled ? "Enabled" : "Ready to enable"}</li>
                      </ul>
                      <div class="cta-row">
                        <button class="button button-secondary" type="button" data-action="telegram-test-alert">Send test Telegram alert</button>
                      </div>
                    `
                    : `
                      <div class="cta-row">
                        <button class="button" type="button" data-action="telegram-link-start">Generate Telegram link code</button>
                      </div>
                      ${
                        telegramReady
                          ? `<div class="notice">Link code: <strong>${escapeHtml(state.runtime.telegramLinkCode)}</strong>${
                              state.runtime.telegramDeepLinkUrl
                                ? ` · <a href="${escapeHtml(state.runtime.telegramDeepLinkUrl)}" target="_blank" rel="noreferrer">Open Telegram bot</a>`
                                : ""
                            }<br><span class="muted">Expires ${escapeHtml(formatDateTime(state.runtime.telegramLinkExpiresAt))}</span></div>`
                          : ""
                      }
                    `
                }
              </article>
            </section>
            <section class="section">
              <div class="notice">Subscription management is coming soon.</div>
            </section>
            <section class="section">
              <article class="panel">
                <h3>Followed intelligence</h3>
                <p class="muted">
                  This is the first website surface for the intelligence layer. Saved teams, leagues, markets, and fixtures now pull in routed and non-routed cards from the published fixture intelligence feed.
                </p>
                ${
                  !followedSignalsConfigured
                    ? `<div class="notice">Add followed teams, leagues, markets, or fixtures below to start shaping your personal intelligence board.</div>`
                    : followedIntelligence.length
                      ? `
                        <div class="card-grid intelligence-grid">
                          ${followedIntelligence.map((entry) => intelligenceCard(entry, Boolean(notificationPreferences?.telegram_enabled))).join("")}
                        </div>
                      `
                      : `<div class="notice">Your follow settings are saved, but no current published fixtures matched this window yet. That will change as covered-fixture and context publishing expands.</div>`
                }
              </article>
            </section>
            <section class="section">
              <article class="panel">
                <h3>My alerts</h3>
                <p class="muted">
                  This queue is the first step from followed intelligence into automatic Telegram delivery. Team and fixture follows take priority. Lower-relevance market-only matches stay in the website feed instead of auto-sending to Telegram.
                </p>
                ${renderNotice(state.runtime.alertsMessage, state.runtime.alertsMessage ? "success" : "default")}
                <div class="cta-row">
                  <button class="button" type="button" data-action="refresh-followed-alerts">Refresh followed alerts</button>
                  <button class="ghost-button" type="button" data-action="dispatch-followed-alerts">Process due Telegram alerts</button>
                </div>
                ${
                  state.runtime.accountAlerts.length
                    ? `<div class="card-grid intelligence-grid">${state.runtime.accountAlerts
                        .slice(0, 6)
                        .map((alert) => notificationAlertCard(alert))
                        .join("")}</div>`
                    : `<div class="notice">No queued or delivered alerts are stored for this account yet.</div>`
                }
              </article>
            </section>
            <section class="section">
              <article class="panel">
                <h3>Intelligence preferences</h3>
                <p class="muted">
                  Choose what kind of intelligence you want on the website and in Telegram. This is the first layer of personalised delivery.
                </p>
                <form id="preferences-form" class="stack-form">
                  <div class="card-grid">
                    <article class="panel">
                      <h4>Channels</h4>
                      <label class="checkbox-row"><input type="checkbox" name="telegram_enabled" ${notificationPreferences?.telegram_enabled ? "checked" : ""} /> Telegram alerts</label>
                      <label class="checkbox-row"><input type="checkbox" name="email_enabled" ${notificationPreferences?.email_enabled ? "checked" : ""} /> Email digests</label>
                      <label class="checkbox-row"><input type="checkbox" name="website_only_mode" ${notificationPreferences?.website_only_mode ? "checked" : ""} /> Website-first only</label>
                    </article>
                    <article class="panel">
                      <h4>Signal alerts</h4>
                      <label class="checkbox-row"><input type="checkbox" name="elite_alerts_enabled" ${notificationPreferences?.elite_alerts_enabled ? "checked" : ""} /> Elite deployments</label>
                      <label class="checkbox-row"><input type="checkbox" name="standard_alerts_enabled" ${notificationPreferences?.standard_alerts_enabled ? "checked" : ""} /> Standard deployments</label>
                      <label class="checkbox-row"><input type="checkbox" name="acca_alerts_enabled" ${notificationPreferences?.acca_alerts_enabled ? "checked" : ""} /> Acca drops</label>
                      <label class="checkbox-row"><input type="checkbox" name="correct_score_alerts_enabled" ${notificationPreferences?.correct_score_alerts_enabled ? "checked" : ""} /> Correct score support</label>
                    </article>
                    <article class="panel">
                      <h4>Intelligence alerts</h4>
                      <label class="checkbox-row"><input type="checkbox" name="injury_alerts_enabled" ${notificationPreferences?.injury_alerts_enabled ? "checked" : ""} /> Injury news</label>
                      <label class="checkbox-row"><input type="checkbox" name="team_news_alerts_enabled" ${notificationPreferences?.team_news_alerts_enabled ? "checked" : ""} /> Major team news</label>
                      <label class="checkbox-row"><input type="checkbox" name="weather_alerts_enabled" ${notificationPreferences?.weather_alerts_enabled ? "checked" : ""} /> Weather alerts</label>
                      <label class="checkbox-row"><input type="checkbox" name="market_movement_alerts_enabled" ${notificationPreferences?.market_movement_alerts_enabled ? "checked" : ""} /> Market movement</label>
                      <label class="checkbox-row"><input type="checkbox" name="volatility_alerts_enabled" ${notificationPreferences?.volatility_alerts_enabled ? "checked" : ""} /> Volatility warnings</label>
                      <label class="checkbox-row"><input type="checkbox" name="allow_non_signal_intelligence" ${notificationPreferences?.allow_non_signal_intelligence ? "checked" : ""} /> Non-signal intelligence updates</label>
                    </article>
                    <article class="panel">
                      <h4>Digests and timing</h4>
                      <label class="checkbox-row"><input type="checkbox" name="daily_digest_enabled" ${notificationPreferences?.daily_digest_enabled ? "checked" : ""} /> Daily digest</label>
                      <label class="checkbox-row"><input type="checkbox" name="results_digest_enabled" ${notificationPreferences?.results_digest_enabled ? "checked" : ""} /> Results digest</label>
                      <label class="checkbox-row"><input type="checkbox" name="weekend_slate_digest_enabled" ${notificationPreferences?.weekend_slate_digest_enabled ? "checked" : ""} /> Weekend slate digest</label>
                      <label class="field-label" for="alert-frequency-mode">Alert frequency</label>
                      <select id="alert-frequency-mode" name="alert_frequency_mode" class="text-input">
                        <option value="mixed" ${notificationPreferences?.alert_frequency_mode === "mixed" ? "selected" : ""}>Mixed</option>
                        <option value="immediate" ${notificationPreferences?.alert_frequency_mode === "immediate" ? "selected" : ""}>Immediate</option>
                        <option value="digest_only" ${notificationPreferences?.alert_frequency_mode === "digest_only" ? "selected" : ""}>Digest only</option>
                      </select>
                      <label class="field-label" for="pre-match-window-minutes">Pre-match window (minutes)</label>
                      <input id="pre-match-window-minutes" name="pre_match_window_minutes" class="text-input" type="number" min="0" max="1440" value="${escapeHtml(notificationPreferences?.pre_match_window_minutes ?? 90)}" />
                    </article>
                  </div>
                  <div class="card-grid">
                    <article class="panel">
                      <h4>Followed teams</h4>
                      <input name="favourite_teams" class="text-input" type="text" placeholder="Comma-separated teams you want to monitor" value="${escapeHtml(joinPreferenceList(notificationPreferences?.favourite_teams))}" />
                      <p class="muted">Example: Arsenal, Liverpool, Porto</p>
                    </article>
                    <article class="panel">
                      <h4>Followed leagues</h4>
                      <input name="favourite_leagues" class="text-input" type="text" placeholder="Comma-separated leagues you want to monitor" value="${escapeHtml(joinPreferenceList(notificationPreferences?.favourite_leagues))}" />
                      <p class="muted">Example: Premier League, Portugal Liga, MLS</p>
                    </article>
                    <article class="panel">
                      <h4>Followed markets</h4>
                      <input name="favourite_markets" class="text-input" type="text" placeholder="Comma-separated market families" value="${escapeHtml(joinPreferenceList(notificationPreferences?.favourite_markets))}" />
                      <p class="muted">Example: BTTS, OU25, FTR</p>
                    </article>
                    <article class="panel">
                      <h4>Followed fixtures</h4>
                      <input name="followed_fixtures" class="text-input" type="text" placeholder="Comma-separated fixture labels" value="${escapeHtml(joinPreferenceList(notificationPreferences?.followed_fixtures))}" />
                      <p class="muted">Example: Arsenal v Chelsea, Benfica v Braga</p>
                    </article>
                  </div>
                  <div class="cta-row">
                    <button class="button" type="submit">Save intelligence preferences</button>
                  </div>
                </form>
              </article>
            </section>
          `
      }
    `;
  };

  const sourceWindowLabel = () => {
    const source = state.summary?.selected_source_csv || "";
    const parts = source.split("/");
    return parts.length > 2 ? parts.slice(-3, -1).join(" / ") : source;
  };

  const dashboardView = () => {
    const signedIn = state.runtime.sessionAuthenticated;
    const entitled = state.runtime.sessionEntitled;
    const accountState = state.runtime.accountState;
    const notificationPreferences = accountState?.notification_preferences || null;
    const followedSignalsConfigured = Boolean(
      parsePreferenceList(notificationPreferences?.favourite_teams).length ||
        parsePreferenceList(notificationPreferences?.favourite_leagues).length ||
        parsePreferenceList(notificationPreferences?.favourite_markets).length ||
        parsePreferenceList(notificationPreferences?.followed_fixtures).length
    );
    const baseMatches = getFollowedIntelligenceMatches(accountState, state.fixtureIntelligence);
    const matches = filteredFollowedIntelligence(baseMatches);
    const classFilter = String(state.runtime.dashboardClassFilter || "ALL").toUpperCase();
    const reasonFilter = String(state.runtime.dashboardReasonFilter || "ALL").toUpperCase();
    const classOptions = ["ALL", "DEPLOY", "OBSERVE", "CONTEXT", "MONITOR"];
    const reasonOptions = [
      ["ALL", "All follows"],
      ["FOLLOWED TEAM", "Team"],
      ["FOLLOWED FIXTURE", "Fixture"],
      ["FOLLOWED LEAGUE", "League"],
      ["FOLLOWED MARKET", "Market"],
    ];

    if (!signedIn) {
      return `
        <section class="section split">
          <article class="hero-main">
            <p class="hero-kicker">Dashboard</p>
            <h1>Personal intelligence hub.</h1>
            <p>Sign in to turn followed teams, followed fixtures, and premium intelligence preferences into a real working board.</p>
            <div class="cta-row">
              <a class="button" href="./account.html">Verify email</a>
              <a class="ghost-button" href="./pricing.html">See founding plan</a>
            </div>
          </article>
          <aside class="hero-side">
            <div class="metric">
              <span class="metric-label">Access</span>
              <span class="metric-value">Locked</span>
            </div>
            <div class="metric">
              <span class="metric-label">Intelligence feed</span>
              <span class="metric-value">${state.fixtureIntelligence.length}</span>
            </div>
          </aside>
        </section>
      `;
    }

    return `
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Dashboard</p>
          <h1>Followed intelligence.</h1>
          <p>This is the first dedicated dashboard surface for the intelligence layer. It combines saved follows, published fixture intelligence, and Telegram delivery posture into one working board.</p>
          <div class="pill-row">
            <span class="stat-chip">${baseMatches.length} matched followed fixtures</span>
            <span class="stat-chip">${entitled ? "Premium active" : "Free / pending"}</span>
            <span class="stat-chip">${notificationPreferences?.telegram_enabled ? "Telegram enabled" : "Website-first mode"}</span>
          </div>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Deploy</span>
            <span class="metric-value">${baseMatches.filter((entry) => String(entry.row.publish_class).toUpperCase() === "DEPLOY").length}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Observe</span>
            <span class="metric-value">${baseMatches.filter((entry) => String(entry.row.publish_class).toUpperCase() === "OBSERVE").length}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Context</span>
            <span class="metric-value">${baseMatches.filter((entry) => String(entry.row.publish_class).toUpperCase() === "CONTEXT").length}</span>
          </div>
        </aside>
      </section>

      <section class="section">
        ${renderNotice(state.runtime.fixtureAlertMessage, state.runtime.fixtureAlertMessage ? "success" : "default")}
        ${renderNotice(state.runtime.alertsMessage, state.runtime.alertsMessage ? "success" : "default")}
        <article class="panel">
          <h3>Filters</h3>
          <p class="muted">Start simple: filter by publish class and why the fixture is relevant to you.</p>
          <div class="pill-row">
            ${classOptions
              .map(
                (option) =>
                  `<button class="${classFilter === option ? "button" : "ghost-button"}" type="button" data-action="dashboard-class-filter" data-value="${option}">${option}</button>`
              )
              .join("")}
          </div>
          <div class="pill-row">
            ${reasonOptions
              .map(
                ([value, label]) =>
                  `<button class="${reasonFilter === value ? "button" : "ghost-button"}" type="button" data-action="dashboard-reason-filter" data-value="${value}">${label}</button>`
              )
              .join("")}
          </div>
        </article>
      </section>

      <section class="section">
        <article class="panel">
          <h3>My alerts</h3>
          <p class="muted">Queue stronger followed alerts from the current intelligence window, then process anything already due for Telegram delivery.</p>
          <div class="cta-row">
            <button class="button" type="button" data-action="refresh-followed-alerts">Refresh followed alerts</button>
            <button class="ghost-button" type="button" data-action="dispatch-followed-alerts">Process due Telegram alerts</button>
          </div>
          ${
            state.runtime.accountAlerts.length
              ? `<div class="card-grid intelligence-grid">${state.runtime.accountAlerts
                  .slice(0, 4)
                  .map((alert) => notificationAlertCard(alert))
                  .join("")}</div>`
              : `<div class="notice">No queued or delivered alerts are stored for this account yet.</div>`
          }
        </article>
      </section>

      <section class="section">
        <article class="panel">
          <h3>Matched fixtures</h3>
          <p class="muted">Each card is generated from the same published intelligence layer that will later drive Telegram alerts, mobile delivery, and deeper followed-team tracking.</p>
          ${
            matches.length
              ? `<div class="card-grid intelligence-grid dashboard-grid">${matches
                  .map((entry) => dashboardFixtureCard(entry, Boolean(notificationPreferences?.telegram_enabled)))
                  .join("")}</div>`
              : !followedSignalsConfigured
                ? `<div class="notice">No followed teams, leagues, markets, or fixtures are saved yet. Add them on the account page, then save your intelligence preferences to start matching live cards.</div>`
                : `<div class="notice">No fixtures match the current filter combination. Change the filters or expand your followed teams, leagues, markets, or fixtures on the account page.</div>`
          }
        </article>
      </section>
    `;
  };

  const fixtureView = () => {
    const fixture = state.fixtureIntelligence.find((row) => String(row.fixture_key || "") === String(selectedFixtureKey || ""));
    if (!fixture) {
      return `
        <section class="section split">
          <article class="hero-main">
            <p class="hero-kicker">Fixture Intelligence</p>
            <h1>Fixture not found.</h1>
            <p>The published intelligence feed does not currently contain the fixture you requested. It may have rolled out of the active window.</p>
            <div class="cta-row">
              <a class="button" href="./dashboard.html">Back to dashboard</a>
              <a class="ghost-button" href="./account.html">Open account</a>
            </div>
          </article>
        </section>
      `;
    }

    const publishClass = String(fixture.publish_class || fixture.fixture_class || "MONITOR").toUpperCase();
    const headline = fixture.signal_summary?.headline || fixture.signal_summary?.summary_text || "Monitoring update published.";
    const notes = Array.isArray(fixture.context_summary?.notes) ? fixture.context_summary.notes : [];
    const odds = fixture.odds_summary || {};
    const accountState = state.runtime.accountState;
    const notificationPreferences = accountState?.notification_preferences || null;
    const matchedEntry = getFollowedIntelligenceMatches(accountState, [fixture])[0] || null;
    const relatedFixtures = state.fixtureIntelligence
      .filter((row) => row.fixture_key !== fixture.fixture_key && row.league === fixture.league)
      .slice(0, 4);

    return `
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Fixture Intelligence</p>
          <h1>${escapeHtml(fixture.home_team)} <span class="muted">vs</span> ${escapeHtml(fixture.away_team)}</h1>
          <p>${escapeHtml(headline)}</p>
          <div class="pill-row">
            <span class="pill">${escapeHtml(publishClass)}</span>
            <span class="chip">${escapeHtml(marketFamilyLabel(fixture.signal_summary?.market_family))}</span>
            <span class="chip">${escapeHtml(fixture.league)}</span>
            <span class="chip">${escapeHtml(formatKickoffLabel(fixture.kickoff_time))}</span>
          </div>
          <div class="cta-row">
            <a class="button" href="./dashboard.html">Back to dashboard</a>
            <a class="ghost-button" href="./premium.html">Open premium board</a>
            <button class="ghost-button" type="button" data-action="telegram-fixture-alert" data-fixture-key="${escapeHtml(String(fixture.fixture_key || ""))}">Send to Telegram</button>
          </div>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Coverage</span>
            <span class="metric-value">${escapeHtml(String(fixture.coverage_status || "covered"))}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Telegram</span>
            <span class="metric-value">${notificationPreferences?.telegram_enabled ? escapeHtml(fixture.follow_relevance?.notification_priority || "ready") : "Preview"}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Follow match</span>
            <span class="metric-value">${matchedEntry ? escapeHtml(matchedEntry.reasons.join(" / ")) : "Not followed"}</span>
          </div>
        </aside>
      </section>

      <section class="section split">
        ${renderNotice(state.runtime.fixtureAlertMessage, state.runtime.fixtureAlertMessage ? "success" : "default")}
        <article class="panel">
          <h3>Signal frame</h3>
          <div class="prediction-meta-grid dashboard-odds-grid">
            <div class="signal-cell">
              <span class="signal-label">1X2</span>
              <span class="signal-value">${escapeHtml(
                odds.home_win_odds && odds.draw_odds && odds.away_win_odds
                  ? `${odds.home_win_odds} / ${odds.draw_odds} / ${odds.away_win_odds}`
                  : "N/A"
              )}</span>
            </div>
            <div class="signal-cell">
              <span class="signal-label">OU25</span>
              <span class="signal-value">${escapeHtml(
                odds.over25_odds && odds.under25_odds ? `${odds.over25_odds} / ${odds.under25_odds}` : "N/A"
              )}</span>
            </div>
            <div class="signal-cell">
              <span class="signal-label">BTTS</span>
              <span class="signal-value">${escapeHtml(
                odds.btts_yes_odds && odds.btts_no_odds ? `${odds.btts_yes_odds} / ${odds.btts_no_odds}` : "N/A"
              )}</span>
            </div>
          </div>
          <div class="card-grid">
            <article class="panel">
              <h4>Published summary</h4>
              <p class="muted">${escapeHtml(fixture.signal_summary?.summary_text || headline)}</p>
            </article>
            <article class="panel">
              <h4>Context tags</h4>
              <div class="pill-row">
                ${(fixture.signal_summary?.context_tags || []).length
                  ? fixture.signal_summary.context_tags.map((tag) => `<span class="chip">${escapeHtml(String(tag).replace(/_/g, " "))}</span>`).join("")
                  : `<span class="muted">No published context tags</span>`}
              </div>
            </article>
          </div>
        </article>
        <article class="panel">
          <h3>Context notes</h3>
          ${
            notes.length
              ? `<ul class="feature-list">${notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>`
              : `<div class="notice">No extra context notes are currently published for this fixture.</div>`
          }
          <div class="dashboard-telegram-preview">
            <span class="metric-label">Telegram alert preview</span>
            <pre>${escapeHtml(
              telegramAlertPreview({
                row: fixture,
                reasons: matchedEntry?.reasons || ["fixture intelligence"],
              })
            )}</pre>
          </div>
        </article>
      </section>

      <section class="section">
        <article class="panel">
          <h3>Related league fixtures</h3>
          ${
            relatedFixtures.length
              ? `<div class="card-grid intelligence-grid">
                  ${relatedFixtures
                    .map((row) =>
                      intelligenceCard(
                        {
                          row,
                          reasons: [row.league === fixture.league ? "same league" : "related"],
                        },
                        Boolean(notificationPreferences?.telegram_enabled)
                      )
                    )
                    .join("")}
                </div>`
              : `<div class="notice">No related fixtures are available from the current published intelligence window.</div>`
          }
        </article>
      </section>
    `;
  };

  const render = () => {
    const views = {
      home: homeView,
      dashboard: dashboardView,
      fixture: fixtureView,
      predictions: predictionsView,
      premium: premiumView,
      results: resultsView,
      pricing: pricingView,
      methodology: methodologyView,
      account: accountView,
    };
    const view = views[page] || homeView;
    app.innerHTML = view();
  };

  const renderError = (error) => {
    app.innerHTML = `
      <section class="section">
        <div class="empty-state">
          <strong>Data bridge unavailable.</strong>
          <p>${escapeHtml(error.message || "Unable to load published JSON.")}</p>
          <p class="muted">
            If you opened these files directly from the filesystem, use a small static server instead so fetch
            requests can resolve <code>frontend/public/data/*.json</code>.
          </p>
        </div>
      </section>
    `;
  };

  const authNoticeFromQuery = () => {
    if (authState === "success") {
      return { message: "Premium access unlocked for this device.", tone: "success" };
    }
    if (authState === "expired") {
      return { message: "This sign-in link has expired. Request a fresh one to continue.", tone: "warning" };
    }
    if (authState === "inactive") {
      return { message: "Your membership is not active yet. If you just checked out, give the webhook a moment and try again.", tone: "warning" };
    }
    if (authState === "invalid") {
      return { message: "This sign-in link is invalid or has expired. Request a new one.", tone: "warning" };
    }
    if (authState === "not_wired") {
      return { message: "Email verification is not configured yet. This flow is scaffolded but not live.", tone: "warning" };
    }
    return { message: "", tone: "default" };
  };

  const loadAuthSession = async () => {
    state.runtime.sessionAuthenticated = false;
    state.runtime.sessionEntitled = false;
    state.runtime.sessionStatus = "";
    state.runtime.sessionAuthMode = "";
    state.runtime.sessionCustomerId = "";
    state.runtime.sessionSubscriptionId = "";
    state.runtime.accountState = null;
    state.runtime.accountAlerts = [];
    state.runtime.accountStateError = "";

    const notice = authNoticeFromQuery();
    state.runtime.authMessage = notice.message;

    if (!workerConfigured()) {
      return;
    }

    let response;
    let payload;
    try {
      ({ response, payload } = await fetchWorkerJson("/api/auth/session", {
        method: "GET",
        withToken: true,
      }));
    } catch (error) {
      state.runtime.sessionStatus = "session_unavailable";
      if (!state.runtime.authMessage) {
        state.runtime.authMessage = "We could not confirm your account session yet. You can request a new sign-in link below.";
      }
      return;
    }

    if (!response.ok || !payload?.ok) {
      state.runtime.sessionStatus = payload?.status || "session_unavailable";
      return;
    }

    state.runtime.sessionAuthenticated = Boolean(payload.authenticated);
    state.runtime.sessionEntitled = Boolean(payload.entitled);
    state.runtime.sessionStatus = String(payload.status || "");
    state.runtime.sessionAuthMode = String(payload.auth_mode || "");
    state.runtime.sessionCustomerId = String(payload.customer_id || "");
    state.runtime.sessionSubscriptionId = String(payload.subscription_id || "");
  };

  const loadAccountState = async () => {
    state.runtime.accountState = null;
    state.runtime.accountStateError = "";

    if (!workerConfigured() || !state.runtime.sessionAuthenticated) {
      return;
    }

    try {
      const { response, payload } = await fetchWorkerJson("/api/account/state", {
        method: "GET",
        withToken: true,
      });

      if (!response.ok || !payload?.ok) {
        state.runtime.accountStateError = payload?.message || "Unable to load account state.";
        return;
      }

      state.runtime.accountState = payload.account || null;
    } catch (error) {
      state.runtime.accountStateError = error.message || "Unable to load account state.";
    }
  };

  const loadAccountAlerts = async () => {
    state.runtime.accountAlerts = [];
    if (!workerConfigured() || !state.runtime.sessionAuthenticated) {
      return;
    }

    try {
      const { response, payload } = await fetchWorkerJson("/api/account/alerts", {
        method: "GET",
        withToken: true,
      });
      if (!response.ok || !payload?.ok) {
        return;
      }
      state.runtime.accountAlerts = Array.isArray(payload.alerts) ? payload.alerts : [];
    } catch {
      return;
    }
  };

  const loadProtectedPremiumPredictions = async () => {
    state.runtime.premiumFetchError = "";
    state.runtime.premiumGeneratedAt = "";
    state.runtime.premiumSubscriberCustomerId = "";
    state.runtime.premiumSourceLabel = "";
    state.securePremiumPredictions = [];

    if (premiumDemoMode || page !== "premium") {
      return;
    }
    if (!workerConfigured() || (!premiumTokenPresent() && !state.runtime.sessionAuthenticated)) {
      return;
    }

    const { response, payload } = await fetchWorkerJson("/api/premium/predictions", {
      method: "GET",
      withToken: true,
    });

    if (!response.ok || !payload?.ok) {
      state.runtime.premiumFetchError = payload?.message || "Protected premium route did not return access.";
      return;
    }

    state.securePremiumPredictions = Array.isArray(payload.predictions) ? payload.predictions : [];
    state.runtime.premiumGeneratedAt = typeof payload.generated_at === "string" ? payload.generated_at : "";
    state.runtime.premiumSubscriberCustomerId =
      typeof payload.subscriber_customer_id === "string" ? payload.subscriber_customer_id : "";
    state.runtime.premiumSourceLabel = workerApiUrl("/api/premium/predictions");
  };

  const startWorkerCheckout = async () => {
    if (!workerConfigured()) {
      window.location.href = checkoutPlaceholderHref;
      return;
    }

    state.runtime.checkoutMessage = "Requesting Stripe Checkout from Worker...";
    render();

    try {
      const { response, payload } = await fetchWorkerJson("/api/stripe/checkout", {
        method: "POST",
        body: {
          reference: "frontend_pricing",
        },
      });

      if (!response.ok || !payload?.url) {
        throw new Error(payload?.message || "Worker checkout route did not return a Stripe URL.");
      }

      window.location.href = payload.url;
    } catch (error) {
      state.runtime.checkoutMessage = error.message || "Unable to start Worker checkout.";
      render();
    }
  };

  const requestMagicLink = async (event) => {
    event.preventDefault();
    if (!workerConfigured()) {
      state.runtime.authMessage = "Email verification is unavailable until the Worker auth routes are configured.";
      render();
      return;
    }

    const formData = new FormData(event.target);
    const email = String(formData.get("email") || "").trim();
    state.runtime.authMessage = "Requesting sign-in link...";
    render();

    try {
      const { response, payload } = await fetchWorkerJson("/api/auth/magic-link/request", {
        method: "POST",
        body: { email },
      });

      if (!response.ok || !payload?.ok) {
        throw new Error(payload?.message || "Unable to request sign-in link.");
      }

      state.runtime.authMessage = payload.message || "If the address is eligible, a sign-in link has been sent.";
    } catch (error) {
      state.runtime.authMessage = error.message || "Unable to request sign-in link.";
    }

    render();
  };

  const startTelegramLink = async (event) => {
    event.preventDefault();
    if (!workerConfigured() || !state.runtime.sessionAuthenticated) {
      state.runtime.telegramMessage = "Verify your email before linking Telegram.";
      render();
      return;
    }

    state.runtime.telegramMessage = "Preparing Telegram link…";
    render();

    try {
      const { response, payload } = await fetchWorkerJson("/api/account/telegram/link/start", {
        method: "POST",
        withToken: true,
      });

      if (!response.ok || !payload?.ok) {
        throw new Error(payload?.message || "Unable to prepare Telegram link.");
      }

      state.runtime.telegramLinkCode = String(payload.code || "");
      state.runtime.telegramLinkExpiresAt = String(payload.expires_at || "");
      state.runtime.telegramBotUsername = String(payload.bot_username || "");
      state.runtime.telegramDeepLinkUrl = String(payload.deep_link_url || "");
      state.runtime.telegramMessage =
        payload.message || "Telegram link code generated. Open the bot or use the code to complete linking.";
    } catch (error) {
      state.runtime.telegramMessage = error.message || "Unable to prepare Telegram link.";
    }

    render();
  };

  const savePreferences = async (event) => {
    event.preventDefault();
    if (!workerConfigured() || !state.runtime.sessionAuthenticated) {
      state.runtime.preferencesMessage = "Verify your email before saving intelligence preferences.";
      render();
      return;
    }

    const formData = new FormData(event.target);
    const payload = {
      telegram_enabled: formData.get("telegram_enabled") === "on",
      email_enabled: formData.get("email_enabled") === "on",
      website_only_mode: formData.get("website_only_mode") === "on",
      elite_alerts_enabled: formData.get("elite_alerts_enabled") === "on",
      standard_alerts_enabled: formData.get("standard_alerts_enabled") === "on",
      acca_alerts_enabled: formData.get("acca_alerts_enabled") === "on",
      correct_score_alerts_enabled: formData.get("correct_score_alerts_enabled") === "on",
      injury_alerts_enabled: formData.get("injury_alerts_enabled") === "on",
      team_news_alerts_enabled: formData.get("team_news_alerts_enabled") === "on",
      weather_alerts_enabled: formData.get("weather_alerts_enabled") === "on",
      market_movement_alerts_enabled: formData.get("market_movement_alerts_enabled") === "on",
      volatility_alerts_enabled: formData.get("volatility_alerts_enabled") === "on",
      allow_non_signal_intelligence: formData.get("allow_non_signal_intelligence") === "on",
      daily_digest_enabled: formData.get("daily_digest_enabled") === "on",
      results_digest_enabled: formData.get("results_digest_enabled") === "on",
      weekend_slate_digest_enabled: formData.get("weekend_slate_digest_enabled") === "on",
      alert_frequency_mode: String(formData.get("alert_frequency_mode") || "mixed"),
      pre_match_window_minutes: Number(formData.get("pre_match_window_minutes") || 90),
      favourite_teams: String(formData.get("favourite_teams") || ""),
      favourite_leagues: String(formData.get("favourite_leagues") || ""),
      favourite_markets: String(formData.get("favourite_markets") || ""),
      followed_fixtures: String(formData.get("followed_fixtures") || ""),
    };

    state.runtime.preferencesMessage = "Saving intelligence preferences…";
    render();

    try {
      const { response, payload: responsePayload } = await fetchWorkerJson("/api/account/preferences", {
        method: "POST",
        withToken: true,
        body: payload,
      });

      if (!response.ok || !responsePayload?.ok) {
        throw new Error(responsePayload?.message || "Unable to save intelligence preferences.");
      }

      state.runtime.accountState = responsePayload.account || state.runtime.accountState;
      state.runtime.preferencesMessage = responsePayload.message || "Intelligence preferences saved.";
      await loadAccountAlerts();
    } catch (error) {
      state.runtime.preferencesMessage = error.message || "Unable to save intelligence preferences.";
    }

    render();
  };

  const sendTelegramTestAlert = async (event) => {
    event.preventDefault();
    if (!workerConfigured() || !state.runtime.sessionAuthenticated) {
      state.runtime.telegramMessage = "Verify your email before sending a Telegram test alert.";
      render();
      return;
    }

    state.runtime.telegramMessage = "Sending Telegram test alert…";
    render();

    try {
      const { response, payload } = await fetchWorkerJson("/api/account/telegram/test-alert", {
        method: "POST",
        withToken: true,
      });

      if (!response.ok || !payload?.ok) {
        throw new Error(payload?.message || "Unable to send Telegram test alert.");
      }

      state.runtime.telegramMessage = payload.message || "Telegram test alert sent.";
    } catch (error) {
      state.runtime.telegramMessage = error.message || "Unable to send Telegram test alert.";
    }

    render();
  };

  const sendTelegramFixtureAlert = async (event, fixtureKey) => {
    event.preventDefault();
    if (!workerConfigured() || !state.runtime.sessionAuthenticated) {
      state.runtime.fixtureAlertMessage = "Verify your email before sending a fixture intelligence alert.";
      render();
      return;
    }

    const key = String(fixtureKey || "").trim();
    if (!key) {
      state.runtime.fixtureAlertMessage = "No fixture key was supplied for this alert.";
      render();
      return;
    }

    state.runtime.fixtureAlertMessage = "Sending fixture intelligence to Telegram…";
    render();

    try {
      const { response, payload } = await fetchWorkerJson("/api/account/telegram/fixture-alert", {
        method: "POST",
        withToken: true,
        body: {
          fixture_key: key,
        },
      });

      if (!response.ok || !payload?.ok) {
        throw new Error(payload?.message || "Unable to send fixture intelligence alert.");
      }

      state.runtime.fixtureAlertMessage = payload.message || "Fixture intelligence alert sent.";
    } catch (error) {
      state.runtime.fixtureAlertMessage = error.message || "Unable to send fixture intelligence alert.";
    }

    render();
  };

  const refreshFollowedAlerts = async (event) => {
    event.preventDefault();
    if (!workerConfigured() || !state.runtime.sessionAuthenticated) {
      state.runtime.alertsMessage = "Verify your email before refreshing followed alerts.";
      render();
      return;
    }

    state.runtime.alertsMessage = "Refreshing followed alerts from the current intelligence window…";
    render();

    try {
      const { response, payload } = await fetchWorkerJson("/api/account/alerts/refresh", {
        method: "POST",
        withToken: true,
      });

      if (!response.ok || !payload?.ok) {
        throw new Error(payload?.message || "Unable to refresh followed alerts.");
      }

      state.runtime.accountAlerts = Array.isArray(payload.alerts) ? payload.alerts : [];
      state.runtime.alertsMessage = payload.message || "Followed alerts refreshed.";
    } catch (error) {
      state.runtime.alertsMessage = error.message || "Unable to refresh followed alerts.";
    }

    render();
  };

  const dispatchFollowedAlerts = async (event) => {
    event.preventDefault();
    if (!workerConfigured() || !state.runtime.sessionAuthenticated) {
      state.runtime.alertsMessage = "Verify your email before dispatching alerts.";
      render();
      return;
    }

    state.runtime.alertsMessage = "Processing due Telegram alerts…";
    render();

    try {
      const { response, payload } = await fetchWorkerJson("/api/account/alerts/dispatch", {
        method: "POST",
        withToken: true,
      });

      if (!response.ok || !payload?.ok) {
        throw new Error(payload?.message || "Unable to dispatch followed alerts.");
      }

      state.runtime.accountAlerts = Array.isArray(payload.alerts) ? payload.alerts : [];
      state.runtime.alertsMessage = payload.message || "Due Telegram alerts processed.";
    } catch (error) {
      state.runtime.alertsMessage = error.message || "Unable to dispatch followed alerts.";
    }

    render();
  };

  const handleTokenSave = async (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    const token = String(formData.get("premium_token") || "").trim();
    writeStoredPremiumToken(token);
    state.runtime.premiumToken = token;
    state.runtime.accountMessage = token
      ? "Developer/test token saved locally. Open the premium page to test Worker-backed access."
      : "Premium token cleared.";
    if (page === "premium") {
      await loadProtectedPremiumPredictions();
    }
    render();
  };

  const handleLogout = async (event) => {
    event.preventDefault();
    if (!workerConfigured()) {
      return;
    }

    try {
      await fetchWorkerJson("/api/auth/logout", {
        method: "POST",
      });
    } catch {
      // Best-effort logout; continue local cleanup.
    }

    writeStoredPremiumToken("");
    state.runtime.premiumToken = "";
    state.runtime.sessionAuthenticated = false;
    state.runtime.sessionEntitled = false;
    state.runtime.sessionStatus = "";
    state.runtime.sessionAuthMode = "";
    state.runtime.sessionCustomerId = "";
    state.runtime.sessionSubscriptionId = "";
    state.runtime.accountState = null;
    state.runtime.accountStateError = "";
    state.runtime.telegramLinkCode = "";
    state.runtime.telegramLinkExpiresAt = "";
    state.runtime.telegramBotUsername = "";
    state.runtime.telegramDeepLinkUrl = "";
    state.runtime.telegramMessage = "";
    state.runtime.fixtureAlertMessage = "";
    state.runtime.alertsMessage = "";
    state.runtime.preferencesMessage = "";
    state.securePremiumPredictions = [];
    state.runtime.premiumFetchError = "";
    state.runtime.authMessage = "You have been signed out from this device.";
    render();
  };

  app.addEventListener("click", async (event) => {
    const checkoutTarget = event.target.closest("[data-action='worker-checkout']");
    if (checkoutTarget) {
      event.preventDefault();
      await startWorkerCheckout();
      return;
    }

    const clearTokenTarget = event.target.closest("[data-action='clear-premium-token']");
    if (clearTokenTarget) {
      event.preventDefault();
      writeStoredPremiumToken("");
      state.runtime.premiumToken = "";
      state.securePremiumPredictions = [];
      state.runtime.accountMessage = "Stored premium token cleared.";
      state.runtime.premiumFetchError = "";
      render();
      return;
    }

    const logoutTarget = event.target.closest("[data-action='auth-logout']");
    if (logoutTarget) {
      await handleLogout(event);
      return;
    }

    const dashboardClassTarget = event.target.closest("[data-action='dashboard-class-filter']");
    if (dashboardClassTarget) {
      event.preventDefault();
      state.runtime.dashboardClassFilter = String(dashboardClassTarget.dataset.value || "ALL").toUpperCase();
      render();
      return;
    }

    const dashboardReasonTarget = event.target.closest("[data-action='dashboard-reason-filter']");
    if (dashboardReasonTarget) {
      event.preventDefault();
      state.runtime.dashboardReasonFilter = String(dashboardReasonTarget.dataset.value || "ALL").toUpperCase();
      render();
      return;
    }

    const telegramTarget = event.target.closest("[data-action='telegram-link-start']");
    if (telegramTarget) {
      await startTelegramLink(event);
      return;
    }

    const telegramTestTarget = event.target.closest("[data-action='telegram-test-alert']");
    if (telegramTestTarget) {
      await sendTelegramTestAlert(event);
      return;
    }

    const telegramFixtureTarget = event.target.closest("[data-action='telegram-fixture-alert']");
    if (telegramFixtureTarget) {
      await sendTelegramFixtureAlert(event, telegramFixtureTarget.dataset.fixtureKey);
      return;
    }

    const refreshAlertsTarget = event.target.closest("[data-action='refresh-followed-alerts']");
    if (refreshAlertsTarget) {
      await refreshFollowedAlerts(event);
      return;
    }

    const dispatchAlertsTarget = event.target.closest("[data-action='dispatch-followed-alerts']");
    if (dispatchAlertsTarget) {
      await dispatchFollowedAlerts(event);
    }
  });

  app.addEventListener("submit", async (event) => {
    if (event.target.id === "premium-token-form") {
      await handleTokenSave(event);
      return;
    }

    if (event.target.id === "magic-link-form") {
      await requestMagicLink(event);
      return;
    }

    if (event.target.id === "preferences-form") {
      await savePreferences(event);
    }
  });

  const boot = async () => {
    let loadingMessage = "Loading published board…";
    if (page === "account" || page === "dashboard") {
      loadingMessage =
        checkoutState === "success"
          ? "Membership confirmed. Please verify your email to continue…"
          : page === "dashboard"
            ? "Loading your intelligence dashboard…"
            : "Loading your account access…";
    } else if (page === "premium") {
      loadingMessage = "Checking premium access…";
    }
    app.innerHTML = `<div class="loading">${escapeHtml(loadingMessage)}</div>`;
    syncActiveNav();
    state.runtime.premiumToken = readStoredPremiumToken();
    await loadAuthSession();
    await loadAccountState();
    await loadAccountAlerts();

    try {
      const [summary, publicPredictions, premiumPredictions] = await Promise.all([
        fetchJson(`${DATA_ROOT}/publish_summary.json`),
        fetchJson(`${DATA_ROOT}/public_predictions.json`),
        premiumDemoMode ? fetchOptionalJson(`${DATA_ROOT}/premium_predictions.json`) : Promise.resolve([]),
      ]);
      const weeklyResults = await fetchOptionalJson(`${DATA_ROOT}/weekly_results.json`);
      const fixtureIntelligence = await fetchOptionalJson(`${DATA_ROOT}/fixture_intelligence_public.json`);
      state.summary = summary;
      state.publicPredictions = publicPredictions;
      state.premiumPredictions = Array.isArray(premiumPredictions) ? premiumPredictions : [];
      state.weeklyResults = weeklyResults;
      state.fixtureIntelligence = Array.isArray(fixtureIntelligence?.fixtures) ? fixtureIntelligence.fixtures : [];
      await loadProtectedPremiumPredictions();
      render();
    } catch (error) {
      renderError(error);
    }
  };

  boot();
})();
