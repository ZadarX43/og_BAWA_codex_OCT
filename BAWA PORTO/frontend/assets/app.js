(function () {
  const DATA_ROOT = "./public/data";
  const PREMIUM_TOKEN_STORAGE_KEY = "og_premium_token";
  const INTERNAL_ADMIN_KEY_STORAGE_KEY = "og_internal_admin_key";
  const INTERNAL_OPERATOR_ID_STORAGE_KEY = "og_internal_operator_id";
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
      accountSessions: [],
      accountAlerts: [],
      accountStateError: "",
      accountSessionsError: "",
      accountSessionsMessage: "",
      dashboardClassFilter: "ALL",
      dashboardReasonFilter: "ALL",
      internalFlagSeverityFilter: "ALL",
      internalFlagStatusFilter: "ALL",
      internalTimelineSourceFilter: "ALL",
      internalReviewPreset: "CUSTOM",
      internalReviewOutcome: "AUTO",
      internalReviewOutcomeNote: "",
      telegramLinkCode: "",
      telegramLinkExpiresAt: "",
      telegramBotUsername: "",
      telegramDeepLinkUrl: "",
      internalAdminKey: "",
      internalOperatorId: "",
      internalLookupMessage: "",
      internalReviewMessage: "",
      internalSelectedUserId: "",
      internalLookupEmail: "",
      internalAccountSummary: null,
      internalFlags: [],
      internalNotes: [],
      internalTimeline: [],
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
  const readStoredInternalAdminKey = () => {
    try {
      return window.localStorage.getItem(INTERNAL_ADMIN_KEY_STORAGE_KEY) || "";
    } catch {
      return "";
    }
  };
  const writeStoredInternalAdminKey = (value) => {
    try {
      if (value) {
        window.localStorage.setItem(INTERNAL_ADMIN_KEY_STORAGE_KEY, value);
      } else {
        window.localStorage.removeItem(INTERNAL_ADMIN_KEY_STORAGE_KEY);
      }
    } catch {
      return;
    }
  };
  const readStoredInternalOperatorId = () => {
    try {
      return window.localStorage.getItem(INTERNAL_OPERATOR_ID_STORAGE_KEY) || "";
    } catch {
      return "";
    }
  };
  const writeStoredInternalOperatorId = (value) => {
    try {
      if (value) {
        window.localStorage.setItem(INTERNAL_OPERATOR_ID_STORAGE_KEY, value);
      } else {
        window.localStorage.removeItem(INTERNAL_OPERATOR_ID_STORAGE_KEY);
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
  const fetchInternalWorkerJson = async (path, options = {}) => {
    const headers = new Headers(options.headers || {});
    headers.set("accept", "application/json");
    if (options.body && !headers.has("content-type")) {
      headers.set("content-type", "application/json");
    }
    if (state.runtime.internalAdminKey) {
      headers.set("x-og-internal-admin", state.runtime.internalAdminKey);
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
      onboarding: "./account.html",
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
  const sessionKindLabel = (value) => {
    const kind = String(value || "browser").trim().toLowerCase();
    if (kind === "browser") return "Browser";
    return titleCase(kind);
  };
  const sessionStatusLabel = (session) => {
    if (!session) return "Recent device";
    if (session.is_current) return "Current device";
    if (session.is_revoked) return "Signed out";
    const expiresAt = Date.parse(String(session.expires_at || ""));
    if (Number.isFinite(expiresAt) && expiresAt <= Date.now()) return "Expired";
    return "Recent device";
  };
  const sessionActivityNote = (session) => {
    if (!session) return "";
    if (session.is_current) {
      return "This is the session you are using right now.";
    }
    if (session.is_revoked) {
      return `Revoked ${formatDateTime(session.revoked_at) || "recently"}.`;
    }
    const lastSeen = formatDateTime(session.last_seen_at);
    return lastSeen ? `Last active ${lastSeen}.` : "Recent verified sign-in.";
  };
  const accountSessionCard = (session) => `
    <article class="panel">
      <div class="pill-row">
        <span class="stat-chip">${escapeHtml(sessionStatusLabel(session))}</span>
        <span class="stat-chip">${escapeHtml(sessionKindLabel(session.session_kind))}</span>
        ${session.is_primary ? `<span class="stat-chip">Primary</span>` : ""}
      </div>
      <h4>${escapeHtml(session.device_label || "Browser session")}</h4>
      <ul class="feature-list">
        <li>${escapeHtml(sessionActivityNote(session))}</li>
        <li>Issued: ${escapeHtml(formatDateTime(session.issued_at) || "Unknown")}</li>
        <li>Expires: ${escapeHtml(formatDateTime(session.expires_at) || "Unknown")}</li>
      </ul>
      <div class="cta-row">
        ${
          session.is_current
            ? `<button class="ghost-button" type="button" data-action="revoke-account-session" data-session-id="${escapeHtml(
                session.id
              )}" data-session-label="${escapeHtml(session.device_label || "Current device")}">Sign out this device</button>`
            : `<button class="ghost-button" type="button" data-action="revoke-account-session" data-session-id="${escapeHtml(
                session.id
              )}" data-session-label="${escapeHtml(session.device_label || "Device")}">Revoke device</button>`
        }
        ${
          session.is_primary || session.is_revoked
            ? ""
            : `<button class="ghost-button" type="button" data-action="make-primary-session" data-session-id="${escapeHtml(
                session.id
              )}" data-session-label="${escapeHtml(session.device_label || "Device")}">Make primary</button>`
        }
      </div>
    </article>
  `;

  const joinPreferenceList = (value) => (Array.isArray(value) ? value.join(", ") : "");
  const normalizeStylePreset = (value) => {
    const preset = String(value || "").trim().toLowerCase();
    if (["analyst", "disciplined_bettor", "tactical_reader", "researcher"].includes(preset)) {
      return preset;
    }
    return "disciplined_bettor";
  };
  const stylePresetLabel = (value) => {
    const preset = normalizeStylePreset(value);
    if (preset === "analyst") return "Analyst";
    if (preset === "disciplined_bettor") return "Disciplined bettor";
    if (preset === "tactical_reader") return "Tactical reader";
    if (preset === "researcher") return "Researcher";
    return "Disciplined bettor";
  };
  const languageLabel = (value) => {
    const key = String(value || "en-GB");
    if (key === "en-US") return "English (US)";
    if (key === "pt-PT") return "Portuguese";
    if (key === "es-ES") return "Spanish";
    return "English (UK)";
  };
  const titleCase = (value) =>
    String(value || "")
      .split(/\s+/)
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
      .join(" ");
  const deriveGreetingName = (accountState) => {
    const email = String(accountState?.user?.email || "").trim().toLowerCase();
    const localPart = email.split("@")[0] || "";
    const direct = localPart.split(/[._-]+/).filter(Boolean)[0] || "";
    if (direct && /[._-]/.test(localPart)) {
      return titleCase(direct);
    }
    for (let pivot = 4; pivot <= 6; pivot += 1) {
      const first = localPart.slice(0, pivot);
      const middle = localPart.slice(pivot, pivot + 1);
      const tail = localPart.slice(pivot + 1);
      if (
        first.length >= 3 &&
        tail.length >= 3 &&
        /[bcdfghjklmnpqrstvwxyz]/.test(middle) &&
        /^[a-z]+$/.test(first) &&
        /^[a-z]+$/.test(tail)
      ) {
        return titleCase(first);
      }
    }
    return "";
  };
  const timeGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return "Good Morning";
    if (hour < 18) return "Good Afternoon";
    return "Good Evening";
  };
  const accountGreeting = (accountState) => {
    const name = deriveGreetingName(accountState);
    return name ? `${timeGreeting()} ${name}` : timeGreeting();
  };
  const stylePresetSummary = (preset) => {
    const key = normalizeStylePreset(preset);
    if (key === "analyst") return "Broad website context, fewer Telegram interruptions, stronger reading posture.";
    if (key === "tactical_reader") return "Team and fixture intelligence first, with calmer follow-led delivery.";
    if (key === "researcher") return "Wide website visibility, richer non-deploy coverage, minimal interruption.";
    return "Selective deploy-led delivery with a higher bar for interruption and action.";
  };
  const feedRoutingExplanation = (preset) => {
    const key = normalizeStylePreset(preset);
    if (key === "analyst") return "Telegram is kept tight. More useful depth stays on the website by design.";
    if (key === "tactical_reader") return "Followed teams and fixtures can interrupt sooner than broad market signals.";
    if (key === "researcher") return "Most intelligence stays on-site unless a direct team or fixture match deserves interruption.";
    return "Telegram is reserved for stronger direct relevance. Broader market context stays website-first.";
  };
  const onboardingStepSummary = (notificationPreferences) => {
    const prefs = notificationPreferences || {};
    const teams = parsePreferenceList(prefs.favourite_teams);
    const leagues = parsePreferenceList(prefs.favourite_leagues);
    const markets = parsePreferenceList(prefs.favourite_markets);
    const fixtures = parsePreferenceList(prefs.followed_fixtures);
    const preset = normalizeStylePreset(prefs.user_style_preset);
    const steps = [
      {
        key: "preset",
        label: "Choose your style preset",
        complete: Boolean(preset),
        detail: stylePresetLabel(preset),
      },
      {
        key: "scope",
        label: "Choose what you care about",
        complete: Boolean(teams.length || leagues.length || markets.length || fixtures.length),
        detail: teams.length || leagues.length || markets.length || fixtures.length ? "Follow scope saved" : "No follows saved yet",
      },
      {
        key: "delivery",
        label: "Set your interruption posture",
        complete: prefs.telegram_enabled || prefs.website_only_mode || prefs.email_enabled,
        detail: prefs.website_only_mode ? "Website-first mode" : prefs.telegram_enabled ? "Selective Telegram enabled" : "Delivery still needs a choice",
      },
      {
        key: "companion",
        label: "Keep decision support active",
        complete: Boolean(prefs.decision_companion_enabled),
        detail: prefs.decision_companion_enabled ? "Decision companion enabled" : "Decision companion not enabled yet",
      },
      {
        key: "reset",
        label: "Turn on reset / clarity mode",
        complete: Boolean(prefs.reset_mode_enabled),
        detail: prefs.reset_mode_enabled ? "Loss-state support enabled" : "Reset mode not enabled yet",
      },
    ];
    return {
      steps,
      completed: steps.filter((step) => step.complete).length,
      total: steps.length,
    };
  };
  const routeExplanation = (entry) => {
    const preset = normalizeStylePreset(entry?.accountState?.notification_preferences?.user_style_preset);
    const priority = dashboardPriorityProfile(entry);
    const reasons = Array.isArray(entry?.reasons) ? entry.reasons : [];
    const publishClass = String(entry?.row?.publish_class || entry?.row?.fixture_class || "MONITOR").toUpperCase();
    if (priority.bucket === "send_now") {
      return "Why this interrupts: direct team or fixture relevance makes this one worth active attention.";
    }
    if (priority.bucket === "watch_closely") {
      return "Why this stays just below interruption: the signal is useful, but it is better read than rushed.";
    }
    if (priority.bucket === "website_only") {
      if (preset === "analyst" || preset === "researcher") {
        return "Why this stays website-only: your preset prefers depth on-site unless the relevance becomes more direct.";
      }
      if (reasons.includes("followed market") && !reasons.includes("followed team") && !reasons.includes("followed fixture")) {
        return "Why this stays website-only: market-only relevance is useful, but not strong enough to break attention.";
      }
      return "Why this stays website-only: it matters, but not enough to justify interruption.";
    }
    if (publishClass === "OBSERVE" || publishClass === "CONTEXT" || publishClass === "MONITOR") {
      return "Why this sits in no edge / monitor: the fixture is informative, but disciplined action is not warranted yet.";
    }
    return "Why this is here: it fits your saved follows, but the website remains the calm first surface.";
  };
  const alertAutoGateLabel = (value) => {
    const key = String(value || "").trim();
    if (key === "direct_follow") return "Direct team or fixture follow";
    if (key === "direct_fixture_follow") return "Direct fixture follow";
    if (key === "direct_team_deploy") return "Direct team deploy match";
    if (key === "direct_team_observe") return "Direct team observe match";
    if (key === "league_market_deploy") return "League and market deploy match";
    if (key === "team_context_follow") return "Team-led context follow";
    if (key === "website_depth_preferred") return "Website depth preferred by preset";
    if (key === "website_only") return "Website-first route";
    return key ? titleCase(key.replace(/_/g, " ")) : "";
  };

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
          accountState,
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
      });
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
        ${autoGate ? `<p class="muted">Why this interrupted: ${escapeHtml(alertAutoGateLabel(autoGate))}</p>` : ""}
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

  const dashboardPriorityProfile = (entry) => {
    const reasons = Array.isArray(entry?.reasons) ? entry.reasons : [];
    const publishClass = String(entry?.row?.publish_class || entry?.row?.fixture_class || "MONITOR").toUpperCase();
    const stylePreset = normalizeStylePreset(entry?.accountState?.notification_preferences?.user_style_preset);
    const reasonSet = new Set(reasons);
    const hasFixture = reasonSet.has("followed fixture");
    const hasTeam = reasonSet.has("followed team");
    const hasLeague = reasonSet.has("followed league");
    const hasMarket = reasonSet.has("followed market");
    const kickoffNear = isNearKickoffWindow(entry?.row?.kickoff_time, 240);
    const eliteFixture = isEliteFixture(entry?.row);

    let score = 0;
    if (hasFixture) score += 100;
    if (hasTeam) score += 80;
    if (hasLeague) score += 35;
    if (hasMarket) score += 10;
    if (publishClass === "DEPLOY") score += 30;
    else if (publishClass === "OBSERVE") score += 15;
    else if (publishClass === "CONTEXT") score += 8;
    else if (publishClass === "MONITOR") score += 4;

    if (stylePreset === "analyst") {
      if (publishClass === "CONTEXT" || publishClass === "MONITOR") score += 8;
      if (hasMarket && !hasTeam && !hasFixture) score -= 10;
    } else if (stylePreset === "disciplined_bettor") {
      if (publishClass === "DEPLOY") score += 18;
      if (publishClass === "OBSERVE") score -= 12;
      if (publishClass === "CONTEXT" || publishClass === "MONITOR") score -= 22;
    } else if (stylePreset === "tactical_reader") {
      if (hasTeam || hasFixture) score += 12;
      if (publishClass === "CONTEXT") score += 10;
      if (hasMarket && !hasLeague && !hasTeam && !hasFixture) score -= 8;
    } else if (stylePreset === "researcher") {
      if (publishClass === "OBSERVE" || publishClass === "CONTEXT" || publishClass === "MONITOR") score += 10;
      if (publishClass === "DEPLOY") score -= 4;
      if (hasMarket && !hasTeam && !hasFixture) score -= 6;
    }

    if (hasFixture) {
      return { label: "Send now", score, bucket: "send_now" };
    }
    if (hasTeam && publishClass === "DEPLOY") {
      return { label: "Send now", score, bucket: "send_now" };
    }
    if (hasTeam && publishClass === "OBSERVE") {
      if (stylePreset === "tactical_reader") {
        return { label: "Watch closely", score, bucket: "watch_closely" };
      }
      if (stylePreset === "disciplined_bettor") {
        return {
          label: kickoffNear ? "Watch closely" : "Website only",
          score,
          bucket: kickoffNear ? "watch_closely" : "website_only",
        };
      }
      return { label: "Website only", score, bucket: "website_only" };
    }
    if (hasTeam && publishClass === "CONTEXT") {
      if (stylePreset === "tactical_reader") {
        return { label: "Watch closely", score, bucket: "watch_closely" };
      }
      if (stylePreset === "analyst") {
        return {
          label: kickoffNear ? "Watch closely" : "Website only",
          score,
          bucket: kickoffNear ? "watch_closely" : "website_only",
        };
      }
      if (stylePreset === "disciplined_bettor") {
        return { label: "No edge", score, bucket: "no_edge" };
      }
      return { label: "Website only", score, bucket: "website_only" };
    }
    if (hasTeam && publishClass === "MONITOR") {
      if (stylePreset === "analyst") {
        return { label: "Website only", score, bucket: "website_only" };
      }
      return { label: "No edge", score, bucket: "no_edge" };
    }
    if (publishClass === "DEPLOY" && hasLeague && hasMarket) {
      if (stylePreset === "tactical_reader") {
        return { label: "Watch closely", score, bucket: "watch_closely" };
      }
      if (stylePreset === "disciplined_bettor") {
        return {
          label: eliteFixture || kickoffNear ? "Watch closely" : "Website only",
          score,
          bucket: eliteFixture || kickoffNear ? "watch_closely" : "website_only",
        };
      }
      return { label: "Website only", score, bucket: "website_only" };
    }
    if (publishClass === "OBSERVE" && (hasLeague || hasMarket)) {
      if (stylePreset === "researcher" || stylePreset === "analyst") {
        return { label: "Website only", score, bucket: "website_only" };
      }
      if (stylePreset === "tactical_reader" && hasLeague && kickoffNear) {
        return { label: "Website only", score, bucket: "website_only" };
      }
      return { label: "No edge", score, bucket: "no_edge" };
    }
    if ((stylePreset === "analyst" || stylePreset === "researcher") && (publishClass === "CONTEXT" || publishClass === "MONITOR")) {
      return { label: "Website only", score, bucket: "website_only" };
    }
    if (stylePreset === "tactical_reader" && publishClass === "CONTEXT" && hasLeague) {
      return { label: "Watch closely", score, bucket: "watch_closely" };
    }
    if (hasLeague || hasMarket) {
      return { label: "Website only", score, bucket: "website_only" };
    }
    return { label: "No edge", score, bucket: publishClass === "MONITOR" || publishClass === "CONTEXT" ? "no_edge" : "website_only" };
  };

  const fixtureClarityProfile = (fixture, matchedEntry) => {
    const publishClass = String(fixture?.publish_class || fixture?.fixture_class || "MONITOR").toUpperCase();
    const reasons = Array.isArray(matchedEntry?.reasons) ? matchedEntry.reasons : [];
    const notes = Array.isArray(fixture?.context_summary?.notes) ? fixture.context_summary.notes.filter(Boolean) : [];
    const headline =
      fixture?.signal_summary?.headline ||
      fixture?.signal_summary?.summary_text ||
      "Fixture intelligence update is available.";

    if (publishClass === "DEPLOY") {
      return {
        action_label: "Deploy signal",
        action_copy: "A deployable edge is live here, but it should still be read with discipline rather than urgency.",
        meaning_title: "Why this reached deployment",
        meaning_copy: headline,
        risk_title: "Why we may be wrong",
        risk_points: notes.length ? notes : ["Market conditions can change quickly around team news, price movement, or fragility factors."],
        decision_title: "What to do now",
        decision_points: [
          "Check whether the edge still makes sense at the current price.",
          "Use stake discipline before acting.",
          "If the setup feels rushed, wait and review again.",
        ],
        reflection_prompt: "Would you still take this without outside noise or social proof?",
        feed_bucket: dashboardPriorityProfile({ row: fixture, reasons }).label,
      };
    }

    if (publishClass === "OBSERVE") {
      return {
        action_label: "Pass / observe",
        action_copy: "There is signal shape here, but not enough support for disciplined deployment. This is a respectable no-edge state, not a missed pick.",
        meaning_title: "What this means",
        meaning_copy: headline,
        risk_title: "Why this stayed out",
        risk_points: notes.length ? notes : ["Structural support remained too weak for deployment."],
        decision_title: "What to do now",
        decision_points: [
          "Treat this as useful context, not a forced action.",
          "Watch for confirming information rather than acting from interest alone.",
          "No bet is a valid outcome here.",
        ],
        reflection_prompt: "Is this structure, or are you being pulled toward action by curiosity?",
        feed_bucket: dashboardPriorityProfile({ row: fixture, reasons }).label,
      };
    }

    return {
      action_label: "No edge / monitor",
      action_copy: "No deployable edge is visible here right now. That is information, not a gap in the product.",
      meaning_title: "What this means",
      meaning_copy:
        headline ||
        "Market appears efficiently priced or structurally incomplete for disciplined action.",
      risk_title: "What weakens conviction",
      risk_points: notes.length
        ? notes
        : ["Current information does not justify a deployable edge or stronger action state."],
      decision_title: "What to do now",
      decision_points: [
        "Monitor only unless stronger structure appears.",
        "Use this fixture for awareness, not urgency.",
        "Passing is a respectable decision when edge is absent.",
      ],
      reflection_prompt: "Can you let this stay a pass without needing to manufacture a bet?",
      feed_bucket: "No edge",
    };
  };

  const kickoffTimestamp = (value) => {
    const parsed = Date.parse(value || "");
    return Number.isFinite(parsed) ? parsed : Number.MAX_SAFE_INTEGER;
  };

  const kickoffMinutesAway = (value) => {
    const timestamp = kickoffTimestamp(value);
    if (!Number.isFinite(timestamp) || timestamp === Number.MAX_SAFE_INTEGER) {
      return Number.POSITIVE_INFINITY;
    }
    return Math.round((timestamp - Date.now()) / 60000);
  };

  const isNearKickoffWindow = (value, minutes = 240) => {
    const diff = kickoffMinutesAway(value);
    return diff >= 0 && diff <= minutes;
  };

  const isEliteFixture = (row) => {
    const tier = String(row?.confidence_tier || row?.premium_tier || "").toUpperCase();
    return tier === "ELITE";
  };

  const fixtureDeskState = (entry) => {
    const publishClass = String(entry?.row?.publish_class || entry?.row?.fixture_class || "MONITOR").toUpperCase();
    const priority = dashboardPriorityProfile(entry);
    if (publishClass === "DEPLOY") {
      return {
        label: "DEPLOY",
        tone: "deploy",
        support_title: "Why this reached deployment",
      };
    }
    if (priority.bucket === "no_edge") {
      return {
        label: "PASS",
        tone: "pass",
        support_title: "Why this stays a pass",
      };
    }
    if (publishClass === "OBSERVE") {
      return {
        label: "OBSERVE",
        tone: "observe",
        support_title: "Why this stays in observe",
      };
    }
    return {
      label: "MONITOR",
      tone: "monitor",
      support_title: "What keeps this in monitor",
    };
  };

  const publicDeskState = (row) => {
    const publishClass = String(row?.publish_class || row?.fixture_class || "MONITOR").toUpperCase();
    if (publishClass === "DEPLOY") {
      return { label: "DEPLOY", tone: "deploy" };
    }
    if (publishClass === "OBSERVE") {
      return { label: "OBSERVE", tone: "observe" };
    }
    return { label: "MONITOR", tone: "monitor" };
  };

  const groupItemsByLeague = (items, getRow = (item) => item) => {
    const sorted = [...items].sort((left, right) => {
      const leftRow = getRow(left);
      const rightRow = getRow(right);
      const timeDiff = kickoffTimestamp(leftRow?.kickoff_time) - kickoffTimestamp(rightRow?.kickoff_time);
      if (timeDiff !== 0) {
        return timeDiff;
      }
      return publishClassRank(rightRow?.publish_class) - publishClassRank(leftRow?.publish_class);
    });

    const groups = [];
    const groupMap = new Map();
    sorted.forEach((item) => {
      const row = getRow(item);
      const key = String(row?.league || "Other fixtures");
      if (!groupMap.has(key)) {
        const group = { league: key, items: [] };
        groupMap.set(key, group);
        groups.push(group);
      }
      groupMap.get(key).items.push(item);
    });
    return groups;
  };

  const leagueGroupTimingLabel = (group, getRow = (item) => item) => {
    const first = group.items[0] ? getRow(group.items[0]) : null;
    if (!first) {
      return "Upcoming fixtures";
    }
    if (group.items.length === 1) {
      return formatKickoffLabel(first.kickoff_time);
    }
    return `${group.items.length} fixtures • from ${formatKickoffLabel(first.kickoff_time)}`;
  };

  const dashboardFixtureCard = (entry, telegramEnabled) => {
    const row = entry.row;
    const publishClass = String(row.publish_class || row.fixture_class || "MONITOR").toUpperCase();
    const priority = dashboardPriorityProfile(entry);
    const deskState = fixtureDeskState(entry);
    const notes = Array.isArray(row.context_summary?.notes) ? row.context_summary.notes.slice(0, 3) : [];
    const odds = row.odds_summary || {};
    const routeValue =
      priority.bucket === "website_only" || priority.bucket === "no_edge"
        ? "Website only"
        : telegramEnabled
          ? row.follow_relevance?.notification_priority || "high"
          : "Website first";
    const reasonLabel = entry.reasons.length ? entry.reasons.join(" • ") : "current intelligence window";
    const supportCopy =
      row.signal_summary?.headline || row.signal_summary?.summary_text || "Monitoring update published.";
    return `
      <details class="panel fixture-stream-card fixture-stream-card-${escapeHtml(deskState.tone)}" ${priority.bucket === "send_now" ? "open" : ""}>
        <summary class="fixture-stream-summary">
          <div class="fixture-stream-summary-main">
            <div class="intelligence-card-head">
              <span class="fixture-state-pill fixture-state-pill-${escapeHtml(deskState.tone)}">${escapeHtml(deskState.label)}</span>
              <span class="chip">${escapeHtml(marketFamilyLabel(row.signal_summary?.market_family))}</span>
              <span class="chip">${escapeHtml(reasonLabel)}</span>
              <span class="muted">${escapeHtml(formatKickoffLabel(row.kickoff_time))}</span>
            </div>
            <strong class="fixture-teamline dashboard-teamline">
              ${badgeMarkup(row.home_team_logo_url, row.home_team)}
              <span class="team-name">${escapeHtml(row.home_team)}</span>
              <span class="versus">vs</span>
              ${badgeMarkup(row.away_team_logo_url, row.away_team)}
              <span class="team-name">${escapeHtml(row.away_team)}</span>
            </strong>
            <p class="fixture-stream-headline">${escapeHtml(supportCopy)}</p>
          </div>
          <div class="fixture-stream-summary-side">
            <span class="metric-label">${escapeHtml(priority.label)}</span>
            <span class="metric-value dashboard-route">${escapeHtml(routeValue)}</span>
            <span class="muted fixture-stream-expand">Open intelligence</span>
          </div>
        </summary>
        <div class="fixture-stream-body">
          <div class="fixture-stream-body-grid">
            <article class="panel">
              <span class="metric-label">${escapeHtml(deskState.support_title)}</span>
              <p class="intelligence-headline">${escapeHtml(supportCopy)}</p>
            </article>
            <article class="panel">
              <span class="metric-label">Why this matches you</span>
              <p class="intelligence-headline">${escapeHtml(
                `This fixture is on your desk because it matched ${reasonLabel}.`
              )}</p>
            </article>
          </div>
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
          <p class="fixture-stream-route">${escapeHtml(routeExplanation(entry))}</p>
          <div class="dashboard-telegram-preview">
            <span class="metric-label">Telegram alert preview</span>
            <pre>${escapeHtml(telegramAlertPreview(entry))}</pre>
          </div>
          <div class="cta-row">
            <a class="button" href="${fixtureDetailHref(row)}">Open fixture view</a>
            <button class="ghost-button" type="button" data-action="telegram-fixture-alert" data-fixture-key="${escapeHtml(String(row.fixture_key || ""))}">Send to Telegram</button>
          </div>
        </div>
      </details>
    `;
  };

  const renderDashboardFixtureGroups = (entries, telegramEnabled, emptyCopy) => {
    if (!entries.length) {
      return `<div class="notice">${escapeHtml(emptyCopy)}</div>`;
    }
    const groups = groupItemsByLeague(entries, (entry) => entry.row);
    return `
      <div class="fixture-stream">
        ${groups
          .map(
            (group) => `
              <article class="panel fixture-league-group">
                <div class="fixture-league-group-head">
                  <div class="fixture-league-group-meta">
                    <span class="league-badge">${escapeHtml(group.league)}</span>
                    <p class="muted">${escapeHtml(leagueGroupTimingLabel(group, (entry) => entry.row))}</p>
                  </div>
                </div>
                <div class="fixture-stream-list">
                  ${group.items.map((entry) => dashboardFixtureCard(entry, telegramEnabled)).join("")}
                </div>
              </article>
            `
          )
          .join("")}
      </div>
    `;
  };

  const publicDeskFixtureCard = (row) => {
    const publishClass = String(row.publish_class || row.fixture_class || "MONITOR").toUpperCase();
    const deskState = publicDeskState(row);
    const notes = Array.isArray(row.context_summary?.notes) ? row.context_summary.notes.slice(0, 2) : [];
    return `
      <details class="panel fixture-stream-card fixture-stream-card-${escapeHtml(deskState.tone)}">
        <summary class="fixture-stream-summary">
          <div class="fixture-stream-summary-main">
            <div class="intelligence-card-head">
              <span class="fixture-state-pill fixture-state-pill-${escapeHtml(deskState.tone)}">${escapeHtml(deskState.label)}</span>
              <span class="chip">${escapeHtml(marketFamilyLabel(row.signal_summary?.market_family))}</span>
              <span class="muted">${escapeHtml(formatKickoffLabel(row.kickoff_time))}</span>
            </div>
            <strong class="fixture-teamline dashboard-teamline">
              ${badgeMarkup(row.home_team_logo_url, row.home_team)}
              <span class="team-name">${escapeHtml(row.home_team)}</span>
              <span class="versus">vs</span>
              ${badgeMarkup(row.away_team_logo_url, row.away_team)}
              <span class="team-name">${escapeHtml(row.away_team)}</span>
            </strong>
            <p class="fixture-stream-headline">${escapeHtml(
              row.signal_summary?.headline || row.signal_summary?.summary_text || "Fixture intelligence update available."
            )}</p>
          </div>
          <div class="fixture-stream-summary-side">
            <span class="metric-label">${escapeHtml(publishClass === "DEPLOY" ? "Deployable" : "Read first")}</span>
            <span class="metric-value dashboard-route">${escapeHtml(deskState.label === "DEPLOY" ? "Actionable" : "Interpretive")}</span>
            <span class="muted fixture-stream-expand">Open intelligence</span>
          </div>
        </summary>
        <div class="fixture-stream-body">
          ${
            notes.length
              ? `<ul class="feature-list compact-list">${notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>`
              : `<p class="muted">This fixture currently carries a concise public-safe intelligence summary only.</p>`
          }
          <div class="cta-row">
            <a class="button" href="${fixtureDetailHref(row)}">Open fixture view</a>
          </div>
        </div>
      </details>
    `;
  };

  const renderPublicFixtureGroups = (rows) => {
    if (!rows.length) {
      return `<div class="notice">No public fixture intelligence is available for the current window yet.</div>`;
    }
    const groups = groupItemsByLeague(rows);
    return `
      <div class="fixture-stream">
        ${groups
          .map(
            (group) => `
              <article class="panel fixture-league-group">
                <div class="fixture-league-group-head">
                  <div class="fixture-league-group-meta">
                    <span class="league-badge">${escapeHtml(group.league)}</span>
                    <p class="muted">${escapeHtml(leagueGroupTimingLabel(group))}</p>
                  </div>
                </div>
                <div class="fixture-stream-list">
                  ${group.items.map((row) => publicDeskFixtureCard(row)).join("")}
                </div>
              </article>
            `
          )
          .join("")}
      </div>
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

  const extractLeagueIdFromLogoUrl = (value) => {
    const raw = String(value || "").trim();
    const match = raw.match(/\/football\/leagues\/(\d+)\.(?:png|svg|webp)(?:\?.*)?$/i);
    return match ? match[1] : "";
  };

  const extractTeamIdFromLogoUrl = (value) => {
    const raw = String(value || "").trim();
    const match = raw.match(/\/football\/teams\/(\d+)\.(?:png|svg|webp)(?:\?.*)?$/i);
    return match ? match[1] : "";
  };

  const inferFootballSeason = (kickoffTime) => {
    const parsed = kickoffTime ? new Date(kickoffTime) : null;
    if (!parsed || Number.isNaN(parsed.getTime())) {
      return "";
    }
    const year = parsed.getUTCFullYear();
    const month = parsed.getUTCMonth() + 1;
    return String(month <= 6 ? year - 1 : year);
  };

  const fixtureStandingsWidgetMarkup = (fixture) => {
    const kickoffDate = String(fixture.kickoff_time || "").slice(0, 10);
    const homeTeamId = extractTeamIdFromLogoUrl(fixture.home_team_logo_url);
    const awayTeamId = extractTeamIdFromLogoUrl(fixture.away_team_logo_url);
    if (!kickoffDate || !fixture.home_team || !fixture.away_team) {
      return `<div class="notice">League table reference is not available for this fixture yet because the widget identity mapping is incomplete.</div>`;
    }
    if (!workerConfigured()) {
      return `<div class="notice">League table reference needs the Worker proxy so the widget can stay branded and keep the API key off the page.</div>`;
    }
    return `
      <div
        class="widget-reference-shell"
        data-role="fixture-standings-reference"
        data-date="${escapeHtml(kickoffDate)}"
        data-home="${escapeHtml(fixture.home_team)}"
        data-away="${escapeHtml(fixture.away_team)}"
        data-home-team-id="${escapeHtml(homeTeamId)}"
        data-away-team-id="${escapeHtml(awayTeamId)}"
      >
        <div class="widget-reference-head">
          <div>
            <span class="metric-label">Reference layer</span>
            <h4>League table</h4>
          </div>
          <span class="pill">Prototype</span>
        </div>
        <p class="muted">Reference context for this fixture. Use this as orientation, not as the decision layer. If the fit is strong, lineups and formations can sit beside it later.</p>
        <div class="widget-reference-frame">
          <div class="notice">Loading league table…</div>
        </div>
      </div>
    `;
  };

  const fixtureLineupsWidgetMarkup = (fixture) => {
    const kickoffDate = String(fixture.kickoff_time || "").slice(0, 10);
    const homeTeamId = extractTeamIdFromLogoUrl(fixture.home_team_logo_url);
    const awayTeamId = extractTeamIdFromLogoUrl(fixture.away_team_logo_url);
    if (!kickoffDate || !fixture.home_team || !fixture.away_team) {
      return `<div class="notice">Lineups and formations are not available for this fixture yet because the widget identity mapping is incomplete.</div>`;
    }
    if (!workerConfigured()) {
      return `<div class="notice">Lineups and formations need the Worker proxy so the fixture resolver and widget API key stay off the page.</div>`;
    }
    return `
      <div
        class="widget-reference-shell"
        data-role="fixture-lineups-widget"
        data-date="${escapeHtml(kickoffDate)}"
        data-home="${escapeHtml(fixture.home_team)}"
        data-away="${escapeHtml(fixture.away_team)}"
        data-home-team-id="${escapeHtml(homeTeamId)}"
        data-away-team-id="${escapeHtml(awayTeamId)}"
      >
        <div class="widget-reference-head">
          <div>
            <span class="metric-label">Reference layer</span>
            <h4>Lineups & formations</h4>
          </div>
          <span class="pill">Prototype</span>
        </div>
        <p class="muted">This sits beside the custom intelligence layer so confirmed teams and shape can support the read without taking over the page.</p>
        <div class="widget-reference-frame">
          <div class="notice">Resolving fixture reference…</div>
        </div>
      </div>
    `;
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

  const homeView = () => {
    const publicDeskRows = [...state.fixtureIntelligence]
      .sort((left, right) => {
        const timeDiff = kickoffTimestamp(left.kickoff_time) - kickoffTimestamp(right.kickoff_time);
        if (timeDiff !== 0) {
          return timeDiff;
        }
        return publishClassRank(right.publish_class) - publishClassRank(left.publish_class);
      })
      .slice(0, 12);

    return `
    <section class="hero">
      <div class="hero-main">
        <div class="hero-copy-stack">
          <p class="hero-kicker">Prediction intelligence system</p>
          <h1>Signal over noise.</h1>
          <p>
            Odds Genius is a calm football intelligence surface built to help users think better under uncertainty.
            It combines selective deployment, bookmaker value, and goal-shape reading without turning every fixture
            into a forced bet.
          </p>
          <div class="pill-row">
            <span class="stat-chip">Better decisions, not more bets</span>
            <span class="stat-chip">Clarity under uncertainty</span>
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
            <h2>Today on the desk</h2>
            <p class="section-copy">
              Fixtures are grouped by competition first so you can orient fast, then expand only the ones worth reading more closely.
            </p>
          </div>
          <a class="ghost-button" href="./dashboard.html">Open dashboard</a>
        </div>
        ${renderPublicFixtureGroups(publicDeskRows)}
      </div>
      <article class="panel">
        <h3>How to read it</h3>
        <ul class="method-list">
          <li>Start with the league group, not a flat prediction table.</li>
          <li>Expand a fixture only when the state, market, or timing makes it worth more attention.</li>
          <li>Deploy is actionable. Observe is interpretive. Monitor keeps awareness high without forcing a bet.</li>
        </ul>
        <div class="cta-row">
          <a class="button" href="./predictions.html">Open predictions board</a>
        </div>
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
        <h3>No edge is also information</h3>
        <ul class="method-list">
          <li>Restraint is part of the product, not a missing feature.</li>
          <li>A pass state should still teach the user something about price, volatility, and uncertainty.</li>
          <li>The goal is cleaner judgment, not constant stimulation.</li>
          <li>If the model doesn't beat the price, it doesn't deploy.</li>
        </ul>
      </article>
    </section>
  `;
  };

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
    const userStylePreset = normalizeStylePreset(notificationPreferences?.user_style_preset);
    const telegramLinked = telegramLink?.link_status === "linked";
    const telegramReady = Boolean(state.runtime.telegramLinkCode);
    const subscriptionStatus = accountState?.subscription?.subscription_status || (entitled ? "active" : "pending");
    const displayEmail = accountState?.user?.email || "";
    const greeting = accountGreeting(accountState);
    const onboardingCompleted = Boolean(notificationPreferences?.calm_onboarding_completed_at);
    const onboardingSummary = onboardingStepSummary(notificationPreferences);
    const followedIntelligence = getFollowedIntelligenceMatches(accountState, state.fixtureIntelligence);
    const accountSessions = Array.isArray(state.runtime.accountSessions) ? state.runtime.accountSessions : [];
    const currentSession = accountSessions.find((session) => session.is_current) || null;
    const recentSessions = accountSessions.filter((session) => !session.is_current);
    const followedSignalsConfigured = Boolean(
      parsePreferenceList(notificationPreferences?.favourite_teams).length ||
        parsePreferenceList(notificationPreferences?.favourite_leagues).length ||
        parsePreferenceList(notificationPreferences?.favourite_markets).length ||
        parsePreferenceList(notificationPreferences?.followed_fixtures).length
    );
    const onboardingStageCount =
      Number(Boolean(parsePreferenceList(notificationPreferences?.favourite_leagues).length)) +
      Number(Boolean(parsePreferenceList(notificationPreferences?.favourite_markets).length)) +
      Number(Boolean(parsePreferenceList(notificationPreferences?.favourite_teams).length)) +
      Number(Boolean(parsePreferenceList(notificationPreferences?.followed_fixtures).length)) +
      Number(Boolean(notificationPreferences?.telegram_enabled));

    return `
      ${
        signedIn
          ? `
            <section class="section split">
              <article class="hero-main">
                <div class="hero-copy-stack">
                  <p class="hero-kicker">Account home</p>
                  <h1>${escapeHtml(greeting)}</h1>
                  <p>Welcome back to your calm intelligence desk. Your account, settings, delivery posture, and followed environment all live here behind your verified sign-in.</p>
                </div>
                <div class="pill-row">
                  <span class="stat-chip">${escapeHtml(stylePresetLabel(userStylePreset))}</span>
                  <span class="stat-chip">${escapeHtml(languageLabel(notificationPreferences?.language_preference || "en-GB"))}</span>
                  <span class="stat-chip">${onboardingCompleted ? "Calm setup saved" : "Calm setup in progress"}</span>
                </div>
                <div class="cta-row">
                  <a class="button" href="${onboardingCompleted ? "#preferences" : "./onboarding.html"}">${onboardingCompleted ? "Preferences" : "Finish calm setup"}</a>
                  <a class="ghost-button" href="#billing">Billing</a>
                  <a class="ghost-button" href="#help">Help</a>
                </div>
              </article>
              <aside class="hero-side">
                <div class="metric">
                  <span class="metric-label">Account</span>
                  <span class="metric-value">Verified</span>
                </div>
                <div class="metric">
                  <span class="metric-label">Preset</span>
                  <span class="metric-value">${escapeHtml(stylePresetLabel(userStylePreset))}</span>
                </div>
                <div class="metric">
                  <span class="metric-label">Delivery posture</span>
                  <span class="metric-value">${notificationPreferences?.telegram_enabled && !notificationPreferences?.website_only_mode ? "Selective" : "Website-first"}</span>
                </div>
              </aside>
            </section>
            ${
              !onboardingCompleted
                ? `
                  <section class="section" id="first-run-onboarding">
                    <article class="panel">
                      <h3>First-run onboarding</h3>
                      <p class="muted">This account is signed in, but your calm setup is not fully locked yet. Finish these steps once so the dashboard and alert system can behave more like your own analyst desk.</p>
                      <div class="stats-grid">
                        ${statPanel("Progress", `${onboardingSummary.completed}/${onboardingSummary.total}`, "Complete your first-run setup")}
                        ${statPanel("Preset", stylePresetLabel(userStylePreset), "This shapes feed ranking and alert selectivity")}
                        ${statPanel("Delivery rule", notificationPreferences?.telegram_enabled && !notificationPreferences?.website_only_mode ? "Telegram selective" : "Website-first", feedRoutingExplanation(userStylePreset))}
                        ${statPanel("Outcome", "Calmer account", "Less noise, clearer relevance, stronger pacing")}
                      </div>
                      <div class="card-grid">
                        ${onboardingSummary.steps
                          .map(
                            (step) => `
                              <article class="panel">
                                <h4>${escapeHtml(step.complete ? "Complete" : "Next step")}</h4>
                                <strong>${escapeHtml(step.label)}</strong>
                                <p class="muted">${escapeHtml(step.detail)}</p>
                              </article>
                            `
                          )
                          .join("")}
                      </div>
                      <div class="cta-row">
                        <a class="button" href="./onboarding.html">Open onboarding flow</a>
                        <a class="ghost-button" href="./dashboard.html">Preview dashboard anyway</a>
                      </div>
                    </article>
                  </section>
                `
                : ""
            }
          `
          : ""
      }
      ${
        !signedIn
          ? `
            <section class="section split">
              <article class="hero-main">
                <p class="hero-kicker">Account</p>
                <h1>${accountHeadline}</h1>
                <p>${accountCopy}</p>
                <div class="cta-row">
                  <a class="button" href="${entitled ? "./premium.html" : "./pricing.html"}">${
                    entitled ? "Open premium board" : "See founding plan"
                  }</a>
                  <a class="ghost-button" href="${entitled ? "./results.html" : "./premium.html"}">${
                    entitled ? "Go to results" : "Open premium page"
                  }</a>
                </div>
              </article>
              <aside class="hero-side">
                <div class="metric">
                  <span class="metric-label">Worker</span>
                  <span class="metric-value">${workerConfigured() ? "Configured" : "Placeholder"}</span>
                </div>
                <div class="metric">
                  <span class="metric-label">Identity</span>
                  <span class="metric-value">Not verified</span>
                </div>
                <div class="metric">
                  <span class="metric-label">Membership</span>
                  <span class="metric-value">${entitled ? "Premium active" : "Premium pending"}</span>
                </div>
                <div class="metric">
                  <span class="metric-label">Telegram</span>
                  <span class="metric-value">Locked</span>
                </div>
              </aside>
            </section>
          `
          : ""
      }

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
                <p class="hero-kicker">Sign up / Log in</p>
                <h3>Verify your email</h3>
                <p class="muted">
                  Use the same email you used for checkout. If the address is eligible, a sign-in link will be sent so you can open your calm intelligence dashboard on this device.
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
                <h3>${accountIntent === "checkout" ? "Getting started" : "Why this matters"}</h3>
                <ul class="feature-list">
                  <li>Checkout confirms billing, not identity.</li>
                  <li>Email verification unlocks this device when your active membership is confirmed.</li>
                  <li>Odds Genius is built to help you think better before you bet, not to rush you into action.</li>
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
            <section class="section">
              <article class="panel">
                <h3>Account navigation</h3>
                <p class="muted">Use this as the stable navigation layer for account state, delivery posture, preferences, billing, and help.</p>
                <div class="pill-row">
                  <a class="ghost-button" href="#account-overview">Account</a>
                  <a class="ghost-button" href="#devices">Devices</a>
                  <a class="ghost-button" href="#settings">Settings</a>
                  <a class="ghost-button" href="#preferences">Preferences</a>
                  <a class="ghost-button" href="#language">Language</a>
                  <a class="ghost-button" href="#billing">Billing</a>
                  <a class="ghost-button" href="#help">Help</a>
                </div>
              </article>
            </section>
            <section class="section split">
              <article class="panel" id="account-overview">
                <h3>Account state</h3>
                <ul class="feature-list">
                  <li>Verified email: ${displayEmail ? escapeHtml(displayEmail) : "Signed in"}</li>
                  <li>Membership status: ${escapeHtml(subscriptionStatus)}</li>
                  <li>D1 profile state: ${accountState ? "Active" : state.runtime.accountStateError ? "Unavailable" : "Pending"}</li>
                </ul>
              </article>
              <article class="panel" id="devices">
                <h3>Devices</h3>
                <p class="muted">This is the first visible device layer on top of magic-link sign-in. Use it to understand where this account is active before revoke and sign-out-other-devices controls go live.</p>
                ${renderNotice(state.runtime.accountSessionsMessage, state.runtime.accountSessionsMessage ? "success" : "default")}
                ${renderNotice(state.runtime.accountSessionsError, state.runtime.accountSessionsError ? "warning" : "default")}
                ${
                  currentSession
                    ? `<div class="card-grid">${accountSessionCard(currentSession)}</div>`
                    : `<div class="notice">No current tracked device session was returned yet.</div>`
                }
                <div class="cta-row">
                  <button class="ghost-button" type="button" data-action="revoke-other-sessions">Sign out other devices</button>
                </div>
                ${
                  recentSessions.length
                    ? `<div class="card-grid">${recentSessions.slice(0, 5).map((session) => accountSessionCard(session)).join("")}</div>`
                    : `<div class="notice">No recent secondary device sessions are stored for this account yet.</div>`
                }
              </article>
              <article class="panel" id="settings">
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
              <article class="panel">
                <h3>Followed intelligence</h3>
                <p class="muted">
                  This is the account preview of your followed intelligence. Use the dashboard for the fuller grouped fixture workspace.
                </p>
                ${
                  !followedSignalsConfigured
                    ? `<div class="notice">Add followed teams, leagues, markets, or fixtures below to start shaping your personal intelligence board.</div>`
                    : followedIntelligence.length
                      ? `
                        <div class="card-grid intelligence-grid">
                          ${followedIntelligence
                            .slice(0, 8)
                            .map((entry) => intelligenceCard(entry, Boolean(notificationPreferences?.telegram_enabled)))
                            .join("")}
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
              <article class="panel" id="preferences">
                <h3>Intelligence preferences</h3>
                <p class="muted">
                  Choose what kind of intelligence you want on the website and in Telegram. This is the first layer of personalised delivery.
                </p>
                <form id="preferences-form" class="stack-form">
                  <div class="card-grid">
                    <article class="panel">
                      <h4>Onboarding and style</h4>
                      <p class="muted">Choose the product posture that should shape your feed, alert thresholds, and interruption style.</p>
                      <label class="field-label" for="user-style-preset">Style preset</label>
                      <select id="user-style-preset" name="user_style_preset" class="text-input">
                        <option value="analyst" ${userStylePreset === "analyst" ? "selected" : ""}>Analyst</option>
                        <option value="disciplined_bettor" ${userStylePreset === "disciplined_bettor" ? "selected" : ""}>Disciplined bettor</option>
                        <option value="tactical_reader" ${userStylePreset === "tactical_reader" ? "selected" : ""}>Tactical reader</option>
                        <option value="researcher" ${userStylePreset === "researcher" ? "selected" : ""}>Researcher</option>
                      </select>
                      <p class="muted">${escapeHtml(stylePresetSummary(userStylePreset))}</p>
                      <label class="checkbox-row"><input type="checkbox" name="decision_companion_enabled" ${notificationPreferences?.decision_companion_enabled ? "checked" : ""} /> Decision companion prompts</label>
                      <label class="checkbox-row"><input type="checkbox" name="reset_mode_enabled" ${notificationPreferences?.reset_mode_enabled ? "checked" : ""} /> Reset / clarity mode after losses or failed deploys</label>
                      <label class="checkbox-row"><input type="checkbox" name="complete_calm_setup" ${onboardingCompleted ? "checked" : ""} /> Mark calm setup as complete</label>
                    </article>
                    <article class="panel" id="language">
                      <h4>Language</h4>
                      <label class="field-label" for="language-preference">Interface language</label>
                      <select id="language-preference" name="language_preference" class="text-input">
                        <option value="en-GB" ${(notificationPreferences?.language_preference || "en-GB") === "en-GB" ? "selected" : ""}>English (UK)</option>
                        <option value="en-US" ${notificationPreferences?.language_preference === "en-US" ? "selected" : ""}>English (US)</option>
                        <option value="pt-PT" ${notificationPreferences?.language_preference === "pt-PT" ? "selected" : ""}>Portuguese</option>
                        <option value="es-ES" ${notificationPreferences?.language_preference === "es-ES" ? "selected" : ""}>Spanish</option>
                      </select>
                      <p class="muted">Language is saved now so the account shell is ready for fuller localisation later.</p>
                    </article>
                  </div>
                  <div class="card-grid">
                    <article class="panel">
                      <h4>Channels</h4>
                      <label class="checkbox-row"><input type="checkbox" name="telegram_enabled" ${notificationPreferences?.telegram_enabled ? "checked" : ""} /> Telegram messages</label>
                      <label class="checkbox-row"><input type="checkbox" name="email_enabled" ${notificationPreferences?.email_enabled ? "checked" : ""} /> Email digests</label>
                      <label class="checkbox-row"><input type="checkbox" name="website_only_mode" ${notificationPreferences?.website_only_mode ? "checked" : ""} /> Website-first mode</label>
                    </article>
                    <article class="panel">
                      <h4>Signal alerts</h4>
                      <label class="checkbox-row"><input type="checkbox" name="elite_alerts_enabled" ${notificationPreferences?.elite_alerts_enabled ? "checked" : ""} /> Elite deployments</label>
                      <label class="checkbox-row"><input type="checkbox" name="standard_alerts_enabled" ${notificationPreferences?.standard_alerts_enabled ? "checked" : ""} /> Standard deployments</label>
                      <label class="checkbox-row"><input type="checkbox" name="acca_alerts_enabled" ${notificationPreferences?.acca_alerts_enabled ? "checked" : ""} /> Acca notes</label>
                      <label class="checkbox-row"><input type="checkbox" name="correct_score_alerts_enabled" ${notificationPreferences?.correct_score_alerts_enabled ? "checked" : ""} /> Correct score support</label>
                    </article>
                    <article class="panel">
                      <h4>Intelligence alerts</h4>
                      <label class="checkbox-row"><input type="checkbox" name="injury_alerts_enabled" ${notificationPreferences?.injury_alerts_enabled ? "checked" : ""} /> Injury news</label>
                      <label class="checkbox-row"><input type="checkbox" name="team_news_alerts_enabled" ${notificationPreferences?.team_news_alerts_enabled ? "checked" : ""} /> Major team news</label>
                      <label class="checkbox-row"><input type="checkbox" name="weather_alerts_enabled" ${notificationPreferences?.weather_alerts_enabled ? "checked" : ""} /> Weather info</label>
                      <label class="checkbox-row"><input type="checkbox" name="market_movement_alerts_enabled" ${notificationPreferences?.market_movement_alerts_enabled ? "checked" : ""} /> Market movement</label>
                      <label class="checkbox-row"><input type="checkbox" name="volatility_alerts_enabled" ${notificationPreferences?.volatility_alerts_enabled ? "checked" : ""} /> Volatility warnings</label>
                      <label class="checkbox-row"><input type="checkbox" name="allow_non_signal_intelligence" ${notificationPreferences?.allow_non_signal_intelligence ? "checked" : ""} /> Observe And Monitor Updates</label>
                    </article>
                    <article class="panel">
                      <h4>Digests and timing</h4>
                      <label class="checkbox-row"><input type="checkbox" name="daily_digest_enabled" ${notificationPreferences?.daily_digest_enabled ? "checked" : ""} /> Daily digest</label>
                      <label class="checkbox-row"><input type="checkbox" name="results_digest_enabled" ${notificationPreferences?.results_digest_enabled ? "checked" : ""} /> Results digest</label>
                      <label class="checkbox-row"><input type="checkbox" name="weekend_slate_digest_enabled" ${notificationPreferences?.weekend_slate_digest_enabled ? "checked" : ""} /> Weekend slate digest</label>
                      <label class="field-label" for="alert-frequency-mode">Alert frequency</label>
                      <select id="alert-frequency-mode" name="alert_frequency_mode" class="text-input">
                        <option value="mixed" ${notificationPreferences?.alert_frequency_mode === "mixed" ? "selected" : ""}>Balanced</option>
                        <option value="immediate" ${notificationPreferences?.alert_frequency_mode === "immediate" ? "selected" : ""}>Interrupt when strong</option>
                        <option value="digest_only" ${notificationPreferences?.alert_frequency_mode === "digest_only" ? "selected" : ""}>Digest only</option>
                      </select>
                      <label class="field-label" for="pre-match-window-minutes">Pre-match window (minutes)</label>
                      <input id="pre-match-window-minutes" name="pre_match_window_minutes" class="text-input" type="number" min="0" max="1440" value="${escapeHtml(notificationPreferences?.pre_match_window_minutes ?? 90)}" />
                      <p class="muted">${escapeHtml(feedRoutingExplanation(userStylePreset))}</p>
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
            <section class="section split">
              <article class="panel" id="billing">
                <h3>Billing</h3>
                <p class="muted">Payments and membership controls will live here once customer self-service is fully wired.</p>
                <ul class="feature-list">
                  <li>Membership status remains controlled by your active subscription.</li>
                  <li>Customer billing self-service is the next commercial polish layer.</li>
                  <li>Board freshness and membership metadata will sit here later.</li>
                </ul>
              </article>
              <article class="panel" id="help">
                <h3>Help</h3>
                <p class="muted">Use this area as the calm support layer: access questions, Telegram linking, and how to interpret product states.</p>
                <ul class="feature-list">
                  <li>Use Telegram for stronger interruptions only.</li>
                  <li>Use the dashboard for broader interpretation and watchlist depth.</li>
                  <li>A pass state is valid. No edge is also information.</li>
                </ul>
              </article>
            </section>
          `
      }
    `;
  };

  const onboardingView = () => {
    const signedIn = state.runtime.sessionAuthenticated;
    const entitled = state.runtime.sessionEntitled;
    const accountState = state.runtime.accountState;
    const notificationPreferences = accountState?.notification_preferences || null;
    const userStylePreset = normalizeStylePreset(notificationPreferences?.user_style_preset);
    const onboardingSummary = onboardingStepSummary(notificationPreferences);
    const followedSignalsConfigured = Boolean(
      parsePreferenceList(notificationPreferences?.favourite_teams).length ||
        parsePreferenceList(notificationPreferences?.favourite_leagues).length ||
        parsePreferenceList(notificationPreferences?.favourite_markets).length ||
        parsePreferenceList(notificationPreferences?.followed_fixtures).length
    );

    if (!signedIn) {
      return `
        <section class="section split">
          <article class="hero-main">
            <p class="hero-kicker">Onboarding</p>
            <h1>Set up your calm intelligence desk.</h1>
            <p>Verify your email first, then come back here to choose your style preset, follows, and delivery posture.</p>
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
              <span class="metric-label">Goal</span>
              <span class="metric-value">Clarity</span>
            </div>
          </aside>
        </section>
      `;
    }

    return `
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Onboarding</p>
          <h1>Build your personal Odds Genius environment.</h1>
          <p>This guided setup is here to help you choose what matters, how often you want interruption, and what kind of analyst desk you want the product to become for you.</p>
          <div class="pill-row">
            <span class="stat-chip">${entitled ? "Premium active" : "Membership pending"}</span>
            <span class="stat-chip">${stylePresetLabel(userStylePreset)}</span>
            <span class="stat-chip">${onboardingSummary.completed}/${onboardingSummary.total} steps complete</span>
          </div>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Preset</span>
            <span class="metric-value">${escapeHtml(stylePresetLabel(userStylePreset))}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Delivery</span>
            <span class="metric-value">${notificationPreferences?.telegram_enabled && !notificationPreferences?.website_only_mode ? "Selective" : "Website-first"}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Why this matters</span>
            <span class="metric-value">Less noise</span>
          </div>
        </aside>
      </section>

      <section class="section">
        ${renderNotice(state.runtime.preferencesMessage, state.runtime.preferencesMessage ? "success" : "default")}
        <article class="panel">
          <h3>First-run path</h3>
          <p class="muted">Move through these steps once, then let the dashboard and alert system adapt to your saved posture.</p>
          <div class="card-grid">
            ${onboardingSummary.steps
              .map(
                (step, index) => `
                  <article class="panel">
                    <h4>${escapeHtml(step.complete ? `Step ${index + 1} complete` : `Step ${index + 1}`)}</h4>
                    <strong>${escapeHtml(step.label)}</strong>
                    <p class="muted">${escapeHtml(step.detail)}</p>
                  </article>
                `
              )
              .join("")}
          </div>
        </article>
      </section>

      <section class="section">
        <article class="panel">
          <h3>Preset guide</h3>
          <p class="muted">Choose the posture that should shape your feed, alert thresholds, and pacing.</p>
          <div class="card-grid">
            <article class="panel">
              <h4>Analyst</h4>
              <p class="muted">Broad website context, fewer Telegram interruptions, more reading than reacting.</p>
            </article>
            <article class="panel">
              <h4>Disciplined bettor</h4>
              <p class="muted">Deploy-led, selective, and built to filter harder before anything interrupts you.</p>
            </article>
            <article class="panel">
              <h4>Tactical reader</h4>
              <p class="muted">Team and fixture intelligence first, with stronger follow-led relevance.</p>
            </article>
            <article class="panel">
              <h4>Researcher</h4>
              <p class="muted">Wider website coverage, deeper non-deploy intelligence, minimal interruption.</p>
            </article>
          </div>
        </article>
      </section>

      <section class="section">
        <article class="panel">
          <h3>Guided setup</h3>
          <p class="muted">Use this to configure the account once in a clearer order, rather than as one large settings screen.</p>
          <form id="onboarding-form" class="stack-form">
            <div class="card-grid">
              <article class="panel">
                <h4>Step 1 — Style preset</h4>
                <label class="field-label" for="onboarding-style-preset">Style preset</label>
                <select id="onboarding-style-preset" name="user_style_preset" class="text-input">
                  <option value="analyst" ${userStylePreset === "analyst" ? "selected" : ""}>Analyst</option>
                  <option value="disciplined_bettor" ${userStylePreset === "disciplined_bettor" ? "selected" : ""}>Disciplined bettor</option>
                  <option value="tactical_reader" ${userStylePreset === "tactical_reader" ? "selected" : ""}>Tactical reader</option>
                  <option value="researcher" ${userStylePreset === "researcher" ? "selected" : ""}>Researcher</option>
                </select>
                <p class="muted">${escapeHtml(stylePresetSummary(userStylePreset))}</p>
              </article>
              <article class="panel">
                <h4>Step 2 — Choose your language</h4>
                <label class="field-label" for="onboarding-language-preference">Interface language</label>
                <select id="onboarding-language-preference" name="language_preference" class="text-input">
                  <option value="en-GB" ${(notificationPreferences?.language_preference || "en-GB") === "en-GB" ? "selected" : ""}>English (UK)</option>
                  <option value="en-US" ${notificationPreferences?.language_preference === "en-US" ? "selected" : ""}>English (US)</option>
                  <option value="pt-PT" ${notificationPreferences?.language_preference === "pt-PT" ? "selected" : ""}>Portuguese</option>
                  <option value="es-ES" ${notificationPreferences?.language_preference === "es-ES" ? "selected" : ""}>Spanish</option>
                </select>
              </article>
              <article class="panel">
                <h4>Step 3 — Choose what you care about</h4>
                <input name="favourite_teams" class="text-input" type="text" placeholder="Teams" value="${escapeHtml(joinPreferenceList(notificationPreferences?.favourite_teams))}" />
                <input name="favourite_leagues" class="text-input" type="text" placeholder="Leagues" value="${escapeHtml(joinPreferenceList(notificationPreferences?.favourite_leagues))}" />
                <input name="favourite_markets" class="text-input" type="text" placeholder="Markets" value="${escapeHtml(joinPreferenceList(notificationPreferences?.favourite_markets))}" />
                <input name="followed_fixtures" class="text-input" type="text" placeholder="Fixtures" value="${escapeHtml(joinPreferenceList(notificationPreferences?.followed_fixtures))}" />
                <p class="muted">${followedSignalsConfigured ? "Your follow environment is already taking shape." : "Start narrow. Relevance beats volume."}</p>
              </article>
              <article class="panel">
                <h4>Step 4 — Set delivery posture</h4>
                <label class="checkbox-row"><input type="checkbox" name="telegram_enabled" ${notificationPreferences?.telegram_enabled ? "checked" : ""} /> Telegram messages</label>
                <label class="checkbox-row"><input type="checkbox" name="email_enabled" ${notificationPreferences?.email_enabled ? "checked" : ""} /> Email digests</label>
                <label class="checkbox-row"><input type="checkbox" name="website_only_mode" ${notificationPreferences?.website_only_mode ? "checked" : ""} /> Website-first mode</label>
                <label class="field-label" for="onboarding-alert-frequency">Alert frequency</label>
                <select id="onboarding-alert-frequency" name="alert_frequency_mode" class="text-input">
                  <option value="mixed" ${notificationPreferences?.alert_frequency_mode === "mixed" ? "selected" : ""}>Balanced</option>
                  <option value="immediate" ${notificationPreferences?.alert_frequency_mode === "immediate" ? "selected" : ""}>Interrupt when strong</option>
                  <option value="digest_only" ${notificationPreferences?.alert_frequency_mode === "digest_only" ? "selected" : ""}>Digest only</option>
                </select>
                <p class="muted">${escapeHtml(feedRoutingExplanation(userStylePreset))}</p>
              </article>
              <article class="panel">
                <h4>Step 5 — Keep the decision layer active</h4>
                <label class="checkbox-row"><input type="checkbox" name="decision_companion_enabled" ${notificationPreferences?.decision_companion_enabled ? "checked" : ""} /> Decision companion prompts</label>
                <label class="checkbox-row"><input type="checkbox" name="reset_mode_enabled" ${notificationPreferences?.reset_mode_enabled ? "checked" : ""} /> Reset / clarity mode after losses or failed deploys</label>
                <label class="checkbox-row"><input type="checkbox" name="complete_calm_setup" ${Boolean(notificationPreferences?.calm_onboarding_completed_at) ? "checked" : ""} /> Mark calm setup as complete</label>
                <p class="muted">This keeps the product calm when a read fails or a result lands badly.</p>
              </article>
            </div>
            <div class="cta-row">
              <button class="button" type="submit">Save onboarding and continue</button>
              <a class="ghost-button" href="./dashboard.html">Skip to dashboard</a>
              <a class="ghost-button" href="./account.html">Back to account</a>
            </div>
          </form>
        </article>
      </section>
    `;
  };

  const internalReviewView = () => {
    const hasKey = Boolean(state.runtime.internalAdminKey);
    const summary = state.runtime.internalAccountSummary;
    const flags = Array.isArray(state.runtime.internalFlags) ? state.runtime.internalFlags : [];
    const notes = Array.isArray(state.runtime.internalNotes) ? state.runtime.internalNotes : [];
    const timeline = Array.isArray(state.runtime.internalTimeline) ? state.runtime.internalTimeline : [];
    const operatorId = String(state.runtime.internalOperatorId || "").trim();
    const severityFilter = String(state.runtime.internalFlagSeverityFilter || "ALL").toUpperCase();
    const statusFilter = String(state.runtime.internalFlagStatusFilter || "ALL").toUpperCase();
    const timelineSourceFilter = String(state.runtime.internalTimelineSourceFilter || "ALL").toUpperCase();
    const reviewPreset = String(state.runtime.internalReviewPreset || "CUSTOM").toUpperCase();
    const reviewOutcome = String(state.runtime.internalReviewOutcome || "AUTO").toUpperCase();
    const severityOptions = ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"];
    const statusOptions = ["ALL", "OPEN", "RESOLVED", "DISMISSED"];
    const timelineSourceOptions = [
      ["ALL", "All activity"],
      ["AUTH_EVENT", "Auth events"],
      ["RISK_FLAG", "Risk flags"],
      ["ADMIN_NOTE", "Admin notes"],
      ["ENFORCEMENT", "Enforcement"],
    ];
    const reviewPresets = [
      ["CUSTOM", "Custom"],
      ["SUSPENSION_REVIEW", "Suspension review"],
      ["SHARING_RISK", "Sharing risk"],
      ["BILLING_CONCERN", "Billing concern"],
    ];
    const reviewOutcomes = [
      ["AUTO", "Auto"],
      ["MONITOR_ONLY", "Monitor only"],
      ["RESTRICT_FOR_REVIEW", "Restrict for review"],
      ["SUSPEND", "Suspend"],
      ["REINSTATE_READY", "Reinstate ready"],
    ];
    const severityCounts = severityOptions.reduce((accumulator, option) => {
      if (option === "ALL") {
        accumulator[option] = flags.length;
      } else {
        accumulator[option] = flags.filter(
          (flag) => String(flag?.severity || "").trim().toUpperCase() === option
        ).length;
      }
      return accumulator;
    }, {});
    const statusCounts = statusOptions.reduce((accumulator, option) => {
      if (option === "ALL") {
        accumulator[option] = flags.length;
      } else {
        accumulator[option] = flags.filter(
          (flag) => String(flag?.flag_status || "open").trim().toUpperCase() === option
        ).length;
      }
      return accumulator;
    }, {});
    const filteredFlags = flags.filter((flag) => {
      const flagSeverity = String(flag?.severity || "").trim().toUpperCase();
      const flagStatus = String(flag?.flag_status || "open").trim().toUpperCase();
      const severityMatches = severityFilter === "ALL" || flagSeverity === severityFilter;
      const statusMatches = statusFilter === "ALL" || flagStatus === statusFilter;
      return severityMatches && statusMatches;
    });
    const actionAuditEntries = timeline
      .filter((item) => {
        const eventType = String(item?.event_type || "");
        return (
          eventType.startsWith("internal_account_") ||
          eventType.startsWith("internal_flag_") ||
          eventType.startsWith("admin_note_")
        );
      })
      .slice(0, 10);
    const timelineSourceCounts = timelineSourceOptions.reduce((accumulator, [value]) => {
      if (value === "ALL") {
        accumulator[value] = timeline.length;
      } else if (value === "ENFORCEMENT") {
        accumulator[value] = timeline.filter((item) =>
          String(item?.event_type || "").startsWith("internal_account_")
        ).length;
      } else {
        accumulator[value] = timeline.filter(
          (item) => String(item?.source_type || "").trim().toUpperCase() === value
        ).length;
      }
      return accumulator;
    }, {});
    const filteredTimeline = timeline.filter((item) => {
      if (timelineSourceFilter === "ALL") {
        return true;
      }
      if (timelineSourceFilter === "ENFORCEMENT") {
        return String(item?.event_type || "").startsWith("internal_account_");
      }
      return String(item?.source_type || "").trim().toUpperCase() === timelineSourceFilter;
    });
    const restrictedEvent = timeline.find((item) => String(item?.event_type || "") === "internal_account_restricted");
    const suspendedEvent = timeline.find((item) => String(item?.event_type || "") === "internal_account_suspended");
    const reinstatedEvent = timeline.find((item) => String(item?.event_type || "") === "internal_account_reinstated");
    const accountHistoryMilestones = [
      restrictedEvent
        ? {
            label: "Restricted",
            when: restrictedEvent.timestamp,
            note: restrictedEvent?.metadata?.reason || "Restriction recorded",
          }
        : null,
      summary?.risk_state?.suspended_at
        ? {
            label: "Suspended",
            when: summary.risk_state.suspended_at,
            note:
              summary.risk_state?.suspension_reason ||
              suspendedEvent?.metadata?.reason ||
              "Suspension recorded",
          }
        : null,
      summary?.risk_state?.reinstated_at
        ? {
            label: "Reinstated",
            when: summary.risk_state.reinstated_at,
            note:
              summary.risk_state?.reinstatement_reason ||
              reinstatedEvent?.metadata?.reason ||
              "Reinstatement recorded",
          }
        : null,
      reinstatedEvent
        ? {
            label: "Review cleared",
            when: reinstatedEvent.timestamp,
            note: reinstatedEvent?.metadata?.reason || "Account returned to active state",
          }
        : null,
    ]
      .filter(Boolean)
      .sort((left, right) => String(right.when || "").localeCompare(String(left.when || "")))
      .slice(0, 5);
    const primaryDeviceMismatch =
      Number(summary?.session_summary?.distinct_device_count ?? 0) > 1 &&
      Number(summary?.session_summary?.active_session_count ?? 0) > 1 &&
      !(summary?.session_summary?.primary_device_label || "");
    const billingConcernFlag = flags.find((flag) =>
      ["billing", "payment", "chargeback", "refund"].some((needle) =>
        `${flag?.flag_type || ""} ${flag?.summary || ""} ${flag?.source || ""}`.toLowerCase().includes(needle)
      )
    );
    const reviewPresetTemplates = {
      CUSTOM: {
        noteType: "support_note",
        noteTitle: "General review note",
        noteBody:
          "Review summary:\n- Key evidence:\n- Open questions:\n- Next step:",
        checklist: [
          "Check the current account state and review status first.",
          "Confirm whether open flags still match the current evidence.",
          "Leave a concise note before taking any stronger action.",
        ],
      },
      SUSPENSION_REVIEW: {
        noteType: "risk_note",
        noteTitle: "Suspension review note",
        noteBody:
          "Suspension review:\n- Reason currently supporting suspension:\n- Evidence checked:\n- Reinstatement blockers:\n- Recommended next move:",
        checklist: [
          "Read the latest enforcement event and suspension reason.",
          "Verify that open flags still support keeping the account suspended.",
          "Confirm whether ownership or billing verification has arrived.",
        ],
      },
      SHARING_RISK: {
        noteType: "risk_note",
        noteTitle: "Sharing-risk review note",
        noteBody:
          "Sharing-risk review:\n- Session/device pattern:\n- IP spread evidence:\n- Follow-up risk questions:\n- Recommended next move:",
        checklist: [
          "Compare active session count, distinct devices, and IP spread.",
          "Check auth-event churn for repeated sign-ins or session rotation.",
          "Decide whether the pattern supports monitoring, restriction, or escalation.",
        ],
      },
      BILLING_CONCERN: {
        noteType: "billing_note",
        noteTitle: "Billing concern review note",
        noteBody:
          "Billing concern review:\n- Subscription/payment context:\n- Billing evidence checked:\n- Outstanding risks:\n- Recommended next move:",
        checklist: [
          "Review billing-related flags, notes, and subscription status together.",
          "Confirm whether the account state still matches current payment standing.",
          "Avoid suspension until billing evidence is clear enough to support it.",
        ],
      },
    };
    const outcomeTemplates = {
      AUTO: null,
      MONITOR_ONLY: {
        noteTitle: "Monitor-only outcome note",
        noteBody:
          "Review outcome: Monitor only\n- Why stronger action is not justified yet:\n- Evidence to keep watching:\n- Next review trigger:",
      },
      RESTRICT_FOR_REVIEW: {
        noteTitle: "Restrict-for-review outcome note",
        noteBody:
          "Review outcome: Restrict for review\n- Evidence supporting restriction:\n- What is still unconfirmed:\n- Conditions for escalation or release:",
      },
      SUSPEND: {
        noteTitle: "Suspend outcome note",
        noteBody:
          "Review outcome: Suspend\n- Core evidence supporting suspension:\n- Risk if access remains active:\n- What would be needed for reinstatement:",
      },
      REINSTATE_READY: {
        noteTitle: "Reinstate-ready outcome note",
        noteBody:
          "Review outcome: Reinstate ready\n- Why reinstatement now looks justified:\n- Evidence checked:\n- Any remaining watchpoints after reinstatement:",
      },
    };
    const presetTemplate = {
      ...(reviewPresetTemplates[reviewPreset] || reviewPresetTemplates.CUSTOM),
      ...(outcomeTemplates[reviewOutcome] || {}),
    };
    const caseBadges = [
      summary?.risk_state?.account_status &&
      String(summary.risk_state.account_status).toLowerCase() !== "active"
        ? titleCase(String(summary.risk_state.account_status).replaceAll("_", " "))
        : null,
      summary?.risk_state?.risk_level ? `${titleCase(String(summary.risk_state.risk_level))} risk` : null,
      Number(summary?.open_flags_count ?? 0) > 0 ? `${summary.open_flags_count} open flags` : null,
      primaryDeviceMismatch ? "Primary device mismatch" : null,
    ].filter(Boolean);
    const recommendedNextAction = (() => {
      if (reviewOutcome === "MONITOR_ONLY") {
        return {
          title: "Hold the account in monitoring only.",
          note: "The selected operator outcome is to avoid stronger enforcement for now and keep the account under observation.",
          bullets: [
            "Leave a clear monitoring note explaining why action is being deferred.",
            "Set the next evidence trigger that would justify reopening the case.",
            "Keep the desk in website-first review mode unless new risk evidence appears.",
          ],
        };
      }
      if (reviewOutcome === "RESTRICT_FOR_REVIEW") {
        return {
          title: "Move toward restriction for review.",
          note: "The selected operator outcome is a controlled restriction rather than immediate suspension.",
          bullets: [
            "Confirm the restriction rationale is captured in a note before acting.",
            "Use restriction when the case is concerning but still incomplete.",
            "Define what evidence would move the account toward suspension or release.",
          ],
        };
      }
      if (reviewOutcome === "SUSPEND") {
        return {
          title: "Prepare for suspension.",
          note: "The selected operator outcome is suspension, so the desk should now support a high-confidence enforcement review.",
          bullets: [
            "Confirm the suspension evidence is specific enough to defend later.",
            "Make sure the enforcement trail and reason are recorded cleanly.",
            "Document what would be required before reinstatement becomes possible.",
          ],
        };
      }
      if (reviewOutcome === "REINSTATE_READY") {
        return {
          title: "Prepare for reinstatement.",
          note: "The selected operator outcome is to move the account back toward active standing.",
          bullets: [
            "Capture the evidence that now supports reinstatement.",
            "Record any remaining monitoring conditions after reinstatement.",
            "Make sure prior suspension or restriction notes are resolved cleanly.",
          ],
        };
      }
      if (String(summary?.risk_state?.account_status || "").toLowerCase() === "suspended") {
        return {
          title: "Hold suspension and verify ownership evidence.",
          note: "Keep the account suspended while you review the latest enforcement trail and any reinstatement evidence.",
          bullets: [
            "Check the enforcement timeline and operator notes for the last suspension reason.",
            "Verify whether any open flags still support keeping the account suspended.",
            "Only reinstate after ownership or billing standing is clearly confirmed.",
          ],
        };
      }
      if (primaryDeviceMismatch || String(summary?.risk_state?.review_status || "").toLowerCase() === "manual_review") {
        return {
          title: "Review possible sharing risk.",
          note: "This account looks like it may need a sharing-risk review before any stronger enforcement decision.",
          bullets: [
            "Compare active sessions, device spread, and recent auth events.",
            "Check whether open flags are clustering around session churn or IP spread.",
            "Restrict first if the pattern is concerning but not yet conclusive.",
          ],
        };
      }
      if (billingConcernFlag) {
        return {
          title: "Check billing context before enforcement.",
          note: "There is billing-shaped risk context here, so confirm payment or subscription history before moving into suspension.",
          bullets: [
            "Read any billing notes or payment-related flags first.",
            "Confirm whether the subscription state still matches the current account posture.",
            "Use restriction or note capture before suspension if the evidence is incomplete.",
          ],
        };
      }
      if (Number(summary?.open_flags_count ?? 0) > 0) {
        return {
          title: "Work through the open flags.",
          note: "This account has unresolved flags, so the clean next move is to resolve, dismiss, or escalate them deliberately.",
          bullets: [
            "Use the filters to isolate the highest-severity open flags first.",
            "Record a note before taking any stronger enforcement action.",
            "Escalate to restrict only if the open evidence now supports it.",
          ],
        };
      }
      return {
        title: "No immediate enforcement needed.",
        note: "The account currently reads as relatively clean, so keep monitoring and document anything new rather than forcing action.",
        bullets: [
          "Use the timeline and notes as a lightweight monitoring desk.",
          "Leave a note if you spot a pattern that may matter later.",
          "Avoid enforcement when the account state is currently clear.",
        ],
      };
    })();

    if (!hasKey) {
      return `
        <section class="section split">
          <article class="hero-main">
            <p class="hero-kicker">Operator review</p>
            <h1>Open the private account review desk.</h1>
            <p>This page is for internal account review only. Add the operator key locally on this device to load the review surface.</p>
          </article>
          <aside class="hero-side">
            <div class="metric">
              <span class="metric-label">Access</span>
              <span class="metric-value">Locked</span>
            </div>
            <div class="metric">
              <span class="metric-label">Use</span>
              <span class="metric-value">Internal only</span>
            </div>
          </aside>
        </section>
        <section class="section">
          ${renderNotice(state.runtime.internalLookupMessage, state.runtime.internalLookupMessage ? "warning" : "default")}
          <article class="panel">
            <h3>Operator access</h3>
            <p class="muted">Store the operator key and your operator identity only on a trusted browser while you are actively using the review desk.</p>
            <form id="internal-admin-form" class="stack-form">
              <label class="field-label" for="internal-operator-id">Operator identity</label>
              <input id="internal-operator-id" name="internal_operator_id" class="text-input" type="text" placeholder="hugh.admin" autocomplete="off" value="${escapeHtml(operatorId)}" />
              <p class="muted">This is the identity that will be stamped onto notes, restrictions, suspensions, reinstatements, and flag actions.</p>
              <label class="field-label" for="internal-admin-key">Operator key</label>
              <input id="internal-admin-key" name="internal_admin_key" class="text-input" type="password" placeholder="Paste operator key" autocomplete="off" />
              <div class="cta-row">
                <button class="button" type="submit">Save operator access</button>
              </div>
            </form>
          </article>
        </section>
      `;
    }

    return `
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Operator review</p>
          <h1>Account review desk.</h1>
          <p>Use this internal page to inspect account posture, review flags, read timeline evidence, and leave operator notes before stronger actions go live.</p>
          ${
            caseBadges.length
              ? `<div class="pill-row">${caseBadges
                  .map((badge) => `<span class="stat-chip">${escapeHtml(badge)}</span>`)
                  .join("")}</div>`
              : ""
          }
          <div class="cta-row">
            <button class="ghost-button" type="button" data-action="clear-internal-admin-key">Clear key</button>
          </div>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Review state</span>
            <span class="metric-value">${summary?.risk_state?.review_status ? escapeHtml(titleCase(String(summary.risk_state.review_status).replaceAll("_", " "))) : "Waiting"}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Risk level</span>
            <span class="metric-value">${summary?.risk_state?.risk_level ? escapeHtml(titleCase(summary.risk_state.risk_level)) : "Unknown"}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Operator</span>
            <span class="metric-value">${escapeHtml(operatorId || "Unset")}</span>
          </div>
        </aside>
      </section>
      <section class="section">
        ${renderNotice(state.runtime.internalLookupMessage, state.runtime.internalLookupMessage ? "warning" : "default")}
        ${renderNotice(state.runtime.internalReviewMessage, state.runtime.internalReviewMessage ? "success" : "default")}
        <article class="panel">
          <h3>Quick review presets</h3>
          <p class="muted">Use these saved desk modes to jump straight into the most common review patterns without rebuilding the filters manually.</p>
          <div class="filter-row">
            ${reviewPresets
              .map(
                ([value, label]) => `
                  <button
                    class="${reviewPreset === value ? "button" : "ghost-button"}"
                    type="button"
                    data-action="internal-review-preset"
                    data-value="${escapeHtml(value)}"
                  >
                    ${escapeHtml(label)}
                  </button>
                `
              )
              .join("")}
          </div>
          <div class="filter-row">
            ${reviewOutcomes
              .map(
                ([value, label]) => `
                  <button
                    class="${reviewOutcome === value ? "button" : "ghost-button"}"
                    type="button"
                    data-action="internal-review-outcome"
                    data-value="${escapeHtml(value)}"
                  >
                    ${escapeHtml(label)}
                  </button>
                `
              )
              .join("")}
          </div>
          <label class="field-label" for="internal-review-outcome-note">Outcome note</label>
          <textarea
            id="internal-review-outcome-note"
            class="text-input"
            rows="4"
            placeholder="Why does this outcome fit the current evidence?"
            data-role="internal-review-outcome-note"
          >${escapeHtml(state.runtime.internalReviewOutcomeNote || "")}</textarea>
          <article class="panel subtle-panel">
            <h4>Last operator decision</h4>
            <ul class="feature-list compact-list">
              <li>Outcome: ${escapeHtml(summary?.risk_state?.last_review_outcome ? titleCase(String(summary.risk_state.last_review_outcome).replaceAll("_", " ")) : "None yet")}</li>
              <li>Preset: ${escapeHtml(summary?.risk_state?.last_review_preset ? titleCase(String(summary.risk_state.last_review_preset).replaceAll("_", " ")) : "Not saved")}</li>
              <li>Operator: ${escapeHtml(summary?.risk_state?.last_review_outcome_by || "Unknown")}</li>
              <li>Timestamp: ${escapeHtml(formatDateTime(summary?.risk_state?.last_review_outcome_at) || "Unknown")}</li>
              <li>Note: ${escapeHtml(summary?.risk_state?.last_review_outcome_note || "No saved note yet")}</li>
            </ul>
          </article>
          <div class="cta-row">
            <button class="ghost-button" type="button" data-action="internal-save-review-outcome">
              Save review outcome
            </button>
          </div>
        </article>
        <article class="panel">
          <h3>Operator access</h3>
          <form id="internal-admin-form" class="stack-form">
            <div class="card-grid">
              <article class="panel">
                <label class="field-label" for="internal-operator-id">Operator identity</label>
                <input id="internal-operator-id" name="internal_operator_id" class="text-input" type="text" placeholder="hugh.admin" autocomplete="off" value="${escapeHtml(operatorId)}" />
                <p class="muted">Use a real operator identity so the audit trail does not fall back to the generic web shell label.</p>
              </article>
              <article class="panel">
                <label class="field-label" for="internal-admin-key">Operator key</label>
                <input id="internal-admin-key" name="internal_admin_key" class="text-input" type="password" placeholder="Stored in this browser" autocomplete="off" />
                <p class="muted">Update the key here if you rotate it later.</p>
              </article>
            </div>
            <div class="cta-row">
              <button class="button" type="submit">Save operator access</button>
            </div>
          </form>
        </article>
      </section>
      <section class="section">
        <article class="panel">
          <h3>Find account</h3>
          <form id="internal-account-lookup-form" class="stack-form">
            <div class="card-grid">
              <article class="panel">
                <label class="field-label" for="internal-lookup-email">Account email</label>
                <input id="internal-lookup-email" name="internal_lookup_email" class="text-input" type="email" placeholder="member@example.com" value="${escapeHtml(state.runtime.internalLookupEmail || "")}" />
                <p class="muted">Use email for the first lookup, then the desk will hold the account id for follow-on reads.</p>
              </article>
              <article class="panel">
                <label class="field-label" for="internal-lookup-user-id">Account id</label>
                <input id="internal-lookup-user-id" name="internal_lookup_user_id" class="text-input" type="text" placeholder="user_..." value="${escapeHtml(state.runtime.internalSelectedUserId || "")}" />
                <p class="muted">Direct id lookup is useful when you already know the account you want to review.</p>
              </article>
            </div>
            <div class="cta-row">
              <button class="button" type="submit">Load account</button>
              ${
                summary?.user?.id
                  ? `<button class="ghost-button" type="button" data-action="refresh-internal-account">Refresh desk</button>`
                  : ""
              }
            </div>
          </form>
        </article>
      </section>
      ${
        summary?.user?.id
          ? `
            <section class="section split">
              <article class="panel">
                <h3>Account summary</h3>
                <ul class="feature-list">
                  <li>Email: ${escapeHtml(summary.user.email || "Unknown")}</li>
                  <li>Account id: ${escapeHtml(summary.user.id || "Unknown")}</li>
                  <li>Membership: ${escapeHtml(summary.subscription?.subscription_status || "Unknown")}</li>
                  <li>Account status: ${escapeHtml(summary.risk_state?.account_status || summary.user.account_status || "Unknown")}</li>
                  <li>Risk level: ${escapeHtml(summary.risk_state?.risk_level || "Unknown")}</li>
                  <li>Risk score: ${escapeHtml(summary.risk_state?.risk_score ?? "0")}</li>
                  <li>Review status: ${escapeHtml(summary.risk_state?.review_status || "clear")}</li>
                  <li>Open flags: ${escapeHtml(summary.open_flags_count ?? 0)}</li>
                </ul>
              </article>
              <article class="panel">
                <h3>Session view</h3>
                <ul class="feature-list">
                  <li>Active sessions: ${escapeHtml(summary.session_summary?.active_session_count ?? 0)}</li>
                  <li>Recent sessions: ${escapeHtml(summary.session_summary?.recent_session_count ?? 0)}</li>
                  <li>Distinct devices: ${escapeHtml(summary.session_summary?.distinct_device_count ?? 0)}</li>
                  <li>Distinct IP hashes: ${escapeHtml(summary.session_summary?.distinct_ip_hash_count ?? 0)}</li>
                  <li>Primary device: ${escapeHtml(summary.session_summary?.primary_device_label || "None")}</li>
                  <li>Telegram: ${escapeHtml(summary.telegram_link?.link_status || "Not linked")}</li>
                </ul>
              </article>
            </section>
            <section class="section split">
              <article class="panel">
                <h3>Restrictions / suspensions</h3>
                <ul class="feature-list">
                  <li>Account status: ${escapeHtml(summary.risk_state?.account_status || "Unknown")}</li>
                  <li>Suspended at: ${escapeHtml(formatDateTime(summary.risk_state?.suspended_at) || "Not suspended")}</li>
                  <li>Suspension reason: ${escapeHtml(summary.risk_state?.suspension_reason || "None recorded")}</li>
                  <li>Reinstated at: ${escapeHtml(formatDateTime(summary.risk_state?.reinstated_at) || "Not reinstated")}</li>
                  <li>Reinstatement reason: ${escapeHtml(summary.risk_state?.reinstatement_reason || "None recorded")}</li>
                </ul>
              </article>
              <article class="panel">
                <h3>Review ownership</h3>
                <ul class="feature-list">
                  <li>Last reviewed at: ${escapeHtml(formatDateTime(summary.risk_state?.last_reviewed_at) || "Unknown")}</li>
                  <li>Last reviewed by: ${escapeHtml(summary.risk_state?.last_reviewed_by || "Unknown")}</li>
                  <li>Last chosen outcome: ${escapeHtml(summary.risk_state?.last_review_outcome ? titleCase(String(summary.risk_state.last_review_outcome).replaceAll("_", " ")) : "None")}</li>
                  <li>Last saved preset: ${escapeHtml(summary.risk_state?.last_review_preset ? titleCase(String(summary.risk_state.last_review_preset).replaceAll("_", " ")) : "None")}</li>
                  <li>Last risk event: ${escapeHtml(formatDateTime(summary.risk_state?.last_risk_event_at) || "Unknown")}</li>
                  <li>Open flags now: ${escapeHtml(summary.open_flags_count ?? 0)}</li>
                  <li>Risk score: ${escapeHtml(summary.risk_state?.risk_score ?? "0")}</li>
                </ul>
              </article>
            </section>
            <section class="section">
              <article class="panel">
                <h3>Recommended next action</h3>
                <strong>${escapeHtml(recommendedNextAction.title)}</strong>
                <p class="muted">${escapeHtml(recommendedNextAction.note)}</p>
                <ul class="feature-list">
                  ${recommendedNextAction.bullets
                    .map((bullet) => `<li>${escapeHtml(bullet)}</li>`)
                    .join("")}
                </ul>
              </article>
            </section>
            <section class="section split">
              <article class="panel">
                <h3>Evidence checklist</h3>
                <p class="muted">This checklist changes with the current review preset so operators can follow a steadier case workflow.</p>
                <ul class="feature-list">
                  ${presetTemplate.checklist
                    .map((item) => `<li>${escapeHtml(item)}</li>`)
                    .join("")}
                </ul>
              </article>
              <article class="panel">
                <h3>Note template</h3>
                <p class="muted">${escapeHtml(presetTemplate.noteTitle)}</p>
                <div class="cta-row">
                  <button
                    class="ghost-button"
                    type="button"
                    data-action="internal-note-template"
                    data-note-type="${escapeHtml(presetTemplate.noteType)}"
                    data-note-body="${escapeHtml(presetTemplate.noteBody)}"
                  >
                    Apply note template
                  </button>
                </div>
              </article>
            </section>
            <section class="section split">
              <article class="panel">
                <h3>Open flags</h3>
                <div class="filter-row">
                  ${severityOptions
                    .map(
                      (option) => `
                        <button
                          class="${severityFilter === option ? "button" : "ghost-button"}"
                          type="button"
                          data-action="internal-flag-filter"
                          data-value="${escapeHtml(option)}"
                        >
                          ${escapeHtml(option === "ALL" ? "All" : titleCase(option.toLowerCase()))}
                          ${escapeHtml(` (${severityCounts[option] || 0})`)}
                        </button>
                      `
                    )
                    .join("")}
                </div>
                <div class="filter-row">
                  ${statusOptions
                    .map(
                      (option) => `
                        <button
                          class="${statusFilter === option ? "button" : "ghost-button"}"
                          type="button"
                          data-action="internal-flag-status-filter"
                          data-value="${escapeHtml(option)}"
                        >
                          ${escapeHtml(option === "ALL" ? "All statuses" : titleCase(option.toLowerCase()))}
                          ${escapeHtml(` (${statusCounts[option] || 0})`)}
                        </button>
                      `
                    )
                    .join("")}
                </div>
                ${
                  filteredFlags.length
                    ? `<div class="card-grid">${filteredFlags
                        .map(
                          (flag) => `
                            <article class="panel">
                              <div class="pill-row">
                                <span class="stat-chip">${escapeHtml(flag.severity || "unknown")}</span>
                                <span class="stat-chip">${escapeHtml(flag.flag_status || "open")}</span>
                              </div>
                              <strong>${escapeHtml(flag.flag_type || "flag")}</strong>
                              <p class="muted">${escapeHtml(flag.summary || "")}</p>
                              <ul class="feature-list">
                                <li>Source: ${escapeHtml(flag.source || "system")}</li>
                                <li>Opened: ${escapeHtml(formatDateTime(flag.opened_at) || "Unknown")}</li>
                              </ul>
                              <div class="cta-row">
                                <button class="ghost-button" type="button" data-action="internal-resolve-flag" data-flag-id="${escapeHtml(flag.id)}" data-flag-type="${escapeHtml(flag.flag_type || "flag")}">Resolve</button>
                                <button class="ghost-button" type="button" data-action="internal-dismiss-flag" data-flag-id="${escapeHtml(flag.id)}" data-flag-type="${escapeHtml(flag.flag_type || "flag")}">Dismiss</button>
                              </div>
                            </article>
                          `
                        )
                        .join("")}</div>`
                    : `<div class="notice">${
                        flags.length
                          ? "No risk flags match the current filter combination."
                          : "No risk flags are open for this account."
                      }</div>`
                }
              </article>
              <article class="panel">
                <h3>Actions</h3>
                <p class="muted">Use these controls carefully. This shell now writes real backend review state and suspension decisions.</p>
                <div class="cta-row">
                  <button class="ghost-button" type="button" data-action="internal-restrict-account">Restrict</button>
                  <button class="ghost-button" type="button" data-action="internal-suspend-account">Suspend</button>
                  <button class="ghost-button" type="button" data-action="internal-reinstate-account">Reinstate</button>
                </div>
              </article>
            </section>
            <section class="section split">
              <article class="panel">
                <h3>Account history</h3>
                ${
                  accountHistoryMilestones.length
                    ? `<div class="card-grid">${accountHistoryMilestones
                        .map(
                          (item) => `
                            <article class="panel">
                              <div class="pill-row">
                                <span class="stat-chip">${escapeHtml(item.label || "Milestone")}</span>
                              </div>
                              <strong>${escapeHtml(formatDateTime(item.when) || "Unknown")}</strong>
                              <p class="muted">${escapeHtml(item.note || "")}</p>
                            </article>
                          `
                        )
                        .join("")}</div>`
                    : `<div class="notice">No internal account milestones have been recorded yet.</div>`
                }
              </article>
              <article class="panel">
                <h3>Action audit</h3>
                ${
                  actionAuditEntries.length
                    ? `<div class="card-grid">${actionAuditEntries
                        .map((item) => {
                          const actor =
                            item?.metadata?.author_id ||
                            item?.metadata?.author ||
                            item?.metadata?.operator_id ||
                            "System";
                          return `
                            <article class="panel">
                              <div class="pill-row">
                                <span class="stat-chip">${escapeHtml(item.source_type || "event")}</span>
                              </div>
                              <strong>${escapeHtml(titleCase(String(item.event_type || "event").replaceAll("_", " ")))}</strong>
                              <p class="muted">${escapeHtml(item.summary || "")}</p>
                              <ul class="feature-list">
                                <li>When: ${escapeHtml(formatDateTime(item.timestamp) || "Unknown")}</li>
                                <li>Who: ${escapeHtml(actor)}</li>
                              </ul>
                            </article>
                          `;
                        })
                        .join("")}</div>`
                    : `<div class="notice">No operator actions have been recorded for this account yet.</div>`
                }
              </article>
              <article class="panel">
                <h3>Timeline</h3>
                <div class="filter-row">
                  ${timelineSourceOptions
                    .map(
                      ([value, label]) => `
                        <button
                          class="${timelineSourceFilter === value ? "button" : "ghost-button"}"
                          type="button"
                          data-action="internal-timeline-filter"
                          data-value="${escapeHtml(value)}"
                        >
                          ${escapeHtml(label)}
                          ${escapeHtml(` (${timelineSourceCounts[value] || 0})`)}
                        </button>
                      `
                    )
                    .join("")}
                </div>
                ${
                  filteredTimeline.length
                    ? `<div class="card-grid">${filteredTimeline
                        .slice(0, 18)
                        .map(
                          (item) => `
                            <article class="panel">
                              <div class="pill-row">
                                <span class="stat-chip">${escapeHtml(item.source_type || "event")}</span>
                              </div>
                              <strong>${escapeHtml(item.event_type || "event")}</strong>
                              <p class="muted">${escapeHtml(item.summary || "")}</p>
                              <ul class="feature-list">
                                <li>When: ${escapeHtml(formatDateTime(item.timestamp) || "Unknown")}</li>
                                ${item.device_label ? `<li>Device: ${escapeHtml(item.device_label)}</li>` : ""}
                              </ul>
                            </article>
                          `
                        )
                        .join("")}</div>`
                    : `<div class="notice">${
                        timeline.length
                          ? "No timeline items match the current source filter."
                          : "No timeline items are available for this account yet."
                      }</div>`
                }
              </article>
              <article class="panel">
                <h3>Notes</h3>
                <form id="internal-note-form" class="stack-form">
                  <label class="field-label" for="internal-note-type">Note type</label>
                  <select id="internal-note-type" name="internal_note_type" class="text-input">
                    <option value="support_note">Support note</option>
                    <option value="billing_note">Billing note</option>
                    <option value="risk_note">Risk note</option>
                    <option value="reinstatement_note">Reinstatement note</option>
                  </select>
                  <p class="muted">Current preset template: ${escapeHtml(presetTemplate.noteTitle)}</p>
                  <label class="field-label" for="internal-note-content">Note</label>
                  <textarea id="internal-note-content" name="internal_note_content" class="text-input" rows="5" placeholder="Add an internal review note"></textarea>
                  <div class="cta-row">
                    <button class="button" type="submit">Add note</button>
                  </div>
                </form>
                ${
                  notes.length
                    ? `<div class="card-grid">${notes
                        .slice(0, 12)
                        .map(
                          (note) => `
                            <article class="panel">
                              <div class="pill-row">
                                <span class="stat-chip">${escapeHtml(note.note_type || "note")}</span>
                              </div>
                              <p>${escapeHtml(note.content || "")}</p>
                              <p class="muted">${escapeHtml(formatDateTime(note.created_at) || "Unknown")} ${note.author_id ? `· ${escapeHtml(note.author_id)}` : ""}</p>
                            </article>
                          `
                        )
                        .join("")}</div>`
                    : `<div class="notice">No internal notes exist for this account yet.</div>`
                }
              </article>
            </section>
          `
          : ""
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
    const sendNowEntries = matches.filter((entry) => dashboardPriorityProfile(entry).bucket === "send_now");
    const watchEntries = matches.filter((entry) => dashboardPriorityProfile(entry).bucket === "watch_closely");
    const websiteOnlyEntries = matches.filter((entry) => dashboardPriorityProfile(entry).bucket === "website_only");
    const noEdgeEntries = matches.filter((entry) => dashboardPriorityProfile(entry).bucket === "no_edge");
    const stylePreset = normalizeStylePreset(notificationPreferences?.user_style_preset);
    const onboardingCompleted = Boolean(notificationPreferences?.calm_onboarding_completed_at);
    const onboardingSummary = onboardingStepSummary(notificationPreferences);
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
      ${
        !onboardingCompleted
          ? `
            <section class="section">
              <article class="panel">
                <h3>Finish calm setup</h3>
                <p class="muted">Your dashboard is live, but your first-run setup is still shaping how sharply the feed and alerts should behave.</p>
                <div class="stats-grid">
                  ${statPanel("Progress", `${onboardingSummary.completed}/${onboardingSummary.total}`, "Preset, follows, delivery, decision support, reset mode")}
                  ${statPanel("Preset", stylePresetLabel(stylePreset), stylePresetSummary(stylePreset))}
                  ${statPanel("Current route", notificationPreferences?.telegram_enabled && !notificationPreferences?.website_only_mode ? "Selective Telegram" : "Website-first", feedRoutingExplanation(stylePreset))}
                  ${statPanel("Next move", "Finish setup", "Lock your preferred posture before relying on the feed")}
                </div>
                <div class="cta-row">
                  <a class="button" href="./onboarding.html">Complete calm setup</a>
                  <a class="ghost-button" href="./account.html#preferences">Open preferences directly</a>
                </div>
              </article>
            </section>
          `
          : ""
      }
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Dashboard</p>
          <h1>Followed intelligence.</h1>
          <p>This is your calm intelligence desk. Fixtures are grouped by league first, then opened only when the signal, timing, or follow relevance makes deeper reading worthwhile.</p>
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
          <p class="muted">Use filters to move from broad monitoring into clearer decision territory.</p>
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
          <h3>Priority map</h3>
          <p class="muted">${escapeHtml(feedRoutingExplanation(stylePreset))} The buckets below are now meant to feel like a fixture workspace rather than a flat alert list.</p>
          <div class="stats-grid">
            ${statPanel("Send now", sendNowEntries.length, "Direct fixture or team relevance")}
            ${statPanel("Watch closely", watchEntries.length, "Higher-quality deploy watchlist")}
            ${statPanel("Website only", websiteOnlyEntries.length, "Useful without interruption")}
            ${statPanel("No edge", noEdgeEntries.length, "Respectable pass or monitor states")}
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
          <h3>Send now</h3>
          <p class="muted">These are the clearest fixtures for active attention right now: direct fixture follows, direct team follows, or the strongest deploys with personal relevance.</p>
          ${renderDashboardFixtureGroups(
            sendNowEntries,
            Boolean(notificationPreferences?.telegram_enabled),
            "Nothing currently qualifies for immediate attention. That is a healthy state, not an empty one."
          )}
        </article>
      </section>

      <section class="section">
        <article class="panel">
          <h3>Watch closely</h3>
          <p class="muted">These fixtures deserve careful reading, but not urgency. Expand them for reasoning, context, and the cleaner interpretation layer.</p>
          ${renderDashboardFixtureGroups(
            watchEntries,
            Boolean(notificationPreferences?.telegram_enabled),
            "No higher-priority watchlist items are active in the current filtered window."
          )}
        </article>
      </section>

      <section class="section">
        <article class="panel">
          <h3>Website only</h3>
          <p class="muted">These fixtures still carry value, but the calmer choice is to leave them on-site and keep Telegram quiet.</p>
          ${renderDashboardFixtureGroups(
            websiteOnlyEntries,
            Boolean(notificationPreferences?.telegram_enabled),
            "No lower-priority website-only items are active in the current filtered window."
          )}
        </article>
      </section>

      <section class="section">
        <article class="panel">
          <h3>No edge / monitor</h3>
          <p class="muted">A pass state should still feel intelligent. These fixtures are here to help you leave something alone with more clarity, not less.</p>
          ${
            noEdgeEntries.length
              ? renderDashboardFixtureGroups(
                  noEdgeEntries,
                  Boolean(notificationPreferences?.telegram_enabled),
                  "No fixtures currently sit in a clear pass / monitor state for this filtered view."
                )
              : !followedSignalsConfigured
                ? `<div class="notice">No followed teams, leagues, markets, or fixtures are saved yet. Add them on the account page, then save your intelligence preferences to start matching live cards.</div>`
                : `<div class="notice">No fixtures currently sit in a clear pass / monitor state for this filtered view.</div>`
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
    const clarity = fixtureClarityProfile(fixture, matchedEntry);
    const matchReasons = Array.isArray(matchedEntry?.reasons) ? matchedEntry.reasons : [];
    const matchCopy = matchReasons.length
      ? `This fixture matches your saved follows through ${matchReasons.join(", ")}.`
      : "This fixture is being shown from the current intelligence window rather than a direct saved follow.";
    const relatedFixtures = state.fixtureIntelligence
      .filter((row) => row.fixture_key !== fixture.fixture_key && row.league === fixture.league)
      .slice(0, 4);

    return `
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Fixture Intelligence</p>
          <h1>${escapeHtml(fixture.home_team)} <span class="muted">vs</span> ${escapeHtml(fixture.away_team)}</h1>
          <p>${escapeHtml(clarity.action_copy)}</p>
          <div class="pill-row">
            <span class="pill">${escapeHtml(publishClass)}</span>
            <span class="chip">${escapeHtml(marketFamilyLabel(fixture.signal_summary?.market_family))}</span>
            <span class="chip">${escapeHtml(fixture.league)}</span>
            <span class="chip">${escapeHtml(formatKickoffLabel(fixture.kickoff_time))}</span>
            <span class="chip">${escapeHtml(clarity.feed_bucket)}</span>
          </div>
          <div class="cta-row">
            <a class="button" href="./dashboard.html">Back to dashboard</a>
            <a class="ghost-button" href="./premium.html">Open premium board</a>
            <button class="ghost-button" type="button" data-action="telegram-fixture-alert" data-fixture-key="${escapeHtml(String(fixture.fixture_key || ""))}">Send to Telegram</button>
          </div>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Action state</span>
            <span class="metric-value">${escapeHtml(clarity.action_label)}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Coverage</span>
            <span class="metric-value">${escapeHtml(String(fixture.coverage_status || "covered"))}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Alert priority</span>
            <span class="metric-value">${notificationPreferences?.telegram_enabled ? escapeHtml(fixture.follow_relevance?.notification_priority || "website only") : "Preview"}</span>
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
          <h3>${escapeHtml(clarity.meaning_title)}</h3>
          <p class="muted">${escapeHtml(clarity.meaning_copy)}</p>
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
          <h3>${escapeHtml(clarity.risk_title)}</h3>
          ${
            clarity.risk_points.length
              ? `<ul class="feature-list">${clarity.risk_points.map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>`
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

      <section class="section split">
        <article class="panel">
          <h3>Why this matches you</h3>
          <p class="muted">${escapeHtml(matchCopy)}</p>
          ${
            matchReasons.length
              ? `<div class="pill-row">${matchReasons.map((reason) => `<span class="chip">${escapeHtml(reason)}</span>`).join("")}</div>`
              : `<div class="notice">No direct saved follow is attached to this fixture right now.</div>`
          }
        </article>
        <article class="panel">
          <h3>${escapeHtml(clarity.decision_title)}</h3>
          <ul class="feature-list">
            ${clarity.decision_points.map((point) => `<li>${escapeHtml(point)}</li>`).join("")}
          </ul>
        </article>
        <article class="panel">
          <h3>Decision companion</h3>
          <ul class="feature-list">
            <li>${escapeHtml(clarity.reflection_prompt)}</li>
            <li>What could invalidate this read before kickoff?</li>
            <li>Is this structure, or are you reacting to noise?</li>
            <li>Would no action be the cleaner decision here?</li>
          </ul>
        </article>
      </section>

      <section class="section">
        <article class="panel">
          ${fixtureStandingsWidgetMarkup(fixture)}
        </article>
      </section>

      <section class="section">
        <article class="panel">
          ${fixtureLineupsWidgetMarkup(fixture)}
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
      onboarding: onboardingView,
      "internal-review": internalReviewView,
    };
    const view = views[page] || homeView;
    app.innerHTML = view();
    if (page === "fixture") {
      hydrateFixtureReferenceWidgets();
    }
  };

  const hydrateFixtureReferenceWidgets = async () => {
    const standingsRoots = Array.from(document.querySelectorAll("[data-role='fixture-standings-reference']"));
    const roots = Array.from(document.querySelectorAll("[data-role='fixture-lineups-widget']"));
    if ((!roots.length && !standingsRoots.length) || !workerConfigured()) {
      return;
    }
    const fixtureLookupCache = new Map();
    const resolveFixtureReference = async (root) => {
      const params = new URLSearchParams({
        date: root.dataset.date || "",
        home: root.dataset.home || "",
        away: root.dataset.away || "",
      });
      if (root.dataset.homeTeamId) {
        params.set("home_team_id", root.dataset.homeTeamId);
      }
      if (root.dataset.awayTeamId) {
        params.set("away_team_id", root.dataset.awayTeamId);
      }
      const cacheKey = params.toString();
      if (!fixtureLookupCache.has(cacheKey)) {
        fixtureLookupCache.set(
          cacheKey,
          fetchWorkerJson(`/api/widgets/football/fixture-lookup?${cacheKey}`, { method: "GET" })
        );
      }
      return fixtureLookupCache.get(cacheKey);
    };
    await Promise.all(
      standingsRoots.map(async (root) => {
        const frame = root.querySelector(".widget-reference-frame");
        if (!frame) {
          return;
        }
        try {
          const fixtureLookup = await resolveFixtureReference(root);
          if (!fixtureLookup?.response?.ok || !fixtureLookup?.payload?.ok) {
            throw new Error(
              fixtureLookup?.payload?.message || "League table reference could not be loaded for this fixture."
            );
          }
          const leagueId = String(fixtureLookup.payload.league_id || "").trim();
          const season = String(fixtureLookup.payload.season || "").trim();
          if (!leagueId || !season) {
            throw new Error("League table reference is not available for this fixture yet.");
          }
          const params = new URLSearchParams({ league: leagueId, season });
          const { response, payload } = await fetchWorkerJson(
            `/api/widgets/football/standings?${params.toString()}`,
            { method: "GET" }
          );
          if (!response.ok || !payload?.response?.length) {
            throw new Error(payload?.message || "League table reference could not be loaded for this fixture.");
          }
          const groups = payload.response?.[0]?.league?.standings || [];
          const tableRows = Array.isArray(groups) && groups.length ? groups[0] : [];
          if (!Array.isArray(tableRows) || !tableRows.length) {
            throw new Error("League table reference is not available for this competition yet.");
          }
          frame.innerHTML = `
            <div class="standings-reference-table">
              <div class="standings-reference-head">
                <span>Pos</span>
                <span>Team</span>
                <span>P</span>
                <span>GD</span>
                <span>Pts</span>
                <span>Form</span>
              </div>
              ${tableRows
                .slice(0, 8)
                .map(
                  (row) => `
                    <div class="standings-reference-row">
                      <span>${escapeHtml(row.rank ?? "")}</span>
                      <strong>${escapeHtml(row.team?.name || "")}</strong>
                      <span>${escapeHtml(row.all?.played ?? "")}</span>
                      <span>${escapeHtml(row.goalsDiff ?? "")}</span>
                      <span>${escapeHtml(row.points ?? "")}</span>
                      <span>${escapeHtml(row.form || "—")}</span>
                    </div>
                  `
                )
                .join("")}
            </div>
          `;
        } catch (error) {
          frame.innerHTML = `<div class="notice">${escapeHtml(
            error.message || "League table reference could not be loaded for this fixture."
          )}</div>`;
        }
      })
    );
    await Promise.all(
      roots.map(async (root) => {
        const frame = root.querySelector(".widget-reference-frame");
        if (!frame) {
          return;
        }
        try {
          const { response, payload } = await resolveFixtureReference(root);
          if (!response.ok || !payload?.ok || !payload.fixture_id) {
            throw new Error(payload?.message || "Unable to resolve fixture lineups for this page.");
          }
          const proxyBase = `${workerApiUrl("/api/widgets/football/").replace(/\/+$/, "")}/`;
          frame.innerHTML = `
            <api-sports-widget
              data-type="config"
              data-sport="football"
              data-key=""
              data-url-football="${escapeHtml(proxyBase)}"
              data-logo-url="https://media.api-sports.io"
              data-theme="grey"
              data-lang="en"
              data-show-error="true"
              data-show-logos="true"
              data-player-injuries="true"
              data-team-squad="true"
              data-game-tab="lineups"
              data-refresh="300"
            ></api-sports-widget>
            <api-sports-widget
              data-type="game"
              data-game-id="${escapeHtml(String(payload.fixture_id))}"
            ></api-sports-widget>
          `;
        } catch (error) {
          frame.innerHTML = `<div class="notice">${escapeHtml(
            error.message || "Lineups and formations are not available for this fixture yet."
          )}</div>`;
        }
      })
    );
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
    state.runtime.accountSessions = [];
    state.runtime.accountAlerts = [];
    state.runtime.accountStateError = "";
    state.runtime.accountSessionsError = "";
    state.runtime.accountSessionsMessage = "";

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

  const loadAccountSessions = async () => {
    state.runtime.accountSessions = [];
    state.runtime.accountSessionsError = "";

    if (!workerConfigured() || !state.runtime.sessionAuthenticated) {
      return;
    }

    try {
      const { response, payload } = await fetchWorkerJson("/api/account/sessions", {
        method: "GET",
        withToken: true,
      });

      if (!response.ok || !payload?.ok) {
        state.runtime.accountSessionsError = payload?.message || "Unable to load account devices.";
        return;
      }

      state.runtime.accountSessions = Array.isArray(payload.sessions) ? payload.sessions : [];
    } catch (error) {
      state.runtime.accountSessionsError = error.message || "Unable to load account devices.";
    }
  };

  const refreshSignedInAccountRuntime = async () => {
    await loadAuthSession();
    if (!state.runtime.sessionAuthenticated) {
      state.runtime.accountState = null;
      state.runtime.accountSessions = [];
      state.runtime.accountAlerts = [];
      return;
    }
    await loadAccountState();
    await loadAccountSessions();
    await loadAccountAlerts();
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
      user_style_preset: String(formData.get("user_style_preset") || "disciplined_bettor"),
      decision_companion_enabled: formData.get("decision_companion_enabled") === "on",
      reset_mode_enabled: formData.get("reset_mode_enabled") === "on",
      complete_calm_setup: formData.get("complete_calm_setup") === "on",
      language_preference: String(formData.get("language_preference") || "en-GB"),
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

  const loadInternalAccountBundle = async (userId) => {
    const targetUserId = String(userId || state.runtime.internalSelectedUserId || "").trim();
    if (!workerConfigured() || !targetUserId || !state.runtime.internalAdminKey) {
      return;
    }
    state.runtime.internalAccountSummary = null;
    state.runtime.internalFlags = [];
    state.runtime.internalNotes = [];
    state.runtime.internalTimeline = [];

    const [summaryResult, flagsResult, notesResult, timelineResult] = await Promise.all([
      fetchInternalWorkerJson(`/internal/accounts/${encodeURIComponent(targetUserId)}`),
      fetchInternalWorkerJson(`/internal/accounts/${encodeURIComponent(targetUserId)}/flags`),
      fetchInternalWorkerJson(`/internal/accounts/${encodeURIComponent(targetUserId)}/notes`),
      fetchInternalWorkerJson(`/internal/accounts/${encodeURIComponent(targetUserId)}/timeline`),
    ]);

    if (!summaryResult.response.ok || !summaryResult.payload?.ok) {
      throw new Error(summaryResult.payload?.message || "Unable to load account summary.");
    }
    if (!flagsResult.response.ok || !flagsResult.payload?.ok) {
      throw new Error(flagsResult.payload?.message || "Unable to load account flags.");
    }
    if (!notesResult.response.ok || !notesResult.payload?.ok) {
      throw new Error(notesResult.payload?.message || "Unable to load account notes.");
    }
    if (!timelineResult.response.ok || !timelineResult.payload?.ok) {
      throw new Error(timelineResult.payload?.message || "Unable to load account timeline.");
    }

    state.runtime.internalAccountSummary = summaryResult.payload.account_summary || null;
    state.runtime.internalFlags = Array.isArray(flagsResult.payload.flags) ? flagsResult.payload.flags : [];
    state.runtime.internalNotes = Array.isArray(notesResult.payload.notes) ? notesResult.payload.notes : [];
    state.runtime.internalTimeline = Array.isArray(timelineResult.payload.timeline) ? timelineResult.payload.timeline : [];
    state.runtime.internalSelectedUserId = targetUserId;
    state.runtime.internalReviewPreset = String(
      summaryResult.payload.account_summary?.risk_state?.last_review_preset || "CUSTOM"
    ).toUpperCase();
    state.runtime.internalReviewOutcome = String(
      summaryResult.payload.account_summary?.risk_state?.last_review_outcome || "AUTO"
    ).toUpperCase();
    state.runtime.internalReviewOutcomeNote = String(
      summaryResult.payload.account_summary?.risk_state?.last_review_outcome_note || ""
    );
  };

  const currentInternalOperatorId = () =>
    String(state.runtime.internalOperatorId || "")
      .replace(/\s+/g, " ")
      .trim();

  const ensureInternalOperatorIdentity = () => {
    const operatorId = currentInternalOperatorId();
    if (!operatorId) {
      return { ok: false, message: "Save your operator identity before using the review desk." };
    }
    if (operatorId.length < 5) {
      return { ok: false, message: "Operator identity must be at least 5 characters." };
    }
    if (["internal:web-shell", "internal:operator", "operator"].includes(operatorId.toLowerCase())) {
      return { ok: false, message: "Use a real operator identity instead of the generic shell label." };
    }
    return { ok: true, operatorId };
  };

  const saveInternalAdminKey = async (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    const key = String(formData.get("internal_admin_key") || "").trim();
    const operatorId = String(formData.get("internal_operator_id") || "")
      .replace(/\s+/g, " ")
      .trim();
    if (!operatorId) {
      state.runtime.internalLookupMessage = "Add your operator identity before opening the review desk.";
      render();
      return;
    }
    if (operatorId.length < 5) {
      state.runtime.internalLookupMessage = "Operator identity must be at least 5 characters.";
      render();
      return;
    }
    if (!key && !state.runtime.internalAdminKey) {
      state.runtime.internalLookupMessage = "Add the operator key before opening the review desk.";
      render();
      return;
    }
    if (key) {
      writeStoredInternalAdminKey(key);
      state.runtime.internalAdminKey = key;
    }
    writeStoredInternalOperatorId(operatorId);
    state.runtime.internalOperatorId = operatorId;
    state.runtime.internalLookupMessage = "";
    state.runtime.internalReviewMessage = "Operator access saved for this browser.";
    render();
  };

  const clearInternalAdminKey = async (event) => {
    event.preventDefault();
    writeStoredInternalAdminKey("");
    writeStoredInternalOperatorId("");
    state.runtime.internalAdminKey = "";
    state.runtime.internalOperatorId = "";
    state.runtime.internalSelectedUserId = "";
    state.runtime.internalLookupEmail = "";
    state.runtime.internalAccountSummary = null;
    state.runtime.internalFlags = [];
    state.runtime.internalNotes = [];
    state.runtime.internalTimeline = [];
    state.runtime.internalLookupMessage = "";
    state.runtime.internalReviewMessage = "Operator key cleared from this browser.";
    render();
  };

  const applyInternalReviewPreset = (preset) => {
    const nextPreset = String(preset || "CUSTOM").toUpperCase();
    state.runtime.internalReviewPreset = nextPreset;
    if (nextPreset === "SUSPENSION_REVIEW") {
      state.runtime.internalFlagSeverityFilter = "HIGH";
      state.runtime.internalFlagStatusFilter = "OPEN";
      state.runtime.internalTimelineSourceFilter = "ENFORCEMENT";
      state.runtime.internalReviewMessage = "Suspension review preset applied.";
      return;
    }
    if (nextPreset === "SHARING_RISK") {
      state.runtime.internalFlagSeverityFilter = "HIGH";
      state.runtime.internalFlagStatusFilter = "OPEN";
      state.runtime.internalTimelineSourceFilter = "AUTH_EVENT";
      state.runtime.internalReviewMessage = "Sharing risk preset applied.";
      return;
    }
    if (nextPreset === "BILLING_CONCERN") {
      state.runtime.internalFlagSeverityFilter = "MEDIUM";
      state.runtime.internalFlagStatusFilter = "OPEN";
      state.runtime.internalTimelineSourceFilter = "ADMIN_NOTE";
      state.runtime.internalReviewMessage = "Billing concern preset applied.";
      return;
    }
    state.runtime.internalFlagSeverityFilter = "ALL";
    state.runtime.internalFlagStatusFilter = "ALL";
    state.runtime.internalTimelineSourceFilter = "ALL";
    state.runtime.internalReviewMessage = "Custom review mode restored.";
  };

  const applyInternalReviewOutcome = (outcome) => {
    state.runtime.internalReviewOutcome = String(outcome || "AUTO").toUpperCase();
    state.runtime.internalReviewMessage =
      state.runtime.internalReviewOutcome === "AUTO"
        ? "Automatic review outcome restored."
        : `${titleCase(state.runtime.internalReviewOutcome.toLowerCase().replaceAll("_", " "))} outcome selected.`;
  };

  const saveInternalReviewOutcome = async (event) => {
    event.preventDefault();
    if (!workerConfigured() || !state.runtime.internalAdminKey || !state.runtime.internalSelectedUserId) {
      state.runtime.internalLookupMessage = "Load an account before saving a review outcome.";
      render();
      return;
    }
    const operatorIdentity = ensureInternalOperatorIdentity();
    if (!operatorIdentity.ok) {
      state.runtime.internalLookupMessage = operatorIdentity.message;
      render();
      return;
    }
    const outcome = String(state.runtime.internalReviewOutcome || "AUTO").toUpperCase();
    if (outcome === "AUTO") {
      state.runtime.internalLookupMessage = "Choose a review outcome before saving it to the backend trail.";
      render();
      return;
    }
    const note = String(state.runtime.internalReviewOutcomeNote || "")
      .replace(/\s+/g, " ")
      .trim();
    if (!note) {
      state.runtime.internalLookupMessage = "Add a review outcome note before saving it.";
      render();
      return;
    }
    if (note.length < 12) {
      state.runtime.internalLookupMessage = "Use at least 12 characters so the saved review outcome is meaningful.";
      render();
      return;
    }
    state.runtime.internalLookupMessage = "";
    state.runtime.internalReviewMessage = "Saving review outcome…";
    render();
    try {
      const { response, payload } = await fetchInternalWorkerJson(
        `/internal/accounts/${encodeURIComponent(state.runtime.internalSelectedUserId)}/review-outcome`,
        {
          method: "POST",
          body: {
            review_outcome: outcome.toLowerCase(),
            review_outcome_note: note,
            review_preset: String(state.runtime.internalReviewPreset || "CUSTOM").toLowerCase(),
            author_id: operatorIdentity.operatorId,
          },
        }
      );
      if (!response.ok || !payload?.ok) {
        throw new Error(payload?.message || "Unable to save the review outcome.");
      }
      if (payload.account_summary) {
        state.runtime.internalAccountSummary = payload.account_summary;
        state.runtime.internalReviewPreset = String(
          payload.account_summary?.risk_state?.last_review_preset || state.runtime.internalReviewPreset || "CUSTOM"
        ).toUpperCase();
        state.runtime.internalReviewOutcome = String(
          payload.account_summary?.risk_state?.last_review_outcome || outcome
        ).toUpperCase();
        state.runtime.internalReviewOutcomeNote = String(
          payload.account_summary?.risk_state?.last_review_outcome_note || note
        );
      }
      await loadInternalAccountBundle(state.runtime.internalSelectedUserId);
      state.runtime.internalReviewMessage = payload.message || "Review outcome saved.";
    } catch (error) {
      state.runtime.internalLookupMessage = error.message || "Unable to save the review outcome.";
      state.runtime.internalReviewMessage = "";
    }
    render();
  };

  const applyInternalNoteTemplate = (noteType, noteBody) => {
    const noteTypeElement = document.getElementById("internal-note-type");
    const noteContentElement = document.getElementById("internal-note-content");
    if (noteTypeElement && noteType) {
      noteTypeElement.value = String(noteType).trim();
    }
    if (noteContentElement && noteBody) {
      noteContentElement.value = String(noteBody);
      noteContentElement.focus();
      noteContentElement.setSelectionRange(noteContentElement.value.length, noteContentElement.value.length);
    }
    state.runtime.internalReviewMessage = "Preset note template applied.";
  };

  const lookupInternalAccount = async (event) => {
    event.preventDefault();
    if (!workerConfigured() || !state.runtime.internalAdminKey) {
      state.runtime.internalLookupMessage = "Save the operator key first.";
      render();
      return;
    }
    const formData = new FormData(event.target);
    const email = String(formData.get("internal_lookup_email") || "").trim();
    const userId = String(formData.get("internal_lookup_user_id") || "").trim();
    if (!email && !userId) {
      state.runtime.internalLookupMessage = "Add an account email or account id to continue.";
      render();
      return;
    }

    state.runtime.internalLookupMessage = "";
    state.runtime.internalReviewMessage = "Loading account review desk…";
    render();

    try {
      if (userId) {
        state.runtime.internalLookupEmail = email;
        await loadInternalAccountBundle(userId);
      } else {
        const { response, payload } = await fetchInternalWorkerJson(
          `/internal/accounts/lookup?email=${encodeURIComponent(email)}`
        );
        if (!response.ok || !payload?.ok) {
          throw new Error(payload?.message || "Unable to find that account.");
        }
        const resolvedUserId = String(payload.account_summary?.user?.id || "").trim();
        state.runtime.internalLookupEmail = email;
        state.runtime.internalSelectedUserId = resolvedUserId;
        state.runtime.internalAccountSummary = payload.account_summary || null;
        await loadInternalAccountBundle(resolvedUserId);
      }
      state.runtime.internalReviewMessage = "Account review desk loaded.";
    } catch (error) {
      state.runtime.internalLookupMessage = error.message || "Unable to load the account review desk.";
      state.runtime.internalReviewMessage = "";
    }

    render();
  };

  const refreshInternalAccount = async (event) => {
    event.preventDefault();
    if (!state.runtime.internalSelectedUserId) {
      state.runtime.internalLookupMessage = "Load an account first.";
      render();
      return;
    }
    state.runtime.internalLookupMessage = "";
    state.runtime.internalReviewMessage = "Refreshing account review desk…";
    render();
    try {
      await loadInternalAccountBundle(state.runtime.internalSelectedUserId);
      state.runtime.internalReviewMessage = "Account review desk refreshed.";
    } catch (error) {
      state.runtime.internalLookupMessage = error.message || "Unable to refresh the account review desk.";
      state.runtime.internalReviewMessage = "";
    }
    render();
  };

  const addInternalNote = async (event) => {
    event.preventDefault();
    if (!workerConfigured() || !state.runtime.internalAdminKey || !state.runtime.internalSelectedUserId) {
      state.runtime.internalLookupMessage = "Load an account before adding a note.";
      render();
      return;
    }
    const operatorIdentity = ensureInternalOperatorIdentity();
    if (!operatorIdentity.ok) {
      state.runtime.internalLookupMessage = operatorIdentity.message;
      render();
      return;
    }
    const formData = new FormData(event.target);
    const noteType = String(formData.get("internal_note_type") || "support_note").trim();
    const content = String(formData.get("internal_note_content") || "").trim();
    if (!content) {
      state.runtime.internalLookupMessage = "Write the note before saving it.";
      render();
      return;
    }

    state.runtime.internalLookupMessage = "";
    state.runtime.internalReviewMessage = "Saving note…";
    render();

    try {
      const { response, payload } = await fetchInternalWorkerJson(
        `/internal/accounts/${encodeURIComponent(state.runtime.internalSelectedUserId)}/notes`,
        {
          method: "POST",
          body: {
            note_type: noteType,
            content,
            author_id: operatorIdentity.operatorId,
          },
        }
      );
      if (!response.ok || !payload?.ok) {
        throw new Error(payload?.message || "Unable to add the note.");
      }
      await loadInternalAccountBundle(state.runtime.internalSelectedUserId);
      state.runtime.internalReviewMessage = "Note added to the account review desk.";
      event.target.reset();
    } catch (error) {
      state.runtime.internalLookupMessage = error.message || "Unable to add the note.";
      state.runtime.internalReviewMessage = "";
    }

    render();
  };

  const runInternalAccountAction = async (path, successMessage, body = {}) => {
    const { response, payload } = await fetchInternalWorkerJson(path, {
      method: "POST",
      body,
    });
    if (!response.ok || !payload?.ok) {
      throw new Error(payload?.message || "Unable to update internal review state.");
    }
    if (payload.account_summary) {
      state.runtime.internalAccountSummary = payload.account_summary;
    }
    if (Array.isArray(payload.flags)) {
      state.runtime.internalFlags = payload.flags;
    }
    if (state.runtime.internalSelectedUserId) {
      await loadInternalAccountBundle(state.runtime.internalSelectedUserId);
    }
    state.runtime.internalReviewMessage = payload.message || successMessage;
  };

  const promptInternalActionReason = (title, defaultText) => {
    const response = window.prompt(title, defaultText || "") || "";
    const reason = response.replace(/\s+/g, " ").trim();
    if (!reason) {
      return { ok: false, message: "A short review reason is required." };
    }
    if (reason.length < 12) {
      return { ok: false, message: "Use at least 12 characters so the review trail stays useful." };
    }
    return { ok: true, reason };
  };

  const restrictInternalAccount = async (event) => {
    event.preventDefault();
    if (!state.runtime.internalSelectedUserId) {
      state.runtime.internalLookupMessage = "Load an account before applying review actions.";
      render();
      return;
    }
    const operatorIdentity = ensureInternalOperatorIdentity();
    if (!operatorIdentity.ok) {
      state.runtime.internalLookupMessage = operatorIdentity.message;
      render();
      return;
    }
    const reasonPrompt = promptInternalActionReason(
      "Restriction reason",
      "Account moved into restricted review state pending further review."
    );
    if (!reasonPrompt.ok) {
      state.runtime.internalLookupMessage = reasonPrompt.message;
      render();
      return;
    }
    state.runtime.internalLookupMessage = "";
    state.runtime.internalReviewMessage = "Applying restriction…";
    render();
    try {
      await runInternalAccountAction(
        `/internal/accounts/${encodeURIComponent(state.runtime.internalSelectedUserId)}/restrict`,
        "Account restricted.",
        { reason: reasonPrompt.reason, author_id: operatorIdentity.operatorId }
      );
    } catch (error) {
      state.runtime.internalLookupMessage = error.message || "Unable to restrict this account.";
      state.runtime.internalReviewMessage = "";
    }
    render();
  };

  const suspendInternalAccount = async (event) => {
    event.preventDefault();
    if (!state.runtime.internalSelectedUserId) {
      state.runtime.internalLookupMessage = "Load an account before applying review actions.";
      render();
      return;
    }
    const operatorIdentity = ensureInternalOperatorIdentity();
    if (!operatorIdentity.ok) {
      state.runtime.internalLookupMessage = operatorIdentity.message;
      render();
      return;
    }
    const reasonPrompt = promptInternalActionReason(
      "Suspension reason",
      "Confirmed misuse after internal review."
    );
    if (!reasonPrompt.ok) {
      state.runtime.internalLookupMessage = reasonPrompt.message;
      render();
      return;
    }
    const confirmation = (window.prompt("Type SUSPEND to confirm", "") || "").trim().toUpperCase();
    if (confirmation !== "SUSPEND") {
      state.runtime.internalLookupMessage = "Suspension cancelled. Type SUSPEND exactly to continue.";
      render();
      return;
    }
    state.runtime.internalLookupMessage = "";
    state.runtime.internalReviewMessage = "Suspending account…";
    render();
    try {
      await runInternalAccountAction(
        `/internal/accounts/${encodeURIComponent(state.runtime.internalSelectedUserId)}/suspend`,
        "Account suspended.",
        { reason: reasonPrompt.reason, confirmation, author_id: operatorIdentity.operatorId }
      );
    } catch (error) {
      state.runtime.internalLookupMessage = error.message || "Unable to suspend this account.";
      state.runtime.internalReviewMessage = "";
    }
    render();
  };

  const reinstateInternalAccount = async (event) => {
    event.preventDefault();
    if (!state.runtime.internalSelectedUserId) {
      state.runtime.internalLookupMessage = "Load an account before applying review actions.";
      render();
      return;
    }
    const operatorIdentity = ensureInternalOperatorIdentity();
    if (!operatorIdentity.ok) {
      state.runtime.internalLookupMessage = operatorIdentity.message;
      render();
      return;
    }
    const reasonPrompt = promptInternalActionReason(
      "Reinstatement reason",
      "Ownership and account standing verified after review."
    );
    if (!reasonPrompt.ok) {
      state.runtime.internalLookupMessage = reasonPrompt.message;
      render();
      return;
    }
    state.runtime.internalLookupMessage = "";
    state.runtime.internalReviewMessage = "Reinstating account…";
    render();
    try {
      await runInternalAccountAction(
        `/internal/accounts/${encodeURIComponent(state.runtime.internalSelectedUserId)}/reinstate`,
        "Account reinstated.",
        { reason: reasonPrompt.reason, author_id: operatorIdentity.operatorId }
      );
    } catch (error) {
      state.runtime.internalLookupMessage = error.message || "Unable to reinstate this account.";
      state.runtime.internalReviewMessage = "";
    }
    render();
  };

  const updateInternalFlagStatus = async (event, flagId, flagType, status) => {
    event.preventDefault();
    if (!state.runtime.internalSelectedUserId) {
      state.runtime.internalLookupMessage = "Load an account before updating flags.";
      render();
      return;
    }
    const operatorIdentity = ensureInternalOperatorIdentity();
    if (!operatorIdentity.ok) {
      state.runtime.internalLookupMessage = operatorIdentity.message;
      render();
      return;
    }
    const reason = window.prompt(
      status === "dismiss"
        ? `Dismiss ${flagType || "this flag"}`
        : `Resolve ${flagType || "this flag"}`,
      status === "dismiss" ? "False positive or no further action needed." : "Review completed and resolved."
    ) || "";
    const normalizedReason = reason.replace(/\s+/g, " ").trim();
    if (!normalizedReason) {
      state.runtime.internalLookupMessage = "A short review note is required.";
      render();
      return;
    }
    if (normalizedReason.length < 12) {
      state.runtime.internalLookupMessage = "Use at least 12 characters so the review trail stays useful.";
      render();
      return;
    }
    state.runtime.internalLookupMessage = "";
    state.runtime.internalReviewMessage =
      status === "dismiss" ? "Dismissing flag…" : "Resolving flag…";
    render();
    try {
      await runInternalAccountAction(
        `/internal/accounts/${encodeURIComponent(state.runtime.internalSelectedUserId)}/flags/${encodeURIComponent(flagId)}/${status === "dismiss" ? "dismiss" : "resolve"}`,
        status === "dismiss" ? "Flag dismissed." : "Flag resolved.",
        { resolution_note: normalizedReason, author_id: operatorIdentity.operatorId }
      );
    } catch (error) {
      state.runtime.internalLookupMessage = error.message || "Unable to update the flag.";
      state.runtime.internalReviewMessage = "";
    }
    render();
  };

  const revokeAccountSession = async (event, sessionId, sessionLabel) => {
    event.preventDefault();
    if (!workerConfigured() || !state.runtime.sessionAuthenticated) {
      state.runtime.accountSessionsError = "Verify your email before managing devices.";
      render();
      return;
    }

    const label = String(sessionLabel || "this device").trim();
    if (!window.confirm(`Sign out ${label}?`)) {
      return;
    }

    state.runtime.accountSessionsError = "";
    state.runtime.accountSessionsMessage = "Updating device access…";
    render();

    try {
      const { response, payload } = await fetchWorkerJson("/api/account/sessions/revoke", {
        method: "POST",
        withToken: true,
        body: { session_id: String(sessionId || "").trim() },
      });

      if (!response.ok || !payload?.ok) {
        throw new Error(payload?.message || "Unable to revoke device session.");
      }

      if (payload.status === "account_current_session_revoked") {
        writeStoredPremiumToken("");
        state.runtime.premiumToken = "";
        await refreshSignedInAccountRuntime();
        state.runtime.authMessage = payload.message || "This device has been signed out.";
      } else {
        state.runtime.accountSessions = Array.isArray(payload.sessions) ? payload.sessions : [];
        state.runtime.accountSessionsMessage = payload.message || "Device session revoked.";
      }
    } catch (error) {
      state.runtime.accountSessionsError = error.message || "Unable to revoke device session.";
    }

    render();
  };

  const revokeOtherAccountSessions = async (event) => {
    event.preventDefault();
    if (!workerConfigured() || !state.runtime.sessionAuthenticated) {
      state.runtime.accountSessionsError = "Verify your email before managing devices.";
      render();
      return;
    }

    if (!window.confirm("Sign out all other devices and keep only this current session active?")) {
      return;
    }

    state.runtime.accountSessionsError = "";
    state.runtime.accountSessionsMessage = "Signing out other devices…";
    render();

    try {
      const { response, payload } = await fetchWorkerJson("/api/account/sessions/revoke-others", {
        method: "POST",
        withToken: true,
      });

      if (!response.ok || !payload?.ok) {
        throw new Error(payload?.message || "Unable to sign out other devices.");
      }

      state.runtime.accountSessions = Array.isArray(payload.sessions) ? payload.sessions : [];
      state.runtime.accountSessionsMessage = payload.message || "Other devices signed out.";
    } catch (error) {
      state.runtime.accountSessionsError = error.message || "Unable to sign out other devices.";
    }

    render();
  };

  const makePrimaryAccountSession = async (event, sessionId, sessionLabel) => {
    event.preventDefault();
    if (!workerConfigured() || !state.runtime.sessionAuthenticated) {
      state.runtime.accountSessionsError = "Verify your email before managing devices.";
      render();
      return;
    }

    const label = String(sessionLabel || "this device").trim();
    if (!window.confirm(`Make ${label} the primary device for this account?`)) {
      return;
    }

    state.runtime.accountSessionsError = "";
    state.runtime.accountSessionsMessage = "Updating primary device…";
    render();

    try {
      const { response, payload } = await fetchWorkerJson("/api/account/sessions/make-primary", {
        method: "POST",
        withToken: true,
        body: { session_id: String(sessionId || "").trim() },
      });

      if (!response.ok || !payload?.ok) {
        throw new Error(payload?.message || "Unable to update primary device.");
      }

      state.runtime.accountSessions = Array.isArray(payload.sessions) ? payload.sessions : [];
      state.runtime.accountSessionsMessage = payload.message || "Primary device updated.";
    } catch (error) {
      state.runtime.accountSessionsError = error.message || "Unable to update primary device.";
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
    state.runtime.accountSessions = [];
    state.runtime.accountStateError = "";
    state.runtime.accountSessionsError = "";
    state.runtime.accountSessionsMessage = "";
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
      return;
    }

    const revokeSessionTarget = event.target.closest("[data-action='revoke-account-session']");
    if (revokeSessionTarget) {
      await revokeAccountSession(
        event,
        revokeSessionTarget.dataset.sessionId,
        revokeSessionTarget.dataset.sessionLabel
      );
      return;
    }

    const revokeOtherSessionsTarget = event.target.closest("[data-action='revoke-other-sessions']");
    if (revokeOtherSessionsTarget) {
      await revokeOtherAccountSessions(event);
      return;
    }

    const makePrimaryTarget = event.target.closest("[data-action='make-primary-session']");
    if (makePrimaryTarget) {
      await makePrimaryAccountSession(
        event,
        makePrimaryTarget.dataset.sessionId,
        makePrimaryTarget.dataset.sessionLabel
      );
      return;
    }

    const clearInternalAdminKeyTarget = event.target.closest("[data-action='clear-internal-admin-key']");
    if (clearInternalAdminKeyTarget) {
      await clearInternalAdminKey(event);
      return;
    }

    const refreshInternalAccountTarget = event.target.closest("[data-action='refresh-internal-account']");
    if (refreshInternalAccountTarget) {
      await refreshInternalAccount(event);
      return;
    }

    const restrictInternalAccountTarget = event.target.closest("[data-action='internal-restrict-account']");
    if (restrictInternalAccountTarget) {
      await restrictInternalAccount(event);
      return;
    }

    const suspendInternalAccountTarget = event.target.closest("[data-action='internal-suspend-account']");
    if (suspendInternalAccountTarget) {
      await suspendInternalAccount(event);
      return;
    }

    const reinstateInternalAccountTarget = event.target.closest("[data-action='internal-reinstate-account']");
    if (reinstateInternalAccountTarget) {
      await reinstateInternalAccount(event);
      return;
    }

    const resolveInternalFlagTarget = event.target.closest("[data-action='internal-resolve-flag']");
    if (resolveInternalFlagTarget) {
      await updateInternalFlagStatus(
        event,
        resolveInternalFlagTarget.dataset.flagId,
        resolveInternalFlagTarget.dataset.flagType,
        "resolve"
      );
      return;
    }

    const dismissInternalFlagTarget = event.target.closest("[data-action='internal-dismiss-flag']");
    if (dismissInternalFlagTarget) {
      await updateInternalFlagStatus(
        event,
        dismissInternalFlagTarget.dataset.flagId,
        dismissInternalFlagTarget.dataset.flagType,
        "dismiss"
      );
      return;
    }

    const internalFlagFilterTarget = event.target.closest("[data-action='internal-flag-filter']");
    if (internalFlagFilterTarget) {
      state.runtime.internalReviewPreset = "CUSTOM";
      state.runtime.internalFlagSeverityFilter = String(
        internalFlagFilterTarget.dataset.value || "ALL"
      ).toUpperCase();
      render();
      return;
    }

    const internalFlagStatusFilterTarget = event.target.closest(
      "[data-action='internal-flag-status-filter']"
    );
    if (internalFlagStatusFilterTarget) {
      state.runtime.internalReviewPreset = "CUSTOM";
      state.runtime.internalFlagStatusFilter = String(
        internalFlagStatusFilterTarget.dataset.value || "ALL"
      ).toUpperCase();
      render();
      return;
    }

    const internalTimelineFilterTarget = event.target.closest("[data-action='internal-timeline-filter']");
    if (internalTimelineFilterTarget) {
      state.runtime.internalReviewPreset = "CUSTOM";
      state.runtime.internalTimelineSourceFilter = String(
        internalTimelineFilterTarget.dataset.value || "ALL"
      ).toUpperCase();
      render();
      return;
    }

    const internalReviewPresetTarget = event.target.closest("[data-action='internal-review-preset']");
    if (internalReviewPresetTarget) {
      state.runtime.internalLookupMessage = "";
      applyInternalReviewPreset(internalReviewPresetTarget.dataset.value || "CUSTOM");
      render();
      return;
    }

    const internalReviewOutcomeTarget = event.target.closest("[data-action='internal-review-outcome']");
    if (internalReviewOutcomeTarget) {
      state.runtime.internalLookupMessage = "";
      applyInternalReviewOutcome(internalReviewOutcomeTarget.dataset.value || "AUTO");
      render();
      return;
    }

    const internalSaveReviewOutcomeTarget = event.target.closest(
      "[data-action='internal-save-review-outcome']"
    );
    if (internalSaveReviewOutcomeTarget) {
      const outcomeNoteElement = document.querySelector("[data-role='internal-review-outcome-note']");
      state.runtime.internalReviewOutcomeNote = String(outcomeNoteElement?.value || "");
      await saveInternalReviewOutcome(event);
      return;
    }

    const internalNoteTemplateTarget = event.target.closest("[data-action='internal-note-template']");
    if (internalNoteTemplateTarget) {
      state.runtime.internalLookupMessage = "";
      applyInternalNoteTemplate(
        internalNoteTemplateTarget.dataset.noteType || "support_note",
        internalNoteTemplateTarget.dataset.noteBody || ""
      );
      render();
      return;
    }
  });

  app.addEventListener("input", (event) => {
    const internalReviewOutcomeNoteTarget = event.target.closest("[data-role='internal-review-outcome-note']");
    if (internalReviewOutcomeNoteTarget) {
      state.runtime.internalReviewOutcomeNote = String(internalReviewOutcomeNoteTarget.value || "");
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

    if (event.target.id === "preferences-form" || event.target.id === "onboarding-form") {
      await savePreferences(event);
      return;
    }

    if (event.target.id === "internal-admin-form") {
      await saveInternalAdminKey(event);
      return;
    }

    if (event.target.id === "internal-account-lookup-form") {
      await lookupInternalAccount(event);
      return;
    }

    if (event.target.id === "internal-note-form") {
      await addInternalNote(event);
    }
  });

  const boot = async () => {
    let loadingMessage = "Loading published board…";
    if (page === "account" || page === "dashboard" || page === "onboarding" || page === "internal-review") {
      loadingMessage =
        checkoutState === "success"
          ? "Membership confirmed. Please verify your email to continue…"
          : page === "dashboard"
            ? "Loading your intelligence dashboard…"
            : page === "onboarding"
              ? "Loading your onboarding flow…"
              : page === "internal-review"
                ? "Loading operator review desk…"
              : "Loading your account access…";
    } else if (page === "premium") {
      loadingMessage = "Checking premium access…";
    }
    app.innerHTML = `<div class="loading">${escapeHtml(loadingMessage)}</div>`;
    syncActiveNav();
    state.runtime.premiumToken = readStoredPremiumToken();
    state.runtime.internalAdminKey = readStoredInternalAdminKey();
    state.runtime.internalOperatorId = readStoredInternalOperatorId();
    await loadAuthSession();
    await loadAccountState();
    await loadAccountSessions();
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
