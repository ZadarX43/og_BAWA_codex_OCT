(function () {
  const DATA_ROOT = "./public/data";
  const PREMIUM_TOKEN_STORAGE_KEY = "og_premium_token";
  const MATCH_FAVOURITES_STORAGE_KEY = "og_match_favourites";
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
  const selectedFixtureTab = String(query.get("tab") || "lineups").toLowerCase();
  const selectedTeam = query.get("team") || "";
  const selectedTeamTab = String(query.get("tab") || "overview").toLowerCase();
  const selectedCompetition = query.get("competition") || "";
  const selectedCompetitionTab = String(query.get("tab") || "overview").toLowerCase();
  const matchesSearchQuery = String(query.get("q") || "").trim();
  const matchesFavouritesOnly = query.get("favs") === "1";
  const runtimeConfig = window.OG_CONFIG || {};
  const workerApiBase = String(runtimeConfig.WORKER_API_BASE || "").replace(/\/+$/, "");
  const siteDataApiBase = String(runtimeConfig.SITE_DATA_API_BASE || runtimeConfig.WORKER_API_BASE || "").replace(/\/+$/, "");
  const checkoutPlaceholderHref = "./account.html?intent=checkout";
  const FIXTURE_HERO_MEDIA_FALLBACK = {
    "2026_05_10_FC_Barcelona_Real_Madrid": {
      src: "https://www.youtube.com/embed/aAdAJEU8_E0?si=F_TvglhZgOaL1oVi",
      title: "FC Barcelona vs Real Madrid highlights",
      label: "",
      heading: "Match Highlights",
      summary: "Goals from Marcus Rashford and Ferran Torres, shown alongside full fixture read below.",
    },
  };

  const state = {
    summary: null,
    publicPredictions: [],
    premiumPredictions: [],
    securePremiumPredictions: [],
    fixtureIntelligence: [],
    weeklyResults: null,
    resultsArchive: null,
    liveResultsFeed: null,
    teamIntelligenceIndex: [],
    clubSquadIntelligenceIndex: [],
    fixtureDecisionIndex: [],
    fixtureLineupIndex: [],
    fixtureH2HIndex: [],
    selectedTeamIntelligence: null,
    selectedTeamSquadIntelligence: null,
    selectedTeamLineupSnapshot: null,
    selectedTeamExternalContent: null,
    selectedTeamExternalContentKey: "",
    selectedFixtureLineupIntelligence: null,
    selectedFixtureDecisionIntelligence: null,
    selectedFixtureDecisionSupport: null,
    selectedFixtureExternalContent: null,
    selectedFixtureExternalContentKey: "",
    selectedFixtureSiteData: null,
    selectedFixtureSiteDataKey: "",
    selectedFixtureStats: null,
    selectedFixtureStatsKey: "",
    selectedFixtureStatsError: "",
    runtime: {
      workerApiBase,
      siteDataApiBase,
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
      sessionAccessTier: "",
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
      matchFavourites: [],
      timelineExpandedFixture: "",
      timelineFixturePayloads: {},
      timelineFixturePayloadLoading: {},
      timelineFixturePayloadErrors: {},
    },
  };

  const OG_ADMIN_FEED_POSTS = [
    {
      id: "founder-window-note",
      timestamp: "2026-05-22T08:00:00Z",
      title: "Founder window note",
      summary:
        "The May 22-26 board is running through the compact Brain publish path: model output, fixture context, H2H fallback, lineup fallback, player-event cards, and public proof.",
      detail:
        "Use the timeline as the calm front door. Open fixtures when you want the full audit view; favourite anything you want to track when alerts go live.",
      cta: "Methodology",
      href: "./methodology.html",
    },
  ];

  const parseJsonResponse = async (response, path, { optional = false } = {}) => {
    const contentType = String(response.headers.get("content-type") || "").toLowerCase();
    const bodyText = await response.text();

    if (!response.ok) {
      if (optional) {
        return null;
      }
      throw new Error(`Failed to load ${path} (${response.status})`);
    }

    const trimmed = bodyText.trim();
    const looksLikeHtml = trimmed.startsWith("<!DOCTYPE") || trimmed.startsWith("<html");
    const looksLikeJson =
      contentType.includes("application/json") ||
      trimmed.startsWith("{") ||
      trimmed.startsWith("[");

    if (!looksLikeJson || looksLikeHtml) {
      if (optional) {
        return null;
      }
      throw new Error(`Expected JSON from ${path} but received HTML or non-JSON content.`);
    }

    try {
      return JSON.parse(bodyText);
    } catch (error) {
      if (optional) {
        return null;
      }
      throw new Error(`Invalid JSON in ${path}: ${error.message}`);
    }
  };

  const fetchJson = async (path) => {
    const response = await fetch(path, { cache: "no-store" });
    return parseJsonResponse(response, path);
  };

  const fetchOptionalJson = async (path) => {
    const response = await fetch(path, { cache: "no-store" });
    return parseJsonResponse(response, path, { optional: true });
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
  const readMatchFavourites = () => {
    try {
      const raw = window.localStorage.getItem(MATCH_FAVOURITES_STORAGE_KEY) || "[]";
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed.map((item) => String(item || "")).filter(Boolean) : [];
    } catch {
      return [];
    }
  };
  const writeMatchFavourites = (items) => {
    try {
      window.localStorage.setItem(MATCH_FAVOURITES_STORAGE_KEY, JSON.stringify([...new Set(items)].filter(Boolean)));
    } catch {
      return;
    }
  };
  const isMatchFavourite = (fixtureKey) =>
    state.runtime.matchFavourites.includes(String(fixtureKey || ""));
  const toggleMatchFavourite = (fixtureKey) => {
    const key = String(fixtureKey || "").trim();
    if (!key) {
      return;
    }
    const next = isMatchFavourite(key)
      ? state.runtime.matchFavourites.filter((item) => item !== key)
      : [...state.runtime.matchFavourites, key];
    state.runtime.matchFavourites = [...new Set(next)];
    writeMatchFavourites(state.runtime.matchFavourites);
  };
  const timelineFixturePayloadState = (fixtureKey) => {
    const key = String(fixtureKey || "").trim();
    return {
      data: state.runtime.timelineFixturePayloads[key] || null,
      loading: Boolean(state.runtime.timelineFixturePayloadLoading[key]),
      error: state.runtime.timelineFixturePayloadErrors[key] || "",
    };
  };
  const loadTimelineFixturePayload = async (fixtureKey) => {
    const key = String(fixtureKey || "").trim();
    if (!key || state.runtime.timelineFixturePayloads[key] || state.runtime.timelineFixturePayloadLoading[key]) {
      return;
    }
    state.runtime.timelineFixturePayloadLoading = {
      ...state.runtime.timelineFixturePayloadLoading,
      [key]: true,
    };
    state.runtime.timelineFixturePayloadErrors = {
      ...state.runtime.timelineFixturePayloadErrors,
      [key]: "",
    };
    render();
    try {
      const payload = await fetchSiteDataJson(`/api/site/fixtures/${encodeURIComponent(key)}`);
      state.runtime.timelineFixturePayloads = {
        ...state.runtime.timelineFixturePayloads,
        [key]: payload?.data || payload || null,
      };
    } catch (error) {
      state.runtime.timelineFixturePayloadErrors = {
        ...state.runtime.timelineFixturePayloadErrors,
        [key]: error.message || "fixture_brain_unavailable",
      };
    } finally {
      state.runtime.timelineFixturePayloadLoading = {
        ...state.runtime.timelineFixturePayloadLoading,
        [key]: false,
      };
      render();
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
  const siteDataApiConfigured = () => Boolean(state.runtime.siteDataApiBase);
  const premiumTokenPresent = () => Boolean(state.runtime.premiumToken);

  const workerApiUrl = (path) => {
    if (!workerConfigured()) {
      return "";
    }
    return new URL(path, `${state.runtime.workerApiBase}/`).toString();
  };

  const siteDataApiUrl = (path) => {
    if (!siteDataApiConfigured()) {
      return "";
    }
    return new URL(path, `${state.runtime.siteDataApiBase}/`).toString();
  };

  const fetchSiteDataJson = async (path) => {
    if (!siteDataApiConfigured()) {
      return null;
    }
    try {
      const response = await fetch(siteDataApiUrl(path), {
        headers: { accept: "application/json" },
        credentials: "omit",
      });
      const payload = await parseJsonResponse(response, path, { optional: true });
      return payload?.ok ? payload : null;
    } catch (error) {
      if (debugMode) {
        console.warn("Site data API fallback:", error);
      }
      return null;
    }
  };

  const fetchProtectedSiteDataJson = async (path, options = {}) => {
    if (!siteDataApiConfigured()) {
      return { response: null, payload: null };
    }
    const headers = new Headers(options.headers || {});
    headers.set("accept", "application/json");
    if (options.withToken && state.runtime.premiumToken) {
      headers.set("authorization", `Bearer ${state.runtime.premiumToken}`);
    }
    const response = await fetch(siteDataApiUrl(path), {
      method: options.method || "GET",
      headers,
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
      matches: "./matches.html",
      live: "./live.html",
      competitions: "./competitions.html",
      teams: "./teams.html",
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
    document.querySelectorAll("[data-mobile-nav]").forEach((select) => {
      if (currentHref && Array.from(select.options).some((option) => option.value === currentHref)) {
        select.value = currentHref;
      }
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

  const externalContentTeamSlug = (value) => normalizePreferenceText(value).replace(/\s+/g, "_");

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
  const teamPageHref = (teamName, tab = "overview") =>
    `./teams.html?team=${encodeURIComponent(String(teamName || ""))}&tab=${encodeURIComponent(String(tab || "overview"))}`;
  const competitionPageHref = (competitionName, tab = "overview") =>
    `./competitions.html?competition=${encodeURIComponent(String(competitionName || ""))}&tab=${encodeURIComponent(
      String(tab || "overview")
    )}`;

  const TEAM_RATING_LABELS = {
    og_power_rating: "OG Power Rating",
    attack_flow_rating: "Attack Flow",
    defensive_lock_rating: "Defensive Lock",
    goal_heat_rating: "Goal Heat",
    btts_pressure_rating: "BTTS Pressure",
    over25_heat_rating: "Over 2.5 Heat",
    control_rating: "Control Rating",
    first_strike_rating: "First Strike",
    corner_pressure_rating: "Corner Pressure",
    card_heat_rating: "Card Heat",
    chaos_rating: "Chaos Rating",
    home_fortress_rating: "Home Fortress",
    away_threat_rating: "Away Threat",
  };

  const stripClubTokens = (value) =>
    String(value || "")
      .split(" ")
      .filter(
        (token) =>
          token &&
          !["fc", "cf", "sc", "afc", "vfl", "sv", "ac", "as", "ss", "fk", "rc", "cd", "ca", "if", "bk", "sk", "nk", "ks"].includes(token)
      )
      .join(" ");

  const fixtureDateToken = (value) => {
    const raw = String(value || "").trim();
    if (!raw) return "";
    const isoMatch = raw.match(/(\d{4})[-_](\d{2})[-_](\d{2})/);
    if (isoMatch) {
      return `${isoMatch[1]} ${isoMatch[2]} ${isoMatch[3]}`;
    }
    const kickoffMatch = raw.match(/(\d{4})-(\d{2})-(\d{2})/);
    if (kickoffMatch) {
      return `${kickoffMatch[1]} ${kickoffMatch[2]} ${kickoffMatch[3]}`;
    }
    return "";
  };

  const findFixtureRowBySelectedKey = () => {
    const direct = state.fixtureIntelligence.find((row) => String(row.fixture_key || "") === String(selectedFixtureKey || ""));
    if (direct) {
      return direct;
    }
    const target = normalizePreferenceText(selectedFixtureKey);
    return state.fixtureIntelligence.find((row) => normalizePreferenceText(row.fixture_key) === target) || null;
  };

  const fixtureIndexScore = (entry, fixture, selectedKey, options = {}) => {
    let score = 0;
    let identityScore = 0;
    let teamPairScore = 0;
    const allowHistoricalPairFallback = Boolean(options.allowHistoricalPairFallback);
    const selectedKeyRaw = String(selectedKey || "").trim();
    const selectedKeyNormalized = normalizePreferenceText(selectedKeyRaw);
    const entryKeyRaw = String(entry?.fixture_key || "").trim();
    const entryKeyNormalized = normalizePreferenceText(entryKeyRaw);
    if (selectedKeyRaw && entryKeyRaw === selectedKeyRaw) {
      score += 100;
      identityScore += 100;
    } else if (selectedKeyNormalized && entryKeyNormalized === selectedKeyNormalized) {
      score += 85;
      identityScore += 85;
    }

    const fixtureId = String(fixture?.api_fixture_id || fixture?.fixture_id || "").trim();
    const entryFixtureId = String(entry?.fixture_id || "").trim();
    if (fixtureId && entryFixtureId && fixtureId === entryFixtureId) {
      score += 100;
      identityScore += 100;
    }

    const fixtureDate = fixtureDateToken(fixture?.kickoff_time || selectedKeyRaw);
    const entryDate = fixtureDateToken(entryKeyRaw);
    if (fixtureDate && entryDate && fixtureDate === entryDate) {
      score += 12;
    }

    const fixtureHome = normalizePreferenceText(fixture?.home_team || "");
    const fixtureAway = normalizePreferenceText(fixture?.away_team || "");
    const fixtureHomeLoose = stripClubTokens(fixtureHome);
    const fixtureAwayLoose = stripClubTokens(fixtureAway);
    const entryHome = normalizePreferenceText(entry?.home_team || "");
    const entryAway = normalizePreferenceText(entry?.away_team || "");
    const entryHomeLoose = stripClubTokens(entryHome);
    const entryAwayLoose = stripClubTokens(entryAway);
    const entryFixtureLabel = normalizePreferenceText(entry?.fixture || "");
    const desiredFixtureLabel = normalizePreferenceText(
      fixture?.home_team && fixture?.away_team ? `${fixture.home_team} vs ${fixture.away_team}` : ""
    );

    if (desiredFixtureLabel && entryFixtureLabel === desiredFixtureLabel) {
      score += 24;
    }
    if (fixtureHome && entryHome && fixtureHome === entryHome) {
      score += 12;
      teamPairScore += 12;
    } else if (fixtureHomeLoose && entryHomeLoose && fixtureHomeLoose === entryHomeLoose) {
      score += 8;
      teamPairScore += 8;
    }
    if (fixtureAway && entryAway && fixtureAway === entryAway) {
      score += 12;
      teamPairScore += 12;
    } else if (fixtureAwayLoose && entryAwayLoose && fixtureAwayLoose === entryAwayLoose) {
      score += 8;
      teamPairScore += 8;
    }

    if (allowHistoricalPairFallback && !identityScore && teamPairScore < 16) {
      return 0;
    }

    if (allowHistoricalPairFallback && teamPairScore > 0 && score < 24) {
      score += 8;
    }

    return score;
  };

  const findFixtureIndexRecord = (rows, fixture, selectedKey, options = {}) => {
    const matches = (rows || [])
      .map((entry) => ({ entry, score: fixtureIndexScore(entry, fixture, selectedKey, options) }))
      .filter((item) => item.score > 0)
      .sort(
        (left, right) =>
          right.score - left.score ||
          String(right.entry?.season || "").localeCompare(String(left.entry?.season || "")) ||
          String(right.entry?.fixture_key || "").localeCompare(String(left.entry?.fixture_key || ""))
      );
    const best = matches[0] || null;
    if (!best) {
      return null;
    }
    if (!options.allowHistoricalPairFallback && best.score < 24) {
      return null;
    }
    return best.entry;
  };

  const loadFixturePayloadFromIndex = async (section, indexRows, fixture, selectedKey, options = {}) => {
    const record = findFixtureIndexRecord(indexRows, fixture, selectedKey, options);
    if (!record?.fixture_key) {
      return null;
    }
    const payload = await fetchOptionalJson(`${DATA_ROOT}/${section}/${encodeURIComponent(record.fixture_key)}.json`);
    return payload ? { payload, record } : null;
  };

  const loadTeamIntelligenceFromStaticBundle = async (record) => {
    if (!record?.competition_key || !record?.season || !record?.team_slug) {
      return null;
    }
    const bundle = await fetchOptionalJson(
      `${DATA_ROOT}/team_intelligence/competitions/${record.competition_key}__${record.season}.json`
    );
    if (Array.isArray(bundle)) {
      return (
        bundle.find(
          (item) =>
            String(item?.team_slug || "") === String(record.team_slug || "") ||
            normalizePreferenceText(item?.team) === normalizePreferenceText(record.team)
        ) || null
      );
    }
    if (Array.isArray(bundle?.teams)) {
      return (
        bundle.teams.find(
          (item) =>
            String(item?.team_slug || "") === String(record.team_slug || "") ||
            normalizePreferenceText(item?.team) === normalizePreferenceText(record.team)
        ) || null
      );
    }
    const legacy = await fetchOptionalJson(
      `${DATA_ROOT}/team_intelligence/teams/${record.competition_key}/${record.season}/${record.team_slug}.json`
    );
    return legacy || null;
  };

  const loadClubSquadFromStaticPayload = async (record) => {
    if (!record?.competition_key || !record?.season || !record?.club_slug) {
      return null;
    }
    if (Array.isArray(record.players) && record.players.length) {
      return record;
    }
    const direct = await fetchOptionalJson(
      `${DATA_ROOT}/player_intelligence/clubs/${record.competition_key}/${record.season}/${record.club_slug}.json`
    );
    if (direct) {
      return direct;
    }
    return (
      state.clubSquadIntelligenceIndex.find(
        (entry) =>
          String(entry?.competition_key || "") === String(record.competition_key || "") &&
          String(entry?.season || "") === String(record.season || "") &&
          String(entry?.club_slug || "") === String(record.club_slug || "") &&
          Array.isArray(entry?.players)
      ) || null
    );
  };

  const loadTeamDetailFromSiteOrStatic = async (teamName, options = {}) => {
    const teamIndexRecord = options.teamIndexRecord || findBestTeamIntelligenceIndexRecord(teamName, options);
    const clubIndexRecord = findBestClubSquadIndexRecord(teamName, { ...options, teamIndexRecord });
    if (teamIndexRecord?.competition_key && teamIndexRecord?.team_slug) {
      const sitePayload = await fetchSiteDataJson(
        `/api/site/teams/${encodeURIComponent(teamIndexRecord.competition_key)}/${encodeURIComponent(teamIndexRecord.team_slug)}`
      );
      if (sitePayload?.data) {
        return {
          team: sitePayload.data.team || null,
          squad: sitePayload.data.squad || null,
          lineupSnapshot: sitePayload.data.lineup_snapshot || null,
          teamIndexRecord,
          clubIndexRecord,
          source: "site_api",
        };
      }
    }
    const [team, squad] = await Promise.all([
      loadTeamIntelligenceFromStaticBundle(teamIndexRecord),
      loadClubSquadFromStaticPayload(clubIndexRecord),
    ]);
    return {
      team,
      squad,
      lineupSnapshot: null,
      teamIndexRecord,
      clubIndexRecord,
      source: "static",
    };
  };

  const loadSelectedFixtureSiteData = async () => {
    if (page !== "fixture" || !selectedFixtureKey) {
      return null;
    }
    if (state.selectedFixtureSiteDataKey === selectedFixtureKey) {
      return state.selectedFixtureSiteData;
    }
    state.selectedFixtureSiteDataKey = selectedFixtureKey;
    state.selectedFixtureSiteData = null;
    const payload = await fetchSiteDataJson(`/api/site/fixtures/${encodeURIComponent(selectedFixtureKey)}`);
    state.selectedFixtureSiteData = payload?.data || null;
    return state.selectedFixtureSiteData;
  };

  const loadSelectedFixtureStats = async () => {
    if (page !== "fixture" || !selectedFixtureKey) {
      return null;
    }
    if (state.selectedFixtureStatsKey === selectedFixtureKey) {
      return state.selectedFixtureStats;
    }
    state.selectedFixtureStatsKey = selectedFixtureKey;
    state.selectedFixtureStats = null;
    state.selectedFixtureStatsError = "";
    if (!siteDataApiConfigured() || (!state.runtime.sessionAuthenticated && !premiumTokenPresent())) {
      return null;
    }
    try {
      const { response, payload } = await fetchProtectedSiteDataJson(`/api/site/fixtures/${encodeURIComponent(selectedFixtureKey)}/stats`, {
        method: "GET",
        withToken: true,
      });
      if (!response?.ok || !payload?.ok) {
        state.selectedFixtureStatsError =
          payload?.status === "tier_locked"
            ? "pro_required"
            : payload?.status || payload?.message || "premium_stats_unavailable";
        return null;
      }
      state.runtime.sessionAccessTier = String(payload.access_tier || state.runtime.sessionAccessTier || "");
      state.selectedFixtureStats = payload;
      return state.selectedFixtureStats;
    } catch (error) {
      state.selectedFixtureStatsError = error.message || "premium_stats_unavailable";
      return null;
    }
  };

  const loadSelectedFixtureExternalContent = async () => {
    if (page !== "fixture" || !selectedFixtureKey) {
      return null;
    }
    if (state.selectedFixtureExternalContentKey === selectedFixtureKey) {
      return state.selectedFixtureExternalContent;
    }
    state.selectedFixtureExternalContentKey = selectedFixtureKey;
    state.selectedFixtureExternalContent = null;
    const sitePayload = await fetchSiteDataJson(`/api/site/fixtures/${encodeURIComponent(selectedFixtureKey)}/context`);
    if (sitePayload?.data) {
      state.selectedFixtureExternalContent = { fixture_key: selectedFixtureKey, ...sitePayload.data };
      return state.selectedFixtureExternalContent;
    }
    const payload = await fetchOptionalJson(
      `${DATA_ROOT}/external_content/fixture_media/${encodeURIComponent(selectedFixtureKey)}.json`
    );
    state.selectedFixtureExternalContent = payload?.fixture_key ? payload : null;
    return state.selectedFixtureExternalContent;
  };

  const loadSelectedTeamExternalContent = async () => {
    if (page !== "teams" || !selectedTeam) {
      return null;
    }
    const slug = externalContentTeamSlug(selectedTeam);
    if (state.selectedTeamExternalContentKey === slug) {
      return state.selectedTeamExternalContent;
    }
    state.selectedTeamExternalContentKey = slug;
    state.selectedTeamExternalContent = null;
    const payload = await fetchOptionalJson(`${DATA_ROOT}/external_content/team_news/${encodeURIComponent(slug)}.json`);
    state.selectedTeamExternalContent = payload?.team_slug ? payload : null;
    return state.selectedTeamExternalContent;
  };

  const loadFixtureIntelligenceRows = async () => {
    const sitePayload = await fetchSiteDataJson("/api/site/fixtures/current?limit=200");
    if (Array.isArray(sitePayload?.fixtures) && sitePayload.fixtures.length) {
      return sitePayload.fixtures;
    }
    const fixtureIntelligence = await fetchOptionalJson(`${DATA_ROOT}/fixture_intelligence_public.json`);
    return Array.isArray(fixtureIntelligence?.fixtures) ? fixtureIntelligence.fixtures : [];
  };

  const scoreTone = (value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return "reference";
    if (numeric >= 80) return "deploy";
    if (numeric >= 55) return "observe";
    return "reference";
  };

  const ogRatingValue = (value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return null;
    if (numeric >= 3 && numeric <= 10) return numeric;
    const clamped = Math.max(0, Math.min(100, numeric));
    return 3 + clamped * 0.07;
  };

  const ogRatingBand = (rating) => {
    if (!Number.isFinite(Number(rating))) return "none";
    if (rating >= 9) return "excellent";
    if (rating >= 8) return "very-good";
    if (rating >= 7) return "good";
    if (rating >= 6.5) return "average";
    if (rating >= 6) return "below-average";
    if (rating >= 3) return "poor";
    return "none";
  };

  const renderOgRatingBadge = (value, size = "medium", label = "OG player rating") => {
    const rating = ogRatingValue(value);
    const text = rating === null ? "—" : rating.toFixed(1);
    return `<span class="og-rating-badge og-rating-badge-${escapeHtml(size)} og-rating-${escapeHtml(ogRatingBand(rating))}" title="${escapeHtml(label)}">${escapeHtml(text)}</span>`;
  };

  const lineupSupportRank = (payload = null) => {
    if (!payload) return 0;
    const status = String(payload.lineup_status || payload.lineup_mode || payload.coverage_status || "").toLowerCase();
    const profileCount =
      (Array.isArray(payload.home_lineup_profiles) ? payload.home_lineup_profiles.length : 0) +
      (Array.isArray(payload.away_lineup_profiles) ? payload.away_lineup_profiles.length : 0);
    if (status.includes("confirmed") || status.includes("official")) return 5;
    if (status.includes("predicted") || status.includes("last_fixture") || profileCount > 0) return 4;
    if (status.includes("published")) return 3;
    if (status.includes("unpublished") || status.includes("placeholder")) return 1;
    return payload.home_units || payload.away_units ? 2 : 1;
  };

  const chooseBestLineupSupport = (...payloads) =>
    payloads.filter(Boolean).sort((left, right) => lineupSupportRank(right) - lineupSupportRank(left))[0] || null;

  const h2hSupportRank = (payload = null) => {
    if (!payload) return 0;
    const fallbackMode = String(payload.fallback_mode || payload.coverage_status || "").toLowerCase();
    const sampleSize = Number(payload.sample_size || 0);
    if (sampleSize > 0 && fallbackMode.includes("historical")) return 4;
    if (sampleSize > 0) return 5;
    if (fallbackMode.includes("historical")) return 3;
    if (fallbackMode.includes("unpublished") || fallbackMode.includes("placeholder")) return 1;
    return payload.summary ? 2 : 1;
  };

  const chooseBestH2HSupport = (...payloads) =>
    payloads.filter(Boolean).sort((left, right) => h2hSupportRank(right) - h2hSupportRank(left))[0] || null;

  const playerProfilePower = (profile = null) => {
    const value =
      profile?.ratings?.og_player_power ??
      profile?.rating_power ??
      profile?.power ??
      profile?.ui?.badge_score ??
      null;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
  };

  const playerProfileRank = (profile = null) => {
    const rank = profile?.ranks?.club_rank ?? profile?.rank_club ?? profile?.club_rank ?? null;
    const numeric = Number(rank);
    return Number.isFinite(numeric) && numeric > 0 ? numeric : null;
  };

  const buildTeamSheetRatingLookup = () => {
    const lookup = new Map();
    const ambiguous = new Set();
    const addKey = (key, profile) => {
      const normalized = normalizePreferenceText(key);
      if (!normalized || !profile) return;
      if (lookup.has(normalized) && lookup.get(normalized) !== profile) {
        ambiguous.add(normalized);
        return;
      }
      lookup.set(normalized, profile);
    };
    const addProfile = (profile, fallbackTeam = "") => {
      const teamKey = normalizePreferenceText(profile?.club || profile?.team || fallbackTeam);
      const names = [
        profile?.name,
        profile?.surname,
        profile?.ui?.pitch_label,
        profile?.player_name,
      ].filter(Boolean);
      names.forEach((name) => {
        addKey(`${teamKey}|${name}`, profile);
        addKey(name, profile);
        const parts = normalizePreferenceText(name).split(" ").filter(Boolean);
        if (parts.length > 1) {
          addKey(`${teamKey}|${parts[parts.length - 1]}`, profile);
        }
      });
    };
    const lineup = state.selectedFixtureLineupIntelligence || {};
    [
      ...(Array.isArray(lineup.home_lineup_profiles) ? lineup.home_lineup_profiles : []),
      ...(Array.isArray(lineup.away_lineup_profiles) ? lineup.away_lineup_profiles : []),
      ...(Array.isArray(lineup.home_bench_profiles) ? lineup.home_bench_profiles : []),
      ...(Array.isArray(lineup.away_bench_profiles) ? lineup.away_bench_profiles : []),
    ].forEach((profile) => addProfile(profile));
    const support = state.selectedFixtureDecisionSupport || {};
    [
      support.homeSquadIntelligence,
      support.awaySquadIntelligence,
      state.selectedTeamSquadIntelligence,
    ].forEach((squad) => {
      const team = squad?.club || squad?.team || "";
      (Array.isArray(squad?.players) ? squad.players : []).forEach((profile) => addProfile(profile, team));
    });
    return { lookup, ambiguous };
  };

  const findTeamSheetRatingProfile = (ratingLookup, teamName = "", playerName = "") => {
    const teamKey = normalizePreferenceText(teamName);
    const playerKey = normalizePreferenceText(playerName);
    if (!playerKey) return null;
    const keys = [`${teamKey}|${playerKey}`, playerKey];
    const parts = playerKey.split(" ").filter(Boolean);
    if (parts.length > 1) {
      keys.push(`${teamKey}|${parts[parts.length - 1]}`);
    }
    for (const key of keys) {
      if (ratingLookup?.ambiguous?.has(key)) continue;
      const profile = ratingLookup?.lookup?.get(key);
      if (profile) return profile;
    }
    return null;
  };

  const collectGoalScorerRows = (source = {}, fixture = {}) => {
    const homeName = normalizePreferenceText(fixture.home_team || fixture.home || fixture?.teams?.home?.name);
    const awayName = normalizePreferenceText(fixture.away_team || fixture.away || fixture?.teams?.away?.name);
    const rawEvents = [
      ...(Array.isArray(source.events) ? source.events : []),
      ...(Array.isArray(source.match_events) ? source.match_events : []),
      ...(Array.isArray(source.goalscorers) ? source.goalscorers : []),
      ...(Array.isArray(source.scorers) ? source.scorers : []),
    ];
    return rawEvents
      .map((event) => {
        const type = String(event?.type || event?.event_type || event?.detail || "").toLowerCase();
        const hasGoalSignal = type.includes("goal") || event?.is_goal === true || event?.goal === true;
        if (!hasGoalSignal || type.includes("missed")) return null;
        const teamName = event?.team?.name || event?.team_name || event?.team || "";
        const teamKey = normalizePreferenceText(teamName);
        const side = teamKey && homeName && teamKey === homeName ? "home" : teamKey && awayName && teamKey === awayName ? "away" : "";
        const elapsed = event?.time?.elapsed ?? event?.minute ?? event?.time ?? "";
        const extra = event?.time?.extra ?? event?.added_time ?? "";
        const minute = elapsed ? `${elapsed}${extra ? `+${extra}` : ""}'` : "";
        return {
          side,
          player: event?.player?.name || event?.player_name || event?.scorer || "Scorer",
          minute,
        };
      })
      .filter(Boolean);
  };

  const renderFixtureScorerStrip = (scorers = []) => {
    const rows = Array.isArray(scorers) ? scorers.filter(Boolean).slice(0, 8) : [];
    if (!rows.length) return "";
    return `
      <div class="fixture-scorer-strip">
        ${rows
          .map(
            (row) => `
              <span class="fixture-scorer-pill fixture-scorer-pill-${escapeHtml(row.side || "neutral")}">
                <strong>${escapeHtml(row.player || "Scorer")}</strong>
                ${row.minute ? `<span>${escapeHtml(row.minute)}</span>` : ""}
              </span>
            `
          )
          .join("")}
      </div>
    `;
  };

  const safeTitleLabel = (value, fallback = "—") => {
    const key = String(value || "").trim();
    return key ? titleCase(key.replace(/_/g, " ")) : fallback;
  };

  const findTeamIntelligenceIndexRecord = (teamName) => {
    const target = normalizePreferenceText(teamName);
    return state.teamIntelligenceIndex.find((entry) => normalizePreferenceText(entry?.team) === target) || null;
  };

  const seasonStartLabel = (value) => {
    const raw = String(value || "").trim();
    if (!raw) return "";
    const match = raw.match(/\d{4}/);
    return match ? match[0] : raw;
  };

  const findBestTeamIntelligenceIndexRecord = (teamName, options = {}) => {
    const target = normalizePreferenceText(teamName);
    const targetCompetition = normalizePreferenceText(options.competitionName);
    const targetSeason = seasonStartLabel(options.season);
    const candidates = state.teamIntelligenceIndex
      .filter((entry) => normalizePreferenceText(entry?.team) === target)
      .map((entry) => {
        let score = 0;
        if (targetCompetition && normalizePreferenceText(entry?.competition) === targetCompetition) {
          score += 4;
        }
        if (targetSeason && seasonStartLabel(entry?.season) === targetSeason) {
          score += 6;
        }
        return { entry, score };
      })
      .sort(
        (left, right) =>
          right.score - left.score ||
          String(right.entry?.season || "").localeCompare(String(left.entry?.season || "")) ||
          String(left.entry?.competition || "").localeCompare(String(right.entry?.competition || ""))
      );
    return candidates[0]?.entry || findTeamIntelligenceIndexRecord(teamName);
  };

  const findClubSquadIndexRecord = (teamName, teamIndexRecord = null) => {
    const target = normalizePreferenceText(teamName);
    return (
      state.clubSquadIntelligenceIndex.find((entry) => {
        return (
          normalizePreferenceText(entry?.club) === target &&
          (!teamIndexRecord ||
            (String(entry?.competition_key || "") === String(teamIndexRecord?.competition_key || "") &&
              String(entry?.season || "") === String(teamIndexRecord?.season || "")))
        );
      }) || null
    );
  };

  const findBestClubSquadIndexRecord = (teamName, options = {}) => {
    const teamIndexRecord =
      options.teamIndexRecord || findBestTeamIntelligenceIndexRecord(teamName, options);
    const target = normalizePreferenceText(teamName);
    const targetCompetition = normalizePreferenceText(options.competitionName || teamIndexRecord?.competition);
    const targetSeason = seasonStartLabel(options.season || teamIndexRecord?.season);
    const candidates = state.clubSquadIntelligenceIndex
      .filter((entry) => normalizePreferenceText(entry?.club) === target)
      .map((entry) => {
        let score = 0;
        if (targetCompetition && normalizePreferenceText(entry?.competition) === targetCompetition) {
          score += 4;
        }
        if (targetSeason && seasonStartLabel(entry?.season) === targetSeason) {
          score += 6;
        }
        return { entry, score };
      })
      .sort(
        (left, right) =>
          right.score - left.score ||
          String(right.entry?.season || "").localeCompare(String(left.entry?.season || "")) ||
          String(left.entry?.competition || "").localeCompare(String(right.entry?.competition || ""))
      );
    return (
      candidates[0]?.entry ||
      findClubSquadIndexRecord(teamName, teamIndexRecord) ||
      (teamIndexRecord?.competition_key && teamIndexRecord?.season && teamIndexRecord?.team_slug
        ? {
            club: teamIndexRecord.team || teamName,
            club_slug: teamIndexRecord.team_slug,
            competition: teamIndexRecord.competition,
            competition_key: teamIndexRecord.competition_key,
            season: teamIndexRecord.season,
          }
        : null)
    );
  };

  const loadSelectedTeamIntelligence = async () => {
    state.selectedTeamIntelligence = null;
    state.selectedTeamSquadIntelligence = null;
    state.selectedTeamLineupSnapshot = null;
    state.selectedTeamExternalContent = null;
    state.selectedTeamExternalContentKey = "";
    if (page !== "teams" || !selectedTeam) {
      return;
    }

    const teamDetail = await loadTeamDetailFromSiteOrStatic(selectedTeam);
    state.selectedTeamIntelligence = teamDetail.team || null;
    state.selectedTeamSquadIntelligence = teamDetail.squad || null;
    state.selectedTeamLineupSnapshot = teamDetail.lineupSnapshot || null;
    await loadSelectedTeamExternalContent();
  };

  const loadSelectedFixtureLineupIntelligence = async () => {
    state.selectedFixtureLineupIntelligence = null;
    if (page !== "fixture" || !selectedFixtureKey) {
      return;
    }
    const siteData = await loadSelectedFixtureSiteData();
    const fixture = findFixtureRowBySelectedKey();
    const direct = await fetchOptionalJson(
      `${DATA_ROOT}/fixture_lineup_intelligence/${encodeURIComponent(selectedFixtureKey)}.json`
    );
    const resolved = await loadFixturePayloadFromIndex(
      "fixture_lineup_intelligence",
      state.fixtureLineupIndex,
      fixture,
      selectedFixtureKey
    );
    state.selectedFixtureLineupIntelligence = chooseBestLineupSupport(siteData?.lineup, direct, resolved?.payload || null);
  };

  const loadSelectedFixtureDecisionIntelligence = async () => {
    state.selectedFixtureDecisionIntelligence = null;
    state.selectedFixtureDecisionSupport = null;
    if (page !== "fixture" || !selectedFixtureKey) {
      return;
    }
    const fixture = findFixtureRowBySelectedKey();
    const siteData = await loadSelectedFixtureSiteData();
    const directDecision = await fetchOptionalJson(
      `${DATA_ROOT}/fixture_decision_intelligence/${encodeURIComponent(selectedFixtureKey)}.json`
    );
    if (directDecision) {
      state.selectedFixtureDecisionIntelligence = directDecision;
    } else if (siteData?.decision) {
      state.selectedFixtureDecisionIntelligence = siteData.decision;
    } else {
      const resolvedDecision = await loadFixturePayloadFromIndex(
        "fixture_decision_intelligence",
        state.fixtureDecisionIndex,
        fixture,
        selectedFixtureKey,
        { allowHistoricalPairFallback: false }
      );
      state.selectedFixtureDecisionIntelligence = resolvedDecision?.payload || null;
    }
    if (!fixture) {
      return;
    }
    const options = {
      competitionName: fixture.league,
      season: fixture.api_season,
    };
    const [homeTeamDetail, awayTeamDetail, directStaticH2H] = await Promise.all([
      loadTeamDetailFromSiteOrStatic(fixture.home_team, options),
      loadTeamDetailFromSiteOrStatic(fixture.away_team, options),
      fetchOptionalJson(`${DATA_ROOT}/fixture_h2h_support/${encodeURIComponent(selectedFixtureKey)}.json`),
    ]);

    let h2hSupport = chooseBestH2HSupport(siteData?.h2h, directStaticH2H);
    if (h2hSupportRank(h2hSupport) < 3) {
      const resolvedH2H = await loadFixturePayloadFromIndex(
        "fixture_h2h_support",
        state.fixtureH2HIndex,
        fixture,
        selectedFixtureKey,
        { allowHistoricalPairFallback: true }
      );
      const resolvedPayload = resolvedH2H?.payload
        ? {
            ...resolvedH2H.payload,
            fallback_mode:
              resolvedH2H.payload?.fallback_mode ||
              (resolvedH2H.record?.fixture_key !== selectedFixtureKey ? "historical_team_pair" : ""),
          }
        : null;
      h2hSupport = chooseBestH2HSupport(h2hSupport, resolvedPayload);
    }

    state.selectedFixtureDecisionSupport = {
      homeTeamIntelligence: homeTeamDetail.team || null,
      awayTeamIntelligence: awayTeamDetail.team || null,
      homeSquadIntelligence: homeTeamDetail.squad || null,
      awaySquadIntelligence: awayTeamDetail.squad || null,
      h2hSupport,
    };
  };

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
    const publishClass = String(row.publish_class || row.fixture_class || "MONITOR").toUpperCase();
    const headline =
      row.signal_summary?.headline ||
      row.signal_summary?.summary_text ||
      "This fixture is being monitored through the intelligence layer.";
    const notes = Array.isArray(row.context_summary?.notes) ? row.context_summary.notes.slice(0, 2) : [];
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

  const dashboardFixtureCard = (entry, telegramEnabled, entryIndex = 0) => {
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
      <details class="panel fixture-stream-card fixture-stream-card-${escapeHtml(deskState.tone)}" style="--enter-index:${entryIndex}" ${priority.bucket === "send_now" ? "open" : ""}>
        <summary class="fixture-stream-summary">
          <div class="fixture-stream-summary-main">
            <div class="intelligence-card-head">
              <span class="fixture-state-pill fixture-state-pill-${escapeHtml(deskState.tone)}">${escapeHtml(deskState.label)}</span>
              <span class="chip">${escapeHtml(marketFamilyLabel(row.signal_summary?.market_family))}</span>
              <span class="chip">${escapeHtml(reasonLabel)}</span>
              <span class="muted">${escapeHtml(formatKickoffLabel(row.kickoff_time))}</span>
            </div>
            <strong class="fixture-teamline dashboard-teamline">
              <span class="team-side team-side-home">
                ${badgeMarkup(row.home_team_logo_url, row.home_team)}
                <span class="team-name">${escapeHtml(teamCardName(row.home_team))}</span>
              </span>
              <span class="versus">vs</span>
              <span class="team-side team-side-away">
                <span class="team-name">${escapeHtml(teamCardName(row.away_team))}</span>
                ${badgeMarkup(row.away_team_logo_url, row.away_team)}
              </span>
            </strong>
            <p class="fixture-stream-headline">${escapeHtml(supportCopy)}</p>
          </div>
          <div class="fixture-stream-summary-side">
            <span class="fixture-route-pill fixture-route-pill-${escapeHtml(deskState.tone)}">${escapeHtml(priority.bucket === "send_now" ? "Send now" : priority.bucket === "watch_closely" ? "Watch closely" : routeValue)}</span>
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
                  ${group.items.map((entry, index) => dashboardFixtureCard(entry, telegramEnabled, index)).join("")}
                </div>
              </article>
            `
          )
          .join("")}
      </div>
    `;
  };

  const publicDeskFixtureCard = (row, entryIndex = 0) => {
    const publishClass = String(row.publish_class || row.fixture_class || "MONITOR").toUpperCase();
    const deskState = publicDeskState(row);
    const notes = Array.isArray(row.context_summary?.notes) ? row.context_summary.notes.slice(0, 2) : [];
    return `
      <details class="panel fixture-stream-card fixture-stream-card-${escapeHtml(deskState.tone)}" style="--enter-index:${entryIndex}">
        <summary class="fixture-stream-summary">
          <div class="fixture-stream-summary-main">
            <div class="intelligence-card-head">
              <span class="fixture-state-pill fixture-state-pill-${escapeHtml(deskState.tone)}">${escapeHtml(deskState.label)}</span>
              <span class="chip">${escapeHtml(marketFamilyLabel(row.signal_summary?.market_family))}</span>
              <span class="muted">${escapeHtml(formatKickoffLabel(row.kickoff_time))}</span>
            </div>
            <strong class="fixture-teamline dashboard-teamline">
              <span class="team-side team-side-home">
                ${badgeMarkup(row.home_team_logo_url, row.home_team)}
                <span class="team-name">${escapeHtml(teamCardName(row.home_team))}</span>
              </span>
              <span class="versus">vs</span>
              <span class="team-side team-side-away">
                <span class="team-name">${escapeHtml(teamCardName(row.away_team))}</span>
                ${badgeMarkup(row.away_team_logo_url, row.away_team)}
              </span>
            </strong>
            <p class="fixture-stream-headline">${escapeHtml(
              row.signal_summary?.headline || row.signal_summary?.summary_text || "Fixture intelligence update available."
            )}</p>
          </div>
          <div class="fixture-stream-summary-side">
            <span class="fixture-route-pill fixture-route-pill-${escapeHtml(deskState.tone)}">${escapeHtml(deskState.label === "DEPLOY" ? "Actionable" : "Read first")}</span>
          </div>
        </summary>
        <div class="fixture-stream-body">
          ${
            notes.length
              ? `<ul class="feature-list compact-list">${notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>`
              : ``
          }
          <div class="cta-row">
            <a class="button" href="${fixtureDetailHref(row)}">Open fixture view</a>
          </div>
        </div>
      </details>
    `;
  };

  const formatTimelineDate = (value) => {
    const parsed = new Date(value || "");
    if (Number.isNaN(parsed.getTime())) {
      return "Date pending";
    }
    return parsed.toLocaleDateString("en-GB", {
      weekday: "short",
      day: "2-digit",
      month: "short",
      year: "numeric",
    });
  };

  const fixtureLocationLabel = (row) => row?.league || "Competition pending";

  const weatherFeedSummary = (row) => {
    const notes = Array.isArray(row?.context_summary?.notes) ? row.context_summary.notes : [];
    const weatherNote = notes.find((note) => /weather|wind|rain|temperature|storm/i.test(String(note || "")));
    if (weatherNote) {
      return weatherNote;
    }
    if (row?.context_summary?.fatigue_note) {
      return "Weather pending. Rotation and travel context is already active.";
    }
    return "Weather context pending for public timeline.";
  };

  const injuryFeedSummary = (row) => {
    const notes = Array.isArray(row?.context_summary?.notes) ? row.context_summary.notes : [];
    const injuryNote = notes.find((note) => /injur|lineup|rotation|sidelined|bench/i.test(String(note || "")));
    if (injuryNote) {
      return injuryNote;
    }
    if (row?.signal_summary?.context_tags?.includes?.("lineup_pending")) {
      return "Lineup layer pending. Check back near kickoff for player movement.";
    }
    return "No public injury shock flag is attached to this fixture yet.";
  };

  const predictionFeedSummary = (row) => {
    const route = row?.deploy_summary;
    if (route?.market && route?.pick) {
      return `${marketFamilyDisplay(route.market)} ${String(route.pick).replace(/_/g, " ")} sits as the published route.`;
    }
    return row?.signal_summary?.headline || row?.signal_summary?.summary_text || "Fixture intelligence update available.";
  };

  const renderTimelineMarketChips = (row) => {
    const odds = row?.odds_summary || {};
    const chips = [
      {
        label: "FTR",
        value:
          odds.home_win_odds || odds.draw_odds || odds.away_win_odds
            ? `H ${odds.home_win_odds || "-"} / D ${odds.draw_odds || "-"} / A ${odds.away_win_odds || "-"}`
            : "Line pending",
      },
      {
        label: "OU25",
        value: odds.over25_odds || odds.under25_odds ? `O ${odds.over25_odds || "-"} / U ${odds.under25_odds || "-"}` : "Line pending",
      },
      {
        label: "BTTS",
        value: odds.btts_yes_odds || odds.btts_no_odds ? `Y ${odds.btts_yes_odds || "-"} / N ${odds.btts_no_odds || "-"}` : "Line pending",
      },
    ];
    return `
      <div class="timeline-market-strip">
        ${chips
          .map(
            (chip) => `
              <span class="timeline-market-chip">
                <b>${escapeHtml(chip.label)}</b>
                <span>${escapeHtml(chip.value)}</span>
              </span>
            `
          )
          .join("")}
      </div>
    `;
  };

  const matchSearchMatchesRow = (row, search) => {
    const target = normalizePreferenceText(
      [row?.home_team, row?.away_team, row?.league, row?.signal_summary?.market_family, row?.deploy_summary?.market].filter(Boolean).join(" ")
    );
    return !search || target.includes(normalizePreferenceText(search));
  };

  const matchesFeedRows = (rows) =>
    orderedFixtureRows(rows).filter(
      (row) =>
        matchSearchMatchesRow(row, matchesSearchQuery) &&
        (!matchesFavouritesOnly || isMatchFavourite(row.fixture_key))
    );

  const timelineLockedPanel = (tier, title, copy, features = []) => `
    <article class="timeline-tier-panel timeline-tier-locked">
      <div class="timeline-tier-panel-head">
        <span>${escapeHtml(tier)}</span>
        <strong>${escapeHtml(title)}</strong>
      </div>
      <p>${escapeHtml(copy)}</p>
      ${
        features.length
          ? `<div class="timeline-tier-pills">${features.map((feature) => `<span>${escapeHtml(feature)}</span>`).join("")}</div>`
          : ""
      }
      <a class="ghost-button" href="./pricing.html">See plans</a>
    </article>
  `;

  const timelineMetricTiles = (items = []) => `
    <div class="timeline-metric-tiles">
      ${items
        .map(
          (item) => `
            <span>
              <b>${escapeHtml(item.label)}</b>
              <strong>${escapeHtml(item.value)}</strong>
            </span>
          `
        )
        .join("")}
    </div>
  `;

  const timelineBrain = (payloadState) => {
    const data = payloadState?.data || null;
    return data?.fixture_brain || data?.brain || data || null;
  };

  const timelineBrainPayload = (section) => section?.payload || section || null;
  const timelineInjurySummary = (injury, row) => {
    const summary = injury?.summary;
    if (typeof summary === "string" && summary.trim()) {
      return summary;
    }
    if (summary && typeof summary === "object") {
      const impacts = summary.market_impacts || {};
      const impactText = ["ftr", "btts", "ou25"]
        .map((key) => impacts[key])
        .filter(Boolean)
        .map((value) => safeTitleLabel(value))
        .join(" / ");
      if (summary.warning_flag) {
        return `Injury shock warning is active${impactText ? `: ${impactText}` : "."}`;
      }
      if (impactText) {
        return `Injury market impact is tracked: ${impactText}.`;
      }
      if (summary.status) {
        return `Injury context status: ${safeTitleLabel(summary.status)}.`;
      }
    }
    return injuryFeedSummary(row);
  };

  const timelineFounderContextPanel = (row, brain, rank) => {
    if (rank < 1) {
      return timelineLockedPanel(
        "Founder / Premium",
        "Fixture context unlock",
        "Founder and above adds the context layer behind the public read: H2H, weather, lineup mode, team intelligence, and injury notes.",
        ["H2H", "Weather", "Lineups", "Team reads", "Injury notes"]
      );
    }
    const h2h = timelineBrainPayload(brain?.h2h);
    const lineup = timelineBrainPayload(brain?.lineup_context || brain?.lineup);
    const injury = brain?.injury_context || null;
    const weather = brain?.weather || null;
    const h2hItems = h2h
      ? [
          { label: "Goal heat", value: h2h.goal_heat ?? "Pending" },
          { label: "OU25 heat", value: h2h.over25_heat ?? "Pending" },
          { label: "BTTS pressure", value: h2h.btts_pressure ?? "Pending" },
          { label: "Chaos", value: h2h.chaos_rating ?? "Pending" },
        ]
      : [];
    return `
      <article class="timeline-tier-panel timeline-tier-open timeline-tier-founder">
        <div class="timeline-tier-panel-head">
          <span>Founder / Premium</span>
          <strong>Fixture context</strong>
        </div>
        <p>${escapeHtml(
          h2h?.summary ||
            row.context_summary?.volatility_note ||
            "Compact fixture context is loaded for this match. H2H and lineup fallbacks stay explicit when source depth is thin."
        )}</p>
        ${h2hItems.length ? timelineMetricTiles(h2hItems) : ""}
        <div class="timeline-context-list">
          <span><b>Weather</b>${escapeHtml(weather?.summary || weatherFeedSummary(row))}</span>
          <span><b>Lineups</b>${escapeHtml(lineup?.summary || safeTitleLabel(lineup?.lineup_status || "Lineup context pending"))}</span>
          <span><b>Injuries</b>${escapeHtml(timelineInjurySummary(injury, row))}</span>
        </div>
      </article>
    `;
  };

  const timelinePremiumPanel = (row, brain, rank) => {
    if (rank < 2) {
      return timelineLockedPanel(
        "Premium",
        "Deeper market posture",
        "Premium opens the supporting market context and team profile that explains why the public read is clean, mixed, or fragile.",
        ["Market posture", "Team context", "Contradictions"]
      );
    }
    const marketRows = Object.entries(brain?.market_cards || {})
      .map(([key, value]) => ({
        key,
        label: key === "ftr" ? "FTR" : key === "btts" ? "BTTS" : key === "ou25" ? "Over 2.5" : "Team Goals",
        lean: value?.model_lean || value?.team_context_lean || value?.selection_label || "Pending",
        state: value?.state || value?.band || "Pending",
        summary: value?.public_summary || "",
      }))
      .slice(0, 4);
    const teams = brain?.team_context || null;
    const homeRating = teams?.home?.meta?.headline_rating;
    const awayRating = teams?.away?.meta?.headline_rating;
    return `
      <article class="timeline-tier-panel timeline-tier-open timeline-tier-premium">
        <div class="timeline-tier-panel-head">
          <span>Premium</span>
          <strong>Market posture</strong>
        </div>
        <div class="timeline-market-read-grid">
          ${marketRows
            .map(
              (item) => `
                <span>
                  <b>${escapeHtml(item.label)}</b>
                  <strong>${escapeHtml(item.lean)}</strong>
                  <small>${escapeHtml(item.state)}</small>
                </span>
              `
            )
            .join("")}
        </div>
        <p>${escapeHtml(marketRows.find((item) => item.summary)?.summary || row.context_summary?.volatility_note || "Premium context is available for this fixture.")}</p>
        ${
          homeRating || awayRating
            ? timelineMetricTiles([
                { label: `${teamCardName(row.home_team)} rating`, value: homeRating ?? "Pending" },
                { label: `${teamCardName(row.away_team)} rating`, value: awayRating ?? "Pending" },
              ])
            : ""
        }
      </article>
    `;
  };

  const timelinePlayerEventPanel = (brain, rank) => {
    if (rank < 3) {
      return timelineLockedPanel(
        "Pro",
        "Player-event watchlists",
        "Pro adds pre-lineup player-event cards for shots, SOT, tackles, fouls, player fouled, key passes, keeper saves, and bookings.",
        ["Shots", "SOT", "Tackles", "Fouls", "Fouled", "Key passes", "Saves", "Bookings"]
      );
    }
    const source = brain?.player_event_cards || null;
    const cards = Array.isArray(source?.cards) ? source.cards.slice(0, 4) : [];
    if (!cards.length) {
      return `
        <article class="timeline-tier-panel timeline-tier-open timeline-tier-pro">
          <div class="timeline-tier-panel-head">
            <span>Pro</span>
            <strong>Player-event watchlists</strong>
          </div>
          <p>Pro access is active, but this fixture has not published player-event cards yet. They will appear after the next player-event compiler refresh.</p>
        </article>
      `;
    }
    return `
      <article class="timeline-tier-panel timeline-tier-open timeline-tier-pro">
        <div class="timeline-tier-panel-head">
          <span>Pro · ${escapeHtml(playerEventPhaseLabel(source.phase))}</span>
          <strong>Player-event watchlists</strong>
        </div>
        <div class="timeline-player-card-grid">
          ${cards
            .map(
              (card) => `
                <section>
                  <div>
                    <b>${escapeHtml(card.card_title || playerEventFamilyTitle(card.event_family))}</b>
                    <small>${escapeHtml(safeTitleLabel(card.lineup_status || source.lineup_status || "Preview"))}</small>
                  </div>
                  <ul>
                    ${(card.shortlist || []).slice(0, 3).map((item) => renderPlayerEventCandidate(item)).join("")}
                  </ul>
                </section>
              `
            )
            .join("")}
        </div>
      </article>
    `;
  };

  const timelineAuditPanel = (brain, rank) => {
    if (rank < 4) {
      return timelineLockedPanel(
        "Pro+",
        "Audit and explainability",
        "Pro+ opens the compact audit trail: route/audit split, source freshness, coverage flags, and downloadable debug payload references.",
        ["Route audit", "Freshness", "Coverage", "Source refs"]
      );
    }
    const decision = timelineBrainPayload(brain?.decision);
    const audit = routeAuditProfile(brain?.fixture_core?.payload || {}, decision);
    const coverage = brain?.coverage || {};
    const freshness = brain?.freshness || {};
    const sourceRefs = brain?.source_refs || {};
    return `
      <article class="timeline-tier-panel timeline-tier-open timeline-tier-pro-plus">
        <div class="timeline-tier-panel-head">
          <span>Pro+</span>
          <strong>Audit dashboard</strong>
        </div>
        ${timelineMetricTiles([
          { label: "Route", value: audit.routeLabel || "Pending" },
          { label: "Audit", value: audit.auditState || "Pending" },
          { label: "Agreement", value: audit.agreement != null ? `${Math.round(Number(audit.agreement))}%` : "Pending" },
          { label: "Freshness", value: freshness.coverage_status || "Compiled" },
        ])}
        <div class="timeline-tier-pills">
          ${Object.entries(coverage)
            .slice(0, 8)
            .map(([key, value]) => `<span>${escapeHtml(`${safeTitleLabel(key)}: ${value ? "yes" : "no"}`)}</span>`)
            .join("")}
        </div>
        <p>${escapeHtml(sourceRefs.site_db ? "Source identity and compact payload references are attached." : "Compact audit source references are pending for this fixture.")}</p>
      </article>
    `;
  };

  const timelineTierPanels = (row) => {
    const payloadState = timelineFixturePayloadState(row.fixture_key);
    const brain = timelineBrain(payloadState);
    const rank = accessTierRank(currentAccessTier());
    if (payloadState.loading) {
      return `<div class="timeline-tier-loading">Loading compact fixture brain...</div>`;
    }
    if (payloadState.error) {
      return `<div class="notice">Could not load the compact fixture brain yet: ${escapeHtml(payloadState.error)}</div>`;
    }
    if (!brain) {
      return `<div class="timeline-tier-loading">Open this read to load the compact fixture brain.</div>`;
    }
    return `
      <div class="timeline-tier-stack">
        <article class="timeline-tier-panel timeline-tier-standard">
          <div class="timeline-tier-panel-head">
            <span>Standard</span>
            <strong>Public read</strong>
          </div>
          <p>${escapeHtml(row.signal_summary?.summary_text || predictionFeedSummary(row))}</p>
        </article>
        ${timelineFounderContextPanel(row, brain, rank)}
        ${timelinePremiumPanel(row, brain, rank)}
        ${timelinePlayerEventPanel(brain, rank)}
        ${timelineAuditPanel(brain, rank)}
      </div>
    `;
  };

  const adminTimelinePost = (post, index = 0) => `
    <article class="x-feed-post x-feed-admin" style="--enter-index:${index}">
      <div class="x-post-rail">
        <span class="x-admin-avatar">OG</span>
      </div>
      <div class="x-post-main">
        <div class="x-post-meta">
          <strong>Odds Genius</strong>
          <span>@oddsgenius</span>
          <span>${escapeHtml(formatKickoffLabel(post.timestamp))}</span>
        </div>
        <h2>${escapeHtml(post.title)}</h2>
        <p>${escapeHtml(post.summary)}</p>
        <details class="x-post-expand">
          <summary>Open update</summary>
          <div>
            <p>${escapeHtml(post.detail)}</p>
            <a class="ghost-button" href="${escapeHtml(post.href)}">${escapeHtml(post.cta || "Open")}</a>
          </div>
        </details>
      </div>
    </article>
  `;

  const matchTimelinePost = (row, index = 0) => {
    const deskState = publicDeskState(row);
    const favourite = isMatchFavourite(row.fixture_key);
    const expanded = state.runtime.timelineExpandedFixture === row.fixture_key;
    const home = teamCardName(row.home_team);
    const away = teamCardName(row.away_team);
    return `
      <article class="x-feed-post x-feed-fixture x-feed-fixture-${escapeHtml(deskState.tone)}" style="--enter-index:${index}">
        <div class="x-post-rail">
          ${badgeMarkup(row.home_team_logo_url, row.home_team, "x-team-avatar")}
          <span class="x-post-rail-line"></span>
          ${badgeMarkup(row.away_team_logo_url, row.away_team, "x-team-avatar")}
        </div>
        <div class="x-post-main">
          <div class="x-post-meta">
            <strong>${escapeHtml(home)} x ${escapeHtml(away)}</strong>
            <span>${escapeHtml(formatKickoffLabel(row.kickoff_time))}</span>
          </div>
          <div class="x-post-title-row">
            <h2>${escapeHtml(row.home_team)} x ${escapeHtml(row.away_team)}</h2>
            <button
              class="x-fav-button ${favourite ? "is-active" : ""}"
              type="button"
              data-action="toggle-match-favourite"
              data-fixture-key="${escapeHtml(row.fixture_key)}"
              aria-label="${favourite ? "Remove fixture from favourites" : "Add fixture to favourites"}"
              title="${favourite ? "Remove from favourites" : "Add to favourites"}"
            >${favourite ? "♥" : "♡"}</button>
          </div>
          <p class="x-post-location">${escapeHtml(formatTimelineDate(row.kickoff_time))} / ${escapeHtml(fixtureLocationLabel(row))}</p>
          <p class="x-post-summary">${escapeHtml(predictionFeedSummary(row))}</p>
          ${renderTimelineMarketChips(row)}
          <div class="x-post-context-grid">
            <div>
              <span class="x-context-label">Weather</span>
              <p>${escapeHtml(weatherFeedSummary(row))}</p>
            </div>
            <div>
              <span class="x-context-label">Player injury news</span>
              <p>${escapeHtml(injuryFeedSummary(row))}</p>
            </div>
          </div>
          <details class="x-post-expand" ${expanded ? "open" : ""}>
            <summary data-action="timeline-expand" data-fixture-key="${escapeHtml(row.fixture_key)}">Expand match read</summary>
            <div>
              <p>${escapeHtml(row.signal_summary?.summary_text || row.context_summary?.volatility_note || predictionFeedSummary(row))}</p>
              ${
                Array.isArray(row.context_summary?.notes) && row.context_summary.notes.length
                  ? `<ul class="feature-list compact-list">${row.context_summary.notes
                      .slice(0, 4)
                      .map((note) => `<li>${escapeHtml(note)}</li>`)
                      .join("")}</ul>`
                  : ""
              }
              ${expanded ? timelineTierPanels(row) : ""}
              <div class="x-post-actions">
                <a class="button" href="${fixtureDetailHref(row)}">Open full fixture</a>
                <a class="ghost-button" href="./premium.html">Unlock deeper cards</a>
              </div>
              <p class="x-swipe-note">On mobile, open the full fixture and use the browser back gesture to slide back into this timeline.</p>
            </div>
          </details>
        </div>
      </article>
    `;
  };

  const renderMatchesTimeline = (rows) => {
    const fixturePosts = matchesFeedRows(rows);
    const fixtureItems = fixturePosts
      .map((row) => ({ type: "fixture", timestamp: row.fixture_kickoff_at || row.kickoff_time, row }))
      .sort((left, right) => kickoffTimestamp(left.timestamp) - kickoffTimestamp(right.timestamp));
    const adminItems = OG_ADMIN_FEED_POSTS.map((post) => ({ type: "admin", timestamp: post.timestamp, post }));
    const posts = fixtureItems.length ? [fixtureItems[0], ...adminItems, ...fixtureItems.slice(1)] : adminItems;
    if (!posts.length) {
      return `<div class="notice">No timeline posts match this search yet.</div>`;
    }
    return `
      <div class="x-feed-shell">
        ${posts
          .map((item, index) =>
            item.type === "admin" ? adminTimelinePost(item.post, index) : matchTimelinePost(item.row, index)
          )
          .join("")}
      </div>
    `;
  };

  const matchesBottomNav = () => `
    <nav class="x-bottom-nav" aria-label="Matches timeline navigation">
      <a href="./index.html"><span>Home</span></a>
      <a href="./matches.html#matches-search"><span>Search</span></a>
      <a href="./premium.html"><span>OG GPT</span></a>
      <a href="./matches.html?favs=1"><span>Favs</span></a>
      <button type="button" data-action="history-back" aria-label="Back page"><span>&lt;</span></button>
      <button type="button" data-action="history-forward" aria-label="Forward page"><span>&gt;</span></button>
    </nav>
  `;

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
                  ${group.items.map((row, index) => publicDeskFixtureCard(row, index)).join("")}
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

  const formatOdds = (value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return "N/A";
    }
    return numeric.toFixed(2).replace(/\.?0+$/, "");
  };

  const impliedProbability = (odds) => {
    const numeric = Number(odds);
    if (!Number.isFinite(numeric) || numeric <= 0) {
      return null;
    }
    return 1 / numeric;
  };

  const formatImpliedProbability = (odds) => {
    const implied = impliedProbability(odds);
    return implied == null ? "N/A" : `${Math.round(implied * 100)}%`;
  };

  const edgeLabel = (row) => row.value_edge_display || row.value_edge || "N/A";
  const confidenceLabel = (row) => row.display_confidence || row.model_prob_display || formatProbability(row.model_prob);
  const tierClass = (tier) => (String(tier || "").toUpperCase() === "STANDARD" ? "standard" : "elite");
  const shouldShowTierChip = (tier) => {
    const value = String(tier || "").toUpperCase();
    return Boolean(value) && value !== "ELITE";
  };
  const edgePointsValue = (row) => {
    const raw = row?.value_edge_display ?? row?.value_edge ?? "";
    const match = String(raw).match(/-?\d+(?:\.\d+)?/);
    if (!match) {
      return null;
    }
    const numeric = Number(match[0]);
    return Number.isFinite(numeric) ? numeric : null;
  };
  const predictionEdgeTone = (row) => {
    const points = edgePointsValue(row);
    if (points == null) {
      return "neutral";
    }
    if (points < 0) {
      return "negative";
    }
    if (points >= 10) {
      return "strong";
    }
    if (points >= 5) {
      return "positive";
    }
    return "marginal";
  };
  const cardReasonText = (row) => {
    const reason = String(row?.short_reason || row?.human_reason || "").trim();
    if (!reason) {
      return "";
    }
    const normalized = reason.toLowerCase();
    if (
      normalized === "cleared value-edge threshold vs market price." ||
      normalized === "qualified premium play." ||
      normalized.includes("cleared live routing checks")
    ) {
      return "";
    }
    return reason;
  };
  const compactMetricText = (row) => {
    const parts = [];
    if (row?.bookie_od != null && row?.bookie_od !== "") {
      parts.push(String(row.bookie_od));
    }
    const confidence = String(confidenceLabel(row) || "").trim();
    if (confidence && confidence !== "N/A") {
      parts.push(confidence);
    }
    if (shouldShowTierChip(row?.confidence_tier)) {
      parts.push(String(row.confidence_tier).trim());
    }
    return parts.join(" · ");
  };
  const teamCardName = (value) =>
    String(value || "")
      .replace(/\s+FC$/i, "")
      .replace(/\s+CF$/i, "")
      .replace(/\s+SC$/i, "")
      .replace(/\s+SV$/i, "")
      .replace(/\s+Revolution$/i, "")
      .replace(/\s+Union$/i, "")
      .trim();

  const marketFamilyDisplay = (value) => {
    const family = String(value || "").toUpperCase();
    if (family === "OU25") {
      return "Over 2.5";
    }
    return marketFamilyLabel(family);
  };

  const deployPickDisplay = (value) => {
    const pick = String(value || "").toUpperCase();
    if (pick === "HOME") {
      return "Home";
    }
    if (pick === "DRAW") {
      return "Draw";
    }
    if (pick === "AWAY") {
      return "Away";
    }
    if (pick === "OVER25") {
      return "Over 2.5";
    }
    if (pick === "UNDER25") {
      return "Under 2.5";
    }
    return pick || "Read pending";
  };

  const confidenceBandDisplay = (tier) => {
    const value = String(tier || "").toUpperCase();
    if (value === "ELITE") {
      return "Elite confidence";
    }
    if (value === "STANDARD") {
      return "Standard confidence";
    }
    return "Watch-first confidence";
  };

  const valueEdgeTone = (fixture) => {
    const label = String(fixture?.deploy_summary?.value_edge_label || "").toLowerCase();
    if (label === "positive") {
      return "positive";
    }
    if (label === "negative") {
      return "fragile";
    }
    return "neutral";
  };

  const valueEdgeDisplay = (fixture) => {
    const tone = valueEdgeTone(fixture);
    if (tone === "positive") {
      return "Positive edge";
    }
    if (tone === "fragile") {
      return "Fragility active";
    }
    return "Public edge pending";
  };

  const hasUsableOdds = (value) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) && numeric > 1;
  };

  const marketVerdictDisplay = (fixture) => {
    const family = String(fixture?.signal_summary?.market_family || "").toUpperCase();
    const pick = String(fixture?.signal_summary?.deploy_pick || fixture?.deploy_summary?.pick || "").toUpperCase();
    if (family === "OU25") {
      if (pick === "OVER25") {
        return "Over 2.5";
      }
      if (pick === "UNDER25") {
        return "Under 2.5";
      }
      return "Totals read pending";
    }
    if (family === "BTTS") {
      if (pick === "YES" || pick === "BTTSYES") {
        return "BTTS Yes";
      }
      if (pick === "NO" || pick === "BTTSNO") {
        return "BTTS No";
      }
      return "BTTS read pending";
    }
    if (family === "FTR") {
      if (pick === "HOME") {
        return "FTR · Home";
      }
      if (pick === "DRAW") {
        return "FTR · Draw";
      }
      if (pick === "AWAY") {
        return "FTR · Away";
      }
      return "Result read pending";
    }
    if (pick) {
      return deployPickDisplay(pick);
    }
    if (family) {
      return `${marketFamilyDisplay(family)} read pending`;
    }
    return "Read pending";
  };

  const bookmakerLineDisplay = (odds) => (hasUsableOdds(odds) ? formatOdds(odds) : "Line pending");

  const impliedLineDisplay = (odds) => (hasUsableOdds(odds) ? `${formatImpliedProbability(odds)} implied` : "Public price unavailable");

  const impliedPercentValue = (odds) => {
    if (!hasUsableOdds(odds)) return null;
    const numeric = Number(formatImpliedProbability(odds).replace("%", ""));
    return Number.isFinite(numeric) ? Math.max(0, Math.min(100, numeric)) : null;
  };

  const compactMarketBarMarkup = ({ activeLabel, activeOdds, oppositionLabel, oppositionOdds, tone = "neutral" }) => {
    const activePercent = impliedPercentValue(activeOdds);
    const oppositionPercent = impliedPercentValue(oppositionOdds);
    const usableValues = [activePercent, oppositionPercent].filter((value) => value !== null);

    if (!usableValues.length) {
      return `
        <div class="market-bar-shell market-bar-shell-neutral">
          <div class="market-bar-header">
            <span class="metric-label">Pricing split</span>
            <span class="muted">Public price unavailable</span>
          </div>
          <div class="market-bar-empty">Compact market bars will appear when usable bookmaker pricing lands.</div>
        </div>
      `;
    }

    const normalizedTotal = usableValues.reduce((sum, value) => sum + value, 0) || 1;
    const activeShare = activePercent === null ? 0 : Math.round((activePercent / normalizedTotal) * 100);
    const oppositionShare = oppositionPercent === null ? 0 : Math.max(0, 100 - activeShare);

    return `
      <div class="market-bar-shell market-bar-shell-${escapeHtml(tone)}">
        <div class="market-bar-header">
          <span class="metric-label">Pricing split</span>
          <span class="muted">Public implied view</span>
        </div>
        <div class="market-bar-legend">
          <div class="market-bar-legend-item">
            <span class="market-bar-dot market-bar-dot-active"></span>
            <span>${escapeHtml(activeLabel)}${activePercent === null ? "" : ` · ${activePercent}%`}</span>
          </div>
          <div class="market-bar-legend-item">
            <span class="market-bar-dot market-bar-dot-opposition"></span>
            <span>${escapeHtml(oppositionLabel)}${oppositionPercent === null ? "" : ` · ${oppositionPercent}%`}</span>
          </div>
        </div>
        <div class="market-bar-track" aria-hidden="true">
          <span class="market-bar-fill market-bar-fill-active" style="width:${activeShare}%"></span>
          <span class="market-bar-fill market-bar-fill-opposition" style="width:${oppositionShare}%"></span>
        </div>
      </div>
    `;
  };

  const marketStructureBarMarkup = (entry, isActive = false) => {
    const value = entry.percent === null || entry.percent === undefined ? null : Math.max(0, Math.min(100, Math.round(entry.percent)));
    const tone = isActive ? "active" : "reference";
    return `
      <div class="market-mini-bar market-mini-bar-${tone}">
        <div class="market-mini-bar-top">
          <span>${escapeHtml(entry.label)}</span>
          <span>${value === null ? "N/A" : `${value}%`}</span>
        </div>
        <div class="market-mini-bar-track" aria-hidden="true">
          <span class="market-mini-bar-fill" style="width:${value === null ? 0 : value}%"></span>
        </div>
      </div>
    `;
  };

  const primaryMarketLine = (fixture) => {
    const family = String(fixture?.signal_summary?.market_family || "").toUpperCase();
    const odds = fixture?.odds_summary || {};
    if (family === "BTTS") {
      return {
        label: "BTTS Yes",
        odds: odds.btts_yes_odds,
        otherLabel: "BTTS No",
        otherOdds: odds.btts_no_odds,
      };
    }
    if (family === "OU25") {
      return {
        label: "Over 2.5",
        odds: odds.over25_odds,
        otherLabel: "Under 2.5",
        otherOdds: odds.under25_odds,
      };
    }
    const pick = String(fixture?.signal_summary?.deploy_pick || fixture?.deploy_summary?.pick || "").toUpperCase();
    const ftrFallbackOdds =
      pick === "HOME"
        ? odds.home_win_odds
        : pick === "DRAW"
          ? odds.draw_odds
          : pick === "AWAY"
            ? odds.away_win_odds
            : null;
    return {
      label: deployPickDisplay(pick),
      odds: fixture?.deploy_summary?.bookie_od ?? ftrFallbackOdds,
      otherLabel: "",
      otherOdds: null,
    };
  };

  const signalStrengthDisplay = (value) => {
    const raw = String(value || "").trim().toLowerCase();
    if (!raw) {
      return "Unspecified";
    }
    return `${raw.charAt(0).toUpperCase()}${raw.slice(1)} strength`;
  };

  const alternativeMarketLine = (fixture) => {
    const family = String(fixture?.signal_summary?.market_family || "").toUpperCase();
    const odds = fixture?.odds_summary || {};
    const primary = primaryMarketLine(fixture);
    if (family === "BTTS" || family === "OU25") {
      return {
        label: primary.otherLabel || "Opposition",
        odds: primary.otherOdds,
      };
    }
    const pick = String(fixture?.signal_summary?.deploy_pick || fixture?.deploy_summary?.pick || "").toUpperCase();
    const candidates = [
      { label: "Home", odds: odds.home_win_odds, key: "HOME" },
      { label: "Draw", odds: odds.draw_odds, key: "DRAW" },
      { label: "Away", odds: odds.away_win_odds, key: "AWAY" },
    ].filter((entry) => entry.key !== pick && Number.isFinite(Number(entry.odds)));
    candidates.sort((left, right) => Number(left.odds) - Number(right.odds));
    const best = candidates[0];
    return {
      label: best?.label || "Best alternative",
      odds: best?.odds ?? null,
    };
  };

  const marketOddsDisplay = (fixture, key) => {
    const odds = fixture?.odds_summary || {};
    if (key === "ftr") {
      return `H ${bookmakerLineDisplay(odds.home_win_odds)} / D ${bookmakerLineDisplay(odds.draw_odds)} / A ${bookmakerLineDisplay(odds.away_win_odds)}`;
    }
    if (key === "ou25") {
      return `Over ${bookmakerLineDisplay(odds.over25_odds)} / Under ${bookmakerLineDisplay(odds.under25_odds)}`;
    }
    if (key === "btts") {
      return `Yes ${bookmakerLineDisplay(odds.btts_yes_odds)} / No ${bookmakerLineDisplay(odds.btts_no_odds)}`;
    }
    return "Team-goals price pending";
  };

  const publishedMarketRow = (fixture, key, pick = "") => {
    const fixtureKey = String(fixture?.fixture_key || "");
    const aliases = {
      ftr: ["FTR"],
      ou25: ["OU25", "OVER25"],
      btts: ["BTTS"],
      team_goals: ["TG15", "TEAM_GOALS", "TEAMGOALS"],
    }[key] || [];
    const wantedPick = String(pick || "").toUpperCase().replace(/[^A-Z0-9]/g, "");
    const rows = [...state.publicPredictions, ...state.premiumPredictions, ...state.securePremiumPredictions];
    return (
      rows.find((row) => {
        const rowFixtureKey = String(row.fixture_key || "");
        const market = String(row.market || "").toUpperCase().replace(/[^A-Z0-9]/g, "");
        const rowPick = String(row.pick || "").toUpperCase().replace(/[^A-Z0-9]/g, "");
        const marketMatches = rowFixtureKey === fixtureKey && aliases.some((alias) => market === alias.replace(/[^A-Z0-9]/g, ""));
        return marketMatches && (!wantedPick || rowPick === wantedPick);
      }) || null
    );
  };

  const publishedModelProbability = (fixture, key, pick = "") => {
    const match = publishedMarketRow(fixture, key, pick);
    if (!match && pick) {
      return "";
    }
    const fallbackMatch =
      match ||
      (() => {
        const fixtureKey = String(fixture?.fixture_key || "");
        const aliases = {
          ftr: ["FTR"],
          ou25: ["OU25", "OVER25"],
          btts: ["BTTS"],
          team_goals: ["TG15", "TEAM_GOALS", "TEAMGOALS"],
        }[key] || [];
        const rows = [...state.publicPredictions, ...state.premiumPredictions, ...state.securePremiumPredictions];
        return rows.find((row) => {
          const rowFixtureKey = String(row.fixture_key || "");
          const market = String(row.market || "").toUpperCase().replace(/[^A-Z0-9]/g, "");
          return rowFixtureKey === fixtureKey && aliases.some((alias) => market === alias.replace(/[^A-Z0-9]/g, ""));
        });
      })();
    if (!fallbackMatch) {
      return "";
    }
    if (fallbackMatch.model_prob_display) {
      return String(fallbackMatch.model_prob_display);
    }
    const numeric = Number(fallbackMatch.model_prob);
    return Number.isFinite(numeric) ? compactPercent(numeric) : "";
  };

  const normalizeMarketPick = (key, pick = "", fixture = null) => {
    const value = normalizeLeanKey(pick);
    if (!value) return "";
    if (key === "ftr") {
      if (value.includes("DRAW")) return "DRAW";
      if (value.includes("AWAY")) return "AWAY";
      if (value.includes("HOME")) return "HOME";
    }
    if (key === "ou25") {
      if (value.includes("UNDER") || value.includes("U25")) return "UNDER25";
      if (value.includes("OVER") || value.includes("O25")) return "OVER25";
    }
    if (key === "btts") {
      if (value.includes("NO")) return "NO";
      if (value.includes("YES")) return "YES";
    }
    if (key === "team_goals") {
      const homeKey = normalizeLeanKey(fixture?.home_team);
      const awayKey = normalizeLeanKey(fixture?.away_team);
      if (value.includes(homeKey) || value.includes("HOME")) return "HOME15";
      if (value.includes(awayKey) || value.includes("AWAY")) return "AWAY15";
    }
    return value;
  };

  const marketKeyFromFamily = (family = "") => {
    const value = String(family || "").toUpperCase();
    if (value === "FTR") return "ftr";
    if (value === "OU25" || value === "OVER25") return "ou25";
    if (value === "BTTS") return "btts";
    if (value === "TG15" || value.includes("TEAM")) return "team_goals";
    return "";
  };

  const explicitRoutedMarketPick = (fixture, key) => {
    const publishClass = String(fixture?.publish_class || fixture?.fixture_class || "").toUpperCase();
    const familyKey = marketKeyFromFamily(fixture?.signal_summary?.market_family || fixture?.deploy_summary?.market || "");
    const rawPick = fixture?.signal_summary?.deploy_pick || fixture?.deploy_summary?.pick || "";
    if (publishClass !== "DEPLOY" || familyKey !== key || !rawPick) {
      return "";
    }
    return normalizeMarketPick(key, rawPick, fixture);
  };

  const selectedMarketPick = (fixture, key) => {
    const published = publishedMarketRow(fixture, key);
    return normalizeMarketPick(key, published?.pick || "", fixture) || explicitRoutedMarketPick(fixture, key);
  };

  const marketPickDisplay = (fixture, key, pick = "") => {
    const normalized = normalizeMarketPick(key, pick, fixture);
    if (key === "ftr") {
      if (normalized === "HOME") return `${teamCardName(fixture?.home_team) || "Home"} Win`;
      if (normalized === "DRAW") return "Draw";
      if (normalized === "AWAY") return `${teamCardName(fixture?.away_team) || "Away"} Win`;
    }
    if (key === "ou25") {
      return normalized === "UNDER25" ? "Under 2.5" : normalized === "OVER25" ? "Over 2.5" : "Totals read";
    }
    if (key === "btts") {
      return normalized === "NO" ? "BTTS No" : normalized === "YES" ? "BTTS Yes" : "BTTS read";
    }
    if (key === "team_goals") {
      return normalized === "HOME15"
        ? `${teamCardName(fixture?.home_team) || "Home"} 1.5+`
        : normalized === "AWAY15"
          ? `${teamCardName(fixture?.away_team) || "Away"} 1.5+`
          : "Team-goals support";
    }
    return deployPickDisplay(pick);
  };

  const fixturePublishedSelection = (fixture) => {
    const familyKey = marketKeyFromFamily(fixture?.signal_summary?.market_family || fixture?.deploy_summary?.market || "");
    const pick = explicitRoutedMarketPick(fixture, familyKey);
    if (!familyKey || !pick) {
      return null;
    }
    return {
      key: familyKey,
      pick,
      label: marketPickDisplay(fixture, familyKey, pick),
      confidence: fixture?.signal_summary?.confidence_tier || fixture?.deploy_summary?.confidence_tier || "",
    };
  };

  const decisionMarketItem = (decision, key) => {
    const normalizedKey = String(key || "").toLowerCase().replace(/[^a-z0-9]/g, "");
    return (
      decisionMarketSuitabilityItems(decision).find((item) => {
        const itemKey = String(item.key || item.label || "").toLowerCase().replace(/[^a-z0-9]/g, "");
        if (normalizedKey === "ftr") return itemKey.includes("ftr");
        if (normalizedKey === "ou25") return itemKey.includes("ou25") || itemKey.includes("over25");
        if (normalizedKey === "btts") return itemKey.includes("btts");
        if (normalizedKey === "teamgoals") return itemKey.includes("teamgoals");
        return itemKey === normalizedKey;
      }) || null
    );
  };

  const normalizeLeanKey = (value) => String(value || "").toUpperCase().replace(/[^A-Z0-9]/g, "");

  const modelLeanForMarket = (fixture, key, intel = null) => {
    const rawLean = normalizeLeanKey(intel?.modelOutput?.pick || intel?.modelLean);
    if (rawLean) {
      if (key === "team_goals") {
        return rawLean;
      }
      if (rawLean.includes("OVER25")) return "OVER25";
      if (rawLean.includes("UNDER25")) return "UNDER25";
      if (rawLean.includes("YES")) return "YES";
      if (rawLean.includes("NO")) return "NO";
      if (rawLean.includes("DRAW")) return "DRAW";
      if (rawLean.includes("AWAY")) return "AWAY";
      if (rawLean.includes("HOME")) return "HOME";
    }
    return "";
  };

  const teamContextLeanForMarket = (fixture, key, intel = null) => {
    const rawLean = normalizeLeanKey(intel?.teamContextLean);
    if (!rawLean) {
      return "";
    }
    if (key === "team_goals") return rawLean;
    if (rawLean.includes("OVER25")) return "OVER25";
    if (rawLean.includes("UNDER25")) return "UNDER25";
    if (rawLean.includes("YES")) return "YES";
    if (rawLean.includes("NO")) return "NO";
    if (rawLean.includes("DRAW")) return "DRAW";
    if (rawLean.includes("AWAY")) return "AWAY";
    if (rawLean.includes("HOME")) return "HOME";
    return rawLean;
  };

  const modelOutputProbability = (intel = null, key = "", pick = "") => {
    const output = intel?.modelOutput || null;
    const normalizedPick = normalizeMarketPick(key, pick);
    const probabilities = output?.probabilities || {};
    const value = probabilities[normalizedPick];
    return Number.isFinite(Number(value)) ? compactPercent(Number(value)) : "";
  };

  const modelSignalText = (fixture, key, pick, intel = null, state = {}) => {
    if (state.selected) {
      const probability = publishedModelProbability(fixture, key, pick);
      return probability ? `Model ${probability}` : "Published pick";
    }
    if (state.modelSelected) {
      const probability = modelOutputProbability(intel, key, pick);
      return probability ? `Model ${probability}` : "Model output";
    }
    if (state.context && Number.isFinite(Number(intel?.rating))) {
      return `Team context ${Math.round(Number(intel.rating))}%`;
    }
    if (state.context) {
      return "Team context";
    }
    return hasUsableOdds(state.odds) ? "Book price" : "No selection";
  };

  const outcomeOddsText = (odds, unavailableCopy = "Odds pending") => (hasUsableOdds(odds) ? bookmakerLineDisplay(odds) : unavailableCopy);

  const outcomeImpliedText = (odds, unavailableCopy = "No public price") => (hasUsableOdds(odds) ? impliedLineDisplay(odds) : unavailableCopy);

  const marketOutcomeRows = (fixture, key, intel = null) => {
    const odds = fixture?.odds_summary || {};
    const selectedPick = selectedMarketPick(fixture, key);
    const modelLean = normalizeMarketPick(key, modelLeanForMarket(fixture, key, intel), fixture);
    const contextLean = normalizeMarketPick(key, teamContextLeanForMarket(fixture, key, intel), fixture);
    const canHighlightContext = key === "team_goals" || !modelLean;
    if (key === "ftr") {
      return [
        { label: "Home", pick: "HOME", odds: odds.home_win_odds },
        { label: "Draw", pick: "DRAW", odds: odds.draw_odds },
        { label: "Away", pick: "AWAY", odds: odds.away_win_odds },
      ].map((row) => {
        const active = selectedPick === row.pick;
        const modelSelected = !active && modelLean === row.pick;
        const context = canHighlightContext && !active && !modelSelected && contextLean === row.pick;
        return {
          ...row,
          active,
          modelSelected,
          context,
          model: modelSignalText(fixture, key, row.pick, intel, { selected: active, modelSelected, context, odds: row.odds }),
          implied: outcomeImpliedText(row.odds),
        };
      });
    }
    if (key === "ou25") {
      return [
        { label: "Over 2.5", pick: "OVER25", odds: odds.over25_odds },
        { label: "Under 2.5", pick: "UNDER25", odds: odds.under25_odds },
      ].map((row) => {
        const active = selectedPick === row.pick;
        const modelSelected = !active && modelLean === row.pick;
        const context = canHighlightContext && !active && !modelSelected && contextLean === row.pick;
        return {
          ...row,
          active,
          modelSelected,
          context,
          model: modelSignalText(fixture, key, row.pick, intel, { selected: active, modelSelected, context, odds: row.odds }),
          implied: outcomeImpliedText(row.odds),
        };
      });
    }
    if (key === "btts") {
      return [
        { label: "Yes", pick: "YES", odds: odds.btts_yes_odds },
        { label: "No", pick: "NO", odds: odds.btts_no_odds },
      ].map((row) => {
        const active = selectedPick === row.pick;
        const modelSelected = !active && modelLean === row.pick;
        const context = canHighlightContext && !active && !modelSelected && contextLean === row.pick;
        return {
          ...row,
          active,
          modelSelected,
          context,
          model: modelSignalText(fixture, key, row.pick, intel, { selected: active, modelSelected, context, odds: row.odds }),
          implied: outcomeImpliedText(row.odds),
        };
      });
    }
    return [
      { label: `${teamCardName(fixture.home_team) || "Home"} 1.5+`, pick: "HOME15" },
      { label: `${teamCardName(fixture.away_team) || "Away"} 1.5+`, pick: "AWAY15" },
    ].map((row) => {
      const active = selectedPick === row.pick;
      const modelSelected = !active && modelLean === row.pick;
      const context = canHighlightContext && !active && !modelSelected && contextLean === row.pick;
      return {
        ...row,
        active,
        modelSelected,
        context,
        odds: null,
        model: context && Number.isFinite(Number(intel?.rating)) ? `Team context ${Math.round(Number(intel.rating))}%` : context ? "Team context" : "Support watch",
        implied: "No odds feed",
      };
    });
  };

  const marketLeadText = (fixture, key, intel = null) => {
    const active = marketOutcomeRows(fixture, key, intel).find((row) => row.active);
    if (active) {
      return `Published ${marketPickDisplay(fixture, key, active.pick)} · ${active.model}`;
    }
    const model = marketOutcomeRows(fixture, key, intel).find((row) => row.modelSelected);
    if (model) {
      return `Model ${model.label} · ${model.model}`;
    }
    const context = marketOutcomeRows(fixture, key, intel).find((row) => row.context);
    if (context) {
      return `Team context ${context.label} · ${context.model}`;
    }
    if (Number.isFinite(Number(intel?.rating))) {
      return `${safeTitleLabel(intel?.band, "Context")} · ${Math.round(Number(intel.rating))}% support`;
    }
    return key === "team_goals" ? "Support-only read" : "Market read pending";
  };

  const marketAccessLabel = (key) => {
    if (key === "team_goals") return "Founder context";
    return "Standard view";
  };

  const marketOutcomeRowsMarkup = (rows) => `
    <div class="fixture-market-outcome-list">
      ${rows
        .map(
          (row) => `
            <div class="fixture-market-outcome-row ${row.active ? "is-active" : row.modelSelected || row.context ? "is-context" : ""}">
              <div>
                <span>${escapeHtml(row.label)}</span>
                <small>${escapeHtml(row.model)}</small>
              </div>
              <div>
                <b>${escapeHtml(outcomeOddsText(row.odds, "No odds"))}</b>
                <small>${escapeHtml(row.implied)}</small>
              </div>
            </div>
          `
        )
        .join("")}
    </div>
  `;

  const fixtureTierDefinitions = () => [
    {
      key: "free",
      label: "Standard",
      title: "Top market cards",
      copy: "FTR, OU25, BTTS, and TG1.5 support cards with public-safe odds and model output state.",
      requiredRank: 0,
    },
    {
      key: "founder",
      label: "Founder / Premium",
      title: "Fixture context",
      copy: "Prediction deck, team reads, H2H, weather, freshness, lineups, news, and market suitability.",
      requiredRank: 1,
    },
    {
      key: "pro",
      label: "Pro",
      title: "Player-event intelligence",
      copy: "Shots, SOT, tackles, fouls, player fouled, key passes, keeper saves, bookings, and team tackles.",
      requiredRank: 3,
    },
    {
      key: "pro_plus",
      label: "Pro+",
      title: "Audit dashboard",
      copy: "Data identity, coverage metadata, model-feature drilldowns, downloadable intelligence, and audit filters.",
      requiredRank: 4,
    },
  ];

  const fixtureTierUnlockRail = () => {
    const rank = accessTierRank(currentAccessTier());
    return `
      <div class="fixture-tier-rail" aria-label="Fixture intelligence tier visibility">
        ${fixtureTierDefinitions()
          .map((tier) => {
            const available = rank >= tier.requiredRank;
            const current =
              rank === tier.requiredRank ||
              (tier.key === "founder" && rank > 1 && rank < 3) ||
              (rank > 4 && tier.key === "pro_plus");
            return `
              <article class="fixture-tier-card ${available ? "fixture-tier-card-open" : "fixture-tier-card-locked"} ${current ? "fixture-tier-card-current" : ""}">
                <div class="fixture-tier-card-top">
                  <span class="metric-label">${escapeHtml(tier.label)}</span>
                  <span class="fixture-tier-status">${escapeHtml(available ? "Available" : "Upgrade")}</span>
                </div>
                <strong>${escapeHtml(tier.title)}</strong>
                <p>${escapeHtml(tier.copy)}</p>
              </article>
            `;
          })
          .join("")}
      </div>
    `;
  };

  const normalizeAccessTierLabel = (value) =>
    String(value || "")
      .trim()
      .toLowerCase()
      .replace(/[\s-]+/g, "_");

  const accessTierRank = (value) => {
    const ranks = {
      free: 0,
      founder: 1,
      founder_early_access: 1,
      premium: 2,
      pro: 3,
      pro_plus: 4,
    };
    return ranks[normalizeAccessTierLabel(value)] ?? 0;
  };

  const currentAccessTier = () =>
    normalizeAccessTierLabel(
      state.runtime.sessionAccessTier ||
        state.runtime.accountState?.subscription?.access_tier ||
        state.runtime.accountState?.subscription?.tier ||
        state.runtime.accountState?.subscription?.plan_tier ||
        ""
    ) || (state.runtime.sessionEntitled ? "founder" : "free");

  const hasTierAccess = (requiredTier) => accessTierRank(currentAccessTier()) >= accessTierRank(requiredTier);

  const fixtureTierGate = (requiredTier, title, copy, features = []) => {
    const tierLabel = safeTitleLabel(requiredTier === "pro_plus" ? "Pro+" : requiredTier);
    return `
      <section class="section section-tight fixture-tier-gate-section">
        <article class="fixture-tier-gate">
          <div>
            <span class="metric-label">${escapeHtml(`${tierLabel} layer`)}</span>
            <h3>${escapeHtml(title)}</h3>
            <p class="muted">${escapeHtml(copy)}</p>
          </div>
          ${
            features.length
              ? `<div class="fixture-tier-gate-grid">${features.map((feature) => `<span>${escapeHtml(feature)}</span>`).join("")}</div>`
              : ""
          }
          <div class="cta-row">
            <a class="button" href="./pricing.html">See access</a>
            <a class="ghost-button" href="./account.html">Account</a>
          </div>
        </article>
      </section>
    `;
  };

  const fixtureContextGate = () =>
    fixtureTierGate("founder", "Fixture context unlocks at Founder.", "Standard keeps the first screen focused on the public market cards. Founder and above opens the deeper read behind the pick.", [
      "Prediction deck",
      "H2H",
      "Weather",
      "Lineups",
      "Team reads",
      "News",
    ]);

  const fixtureTabRequiredTier = (tabKey) => {
    if (tabKey === "stats") return "pro_plus";
    return "founder";
  };

  const fixtureLockedTabContent = (tabKey) => {
    if (tabKey === "stats") {
      return fixtureTierGate("pro_plus", "Audit dashboard unlocks at Pro+.", "This tab is for data identity, coverage metadata, and the audit view behind the fixture payload.", [
        "Coverage metadata",
        "Source identity",
        "Freshness audit",
        "Feature drilldowns",
      ]);
    }
    return fixtureTierGate("founder", "Deeper fixture context unlocks at Founder.", "The public view shows the top market cards. Founder and above adds the full fixture read and supporting context.", [
      "Prediction",
      "Markets",
      "Lineups",
      "H2H",
      "Form",
      "News",
    ]);
  };

  const playerEventPhaseLabel = (value) => {
    const normalized = String(value || "").toLowerCase();
    if (normalized === "lineup_confirmed_refresh") return "Confirmed lineup refresh";
    if (normalized === "pre_tournament_projection") return "Pre-tournament projection";
    if (normalized === "lineup_pending") return "Lineup pending";
    return "Pre-lineup preview";
  };

  const playerEventFamilyTitle = (family) => {
    const labels = {
      shots: "Player Shots",
      shots_on_target: "Shots On Target",
      key_passes: "Key Passes",
      tackles: "Player Tackles",
      fouls: "Player Fouls",
      player_fouled: "Player Fouled",
      bookings: "Bookings Watch",
      cards: "Bookings Watch",
      yellow_cards: "Bookings Watch",
      keeper_saves: "Keeper Saves",
      goalkeeper_saves: "Keeper Saves",
      saves: "Keeper Saves",
      team_tackles: "Team / Match Tackles",
    };
    return labels[String(family || "").trim()] || safeTitleLabel(family || "Player event");
  };

  const normalizePlayerEventFamily = (item) => {
    const raw = String(item?.event_family || item?.event_key || "").trim();
    const aliases = {
      cards: "bookings",
      yellow_cards: "bookings",
      goalkeeper_saves: "keeper_saves",
      saves: "keeper_saves",
      passes_key: "key_passes",
    };
    if (aliases[raw]) return aliases[raw];
    const key = String(item?.event_key || "").toLowerCase();
    if (key.includes("card") || key.includes("booking")) return "bookings";
    if (key.includes("save")) return "keeper_saves";
    if (key.includes("key_pass") || key.includes("passes_key")) return "key_passes";
    return raw;
  };

  const buildPlayerEventCardsFromShortlists = (shortlists = [], source = {}) => {
    const families = ["shots", "shots_on_target", "key_passes", "tackles", "fouls", "player_fouled", "bookings", "keeper_saves", "team_tackles"];
    const grouped = new Map();
    shortlists.forEach((item) => {
      const family = normalizePlayerEventFamily(item);
      if (!families.includes(family)) {
        return;
      }
      if (!grouped.has(family)) {
        grouped.set(family, []);
      }
      grouped.get(family).push(item);
    });
    const cards = families
      .map((family) => {
        const rows = (grouped.get(family) || [])
          .slice()
          .sort(
            (left, right) =>
              Number(right.shortlist_score ?? right.score ?? 0) - Number(left.shortlist_score ?? left.score ?? 0) ||
              Number(left.shortlist_rank ?? left.rank ?? 999) - Number(right.shortlist_rank ?? right.rank ?? 999)
          );
        const seen = new Set();
        const shortlist = rows
          .filter((item) => {
            const key = `${normalizePreferenceText(item.team_name)}:${normalizePreferenceText(item.player_name)}`;
            if (seen.has(key)) {
              return false;
            }
            seen.add(key);
            return true;
          })
          .slice(0, 4)
          .map((item) => ({
            player_name: item.player_name || item.player || "",
            team_name: item.team_name || item.team || "",
            is_home: Boolean(item.is_home),
            position_group: item.position_group || item.position || "",
            rank: item.shortlist_rank ?? item.rank ?? "",
            score: item.shortlist_score ?? item.score ?? "",
            confidence_label: item.confidence_label || "manual_review",
            sample_size: item.sample_size ?? "",
            minutes_sample: item.minutes_sample ?? "",
            reason: item.reason || "",
          }));
        if (!shortlist.length) {
          return null;
        }
        return {
          event_family: family,
          card_title: playerEventFamilyTitle(family),
          beta_status: "beta_shortlist",
          lineup_status: source.lineup_status || rows[0]?.source_lineup_status || "last_fixture_snapshot",
          shortlist,
        };
      })
      .filter(Boolean);
    return {
      status: cards.length ? "available" : "missing",
      phase: source.phase || (shortlists.some((item) => String(item?.source_lineup_status || "").includes("confirmed")) ? "lineup_confirmed_refresh" : "pre_lineup_preview"),
      lineup_status: source.lineup_status || shortlists.find((item) => item?.source_lineup_status)?.source_lineup_status || "",
      cards,
      missing_event_families: families.filter((family) => !grouped.has(family)),
    };
  };

  const fixturePlayerEventCardsSource = () => {
    const brainCards = state.selectedFixtureSiteData?.player_event_cards;
    if (brainCards?.status === "available" && Array.isArray(brainCards.cards)) {
      return brainCards;
    }
    const protectedShortlists = state.selectedFixtureStats?.data?.player_event_shortlists;
    if (Array.isArray(protectedShortlists) && protectedShortlists.length) {
      return buildPlayerEventCardsFromShortlists(protectedShortlists, {
        phase: "pre_lineup_preview",
      });
    }
    const fixtureStatsShortlists = state.selectedFixtureSiteData?.stats?.player_event_shortlists;
    if (Array.isArray(fixtureStatsShortlists) && fixtureStatsShortlists.length && accessTierRank(currentAccessTier()) >= 3) {
      return buildPlayerEventCardsFromShortlists(fixtureStatsShortlists, {
        phase: "pre_lineup_preview",
      });
    }
    return null;
  };

  const renderPlayerEventCandidate = (item) => {
    const score = Number(item?.score);
    const sample = item?.sample_size ? `${item.sample_size} samples` : item?.minutes_sample ? `${item.minutes_sample} mins` : "Sample tracked";
    return `
      <li class="player-event-candidate">
        <div>
          <strong>${escapeHtml(item?.player_name || "Player pending")}</strong>
          <span>${escapeHtml([item?.team_name, safeTitleLabel(item?.position_group || "")].filter(Boolean).join(" · "))}</span>
        </div>
        <div>
          <b>${escapeHtml(Number.isFinite(score) ? Math.round(score).toString() : "Watch")}</b>
          <small>${escapeHtml(sample)}</small>
        </div>
      </li>
    `;
  };

  const fixturePlayerEventTeaser = (fixture) => `
    <section class="section section-tight fixture-player-event-section">
      <div class="section-head">
        <div>
          <h2>Player-event intelligence</h2>
          <p class="section-copy">Preview shortlists for shots, SOT, tackles, fouls, player fouled, key passes, keeper saves, and bookings sit behind Pro access. They refresh again when confirmed lineups land.</p>
        </div>
        <span class="pill">Pro preview</span>
      </div>
      <article class="fixture-player-event-lock">
        <div>
          <span class="metric-label">Standard view</span>
          <strong>Goal markets stay public. Player events unlock at Pro.</strong>
          <p class="muted">The public page keeps the top market cards clean. Pro adds pre-lineup player-event watchlists immediately below, then updates around T-60 when lineup automation confirms starters and bench.</p>
        </div>
        <div class="player-event-lock-grid" aria-label="Locked player-event families">
          ${["Shots", "SOT", "Tackles", "Fouls", "Fouled", "Key passes", "Saves", "Bookings"]
            .map((label) => `<span>${escapeHtml(label)}</span>`)
            .join("")}
        </div>
        <div class="cta-row">
          <a class="button" href="./pricing.html">See Pro access</a>
          <a class="ghost-button" href="./methodology.html">Read methodology</a>
        </div>
      </article>
    </section>
  `;

  const fixturePlayerEventCardsMarkup = (fixture) => {
    const accessTier = currentAccessTier();
    const hasProAccess = accessTierRank(accessTier) >= 3;
    const cardSource = fixturePlayerEventCardsSource();
    if (!hasProAccess) {
      return fixturePlayerEventTeaser(fixture);
    }
    if (!cardSource?.cards?.length) {
      return `
        <section class="section section-tight fixture-player-event-section">
          <div class="section-head">
            <div>
              <h2>Player-event intelligence</h2>
              <p class="section-copy">Pro access is active, but this fixture does not have a publish-safe player-event shortlist yet.</p>
            </div>
            <span class="pill">Pro</span>
          </div>
          <div class="notice">Player-event cards will appear after the local player/team/lineup compiler publishes shortlists for this fixture.</div>
        </section>
      `;
    }
    return `
      <section class="section section-tight fixture-player-event-section">
        <div class="section-head">
          <div>
            <h2>Player-event intelligence</h2>
            <p class="section-copy">Beta shortlists based on recent player/team event rates and current lineup context. These are watchlists, not priced picks.</p>
          </div>
          <span class="pill">Pro · ${escapeHtml(playerEventPhaseLabel(cardSource.phase))}</span>
        </div>
        <div class="fixture-player-event-grid">
          ${cardSource.cards
            .map(
              (card) => `
                <article class="fixture-player-event-card">
                  <div class="fixture-player-event-head">
                    <div>
                      <span class="metric-label">${escapeHtml(card.beta_status || "Beta shortlist")}</span>
                      <strong>${escapeHtml(card.card_title || playerEventFamilyTitle(card.event_family))}</strong>
                    </div>
                    <span>${escapeHtml(safeTitleLabel(card.lineup_status || cardSource.lineup_status || "Preview"))}</span>
                  </div>
                  <ul class="player-event-candidate-list">
                    ${(card.shortlist || []).map((item) => renderPlayerEventCandidate(item)).join("")}
                  </ul>
                  <p class="muted">${escapeHtml((card.shortlist || [])[0]?.reason || "Shortlist reason will refresh with the next player-event compile.")}</p>
                </article>
              `
            )
            .join("")}
        </div>
        ${
          (cardSource.missing_event_families || []).length
            ? `<p class="fixture-player-event-missing">Missing producer coverage: ${escapeHtml(
                cardSource.missing_event_families.map((family) => playerEventFamilyTitle(family)).join(", ")
              )}.</p>`
            : ""
        }
      </section>
    `;
  };

  const fixtureMarketCardsMarkup = (fixture, decision = null) => {
    const deployFamily = String(fixture?.signal_summary?.market_family || "").toUpperCase();
    const cardDefs = [
      {
        key: "ftr",
        title: "Full Time Result",
        copy: "Home / Draw / Away odds with the actual model output surfaced clearly.",
        active: deployFamily === "FTR",
      },
      {
        key: "ou25",
        title: "Over 2.5 Match Goals",
        copy: "Totals posture with Over/Under pricing and model output state.",
        active: deployFamily === "OU25",
      },
      {
        key: "btts",
        title: "BTTS",
        copy: "Both-teams-to-score pricing with Yes/No model output.",
        active: deployFamily === "BTTS",
      },
      {
        key: "team_goals",
        title: "Team 1.5 Goals",
        copy: "Support-only TG1.5 read. Odds feed is not live for this market yet.",
        active: deployFamily === "TG15" || deployFamily.includes("TEAM"),
      },
    ];
    return `
      <section class="section section-tight fixture-market-card-section">
        <div class="section-head">
          <div>
            <h2>Standard market cards</h2>
            <p class="section-copy">The top fixture view is intentionally simple: odds, model output, and confidence posture for the launch markets. Deeper context sits behind the tiered tabs below.</p>
          </div>
          <span class="pill">Standard view</span>
        </div>
        <div class="fixture-market-card-grid">
          ${cardDefs
            .map((def) => {
              const intel = decisionMarketItem(decision, def.key) || null;
              const stateLabel = safeTitleLabel(intel?.band || intel?.state || fixture?.signal_summary?.signal_state, "Pending");
              const outcomeRows = marketOutcomeRows(fixture, def.key, intel);
              const hasPublishedSelection = outcomeRows.some((row) => row.active);
              const hasModelOutput = outcomeRows.some((row) => row.modelSelected);
              const hasContextLean = outcomeRows.some((row) => row.context);
              const intelligenceState = String(intel?.band || intel?.state || "").toUpperCase();
              const tone = hasPublishedSelection
                ? intelligenceState === "AVOID"
                  ? "fragile"
                  : valueEdgeTone(fixture)
                : decisionStateTone(intel?.band || intel?.state || "");
              const stateCopy = hasPublishedSelection
                ? "Published pick"
                : hasModelOutput
                  ? "Model output"
                  : hasContextLean
                  ? intelligenceState === "AVOID"
                    ? "Team context caution"
                    : "Team context"
                  : def.key === "team_goals"
                    ? "Support only"
                    : "No pick";
              return `
                <article class="fixture-market-card fixture-market-card-${escapeHtml(tone)} ${hasPublishedSelection ? "fixture-market-card-active" : ""}">
                  <div class="fixture-market-card-head">
                    <div>
                      <span class="metric-label">${escapeHtml(def.title)}</span>
                      <strong>${escapeHtml(marketLeadText(fixture, def.key, intel))}</strong>
                    </div>
                    <span class="fixture-market-state">${escapeHtml(`${stateCopy}${stateLabel && !hasPublishedSelection ? ` · ${stateLabel}` : ""}`)}</span>
                  </div>
                  <p class="fixture-market-card-copy">${escapeHtml(def.copy)}</p>
                  ${marketOutcomeRowsMarkup(outcomeRows)}
                  <div class="fixture-market-card-meta fixture-market-card-meta-access">
                    <span>${escapeHtml(marketAccessLabel(def.key))}</span>
                    <b>${escapeHtml(def.key === "team_goals" ? "No odds feed" : marketOddsDisplay(fixture, def.key))}</b>
                  </div>
                  <p class="muted">${escapeHtml(intel?.read || "No published expert read for this family yet.")}</p>
                </article>
              `;
            })
            .join("")}
        </div>
        ${fixtureTierUnlockRail()}
      </section>
    `;
  };

  const computeRouteTone = (entry) => {
    const priority = dashboardPriorityProfile(entry);
    if (priority.bucket === "send_now") {
      return { label: "Deploy", tone: "deploy", read: "Route live" };
    }
    if (priority.bucket === "watch_closely") {
      return { label: "Observe", tone: "observe", read: "Hold and watch" };
    }
    if (priority.bucket === "website_only") {
      return { label: "Context", tone: "monitor", read: "Website-first" };
    }
    return { label: "Pass", tone: "pass", read: "No forced route" };
  };

  const compactKickoffLabel = (value) => {
    if (!value) return "Pending kickoff";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return String(value);
    return parsed.toLocaleString("en-GB", {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const dashboardComputePanel = (entries) => {
    if (!entries.length) {
      return `
        <article class="compute-panel compute-panel-empty">
          <div class="compute-panel-head">
            <div class="compute-panel-copy">
              <span class="metric-label">Odds Genius - Computing...</span>
              <h3>Routing window is calm.</h3>
              <p class="muted">When followed intelligence comes into view, live route rows will appear here with league flow, market posture, and public pricing pressure.</p>
            </div>
            <div class="compute-panel-pulse">
              <span class="compute-panel-dot" aria-hidden="true"></span>
              <span>Standby</span>
            </div>
          </div>
        </article>
      `;
    }

    const rows = entries.slice(0, 4);
    const visibleLeagues = new Set(rows.map((entry) => String(entry.row.league || "").trim()).filter(Boolean)).size;
    const deployCount = rows.filter((entry) => computeRouteTone(entry).tone === "deploy").length;
    const observeCount = rows.filter((entry) => computeRouteTone(entry).tone === "observe").length;

    return `
      <article class="compute-panel">
        <div class="compute-panel-head">
          <div class="compute-panel-copy">
            <span class="metric-label">Odds Genius - Computing...</span>
            <h3>Live routing surface.</h3>
            <p class="muted">A quiet window into the current followed-intelligence layer: league flow, active market pressure, and route posture updating in sequence.</p>
          </div>
          <div class="compute-panel-pulse">
            <span class="compute-panel-dot" aria-hidden="true"></span>
            <span>Live</span>
          </div>
        </div>
        <div class="compute-panel-meta">
          <span>${visibleLeagues} leagues visible</span>
          <span>${rows.length} live rows</span>
          <span>${deployCount} deploy / ${observeCount} observe</span>
        </div>
        <div class="compute-panel-rows">
          ${rows
            .map((entry, index) => {
              const row = entry.row;
              const route = computeRouteTone(entry);
              const marketLine = primaryMarketLine(row);
              const opposition = alternativeMarketLine(row);
              const activePercent = impliedPercentValue(marketLine.odds);
              const oppositionPercent = impliedPercentValue(opposition.odds);
              const activeShare = activePercent == null ? 0 : Math.max(12, activePercent);
              const oppositionShare = oppositionPercent == null ? 0 : Math.max(8, oppositionPercent);
              return `
                <article class="compute-row compute-row-${escapeHtml(route.tone)}" style="--compute-index:${index}">
                  <div class="compute-row-top">
                    <span class="fixture-route-pill fixture-route-pill-${escapeHtml(route.tone)}">${escapeHtml(route.label)}</span>
                    <span class="compute-row-meta">${escapeHtml(marketFamilyLabel(row.signal_summary?.market_family))} · ${escapeHtml(compactKickoffLabel(row.kickoff_time))}</span>
                  </div>
                  <strong class="compute-row-fixture">
                    <span class="compute-row-team compute-row-team-home">${badgeMarkup(row.home_team_logo_url, row.home_team)}<span>${escapeHtml(teamCardName(row.home_team))}</span></span>
                    <span class="versus">vs</span>
                    <span class="compute-row-team compute-row-team-away"><span>${escapeHtml(teamCardName(row.away_team))}</span>${badgeMarkup(row.away_team_logo_url, row.away_team)}</span>
                  </strong>
                  <div class="compute-row-read">
                    <span>${escapeHtml(`${marketVerdictDisplay(row)} · ${bookmakerLineDisplay(marketLine.odds)}`)}</span>
                    <span class="edge-tone-${escapeHtml(valueEdgeTone(row))}">${escapeHtml(valueEdgeDisplay(row))}</span>
                  </div>
                  <div class="compute-market-track" aria-hidden="true">
                    <span class="compute-market-bar compute-market-bar-active" style="width:${activeShare}%"></span>
                    <span class="compute-market-bar compute-market-bar-opposition" style="width:${oppositionShare}%"></span>
                  </div>
                  <div class="compute-row-foot">
                    <span>${escapeHtml(route.read)}</span>
                    <span>${escapeHtml(
                      activePercent == null
                        ? "Public pricing pending"
                        : `${activePercent}% active / ${oppositionPercent == null ? "N/A" : `${oppositionPercent}%`} opposition`
                    )}</span>
                  </div>
                </article>
              `;
            })
            .join("")}
        </div>
      </article>
    `;
  };

  const marketStructureRows = (odds) => [
    {
      key: "FTR",
      label: "1X2",
      value:
        odds.home_win_odds && odds.draw_odds && odds.away_win_odds
          ? `${formatOdds(odds.home_win_odds)} / ${formatOdds(odds.draw_odds)} / ${formatOdds(odds.away_win_odds)}`
          : "N/A",
      meta:
        odds.home_win_odds && odds.draw_odds && odds.away_win_odds
          ? `H ${formatImpliedProbability(odds.home_win_odds)} • D ${formatImpliedProbability(odds.draw_odds)} • A ${formatImpliedProbability(odds.away_win_odds)}`
          : "No current 1X2 snapshot",
      percent:
        odds.home_win_odds && odds.draw_odds && odds.away_win_odds
          ? Math.max(impliedPercentValue(odds.home_win_odds) || 0, impliedPercentValue(odds.draw_odds) || 0, impliedPercentValue(odds.away_win_odds) || 0)
          : null,
    },
    {
      key: "OU25",
      label: "OU25",
      value:
        odds.over25_odds && odds.under25_odds
          ? `${formatOdds(odds.over25_odds)} / ${formatOdds(odds.under25_odds)}`
          : "N/A",
      meta:
        odds.over25_odds && odds.under25_odds
          ? `Over ${formatImpliedProbability(odds.over25_odds)} • Under ${formatImpliedProbability(odds.under25_odds)}`
          : "No current totals snapshot",
      percent:
        odds.over25_odds && odds.under25_odds ? Math.max(impliedPercentValue(odds.over25_odds) || 0, impliedPercentValue(odds.under25_odds) || 0) : null,
    },
    {
      key: "BTTS",
      label: "BTTS",
      value:
        odds.btts_yes_odds && odds.btts_no_odds
          ? `${formatOdds(odds.btts_yes_odds)} / ${formatOdds(odds.btts_no_odds)}`
          : "N/A",
      meta:
        odds.btts_yes_odds && odds.btts_no_odds
          ? `Yes ${formatImpliedProbability(odds.btts_yes_odds)} • No ${formatImpliedProbability(odds.btts_no_odds)}`
          : "No current BTTS snapshot",
      percent:
        odds.btts_yes_odds && odds.btts_no_odds ? Math.max(impliedPercentValue(odds.btts_yes_odds) || 0, impliedPercentValue(odds.btts_no_odds) || 0) : null,
    },
  ];

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
    const homeTeamId = String(fixture.api_home_team_id || "").trim() || extractTeamIdFromLogoUrl(fixture.home_team_logo_url);
    const awayTeamId = String(fixture.api_away_team_id || "").trim() || extractTeamIdFromLogoUrl(fixture.away_team_logo_url);
    const apiLeagueId = String(fixture.api_league_id || "").trim();
    const apiSeason = String(fixture.api_season || "").trim();
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
        data-api-league-id="${escapeHtml(apiLeagueId)}"
        data-api-season="${escapeHtml(apiSeason)}"
      >
        <div class="widget-reference-head fixture-table-reference-head">
          <div>
            <h4>League table</h4>
            <p class="muted">Reference context for this fixture. Use this as orientation, not as the decision layer.</p>
          </div>
        </div>
        <div class="widget-reference-frame">
          <div class="notice reference-loading">Loading league table…</div>
        </div>
      </div>
    `;
  };

  const fixtureLineupsWidgetMarkup = (fixture) => {
    const kickoffDate = String(fixture.kickoff_time || "").slice(0, 10);
    const homeTeamId = String(fixture.api_home_team_id || "").trim() || extractTeamIdFromLogoUrl(fixture.home_team_logo_url);
    const awayTeamId = String(fixture.api_away_team_id || "").trim() || extractTeamIdFromLogoUrl(fixture.away_team_logo_url);
    const apiFixtureId = String(fixture.api_fixture_id || "").trim();
    if (!kickoffDate || !fixture.home_team || !fixture.away_team) {
      return `<div class="notice">Lineups and formations are not available for this fixture yet because the widget identity mapping is incomplete.</div>`;
    }
    if (!workerConfigured()) {
      return `<div class="notice">Lineups and formations need the Worker proxy so the fixture resolver and widget API key stay off the page.</div>`;
    }
    return `
      <div
        class="widget-reference-shell"
        data-role="fixture-lineups-reference"
        data-date="${escapeHtml(kickoffDate)}"
        data-home="${escapeHtml(fixture.home_team)}"
        data-away="${escapeHtml(fixture.away_team)}"
        data-home-team-id="${escapeHtml(homeTeamId)}"
        data-away-team-id="${escapeHtml(awayTeamId)}"
        data-api-fixture-id="${escapeHtml(apiFixtureId)}"
      >
        <div class="widget-reference-head">
          <div>
            <span class="metric-label">Confirmed provider lineup</span>
            <h4>Official team sheets</h4>
          </div>
        </div>
        <p class="muted">When the provider publishes official teams, this section is the confirmed lineup. Until then, use the predicted lineup from last fixture below.</p>
        <div class="widget-reference-frame">
          <div class="lineup-empty-state">
            <div class="lineup-empty-grid">
              <article class="lineup-empty-card">
                <span class="metric-label">${escapeHtml(fixture.home_team)}</span>
                <div class="lineup-empty-skeleton"></div>
                <div class="lineup-empty-skeleton lineup-empty-skeleton-short"></div>
                <div class="lineup-empty-skeleton"></div>
              </article>
              <article class="lineup-empty-card">
                <span class="metric-label">${escapeHtml(fixture.away_team)}</span>
                <div class="lineup-empty-skeleton"></div>
                <div class="lineup-empty-skeleton lineup-empty-skeleton-short"></div>
                <div class="lineup-empty-skeleton"></div>
              </article>
            </div>
            <p class="muted">Confirmed lineups usually publish closer to kickoff. If team sheets are not available yet, the intelligence, table, and context layers stay live.</p>
          </div>
        </div>
      </div>
    `;
  };

  const fixtureTabHref = (tabKey) => {
    const params = new URLSearchParams();
    if (selectedFixtureKey) {
      params.set("fixture", selectedFixtureKey);
    }
    if (premiumDemoMode) {
      params.set("demo", "1");
    }
    if (debugMode) {
      params.set("debug", "1");
    }
    params.set("tab", String(tabKey || "intelligence"));
    return `./fixture.html?${params.toString()}#fixture-tab-${encodeURIComponent(String(tabKey || "intelligence"))}`;
  };

  const formatKickoffClock = (value) => {
    if (!value) {
      return "Time pending";
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

  const fixtureTimeState = (value) => {
    const timestamp = kickoffTimestamp(value);
    if (!Number.isFinite(timestamp)) {
      return {
        label: "Scheduled",
        detail: "Kickoff time pending",
        tone: "scheduled",
      };
    }
    const now = Date.now();
    const diff = timestamp - now;
    if (diff <= -3 * 60 * 60 * 1000) {
      return {
        label: "Final window",
        detail: formatKickoffClock(value),
        tone: "final",
      };
    }
    if (diff <= 30 * 60 * 1000) {
      return {
        label: "Kickoff window",
        detail: formatKickoffClock(value),
        tone: "live",
      };
    }
    return {
      label: "Upcoming",
      detail: formatKickoffClock(value),
      tone: "scheduled",
    };
  };

  const formatFixtureResultChip = (fixtureRow, teamSide) => {
    const goals = fixtureRow?.goals || {};
    const home = Number(goals.home);
    const away = Number(goals.away);
    if (!Number.isFinite(home) || !Number.isFinite(away)) {
      return { label: "PENDING", tone: "pending" };
    }
    if (home === away) {
      return { label: "DRAW", tone: "draw" };
    }
    const teamWon = teamSide === "home" ? home > away : away > home;
    return teamWon ? { label: "WIN", tone: "win" } : { label: "LOSS", tone: "loss" };
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
      <span class="${className} ${safeUrl ? "badge-has-image" : "badge-fallback-only"}" aria-hidden="true">
        <span class="badge-fallback">${initials}</span>
        ${safeUrl ? `<img src="${escapeHtml(safeUrl)}" alt="" loading="lazy" decoding="async" onerror="this.remove()" />` : ""}
      </span>
      <span class="sr-only">${label}</span>
    `;
  };

  const decisionStateTone = (signalState) => {
    const normalized = String(signalState || "").toUpperCase();
    if (normalized === "SUPPORTED") {
      return "deploy";
    }
    if (normalized === "WATCHLIST" || normalized === "MIXED") {
      return "observe";
    }
    return "reference";
  };

  const routeAuditProfile = (fixture, decision = null) => {
    const publishedSelection = fixturePublishedSelection(fixture);
    const routeActive = typeof decision?.route_active === "boolean" ? decision.route_active : Boolean(publishedSelection);
    const routeMarket = String(decision?.route_market || fixture?.signal_summary?.market_family || fixture?.deploy_summary?.market || "").toUpperCase();
    const routePick = String(decision?.route_pick || publishedSelection?.pick || fixture?.signal_summary?.deploy_pick || fixture?.deploy_summary?.pick || "").toUpperCase();
    const routeState = String(decision?.route_state || fixture?.publish_class || fixture?.signal_summary?.signal_state || "").toUpperCase();
    const auditState = String(decision?.audit_state || decision?.signal_state || "").toUpperCase();
    const conflictLevel = String(decision?.conflict_level || "").toUpperCase();
    const agreement = decision?.audit_agreement_score ?? decision?.agreement_score;
    const routeLabel = routeActive
      ? decision?.route_label || decision?.primary_signal || publishedSelection?.label || marketVerdictDisplay(fixture)
      : "No published pick";
    const contextLabel = decision?.context_signal || (!routeActive ? decision?.primary_signal : "") || "Fixture context";
    const conflictCopy =
      conflictLevel === "HARD_CONFLICT"
        ? "Hard conflict: route pick and context lean disagree."
        : conflictLevel === "CAUTION"
          ? "Caution: route remains live, but context audit is fragile."
          : routeActive
            ? "No route/audit conflict flagged."
            : "Context only: no published route is active.";
    return {
      routeActive,
      routeState,
      routeMarket,
      routePick,
      routeLabel,
      contextLabel,
      auditState,
      conflictLevel: conflictLevel || "NONE",
      agreement,
      confidence: decision?.route_confidence_tier || decision?.confidence_band || publishedSelection?.confidence || "",
      routeOdds: decision?.route_bookie_od || fixture?.deploy_summary?.bookie_od,
      conflictCopy,
      conflictLabel: conflictLevel === "HARD_CONFLICT" ? "Hard conflict" : conflictLevel === "CAUTION" ? "Caution conflict" : "No conflict",
      routeTone: routeActive ? "deploy" : "reference",
      auditTone: decisionStateTone(auditState),
      conflictTone: conflictLevel === "HARD_CONFLICT" ? "reference" : conflictLevel === "CAUTION" ? "observe" : "deploy",
    };
  };

  const decisionReasonRows = (decision, limit = 4) => {
    const supportRows = (decision?.supporting_layers || []).slice(0, 2).map((token) => ({
      tone: "support",
      text: reasonTokenLabel(token),
    }));
    const cautionRows = (decision?.caution_layers || []).slice(0, 2).map((token) => ({
      tone: "contradict",
      text: reasonTokenLabel(token),
    }));
    return [...supportRows, ...cautionRows].slice(0, limit);
  };

  const decisionTopCaution = (decision) => {
    const token = Array.isArray(decision?.caution_layers) ? decision.caution_layers[0] : null;
    return token ? reasonTokenLabel(token) : "No major caution has been published for this fixture yet.";
  };

  const decisionMarketSuitabilityItems = (decision) => {
    const marketIntelligence = decision?.market_intelligence || null;
    if (marketIntelligence && typeof marketIntelligence === "object") {
      return Object.entries(marketIntelligence)
        .map(([key, value]) => ({
          key,
          label: key === "ftr" ? "FTR" : key === "btts" ? "BTTS" : key === "ou25" ? "Over 2.5" : safeTitleLabel(key),
          rating: Number(value?.alignment_score),
          read: value?.public_summary || "No published read yet.",
          band: value?.state || "Pending",
          modelLean: value?.model_lean || "",
          modelOutput: value?.model_output || null,
          teamContextLean: value?.team_context_lean || "",
          support: Array.isArray(value?.structural_support) ? value.structural_support : [],
          cautions: Array.isArray(value?.cautions) ? value.cautions : [],
        }))
        .filter((entry) => Number.isFinite(entry.rating))
        .sort((left, right) => right.rating - left.rating || left.label.localeCompare(right.label));
    }
    const markets = decision?.market_suitability || null;
    if (!markets) {
      return [];
    }
    return [
      { label: "FTR", data: markets.ftr },
      { label: "BTTS", data: markets.btts },
      { label: "Over 2.5", data: markets.ou25 },
      { label: "Team Goals", data: markets.team_goals },
      { label: "Correct Score", data: markets.correct_score },
      { label: "Corners", data: markets.corners },
      { label: "Cards", data: markets.cards },
    ]
      .map((entry) => ({
        label: entry.label,
        rating: Number(entry.data?.rating),
        read: entry.data?.read || "No published read yet.",
        band: entry.data?.label || "Pending",
        modelLean: entry.data?.model_lean || "",
        modelOutput: entry.data?.model_output || null,
        teamContextLean: entry.data?.team_context_lean || "",
        support: Array.isArray(entry.data?.structural_support) ? entry.data.structural_support : [],
        cautions: Array.isArray(entry.data?.cautions) ? entry.data.cautions : [],
      }))
      .filter((entry) => Number.isFinite(entry.rating))
      .sort((left, right) => right.rating - left.rating || left.label.localeCompare(right.label));
  };

  const decisionMarketPosture = (decision) => {
    const items = decisionMarketSuitabilityItems(decision);
    if (!items.length) {
      return null;
    }
    const best = items[0] || null;
    const secondary = items[1] || null;
    const avoidCandidate =
      items.find((item) => String(item.band || "").toUpperCase() === "AVOID") ||
      [...items].reverse().find((item) => Number.isFinite(item.rating) && item.rating <= 44) ||
      null;
    const weakCandidate =
      items.find((item) => String(item.band || "").toUpperCase() === "FRAGILE") ||
      items.find((item) => String(item.band || "").toUpperCase() === "MIXED") ||
      items[Math.max(items.length - 1, 0)] ||
      null;
    return {
      best,
      secondary,
      weak: weakCandidate,
      avoid: avoidCandidate,
    };
  };

  const tokenHas = (items = [], needle = "") =>
    items.some((item) => String(item || "").toUpperCase().includes(String(needle || "").toUpperCase()));

  const allNumericValuesZero = (values = []) => {
    const usable = values.map((value) => Number(value)).filter((value) => Number.isFinite(value));
    return usable.length > 0 && usable.every((value) => value === 0);
  };

  const lineupCoverageProfile = (payload = null, decision = null) => {
    const homeProfiles = Array.isArray(payload?.home_lineup_profiles) ? payload.home_lineup_profiles : [];
    const awayProfiles = Array.isArray(payload?.away_lineup_profiles) ? payload.away_lineup_profiles : [];
    const homeUnits = payload?.home_units && typeof payload.home_units === "object" ? Object.values(payload.home_units) : [];
    const awayUnits = payload?.away_units && typeof payload.away_units === "object" ? Object.values(payload.away_units) : [];
    const publishedProfiles = homeProfiles.length + awayProfiles.length;
    const unitsZero = allNumericValuesZero([...homeUnits, ...awayUnits]);
    const statusText = String(
      payload?.lineup_status || payload?.lineup_mode || payload?.coverage_status || payload?.fallback_mode || ""
    ).toLowerCase();
    const explicitUnpublished = statusText.includes("unpublished") || statusText.includes("unavailable");
    const confirmed = statusText.includes("confirmed");
    const predicted = statusText.includes("predicted");
    const decisionMissing = tokenHas([...(decision?.caution_layers || []), ...(decision?.internal_reason_tokens || [])], "LINEUP_DATA_MISSING");
    const placeholder = !payload || explicitUnpublished || publishedProfiles === 0 || unitsZero;
    const status = placeholder ? "fallback" : confirmed ? "confirmed" : predicted ? "predicted" : "published";
    return {
      status,
      label: placeholder ? "Lineup unavailable" : confirmed ? "Confirmed lineup" : predicted ? "Predicted from last fixture" : "Lineup published",
      tone: placeholder ? "reference" : confirmed ? "deploy" : predicted ? "observe" : "deploy",
      summary:
        payload?.summary ||
        (placeholder || decisionMissing
          ? "No publish-safe lineup snapshot is available for this fixture key yet, so the page leans on squad and team intelligence."
          : predicted
            ? "Predicted lineups are built from each team's most recent published lineup and bench snapshot."
            : "Published lineup profiles are active for this fixture."),
      profileCount: publishedProfiles,
    };
  };

  const h2hCoverageProfile = (payload = null, decision = null) => {
    const context = decision?.h2h_context || null;
    const fallbackMode = String(payload?.fallback_mode || "").toLowerCase();
    const directAvailable = Boolean(context?.available || (payload && Number(payload.sample_size || 0) > 0 && fallbackMode !== "unpublished"));
    const historicalFallback = fallbackMode === "historical_team_pair";
    const unpublished = !payload || fallbackMode === "unpublished" || context?.available === false;
    return {
      status: directAvailable ? (historicalFallback ? "historical" : "published") : "fallback",
      label: directAvailable ? (historicalFallback ? "Historical pair" : "H2H published") : "H2H unpublished",
      tone: directAvailable ? (historicalFallback ? "observe" : "deploy") : "reference",
      summary:
        payload?.summary ||
        context?.summary ||
        (unpublished
          ? "No publish-safe H2H regime summary is available for this fixture key yet; history stays supporting-only."
          : "Published H2H context is available as supporting evidence."),
      sampleSize: payload?.sample_size ?? context?.sample_size ?? 0,
    };
  };

  const renderFixtureCoverageTruthStrip = (fixture, decision = null, lineup = null, h2hSupport = null) => {
    const lineupProfile = lineupCoverageProfile(lineup, decision);
    const h2hProfile = h2hCoverageProfile(h2hSupport, decision);
    const playerDriverCount = Array.isArray(decision?.key_player_drivers) ? decision.key_player_drivers.length : 0;
    const squadFallbackActive = playerDriverCount === 0 || tokenHas(decision?.caution_layers || [], "LINEUP_DATA_MISSING");
    const marketCount = decisionMarketSuitabilityItems(decision).length;
    const lineupModelActive = ["published", "confirmed", "predicted"].includes(lineupProfile.status);
    const lineupCopy =
      lineupProfile.status === "confirmed"
        ? "Confirmed provider lineup"
        : lineupProfile.status === "predicted"
          ? "Predicted lineups from last fixture"
          : "Lineup fallback available";
    const h2hCopy =
      h2hProfile.status === "historical"
        ? "Historical matchup context"
        : h2hProfile.status === "published"
          ? "H2H context active"
          : "H2H context unavailable";
    return `
      <section class="section section-tight">
        <article class="intel-coverage-strip" aria-label="Published intelligence coverage">
          <div class="intel-coverage-copy">
            <span class="metric-label">Intelligence status</span>
            <strong>${escapeHtml(lineupModelActive && h2hProfile.status !== "fallback" ? "Supporting layers active" : "Decision read remains live")}</strong>
            <p>${escapeHtml("The page explains what is real, what is predicted, and what is deliberately left out.")}</p>
          </div>
          <div class="intel-coverage-grid">
            <article class="intel-coverage-item intel-coverage-item-${escapeHtml(lineupProfile.tone)}">
              <span>Lineups</span>
              <strong>${escapeHtml(lineupCopy)}</strong>
            </article>
            <article class="intel-coverage-item intel-coverage-item-${escapeHtml(h2hProfile.tone)}">
              <span>H2H</span>
              <strong>${escapeHtml(h2hCopy)}</strong>
            </article>
            <article class="intel-coverage-item intel-coverage-item-${escapeHtml(squadFallbackActive ? "observe" : "deploy")}">
              <span>Player drivers</span>
              <strong>${escapeHtml(squadFallbackActive ? "From squad model" : "Fixture player layer")}</strong>
            </article>
            <article class="intel-coverage-item intel-coverage-item-${escapeHtml(marketCount ? "deploy" : "reference")}">
              <span>Markets</span>
              <strong>${escapeHtml(marketCount ? "Market read active" : "Market read pending")}</strong>
            </article>
          </div>
        </article>
      </section>
    `;
  };

  const fixtureFreshnessMeta = (fixture, decision = null, lineup = null, h2hSupport = null) => {
    const lastUpdated =
      fixture?.updated_at ||
      fixture?.capture_generated_at ||
      fixture?.source_data_cutoff_at ||
      decision?.generated_at ||
      state.summary?.generated_at ||
      "";
    const nextRefresh =
      fixture?.next_refresh_at ||
      decision?.next_refresh_at ||
      state.summary?.next_refresh_at ||
      "Next publish automation";
    const coverage = String(fixture?.coverage_status || decision?.coverage_status || "coverage pending").replace(/_/g, " ");
    const sourceCutoff = fixture?.source_data_cutoff_at || fixture?.capture_generated_at || state.summary?.selected_source_mtime_utc || "";
    return [
      {
        label: "Last updated",
        value: formatDateTime(lastUpdated) || "Not published",
        note: fixture?.snapshot_phase ? `Snapshot ${String(fixture.snapshot_phase).replace(/_/g, " ")}` : "Published fixture feed",
      },
      {
        label: "Next refresh",
        value: formatDateTime(nextRefresh) || nextRefresh,
        note: "Refresh timing is shown from publish metadata when available.",
      },
      {
        label: "Coverage",
        value: safeTitleLabel(coverage, "Coverage pending"),
        note: [
          lineup ? "lineups" : "lineup fallback",
          h2hSupport ? "h2h" : "h2h fallback",
          decision ? "decision" : "decision fallback",
        ].join(" · "),
      },
      {
        label: "Data cutoff",
        value: formatDateTime(sourceCutoff) || "Source cutoff pending",
        note: fixture?.fixture_kickoff_source ? `Kickoff from ${String(fixture.fixture_kickoff_source).replace(/_/g, " ")}` : "Website-safe export",
      },
    ];
  };

  const renderFixtureFreshnessPanel = (fixture, decision = null, lineup = null, h2hSupport = null) => `
    <section class="section section-tight">
      <article class="panel freshness-panel">
        <div class="section-head">
          <div>
            <span class="metric-label">Freshness</span>
            <h2>Data status and coverage</h2>
          </div>
        </div>
        <div class="prediction-meta-grid dashboard-odds-grid freshness-grid">
          ${fixtureFreshnessMeta(fixture, decision, lineup, h2hSupport)
            .map(
              (entry) => `
                <div class="signal-cell signal-cell-model">
                  <span class="signal-label">${escapeHtml(entry.label)}</span>
                  <span class="signal-value">${escapeHtml(entry.value)}</span>
                  <span class="muted">${escapeHtml(entry.note)}</span>
                </div>
              `
            )
            .join("")}
        </div>
      </article>
    </section>
  `;

  const renderTeamOverviewDrivers = (payload) => {
    if (!payload || !Array.isArray(payload.players) || !payload.players.length) {
      return "";
    }
    const featuredPlayers = payload.players.slice(0, 3);
    return `
      <section class="section">
        <div class="split">
          <article class="panel">
            <h3>Player drivers</h3>
            <p class="section-copy">This is the first player layer for the team desk: who is carrying the squad profile, where the threat sits, and where the caution lives.</p>
            <ul class="feature-list compact-list">
              ${renderSquadLeaderList("Power", payload?.leaders?.power)}
              ${renderSquadLeaderList("Goal threat", payload?.leaders?.goal_threat)}
              ${renderSquadLeaderList("Creative spark", payload?.leaders?.creative_spark)}
              ${renderSquadLeaderList("Discipline risk", payload?.leaders?.discipline_risk)}
            </ul>
          </article>
          <article class="panel">
            <h3>Featured squad profiles</h3>
            <div class="card-grid card-grid-compact">
              ${featuredPlayers.map((player) => renderPlayerIntelligenceCard(player)).join("")}
            </div>
          </article>
        </div>
      </section>
    `;
  };

  const renderFixtureHeroScoreboard = (fixture, clarity) => {
    const leagueBadge = safeLogoUrl(fixture.league_logo_url || fixture.league_flag_url);
    const timing = fixtureTimeState(fixture.kickoff_time);
    const heroMode = timing.tone === "scheduled" ? "editorial" : "scoreboard";
    const homeTeamId = String(fixture.api_home_team_id || "").trim() || extractTeamIdFromLogoUrl(fixture.home_team_logo_url);
    const awayTeamId = String(fixture.api_away_team_id || "").trim() || extractTeamIdFromLogoUrl(fixture.away_team_logo_url);
    return `
      <div
        class="fixture-hero-scoreboard fixture-hero-scoreboard-${escapeHtml(heroMode)}"
        data-role="fixture-scoreboard"
        data-hero-mode="${escapeHtml(heroMode)}"
        data-api-fixture-id="${escapeHtml(String(fixture.api_fixture_id || ""))}"
        data-kickoff-time="${escapeHtml(String(fixture.kickoff_time || ""))}"
        data-date="${escapeHtml(String(fixture.kickoff_time || "").slice(0, 10))}"
        data-home="${escapeHtml(fixture.home_team || "")}"
        data-away="${escapeHtml(fixture.away_team || "")}"
        data-home-team-id="${escapeHtml(homeTeamId)}"
        data-away-team-id="${escapeHtml(awayTeamId)}"
      >
        <div class="fixture-hero-score-row">
          <div class="fixture-hero-side">
            <a class="fixture-entity-link" href="${teamPageHref(fixture.home_team)}"><strong>${escapeHtml(fixture.home_team)}</strong></a>
            ${badgeMarkup(fixture.home_team_logo_url, fixture.home_team, "match-hero-badge")}
          </div>
          <div class="fixture-hero-center">
            <span class="metric-label">${escapeHtml(timing.label)}</span>
            <strong class="fixture-hero-score">${escapeHtml(heroMode === "editorial" ? "vs" : "—")}</strong>
            <span class="muted">${escapeHtml(timing.detail)}</span>
          </div>
          <div class="fixture-hero-side fixture-hero-side-end">
            ${badgeMarkup(fixture.away_team_logo_url, fixture.away_team, "match-hero-badge")}
            <a class="fixture-entity-link" href="${teamPageHref(fixture.away_team)}"><strong>${escapeHtml(fixture.away_team)}</strong></a>
          </div>
        </div>
        <div class="fixture-scorer-slot" data-role="fixture-scorers">${renderFixtureScorerStrip(collectGoalScorerRows(fixture, fixture))}</div>
        <div class="fixture-hero-meta fixture-hero-meta-bottom">
          <span class="fixture-hero-meta-item">${escapeHtml(timing.detail)}</span>
          <a class="fixture-competition-mark" href="${competitionPageHref(fixture.league)}">
            ${
              leagueBadge
                ? `<img class="league-badge" src="${escapeHtml(leagueBadge)}" alt="" loading="lazy" decoding="async" onerror="this.remove()" />`
                : ""
            }
            <span>${escapeHtml(fixture.league || "Competition")}</span>
          </a>
          <span class="fixture-status-badge fixture-status-badge-${escapeHtml(timing.tone)}">${escapeHtml(timing.label)}</span>
        </div>
      </div>
    `;
  };

  const renderFixtureHeroMedia = (fixture) => {
    const fixtureKey = String(fixture?.fixture_key || "");
    const externalMedia = state.selectedFixtureExternalContent?.fixture_key === fixtureKey
      ? state.selectedFixtureExternalContent?.media
      : null;
    const media =
      (Array.isArray(externalMedia) ? externalMedia.find((item) => item?.type === "youtube_embed" && item?.embed_url) : null) ||
      FIXTURE_HERO_MEDIA_FALLBACK[fixtureKey];
    const embedSrc = media?.embed_url || media?.src;
    if (!embedSrc) {
      return "";
    }
    return `
      <section class="section fixture-hero-media-section">
        <div class="fixture-hero-media-copy">
          ${media.label ? `<span>${escapeHtml(media.label)}</span>` : ""}
          <strong>${escapeHtml(media.heading || "Fixture video")}</strong>
          ${media.summary ? `<p>${escapeHtml(media.summary)}</p>` : ""}
        </div>
        <div class="fixture-hero-media-frame">
          <iframe
            src="${escapeHtml(embedSrc)}"
            title="${escapeHtml(media.title || media.heading || "Fixture video")}"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            referrerpolicy="strict-origin-when-cross-origin"
            loading="lazy"
            allowfullscreen
          ></iframe>
        </div>
      </section>
    `;
  };

  const weatherBadgeIcon = (badge) => {
    const key = String(badge || "unknown").toLowerCase();
    const iconClass = `og-weather-icon og-weather-icon-${escapeHtml(key)}`;
    const sun = `<circle class="og-weather-sun" cx="34" cy="28" r="11"></circle><path class="og-weather-ray" d="M34 8v7M34 41v7M14 28h7M47 28h7M20 14l5 5M48 14l-5 5M20 42l5-5M48 42l-5-5"></path>`;
    const sunCloudSun = `<circle class="og-weather-sun" cx="52" cy="29" r="9"></circle><path class="og-weather-ray" d="M52 13v5M52 40v5M36 29h5M63 29h5M40 17l4 4M64 17l-4 4M40 41l4-4M64 41l-4-4"></path>`;
    const cloud = `<path class="og-weather-cloud" d="M23 43h25c6 0 10-4 10-9s-4-9-9-9c-2-8-9-13-17-11-7 1-12 7-13 14-6 1-10 5-10 10 0 3 3 5 7 5h7z"></path>`;
    const rain = `<path class="og-weather-rain" d="M23 51l-4 8M36 51l-4 8M49 51l-4 8"></path>`;
    const snow = `<path class="og-weather-snow" d="M25 54h10M30 49v10M45 54h10M50 49v10"></path>`;
    const wind = `<path class="og-weather-wind" d="M11 29h34c5 0 8-3 8-7s-3-7-7-7c-3 0-5 1-7 4M15 41h28c5 0 8 3 8 7s-3 7-7 7c-3 0-5-1-7-4"></path>`;
    const bolt = `<path class="og-weather-bolt" d="M36 33l-8 16h9l-5 13 14-19h-9l6-10z"></path>`;
    const thermometer = `<path class="og-weather-thermo" d="M34 13v26M25 47a9 9 0 1 0 18 0 9 9 0 0 0-4-7V13a5 5 0 0 0-10 0v27a9 9 0 0 0-4 7z"></path>`;
    const question = `<path class="og-weather-unknown" d="M28 25c1-7 12-8 15-2 4 8-7 10-7 17M36 50v2"></path>`;
    const content = {
      sunny: sun,
      "sun-cloud": `${sunCloudSun}${cloud}`,
      cloudy: cloud,
      rain: `${cloud}${rain}`,
      snow: `${cloud}${snow}`,
      wind,
      storm: `${cloud}${bolt}${rain}`,
      cold: `${thermometer}${snow}`,
      hot: `${sun}${thermometer}`,
      unknown: question,
    }[key] || question;
    return `<svg class="${iconClass}" viewBox="0 0 68 68" aria-hidden="true" focusable="false">${content}</svg>`;
  };

  const renderWeatherBadge = (weather, size = "large") => {
    const badge = String(weather?.badge || weather?.condition || "unknown").toLowerCase();
    const label = weather?.label || weather?.condition || "Weather monitored";
    return `
      <div class="og-weather-badge og-weather-badge-${escapeHtml(size)} og-weather-badge-${escapeHtml(badge)}" title="${escapeHtml(label)}">
        ${weatherBadgeIcon(badge)}
        <span>${escapeHtml(label)}</span>
      </div>
    `;
  };

  const weatherDragLabel = (weather) => {
    if (weather?.drag_label) return weather.drag_label;
    if (weather?.weather_drag_label) return weather.weather_drag_label;
    const severity = Number(weather?.severity_score || 0);
    if (!Number.isFinite(severity) || severity <= 1) return "Low";
    if (severity <= 3) return "Raised";
    if (severity <= 5) return "High";
    return "Severe";
  };

  const renderSpaceWeatherBadge = (spaceWeather) => {
    const label = spaceWeather?.alert_level || "Monitor";
    return `
      <div class="og-space-weather-badge" title="${escapeHtml(label)}">
        <svg class="og-space-weather-icon" viewBox="0 0 72 72" aria-hidden="true" focusable="false">
          <circle class="og-space-weather-core" cx="36" cy="36" r="7"></circle>
          <ellipse class="og-space-weather-orbit" cx="36" cy="36" rx="24" ry="10" transform="rotate(-24 36 36)"></ellipse>
          <ellipse class="og-space-weather-orbit" cx="36" cy="36" rx="24" ry="10" transform="rotate(24 36 36)"></ellipse>
          <path class="og-space-weather-pulse" d="M20 52c4 4 9 6 16 6s12-2 16-6"></path>
          <circle class="og-space-weather-dot" cx="55" cy="28" r="3"></circle>
        </svg>
        <span>${escapeHtml(label)}</span>
      </div>
    `;
  };

  const renderFixtureWeatherContext = (fixture) => {
    const fixtureKey = String(fixture?.fixture_key || "");
    const context = state.selectedFixtureExternalContent?.fixture_key === fixtureKey ? state.selectedFixtureExternalContent : null;
    const weather = Array.isArray(context?.weather_signals) ? context.weather_signals[0] : null;
    const spaceWeather = Array.isArray(context?.space_weather_signals) ? context.space_weather_signals[0] : null;
    if (!weather && !spaceWeather) {
      return "";
    }
    const weatherMetrics = weather
      ? [
          ["Temp", weather.temperature_c !== undefined ? `${Number(weather.temperature_c).toFixed(0)}°C` : ""],
          ["Rain", weather.precip_mm !== undefined ? `${Number(weather.precip_mm).toFixed(1)}mm` : ""],
          ["Wind", weather.wind_kmh !== undefined ? `${Number(weather.wind_kmh).toFixed(0)} km/h` : ""],
          ["Weather Drag", weatherDragLabel(weather)],
        ].filter((item) => item[1])
      : [];
    const weatherNotes = Array.isArray(weather?.interpretation) ? weather.interpretation.slice(0, 3) : [];
    const spaceNotes = Array.isArray(spaceWeather?.interpretation) ? spaceWeather.interpretation.slice(0, 2) : [];
    const spaceHeading = spaceWeather?.heading || "Space Weather";
    const spaceHeadingMarkup = String(spaceHeading).trim().toLowerCase() === "space weather" ? "Space<br>Weather" : escapeHtml(spaceHeading);
    return `
      <section class="section fixture-context-weather-section">
        <article class="fixture-weather-card">
          <div class="fixture-weather-copy">
            <span class="metric-label">${escapeHtml(weather?.provider || "Weather overlay")}</span>
            <h2>${escapeHtml(weather?.heading || "Weather Forecast")}</h2>
            ${weather ? renderWeatherBadge(weather, "compact") : ""}
            <p>${escapeHtml(weather?.summary || "Weather context is monitored as a soft fixture layer.")}</p>
          </div>
          ${weather ? renderWeatherBadge(weather, "large") : ""}
          ${
            weatherMetrics.length
              ? `<div class="fixture-weather-metrics">${weatherMetrics
                  .map(
                    ([label, value]) => `
                      <div class="fixture-weather-metric">
                        <strong>${escapeHtml(value)}</strong>
                        <span>${escapeHtml(label)}</span>
                      </div>
                    `
                  )
                  .join("")}</div>`
              : ""
          }
          ${weatherNotes.length ? `<ul class="fixture-weather-notes">${weatherNotes.map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>` : ""}
        </article>
        <article class="fixture-space-weather-card">
          <div class="fixture-space-weather-head">
            <div>
              <span class="metric-label">${escapeHtml(spaceWeather?.provider || "Space weather")}</span>
              <h3>${spaceHeadingMarkup}</h3>
            </div>
            ${renderSpaceWeatherBadge(spaceWeather)}
          </div>
          <p>${escapeHtml(spaceWeather?.summary || "No environmental volatility alert is applied.")}</p>
          ${spaceNotes.length ? `<ul class="fixture-weather-notes fixture-weather-notes-compact">${spaceNotes.map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>` : ""}
        </article>
      </section>
    `;
  };

  const sourceHostname = (value) => {
    try {
      return new URL(String(value || "")).hostname.replace(/^www\./, "");
    } catch {
      return "";
    }
  };

  const renderNewsSignalCards = (signals, emptyCopy = "No source-linked news signals are published for this page yet.") => {
    const rows = Array.isArray(signals) ? signals.filter((item) => item?.title || item?.source_url).slice(0, 8) : [];
    if (!rows.length) {
      return `<div class="notice">${escapeHtml(emptyCopy)}</div>`;
    }
    return `
      <div class="card-grid news-signal-grid">
        ${rows
          .map((item) => {
            const url = item.source_url || item.url || "";
            const host = sourceHostname(url);
            const tags = Array.isArray(item.tags) ? item.tags.slice(0, 3) : [];
            return `
              <article class="panel news-signal-card">
                <div class="news-signal-head">
                  <span class="metric-label">${escapeHtml(item.provider || host || "Source")}</span>
                  <span class="stat-chip">${escapeHtml(String(item.usage_mode || item.type || "source").replace(/_/g, " "))}</span>
                </div>
                <h3>${escapeHtml(item.title || "Source-linked football update")}</h3>
                <p class="muted">${escapeHtml(item.summary || "Headline and source link are stored as context. Full article remains with the publisher.")}</p>
                ${
                  tags.length
                    ? `<div class="pill-row">${tags.map((tag) => `<span class="chip chip-reference">${escapeHtml(String(tag).replace(/_/g, " "))}</span>`).join("")}</div>`
                    : ""
                }
                ${
                  url
                    ? `<a class="ghost-button news-source-link" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">Open source${host ? ` · ${escapeHtml(host)}` : ""}</a>`
                    : ""
                }
              </article>
            `;
          })
          .join("")}
      </div>
    `;
  };

  const renderFixtureNewsSection = (fixture) => {
    const fixtureKey = String(fixture?.fixture_key || "");
    const context = state.selectedFixtureExternalContent?.fixture_key === fixtureKey ? state.selectedFixtureExternalContent : null;
    const signals = Array.isArray(context?.news_signals) ? context.news_signals : [];
    return `
      <section class="section">
        <div class="section-head">
          <div>
            <h2>News Signals</h2>
            <p class="section-copy">Source-linked club and publisher context for this fixture. Odds Genius stores headlines, links, and interpretation only; full articles stay with the original publisher.</p>
          </div>
        </div>
        ${renderNewsSignalCards(signals, "No source-linked news signals are published for this fixture yet.")}
      </section>
    `;
  };

  const renderTeamNewsSection = (teamName) => {
    const signals = Array.isArray(state.selectedTeamExternalContent?.news_signals) ? state.selectedTeamExternalContent.news_signals : [];
    return `
      <section class="section">
        <div class="section-head">
          <div>
            <h2>${escapeHtml(teamName)} News Signals</h2>
            <p class="section-copy">Official club sources and publisher links for team context. This is an intelligence feed, not a republished news site.</p>
          </div>
        </div>
        ${renderNewsSignalCards(signals, "No source-linked news signals are published for this team yet.")}
      </section>
    `;
  };

  const renderFixturePredictionDeck = (fixture, clarity, matchedEntry, publishClass) => {
    const decision = state.selectedFixtureDecisionIntelligence || null;
    const verdictLabel = marketVerdictDisplay(fixture);
    const routeAudit = routeAuditProfile(fixture, decision);
    const deckLabel = routeAudit.routeActive ? "Published prediction" : "Fixture context";
    const deckTitle = routeAudit.routeLabel || verdictLabel;
    const deckCopy = routeAudit.routeActive
      ? routeAudit.conflictLevel === "CAUTION" || routeAudit.conflictLevel === "HARD_CONFLICT"
        ? `${fixture.signal_summary?.summary_text || clarity.action_copy} Context audit flags ${safeTitleLabel(routeAudit.auditState, "caution")} at ${routeAudit.agreement ?? "—"}% agreement, so the deeper cards are caution context rather than a second selection.`
        : fixture.signal_summary?.summary_text || decision?.preview?.short_summary || decision?.public_safe_summary || clarity.action_copy
      : `This fixture is ${safeTitleLabel(publishClass || fixture.publish_class || "observe", "context")} only. Market cards show context posture, pricing, and cautions, but no Odds Genius pick is published for this fixture.`;
    return `
      <section class="section fixture-prediction-section">
        <article class="panel fixture-prediction-card">
          <div>
            <span class="metric-label">${escapeHtml(deckLabel)}</span>
            <h2>${escapeHtml(deckTitle || verdictLabel)}</h2>
            <p>${escapeHtml(deckCopy)}</p>
          </div>
          <div class="cta-row">
            <a class="button" href="./dashboard.html">Back to dashboard</a>
            <a class="ghost-button" href="./premium.html">Open premium board</a>
            <button class="ghost-button" type="button" data-action="telegram-fixture-alert" data-fixture-key="${escapeHtml(String(fixture.fixture_key || ""))}">Send to Telegram</button>
          </div>
        </article>
        <div class="fixture-prediction-support-grid">
          ${renderFixtureHeroDecisionAside(fixture, clarity, matchedEntry)}
        </div>
      </section>
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
          <span class="team-side team-side-home">
            ${badgeMarkup(row.home_team_logo_url, row.home_team)}
            <span class="team-name">${escapeHtml(teamCardName(row.home_team))}</span>
          </span>
          <span class="versus">vs</span>
          <span class="team-side team-side-away">
            <span class="team-name">${escapeHtml(teamCardName(row.away_team))}</span>
            ${badgeMarkup(row.away_team_logo_url, row.away_team)}
          </span>
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

  const predictionCard = (row, locked, entryIndex = 0) => {
    const shortlist = Array.isArray(row.correct_score_shortlist) ? row.correct_score_shortlist : [];
    const edge = locked ? row.value_edge_display || edgeLabel(row) : edgeLabel(row);
    const edgeTone = predictionEdgeTone(row);
    const metricLine = compactMetricText(row);
    const reasonText = cardReasonText(row);
    return `
      <article class="card prediction-card prediction-card-${escapeHtml(edgeTone)}" style="--enter-index:${entryIndex}">
        <div class="prediction-top">
          ${fixtureTeamsMarkup(row)}
          <div class="pill-row">
            <span class="market-badge">${escapeHtml(row.market)}</span>
            ${
              shouldShowTierChip(row.confidence_tier)
                ? `<span class="confidence-badge ${tierClass(row.confidence_tier)}">${escapeHtml(row.confidence_tier)}</span>`
                : ""
            }
          </div>
        </div>
        <div class="prediction-core">
          <div class="prediction-call">
            <div>
              <strong class="prediction-pick">${escapeHtml(row.pick)}</strong>
              ${metricLine ? `<span class="prediction-metric-line">${escapeHtml(metricLine)}</span>` : ""}
            </div>
            <div class="prediction-edge">
              <span class="prediction-edge-chip prediction-edge-chip-${escapeHtml(edgeTone)}">${escapeHtml(`EV ${edge}`)}</span>
            </div>
          </div>
        </div>
        ${reasonText ? `<p class="muted prediction-rationale">${escapeHtml(reasonText)}</p>` : ""}
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

  const capabilityPill = (label, detail = "") => `
    <article class="capability-pill">
      <strong>${escapeHtml(label)}</strong>
      ${detail ? `<span>${escapeHtml(detail)}</span>` : ""}
    </article>
  `;

  const launchCapabilityGrid = (title, items, note = "") => `
    <article class="panel launch-capability-panel">
      <span class="metric-label">${escapeHtml(title)}</span>
      <div class="launch-capability-grid">
        ${items.map((item) => capabilityPill(item.label || item, item.detail || "")).join("")}
      </div>
      ${note ? `<p class="muted">${escapeHtml(note)}</p>` : ""}
    </article>
  `;

  const worldCupFounderModule = () => `
    <section class="section world-cup-founder-module">
      <article class="world-cup-founder-copy">
        <span class="metric-label">OG Founder Early Access</span>
        <h2>World Cup + pre-season edition.</h2>
        <p>
          A launch window for public proof, premium fixture context, and player-event beta intelligence while the product hardens in the open.
        </p>
        <div class="pill-row">
          <span class="stat-chip">First 250 founders</span>
          <span class="stat-chip">£20/month while active</span>
          <span class="stat-chip">Protected premium route</span>
        </div>
      </article>
      <article class="world-cup-founder-actions">
        <div class="metric">
          <span class="metric-label">Launch edition</span>
          <span class="metric-value">Football v0.12</span>
        </div>
        <div class="cta-row">
          <a class="button" href="./pricing.html">Secure founder access</a>
          <a class="ghost-button" href="./methodology.html">Read methodology</a>
          <a class="ghost-button" href="./results.html">See live proof</a>
        </div>
      </article>
    </section>
  `;

  const marketLabelCanonical = (value) => {
    const key = String(value || "").toUpperCase().replace(/[^A-Z0-9]/g, "");
    if (key === "OU25" || key === "OVER25") return "OU25";
    if (key === "BTTS") return "BTTS";
    if (key === "FTR") return "FTR";
    if (key === "TG15" || key === "TEAMGOALS" || key === "TEAMGOALS15") return "TG1.5";
    return value || "Market";
  };

  const resultItemsByMarket = (window = {}) => {
    const rows = Array.isArray(window.featured_results) ? window.featured_results : Array.isArray(window.items) ? window.items : [];
    return rows.reduce((groups, row) => {
      const market = marketLabelCanonical(row.market);
      groups[market] = groups[market] || [];
      groups[market].push(row);
      return groups;
    }, {});
  };

  const recentResultPreview = (feed) => {
    const window = Array.isArray(feed?.windows) ? feed.windows[0] || null : null;
    if (!window) {
      return `<div class="notice">Recent live proof will appear here after the next settlement publish.</div>`;
    }
    const grouped = resultItemsByMarket(window);
    const cards = ["FTR", "BTTS", "OU25", "TG1.5"]
      .map((market) => {
        const rows = grouped[market] || [];
        const settled = rows.filter((row) => ["won", "lost", "void", "cashed"].includes(String(row.result_status || "").toLowerCase())).length;
        const wins = rows.filter((row) => ["won", "cashed"].includes(String(row.result_status || "").toLowerCase())).length;
        const pending = rows.filter((row) => String(row.result_status || "").toLowerCase() === "pending").length;
        const rate = settled ? compactPercent(wins / settled) : "Pending";
        return `
          <article class="proof-market-split-card">
            <span class="metric-label">${escapeHtml(market)}</span>
            <strong>${escapeHtml(rate)}</strong>
            <span class="muted">${escapeHtml(`${wins}/${settled} settled${pending ? `, ${pending} pending` : ""}`)}</span>
          </article>
        `;
      })
      .join("");
    return `
      <div class="proof-market-split">${cards}</div>
      <p class="muted">Markets stay separated so totals, BTTS, result, and TG1.5 proof are never blended into one vague headline.</p>
    `;
  };

  const renderResultsMarketSplit = (weekly = {}) => {
    const weeklyMarketRollups = (weekly.by_market || []).reduce((acc, item) => {
      acc[marketLabelCanonical(item.market)] = item;
      return acc;
    }, {});
    return `
      <div class="stats-grid">
        ${["FTR", "BTTS", "OU25", "TG1.5"]
          .map((market) => {
            const item = weeklyMarketRollups[market] || {
              market,
              hit_rate: null,
              settled_picks: 0,
              total_picks: 0,
            };
            return `
              <article class="panel market-proof-card market-proof-card--${resultsStatusTone(item.hit_rate)}">
                <span class="muted">${escapeHtml(market)}</span>
                <strong>${escapeHtml(item.hit_rate == null ? "Pending" : compactPercent(item.hit_rate))}</strong>
                <span>${escapeHtml(`${item.settled_picks || 0}/${item.total_picks || 0} settled`)}</span>
              </article>
            `;
          })
          .join("")}
      </div>
    `;
  };

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
    <section class="hero launch-hero">
      <div class="hero-main launch-hero-main">
        <div class="launch-hero-copy">
          <p class="hero-kicker launch-system-title">Sports Prediction Intelligence System</p>
          <h1 class="launch-title">Football <span>v0.1.2</span></h1>
          <div class="launch-proof-lines" aria-label="System positioning">
            <strong>Advanced Machine Learning Systems.</strong>
            <strong>Exclusive Modelling Architecture.</strong>
            <strong>Industry Leading Benchmarks.</strong>
          </div>
          <div class="launch-founder-headline">
            <span>OG Founder<br class="mobile-break" /> Early Access</span>
            <strong>Memberships<br class="mobile-break" /> Now Open.</strong>
          </div>
          <h2 class="launch-window-title">World Cup 2026 <span>+ 26/27 Pre-Season<br class="mobile-break" /> Membership</span></h2>
          <div class="hero-actions launch-hero-actions">
            <a class="button button-large" href="./pricing.html">Secure founder access</a>
            <a class="ghost-button" href="./results.html">See public proof</a>
            <a class="ghost-button" href="./matches.html">Open matches desk</a>
          </div>
          <p class="footer-note">Historical walk-forward validation. Not a guarantee of future results.</p>
        </div>
      </div>
      <aside class="hero-side launch-founder-side">
        <article class="launch-founder-card">
          <span class="metric-label">Founder access</span>
          <h3>Memberships now open.</h3>
          <div class="launch-founder-metrics">
            <div>
              <strong>First 250</strong>
              <span>founders</span>
            </div>
            <div>
              <strong>£20/month</strong>
              <span>while active</span>
            </div>
            <div>
              <strong>Protected</strong>
              <span>premium route</span>
            </div>
            <div>
              <strong>Launch edition</strong>
              <span>Football v0.1.2</span>
            </div>
          </div>
        </article>
      </aside>
    </section>

    <section class="section split launch-capabilities-section">
      ${launchCapabilityGrid(
        "Models",
        [
          { label: "FTR", detail: "Home / draw / away posture" },
          { label: "BTTS", detail: "Two-way scoring pressure" },
          { label: "Over 2.5", detail: "Match goal total shape" },
          { label: "TG1.5", detail: "Team over 1.5 support" },
          { label: "Goal combos", detail: "Aligned market families" },
        ],
        "Public pages show the approved signal. Premium adds the fixture intelligence behind the read."
      )}
      ${launchCapabilityGrid(
        "Player Events",
        [
          "Shots",
          "Shots on Target",
          "Tackles",
          "Fouls",
          "Player Fouled",
          "Key Passes",
          "Goalkeeper Saves",
          "Corners",
          "Bookings",
        ],
        "Player-event surfaces are beta intelligence cards for review, not public-priced prop tips."
      )}
    </section>

    <section class="section proof-command launch-proof-section">
      <div class="section-head home-proof-head">
        <div>
          <h2>Walk-forward proof, not vibes</h2>
          <p class="section-copy">Benchmark-safe proof across the current football intelligence stack, with live public results settled separately.</p>
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
    </section>

    <section class="section split">
      <article class="panel featured-proof-panel">
        <span class="metric-label">Recent public live results</span>
        <h3>Settled outcomes stay visible.</h3>
        ${recentResultPreview(state.liveResultsFeed)}
        <div class="cta-row">
          <a class="button" href="./results.html">Open results page</a>
        </div>
      </article>
      <article class="panel">
        <span class="metric-label">Founder Early Access</span>
        <h3>World Cup + pre-season edition.</h3>
        <p class="muted">
          Founder access is a discounted early seat for the football intelligence system: core fixture reads,
          proof archive, premium market posture, and selected beta surfaces as they harden.
        </p>
        <ul class="method-list">
          <li>Free users see public proof and a limited board.</li>
          <li>Founder/Premium users see the core premium fixture intelligence.</li>
          <li>Pro and Pro+ expand into player events, filters, downloads, and audit-style dashboards.</li>
        </ul>
        <div class="cta-row">
          <a class="button" href="./pricing.html">See access tiers</a>
        </div>
      </article>
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

  const orderedFixtureRows = (rows = state.fixtureIntelligence) =>
    [...rows].sort((left, right) => {
      const timeDiff = kickoffTimestamp(left.kickoff_time) - kickoffTimestamp(right.kickoff_time);
      if (timeDiff !== 0) {
        return timeDiff;
      }
      return publishClassRank(right.publish_class) - publishClassRank(left.publish_class);
    });

  const isCompletedWindowFixture = (row) => {
    const scoreline = String(row?.scoreline || "").trim();
    if (scoreline) {
      return true;
    }
    const ts = kickoffTimestamp(row?.kickoff_time);
    return Number.isFinite(ts) ? ts < Date.now() : false;
  };

  const splitFixtureWindow = (rows) => {
    const ordered = orderedFixtureRows(rows);
    return {
      results: ordered.filter((row) => isCompletedWindowFixture(row)),
      upcoming: ordered.filter((row) => !isCompletedWindowFixture(row)),
    };
  };

  const collectTeamRows = (teamName) => {
    const target = normalizePreferenceText(teamName);
    return orderedFixtureRows().filter((row) => {
      return normalizePreferenceText(row.home_team) === target || normalizePreferenceText(row.away_team) === target;
    });
  };

  const collectCompetitionRows = (competitionName) => {
    const target = normalizePreferenceText(competitionName);
    return orderedFixtureRows().filter((row) => normalizePreferenceText(row.league) === target);
  };

  const collectTeamEntity = (teamName) => {
    const rows = collectTeamRows(teamName);
    if (!rows.length) {
      return null;
    }
    const teamNormalized = normalizePreferenceText(teamName);
    const logo =
      rows.find((row) => normalizePreferenceText(row.home_team) === teamNormalized)?.home_team_logo_url ||
      rows.find((row) => normalizePreferenceText(row.away_team) === teamNormalized)?.away_team_logo_url ||
      "";
    const apiTeamId =
      rows.find((row) => normalizePreferenceText(row.home_team) === teamNormalized)?.api_home_team_id ||
      rows.find((row) => normalizePreferenceText(row.away_team) === teamNormalized)?.api_away_team_id ||
      "";
    const relatedCompetitions = Array.from(new Set(rows.map((row) => row.league).filter(Boolean)));
    return {
      name: teamName,
      logo,
      apiTeamId: String(apiTeamId || "").trim(),
      rows,
      fixtures: splitFixtureWindow(rows),
      relatedCompetitions,
      deployCount: rows.filter((row) => String(row.publish_class || row.fixture_class || "").toUpperCase() === "DEPLOY").length,
      observeCount: rows.filter((row) => String(row.publish_class || row.fixture_class || "").toUpperCase() === "OBSERVE").length,
      contextCount: rows.filter((row) => {
        const key = String(row.publish_class || row.fixture_class || "").toUpperCase();
        return key === "CONTEXT" || key === "MONITOR";
      }).length,
    };
  };

  const collectCompetitionEntity = (competitionName) => {
    const rows = collectCompetitionRows(competitionName);
    if (!rows.length) {
      return null;
    }
    const teams = new Set();
    rows.forEach((row) => {
      if (row.home_team) teams.add(row.home_team);
      if (row.away_team) teams.add(row.away_team);
    });
    return {
      name: competitionName,
      logo: rows[0]?.league_logo_url || "",
      rows,
      fixtures: splitFixtureWindow(rows),
      teamCount: teams.size,
      deployCount: rows.filter((row) => String(row.publish_class || row.fixture_class || "").toUpperCase() === "DEPLOY").length,
      observeCount: rows.filter((row) => String(row.publish_class || row.fixture_class || "").toUpperCase() === "OBSERVE").length,
      apiLeagueId: String(rows[0]?.api_league_id || "").trim(),
      apiSeason: String(rows[0]?.api_season || "").trim(),
    };
  };

  const publishClassKeyForRow = (row) => {
    const key = String(row?.publish_class || row?.fixture_class || "").toUpperCase();
    if (key === "DEPLOY" || key === "OBSERVE" || key === "CONTEXT" || key === "MONITOR") {
      return key;
    }
    return "OTHER";
  };

  const collectPublishClassMix = (rows) => {
    const counts = { DEPLOY: 0, OBSERVE: 0, CONTEXT: 0, MONITOR: 0, OTHER: 0 };
    rows.forEach((row) => {
      counts[publishClassKeyForRow(row)] += 1;
    });
    return counts;
  };

  const collectMarketFamilyMix = (rows) => {
    const counts = new Map();
    rows.forEach((row) => {
      const label = marketFamilyLabel(row?.signal_summary?.market_family);
      counts.set(label, (counts.get(label) || 0) + 1);
    });
    return Array.from(counts.entries())
      .map(([label, value]) => ({ label, value }))
      .sort((left, right) => right.value - left.value || left.label.localeCompare(right.label));
  };

  const renderEntitySurfaceTiles = (items) => `
    <div class="entity-surface-grid">
      ${items
        .map(
          (item) => `
            <article class="entity-surface-tile entity-surface-tile-${escapeHtml(item.tone || "reference")}">
              <span class="entity-surface-label">${escapeHtml(item.label)}</span>
              <strong class="entity-surface-value">${escapeHtml(item.value)}</strong>
              ${item.meta ? `<span class="entity-surface-meta">${escapeHtml(item.meta)}</span>` : ""}
            </article>
          `
        )
        .join("")}
    </div>
  `;

  const renderEntityBreakdown = (items, total, emptyCopy) => {
    if (!items.length || !total) {
      return `<div class="notice">${escapeHtml(emptyCopy)}</div>`;
    }
    return `
      <div class="entity-breakdown">
        ${items
          .map((item) => {
            const share = Math.max(6, Math.round((item.value / total) * 100));
            return `
              <div class="entity-breakdown-row">
                <div class="entity-breakdown-copy">
                  <span class="entity-breakdown-label">${escapeHtml(item.label)}</span>
                  <span class="entity-breakdown-meta">${escapeHtml(`${item.value} rows • ${Math.round((item.value / total) * 100)}%`)}</span>
                </div>
                <div class="entity-breakdown-bar">
                  <span class="entity-breakdown-fill entity-breakdown-fill-${escapeHtml(item.tone || "reference")}" style="width:${share}%"></span>
                </div>
              </div>
            `;
          })
          .join("")}
      </div>
    `;
  };

  const renderScoreBreakdown = (items, emptyCopy) => {
    if (!items.length) {
      return `<div class="notice">${escapeHtml(emptyCopy)}</div>`;
    }
    return `
      <div class="entity-breakdown">
        ${items
          .map((item) => {
            const numeric = Number(item.value);
            const share = Math.max(6, Math.min(100, Math.round(Number.isFinite(numeric) ? numeric : 0)));
            return `
              <div class="entity-breakdown-row">
                <div class="entity-breakdown-copy">
                  <span class="entity-breakdown-label">${escapeHtml(item.label)}</span>
                  <span class="entity-breakdown-meta">${escapeHtml(`${share}% • ${item.meta || safeTitleLabel(item.band)}`)}</span>
                </div>
                <div class="entity-breakdown-bar">
                  <span class="entity-breakdown-fill entity-breakdown-fill-${escapeHtml(item.tone || scoreTone(share))}" style="width:${share}%"></span>
                </div>
              </div>
            `;
          })
          .join("")}
      </div>
    `;
  };

  const renderSquadLeaderList = (label, names) => {
    if (!Array.isArray(names) || !names.length) {
      return "";
    }
    return `
      <li>
        <strong>${escapeHtml(label)}</strong><br />
        <span class="muted">${escapeHtml(names.join(" • "))}</span>
      </li>
    `;
  };

  const renderPlayerIntelligenceCard = (player) => {
    const power = player?.ratings?.og_player_power;
    const goalThreat = player?.ratings?.goal_threat;
    const creativeSpark = player?.ratings?.creative_spark;
    const disciplineRisk = player?.ratings?.discipline_risk;
    return `
      <article class="panel">
        <h4>${escapeHtml(player?.name || "Player")}</h4>
        <div class="pill-row">
          <span class="chip chip-reference">${escapeHtml(`${safeTitleLabel(player?.position_group, "Utility")} • ${power ?? "—"}%`)}</span>
          <span class="chip chip-reference">${escapeHtml(player?.minutes_confidence?.label || "Sample pending")}</span>
        </div>
        <ul class="feature-list compact-list">
          <li>Goal Threat: ${escapeHtml(goalThreat ?? "—")}%</li>
          <li>Creative Spark: ${escapeHtml(creativeSpark ?? "—")}%</li>
          <li>Discipline Risk: ${escapeHtml(disciplineRisk ?? "—")}%</li>
          <li>League Rank: ${escapeHtml(player?.ranks?.league_overall_rank ? `#${player.ranks.league_overall_rank}` : "—")}</li>
        </ul>
        <p class="section-copy">${escapeHtml((player?.tags || []).slice(0, 3).join(" • ") || "No public player tags yet.")}</p>
      </article>
    `;
  };

  const renderIntelligenceHeadline = (headline, body, tone = "reference") => `
    <article class="panel intelligence-callout intelligence-callout-${escapeHtml(tone)}">
      <h3>${escapeHtml(headline)}</h3>
      <p class="section-copy">${escapeHtml(body)}</p>
    </article>
  `;

  const summarizeSquadSnapshot = (payload) => {
    const players = Array.isArray(payload?.players) ? payload.players : [];
    const powers = players.map((player) => Number(player?.ratings?.og_player_power || 0)).filter(Number.isFinite);
    return {
      totalProfiles: players.length,
      eliteCount: powers.filter((value) => value >= 90).length,
      strongCount: powers.filter((value) => value >= 80).length,
      highRiskCount: players.filter((player) => Number(player?.ratings?.discipline_risk || 0) >= 70).length,
    };
  };

  const renderSquadDepthTiles = (payload) => {
    const snapshot = summarizeSquadSnapshot(payload);
    return renderEntitySurfaceTiles([
      { label: "Squad profiles", value: snapshot.totalProfiles || 0, meta: "Publish-safe player profiles", tone: "reference" },
      { label: "Elite profiles", value: snapshot.eliteCount || 0, meta: "90%+ OG Player Power", tone: "deploy" },
      { label: "Strong profiles", value: snapshot.strongCount || 0, meta: "80%+ OG Player Power", tone: "observe" },
      { label: "Discipline risk", value: snapshot.highRiskCount || 0, meta: "70%+ risk profiles", tone: snapshot.highRiskCount ? "observe" : "reference" },
    ]);
  };

  const renderMarketTendencyList = (marketTendencies) => {
    const items = [
      ["FTR Lean", marketTendencies?.ftr_lean, "Full-result posture"],
      ["BTTS Lean", marketTendencies?.btts_lean, "Two-way scoring pressure"],
      ["Over 2.5", marketTendencies?.over25_lean, "Goal-total profile"],
      ["Under Control", marketTendencies?.under_control, "Suppression/read discipline"],
      ["Team Goals 1.5+", marketTendencies?.team_goals_15, "Scoring-repeat pressure"],
      ["Corners", marketTendencies?.corners, "Territory / wide pressure"],
      ["Cards", marketTendencies?.cards, "Discipline / card heat"],
    ];
    return `
      <ul class="feature-list compact-list">
        ${items
          .map(
            ([label, value, note]) => `
              <li>
                <strong>${escapeHtml(label)}</strong><br />
                <span class="muted">${escapeHtml(`${safeTitleLabel(value)} • ${note}`)}</span>
              </li>
            `
          )
          .join("")}
      </ul>
    `;
  };

  const renderProfileSurface = (label, profile = {}) => `
    <article class="panel">
      <h3>${escapeHtml(label)}</h3>
      ${renderEntitySurfaceTiles([
        { label: "Power", value: `${profile.power ?? "—"}%`, meta: "Overall profile", tone: scoreTone(profile.power) },
        { label: "Attack", value: `${profile.attack ?? "—"}%`, meta: "Chance / scoring shape", tone: scoreTone(profile.attack) },
        { label: "Defence", value: `${profile.defence ?? "—"}%`, meta: "Resistance / suppression", tone: scoreTone(profile.defence) },
        { label: "Goal Heat", value: `${profile.goal_heat ?? "—"}%`, meta: "Total-goals environment", tone: scoreTone(profile.goal_heat) },
      ])}
    </article>
  `;

  const formationBandPlan = (formation, players = []) => {
    const digits = String(formation || "")
      .split(/[^0-9]+/)
      .filter(Boolean)
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value) && value > 0);
    const goalkeeper = players.find((player) => String(player?.lineup_position || "").toUpperCase() === "G") || null;
    const outfield = players.filter((player) => player !== goalkeeper);
    if (!digits.length || digits.reduce((sum, value) => sum + value, 0) !== outfield.length) {
      const fallbackCounts = [];
      const remaining = outfield.length;
      if (remaining <= 0) {
        return goalkeeper ? [[goalkeeper]] : [];
      }
      if (remaining <= 3) {
        fallbackCounts.push(remaining);
      } else {
        const defence = Math.max(3, Math.round(remaining * 0.36));
        const midfield = Math.max(2, Math.round(remaining * 0.32));
        const attack = Math.max(1, remaining - defence - midfield);
        const correction = remaining - (defence + midfield + attack);
        fallbackCounts.push(defence, midfield, attack + correction);
      }
      const bands = goalkeeper ? [[goalkeeper]] : [];
      let cursor = 0;
      fallbackCounts.forEach((count) => {
        bands.push(outfield.slice(cursor, cursor + count));
        cursor += count;
      });
      if (cursor < outfield.length) {
        bands[bands.length - 1] = (bands[bands.length - 1] || []).concat(outfield.slice(cursor));
      }
      return bands;
    }
    const bands = goalkeeper ? [[goalkeeper]] : [];
    let cursor = 0;
    digits.forEach((count) => {
      bands.push(outfield.slice(cursor, cursor + count));
      cursor += count;
    });
    return bands;
  };

  const lineupTopMetric = (profile = {}) => {
    const metrics = [
      ["Goal Threat", Number(profile.goal_threat || 0)],
      ["Creative Spark", Number(profile.creative_spark || 0)],
      ["Midfield Engine", Number(profile.midfield_engine || 0)],
      ["Defensive Lock", Number(profile.defensive_lock || 0)],
      ["Pressing Heat", Number(profile.pressing_heat || 0)],
      ["Ball Progression", Number(profile.ball_progression || 0)],
      ["Aerial Dominance", Number(profile.aerial_dominance || 0)],
      ["Goalkeeper Shield", Number(profile.goalkeeper_shield || 0)],
    ]
      .filter((entry) => Number.isFinite(entry[1]))
      .sort((left, right) => right[1] - left[1]);
    return metrics[0] || ["Profile", Number(profile.power || 0)];
  };

  const renderFormationPitchCard = (teamName, formation, profiles = [], units = {}, teamLogoUrl = "") => {
    const bands = formationBandPlan(formation, profiles);
    const lineCount = Math.max(1, bands.length);
    const unitChips = [
      ["Attack", units.attack_unit],
      ["Midfield", units.midfield_control],
      ["Defence", units.defensive_unit],
    ];
    return `
      <article class="panel formation-pitch-card">
        <div class="formation-pitch-head">
          <div class="formation-pitch-title">
            ${badgeMarkup(teamLogoUrl, teamName, "lineup-team-badge")}
            <div>
              <span class="metric-label">${escapeHtml(teamName || "Team")}</span>
              <h4>${escapeHtml(formation || "Formation pending")}</h4>
            </div>
          </div>
          <div class="formation-unit-strip">
            ${unitChips
              .map(
                ([label, value]) => `
                  <span class="formation-unit-chip formation-unit-chip-${escapeHtml(scoreTone(value))}">
                    ${escapeHtml(`${label} ${value ?? "—"}%`)}
                  </span>
                `
              )
              .join("")}
          </div>
        </div>
        <div class="formation-pitch">
          ${bands
            .map((band, bandIndex) => {
              const top = lineCount === 1 ? 50 : 88 - (bandIndex * 68) / Math.max(1, lineCount - 1);
              const count = Math.max(1, band.length);
              return band
                .map((player, playerIndex) => {
                  const left = count === 1 ? 50 : 14 + (playerIndex * 72) / Math.max(1, count - 1);
                  const [metricLabel, metricValue] = lineupTopMetric(player);
                  return `
                    <button
                      class="formation-player formation-player-${escapeHtml(scoreTone(player.power))}"
                      type="button"
                      style="left:${left}%; top:${top}%;"
                    >
                      ${renderOgRatingBadge(player.power, "medium", `${player.name || player.surname || "Player"} OG rating`)}
                      <span class="formation-player-name">${escapeHtml(player.surname || player.name || "Player")}</span>
                      <span class="formation-player-role">${escapeHtml(safeTitleLabel(player.position_group, "Utility"))}</span>
                      <span class="formation-player-tooltip">
                        <strong>${escapeHtml(player.name || player.surname || "Player")}</strong>
                        <span>${escapeHtml(`OG Power ${player.power ?? "—"}%`)}</span>
                        <span>${escapeHtml(`Display rating ${ogRatingValue(player.power)?.toFixed(1) || "—"}`)}</span>
                        <span>${escapeHtml(`${metricLabel} ${metricValue ?? "—"}%`)}</span>
                        <span>${escapeHtml(`Discipline ${player.discipline_risk ?? "—"}%`)}</span>
                      </span>
                    </button>
                  `;
                })
                .join("");
            })
            .join("")}
        </div>
      </article>
    `;
  };

  const horizontalPitchPlayers = (side, formation, profiles = []) => {
    const bands = formationBandPlan(formation, profiles);
    const lineCount = Math.max(1, bands.length);
    return bands.flatMap((band, bandIndex) => {
      const progress = lineCount === 1 ? 0.5 : bandIndex / Math.max(1, lineCount - 1);
      const left = side === "home" ? 9 + progress * 34 : 91 - progress * 34;
      const count = Math.max(1, band.length);
      return band.map((player, playerIndex) => {
        const top = count === 1 ? 50 : 16 + (playerIndex * 68) / Math.max(1, count - 1);
        return { player, left, top, side };
      });
    });
  };

  const renderHorizontalPitchPlayer = ({ player, left, top, side }) => {
    const [metricLabel, metricValue] = lineupTopMetric(player);
    return `
      <button
        class="formation-player formation-player-horizontal formation-player-${escapeHtml(scoreTone(player.power))} formation-player-${escapeHtml(side)}"
        type="button"
        style="left:${left}%; top:${top}%;"
      >
        ${renderOgRatingBadge(player.power, "small", `${player.name || player.surname || "Player"} OG rating`)}
        <span class="formation-player-name">${escapeHtml(player.surname || player.name || "Player")}</span>
        <span class="formation-player-tooltip">
          <strong>${escapeHtml(player.name || player.surname || "Player")}</strong>
          <span>${escapeHtml(`OG Power ${player.power ?? "—"}%`)}</span>
          <span>${escapeHtml(`Display rating ${ogRatingValue(player.power)?.toFixed(1) || "—"}`)}</span>
          <span>${escapeHtml(`${metricLabel} ${metricValue ?? "—"}%`)}</span>
          <span>${escapeHtml(`Discipline ${player.discipline_risk ?? "—"}%`)}</span>
        </span>
      </button>
    `;
  };

  const renderCombinedFormationPitch = (payload, fixture = null) => {
    const homePlayers = horizontalPitchPlayers("home", payload.home_formation || "", payload.home_lineup_profiles || []);
    const awayPlayers = horizontalPitchPlayers("away", payload.away_formation || "", payload.away_lineup_profiles || []);
    return `
      <article class="panel formation-pitch-card formation-pitch-card-wide">
        <div class="formation-match-head">
          <div class="formation-match-team">
            ${badgeMarkup(fixture?.home_team_logo_url, payload.home_team || "Home", "lineup-team-badge")}
            <div>
              <span class="formation-match-formation">${escapeHtml(payload.home_formation || "Shape pending")}</span>
              <h4>${escapeHtml(payload.home_team || "Home")}</h4>
            </div>
          </div>
          <div class="formation-match-team formation-match-team-away">
            <div>
              <span class="formation-match-formation">${escapeHtml(payload.away_formation || "Shape pending")}</span>
              <h4>${escapeHtml(payload.away_team || "Away")}</h4>
            </div>
            ${badgeMarkup(fixture?.away_team_logo_url, payload.away_team || "Away", "lineup-team-badge")}
          </div>
        </div>
        <div class="formation-pitch-scroll">
          <div class="formation-pitch formation-pitch-horizontal">
            <span class="pitch-line pitch-line-half"></span>
            <span class="pitch-line pitch-line-circle"></span>
            <span class="pitch-line pitch-line-box pitch-line-box-home"></span>
            <span class="pitch-line pitch-line-box pitch-line-box-away"></span>
            <span class="pitch-line pitch-line-six pitch-line-six-home"></span>
            <span class="pitch-line pitch-line-six pitch-line-six-away"></span>
            ${homePlayers.concat(awayPlayers).map(renderHorizontalPitchPlayer).join("")}
          </div>
        </div>
      </article>
    `;
  };

  const renderFormationMismatchSurface = (payload) => {
    const mismatches = Array.isArray(payload?.key_mismatches) ? payload.key_mismatches.slice(0, 4) : [];
    if (!mismatches.length) {
      return `<div class="notice">No publish-safe mismatch surface is available for this fixture yet.</div>`;
    }
    return `
      <div class="formation-mismatch-grid">
        ${mismatches
          .map(
            (item) => `
              <article class="formation-mismatch-card">
                <span class="metric-label">Key mismatch</span>
                <h4>${escapeHtml(item.summary || item.zone || "Mismatch edge")}</h4>
                <div class="formation-mismatch-values">
                  <div>
                    <span>${escapeHtml(item.left_label || "Left side")}</span>
                    <strong>${escapeHtml(`${item.left_value ?? "—"}%`)}</strong>
                  </div>
                  <div>
                    <span>${escapeHtml(item.right_label || "Right side")}</span>
                    <strong>${escapeHtml(`${item.right_value ?? "—"}%`)}</strong>
                  </div>
                </div>
                <p class="muted">${escapeHtml(`${item.advantage || "Advantage pending"} +${item.mismatch_score ?? "—"}`)}</p>
              </article>
            `
          )
          .join("")}
      </div>
    `;
  };

  const renderFormationPlayerMatchups = (payload) => {
    const matchups = Array.isArray(payload?.player_matchups) ? payload.player_matchups.slice(0, 3) : [];
    if (!matchups.length) {
      return "";
    }
    return `
      <article class="panel">
        <h3>Player matchups</h3>
        <div class="fixture-player-matchups">
          ${matchups
            .map(
              (item) => `
                <article class="fixture-player-matchup-card">
                  <div class="fixture-player-matchup-head">
                    <div>
                      <span class="metric-label">${escapeHtml("Attacker / creator")}</span>
                      <h4>${escapeHtml(item.home_player || "Player")}</h4>
                    </div>
                    <div class="fixture-player-matchup-delta">${escapeHtml(`${item.mismatch_score ?? "—"} pts`)}</div>
                    <div class="fixture-player-matchup-away">
                      <span class="metric-label">${escapeHtml("Defender / counter")}</span>
                      <h4>${escapeHtml(item.away_player || "Player")}</h4>
                    </div>
                  </div>
                  <div class="fixture-player-metric-grid">
                    <div class="fixture-player-metric"><span>${escapeHtml(item.home_metric_label || "Metric")}</span><strong>${escapeHtml(`${item.home_metric_value ?? "—"}%`)}</strong></div>
                    <div class="fixture-player-metric"><span>${escapeHtml(item.away_metric_label || "Metric")}</span><strong>${escapeHtml(`${item.away_metric_value ?? "—"}%`)}</strong></div>
                  </div>
                  <p class="muted">${escapeHtml(item.summary || `${item.advantage || "Advantage"} matchup edge`)}</p>
                </article>
              `
            )
            .join("")}
        </div>
      </article>
    `;
  };

  const renderBenchSnapshot = (teamName, players = []) => {
    const bench = Array.isArray(players) ? players.slice(0, 7) : [];
    if (!bench.length) {
      return "";
    }
    return `
      <article class="lineup-fallback-card">
        <span class="metric-label">${escapeHtml(teamName || "Team")} bench</span>
        <div class="fixture-bench-list">
          ${bench
            .map(
              (player) => `
                <span class="fixture-bench-player">
                  ${renderOgRatingBadge(player.power, "small", `${player.name || player.surname || "Player"} OG rating`)}
                  <span>
                    <strong>${escapeHtml(player.surname || player.name || "Player")}</strong>
                    <small>${escapeHtml(`${safeTitleLabel(player.position_group, "Utility")} · ${player.power ?? "—"}%`)}</small>
                  </span>
                </span>
              `
            )
            .join("")}
        </div>
      </article>
    `;
  };

  const renderTeamLineupSnapshot = (snapshot) => {
    if (!snapshot) {
      return "";
    }
    const starters = Array.isArray(snapshot.starters) ? snapshot.starters.slice(0, 11) : [];
    const bench = Array.isArray(snapshot.bench) ? snapshot.bench.slice(0, 9) : [];
    const units = snapshot.units && typeof snapshot.units === "object" ? snapshot.units : {};
    return `
      <section class="section">
        <article class="panel">
          <div class="intel-placeholder-head">
            <div>
              <span class="metric-label">Team sheet snapshot</span>
              <h3>Most recent lineup and bench</h3>
            </div>
            <span class="chip chip-signal">${escapeHtml(snapshot.formation || "Shape pending")}</span>
          </div>
          <p class="section-copy">${escapeHtml(
            snapshot.summary ||
              `Latest publish-safe team sheet snapshot for ${snapshot.team || "this team"}${
                snapshot.source_match_date ? ` from ${snapshot.source_match_date}` : ""
              }.`
          )}</p>
          <div class="split">
            <article class="lineup-fallback-card">
              <span class="metric-label">Starting XI</span>
              <div class="fixture-bench-list">
                ${starters
                  .map(
                    (player) => `
                      <span class="fixture-bench-player">
                        ${renderOgRatingBadge(player.power, "small", `${player.name || player.surname || "Player"} OG rating`)}
                        <span>
                          <strong>${escapeHtml(player.surname || player.name || "Player")}</strong>
                          <small>${escapeHtml(`${safeTitleLabel(player.position_group, "Utility")} · ${player.power ?? "—"}%`)}</small>
                        </span>
                      </span>
                    `
                  )
                  .join("") || `<span class="muted">Starting XI snapshot is not available yet.</span>`}
              </div>
            </article>
            <article class="lineup-fallback-card">
              <span class="metric-label">Bench</span>
              <div class="fixture-bench-list">
                ${bench
                  .map(
                    (player) => `
                      <span class="fixture-bench-player">
                        ${renderOgRatingBadge(player.power, "small", `${player.name || player.surname || "Player"} OG rating`)}
                        <span>
                          <strong>${escapeHtml(player.surname || player.name || "Player")}</strong>
                          <small>${escapeHtml(`${safeTitleLabel(player.position_group, "Utility")} · ${player.power ?? "—"}%`)}</small>
                        </span>
                      </span>
                    `
                  )
                  .join("") || `<span class="muted">Bench snapshot is not available yet.</span>`}
              </div>
            </article>
          </div>
          ${renderScoreBreakdown(
            Object.entries(units).map(([key, value]) => ({
              label: safeTitleLabel(key),
              value,
              band: key === "discipline_risk" ? "Risk profile" : "Unit score",
              tone: key === "discipline_risk" ? scoreTone(100 - Number(value || 0)) : scoreTone(value),
            })),
            "No unit scores are available for this lineup snapshot yet."
          )}
        </article>
      </section>
    `;
  };

  const renderFixtureLineupIntelligence = (payload, fixture = null) => {
    const decision = state.selectedFixtureDecisionIntelligence || null;
    const coverage = lineupCoverageProfile(payload, decision);
    if (!payload || coverage.status === "fallback" || !payload.home_units || !payload.away_units) {
      return `
        <section class="section">
          <article class="panel intel-placeholder-panel">
            <div class="intel-placeholder-head">
              <div>
                <span class="metric-label">Formation intelligence</span>
                <h3>Lineup unavailable</h3>
              </div>
              <span class="chip chip-reference">Unavailable</span>
            </div>
            <p class="section-copy">${escapeHtml(coverage.summary)}</p>
            <div class="lineup-fallback-grid">
              <article class="lineup-fallback-card">
                <span class="metric-label">What is unavailable</span>
                <strong>Last-fixture team sheet</strong>
                <p class="muted">No compact latest-lineup snapshot has been emitted for at least one team in this fixture.</p>
              </article>
              <article class="lineup-fallback-card">
                <span class="metric-label">What stays live</span>
                <strong>Team and squad intelligence</strong>
                <p class="muted">The page keeps the decision, team ratings, market posture, and squad-driver fallback visible.</p>
              </article>
              <article class="lineup-fallback-card">
                <span class="metric-label">Decision impact</span>
                <strong>${escapeHtml(tokenHas(decision?.caution_layers || [], "LINEUP_DATA_MISSING") ? "Caution applied" : "Context only")}</strong>
                <p class="muted">Missing lineups are treated as a caution layer, never as a reason to invent a stronger read.</p>
              </article>
            </div>
          </article>
        </section>
      `;
    }
    const unitLabelMap = {
      attack_unit: "Attack Unit",
      midfield_control: "Midfield Control",
      defensive_unit: "Defensive Unit",
      wide_threat: "Wide Threat",
      central_threat: "Central Threat",
      discipline_risk: "Discipline Risk",
    };
    const toItems = (unitMap = {}) =>
      Object.entries(unitMap).map(([key, value]) => ({
        label: unitLabelMap[key] || safeTitleLabel(key),
        value,
        band: value >= 80 ? "Strong unit" : value >= 55 ? "Live support" : "Needs context",
        tone: key === "discipline_risk" ? scoreTone(100 - Number(value || 0)) : scoreTone(value),
      }));
    const lineupHeading =
      coverage.status === "confirmed"
        ? "Confirmed lineups"
        : coverage.status === "predicted"
          ? "Predicted lineups"
          : "Published lineup structure";
    const lineupChip =
      coverage.status === "confirmed"
        ? "Team sheets confirmed"
        : coverage.status === "predicted"
          ? "From last fixture"
          : `${coverage.profileCount} profiles`;
    return `
      <section class="section">
        <article class="panel">
          <div class="intel-placeholder-head">
            <div>
              <span class="metric-label">Formation intelligence</span>
              <h3>${escapeHtml(lineupHeading)}</h3>
            </div>
            <span class="chip chip-signal">${escapeHtml(lineupChip)}</span>
          </div>
          <p class="section-copy">${escapeHtml(
            coverage.status === "predicted" || coverage.status === "confirmed"
              ? coverage.summary
              : "This layer turns the actual XI into a visual judgement surface: team shape, player strength, unit balance, and the mismatch zones most likely to drive the read."
          )}</p>
          ${renderCombinedFormationPitch(payload, fixture)}
          ${
            coverage.status === "predicted"
              ? `
                <div class="lineup-fallback-grid lineup-bench-grid">
                  ${renderBenchSnapshot(payload.home_team || "Home", payload.home_bench_profiles || [])}
                  ${renderBenchSnapshot(payload.away_team || "Away", payload.away_bench_profiles || [])}
                </div>
              `
              : ""
          }
        </article>
      </section>
      <section class="section">
        <div class="split">
          <article class="panel">
            <h3>Unit strength</h3>
            <div class="split split-tight">
              <div>
                <span class="metric-label">${escapeHtml(payload.home_team || "Home")}</span>
                <h4>${escapeHtml(payload.home_formation || "Formation pending")}</h4>
                ${renderScoreBreakdown(toItems(payload.home_units), "No home unit ratings published yet.")}
              </div>
              <div>
                <span class="metric-label">${escapeHtml(payload.away_team || "Away")}</span>
                <h4>${escapeHtml(payload.away_formation || "Formation pending")}</h4>
                ${renderScoreBreakdown(toItems(payload.away_units), "No away unit ratings published yet.")}
              </div>
            </div>
          </article>
          <article class="panel">
            <h3>Key mismatch zones</h3>
            ${renderFormationMismatchSurface(payload)}
          </article>
        </div>
      </section>
      <section class="section">
        ${renderFormationPlayerMatchups(payload)}
      </section>
    `;
  };

  const findLineupProfileByName = (profiles = [], playerName = "") => {
    const target = normalizePreferenceText(playerName);
    return (
      profiles.find((profile) => normalizePreferenceText(profile?.name) === target || normalizePreferenceText(profile?.surname) === target) || null
    );
  };

  const findSquadProfileByName = (payload, playerName = "") => {
    if (!payload || !Array.isArray(payload.players)) {
      return null;
    }
    const target = normalizePreferenceText(playerName);
    return (
      payload.players.find((profile) => normalizePreferenceText(profile?.name) === target || normalizePreferenceText(profile?.surname) === target) || null
    );
  };

  const fixtureSignalProfile = (fixture) => {
    const family = String(fixture?.signal_summary?.market_family || "").toUpperCase();
    const copy = `${fixture?.signal_summary?.headline || ""} ${fixture?.signal_summary?.summary_text || ""}`.toLowerCase();
    let pick = String(fixture?.signal_summary?.deploy_pick || fixture?.deploy_summary?.pick || "").toUpperCase();
    if (!pick) {
      if (family === "BTTS") {
        pick = copy.includes("btts no") ? "NO" : "YES";
      } else if (family === "OU25") {
        pick = copy.includes("under") ? "UNDER25" : "OVER25";
      } else if (family === "FTR") {
        if (copy.includes("draw")) {
          pick = "DRAW";
        } else if (copy.includes("away")) {
          pick = "AWAY";
        } else {
          pick = "HOME";
        }
      }
    }
    return { family, pick };
  };

  const signalRelevantRatingKeys = ({ family, pick }) => {
    if (family === "BTTS") {
      return pick === "NO"
        ? ["defensive_lock_rating", "control_rating", "goal_heat_rating", "btts_pressure_rating", "chaos_rating"]
        : ["goal_heat_rating", "btts_pressure_rating", "attack_flow_rating", "defensive_lock_rating", "first_strike_rating", "chaos_rating"];
    }
    if (family === "OU25") {
      return pick === "UNDER25"
        ? ["control_rating", "defensive_lock_rating", "goal_heat_rating", "over25_heat_rating", "chaos_rating", "first_strike_rating"]
        : ["goal_heat_rating", "over25_heat_rating", "attack_flow_rating", "defensive_lock_rating", "chaos_rating", "first_strike_rating"];
    }
    if (family === "FTR") {
      if (pick === "AWAY") {
        return ["og_power_rating", "away_threat_rating", "home_fortress_rating", "defensive_lock_rating", "first_strike_rating", "control_rating"];
      }
      if (pick === "DRAW") {
        return ["control_rating", "defensive_lock_rating", "chaos_rating", "goal_heat_rating", "first_strike_rating"];
      }
      return ["og_power_rating", "home_fortress_rating", "away_threat_rating", "defensive_lock_rating", "first_strike_rating", "control_rating"];
    }
    return ["og_power_rating", "attack_flow_rating", "defensive_lock_rating", "goal_heat_rating"];
  };

  const evaluateDecisionAlignment = (fixture, homeIntelligence, awayIntelligence) => {
    if (!homeIntelligence?.ratings || !awayIntelligence?.ratings) {
      return null;
    }
    const { family, pick } = fixtureSignalProfile(fixture);
    const home = homeIntelligence.ratings;
    const away = awayIntelligence.ratings;
    const avg = (...values) => {
      const usable = values.map((value) => Number(value)).filter((value) => Number.isFinite(value));
      return usable.length ? usable.reduce((sum, value) => sum + value, 0) / usable.length : null;
    };
    const rows = [];
    const push = (tone, text) => rows.push({ tone, text });
    const toneFor = (supported, contradicted) => (supported ? "support" : contradicted ? "contradict" : "neutral");
    const scoreForTone = (tone) => (tone === "support" ? 1 : tone === "contradict" ? -1 : 0);

    if (family === "BTTS") {
      const combinedHeat = avg(home.goal_heat_rating, away.goal_heat_rating);
      const combinedBtts = avg(home.btts_pressure_rating, away.btts_pressure_rating);
      const combinedAttack = avg(home.attack_flow_rating, away.attack_flow_rating);
      const maxLock = Math.max(Number(home.defensive_lock_rating || 0), Number(away.defensive_lock_rating || 0));
      if (pick === "NO") {
        push(toneFor(combinedBtts !== null && combinedBtts <= 44, combinedBtts !== null && combinedBtts >= 58), `BTTS pressure averages ${Math.round(combinedBtts ?? 0)} across both sides.`);
        push(toneFor(maxLock >= 72, maxLock <= 58), `Defensive lock peaks at ${Math.round(maxLock)} across the matchup.`);
        push(toneFor(combinedHeat !== null && combinedHeat <= 48, combinedHeat !== null && combinedHeat >= 62), `Goal heat sits at ${Math.round(combinedHeat ?? 0)} combined.`);
        push(toneFor(combinedAttack !== null && combinedAttack <= 54, combinedAttack !== null && combinedAttack >= 66), `Attack flow averages ${Math.round(combinedAttack ?? 0)} across the front lines.`);
      } else {
        push(toneFor(combinedHeat !== null && combinedHeat >= 62, combinedHeat !== null && combinedHeat <= 48), `Goal heat averages ${Math.round(combinedHeat ?? 0)} across both teams.`);
        push(toneFor(combinedBtts !== null && combinedBtts >= 58, combinedBtts !== null && combinedBtts <= 44), `BTTS pressure combines to ${Math.round(combinedBtts ?? 0)} in this pairing.`);
        push(toneFor(maxLock <= 62, maxLock >= 78), `Defensive lock tops out at ${Math.round(maxLock)} across the matchup.`);
        push(toneFor(combinedAttack !== null && combinedAttack >= 60, combinedAttack !== null && combinedAttack <= 50), `Attack flow averages ${Math.round(combinedAttack ?? 0)} across both sides.`);
      }
    } else if (family === "OU25") {
      const combinedHeat = avg(home.goal_heat_rating, away.goal_heat_rating);
      const combinedOver = avg(home.over25_heat_rating, away.over25_heat_rating);
      const combinedControl = avg(home.control_rating, away.control_rating);
      const combinedLock = avg(home.defensive_lock_rating, away.defensive_lock_rating);
      const combinedChaos = avg(home.chaos_rating, away.chaos_rating);
      if (pick === "UNDER25") {
        push(toneFor(combinedControl !== null && combinedControl >= 64, combinedControl !== null && combinedControl <= 48), `Control rating averages ${Math.round(combinedControl ?? 0)} across the fixture.`);
        push(toneFor(combinedLock !== null && combinedLock >= 66, combinedLock !== null && combinedLock <= 54), `Defensive lock sits at ${Math.round(combinedLock ?? 0)} combined.`);
        push(toneFor(combinedHeat !== null && combinedHeat <= 48, combinedHeat !== null && combinedHeat >= 62), `Goal heat runs at ${Math.round(combinedHeat ?? 0)} across both sides.`);
        push(toneFor(combinedChaos !== null && combinedChaos <= 46, combinedChaos !== null && combinedChaos >= 62), `Chaos rating averages ${Math.round(combinedChaos ?? 0)} in this matchup.`);
      } else {
        push(toneFor(combinedHeat !== null && combinedHeat >= 60, combinedHeat !== null && combinedHeat <= 48), `Goal heat averages ${Math.round(combinedHeat ?? 0)} across the pairing.`);
        push(toneFor(combinedOver !== null && combinedOver >= 58, combinedOver !== null && combinedOver <= 46), `Over 2.5 heat combines to ${Math.round(combinedOver ?? 0)} here.`);
        push(toneFor(combinedLock !== null && combinedLock <= 58, combinedLock !== null && combinedLock >= 72), `Defensive lock averages ${Math.round(combinedLock ?? 0)} across both back lines.`);
        push(toneFor(combinedChaos !== null && combinedChaos >= 52, combinedChaos !== null && combinedChaos <= 40), `Chaos rating runs at ${Math.round(combinedChaos ?? 0)} across the match state.`);
      }
    } else if (family === "FTR") {
      const homePowerEdge = Number(home.og_power_rating || 0) - Number(away.og_power_rating || 0);
      const fortressEdge = Number(home.home_fortress_rating || 0) - Number(away.away_threat_rating || 0);
      const awayFortressEdge = Number(away.away_threat_rating || 0) - Number(home.home_fortress_rating || 0);
      const firstStrikeEdge = Number(home.first_strike_rating || 0) - Number(away.first_strike_rating || 0);
      const lockEdge = Number(home.defensive_lock_rating || 0) - Number(away.defensive_lock_rating || 0);
      if (pick === "DRAW") {
        const parity = Math.abs(homePowerEdge);
        const combinedControl = avg(home.control_rating, away.control_rating);
        const combinedChaos = avg(home.chaos_rating, away.chaos_rating);
        push(toneFor(parity <= 8, parity >= 18), `Power difference sits at ${Math.round(Math.abs(homePowerEdge))} points between the sides.`);
        push(toneFor(combinedControl !== null && combinedControl >= 58, combinedControl !== null && combinedControl <= 44), `Control rating averages ${Math.round(combinedControl ?? 0)} across the fixture.`);
        push(toneFor(combinedChaos !== null && combinedChaos <= 48, combinedChaos !== null && combinedChaos >= 62), `Chaos rating averages ${Math.round(combinedChaos ?? 0)} in the matchup.`);
      } else if (pick === "AWAY") {
        push(toneFor(homePowerEdge <= -8, homePowerEdge >= 6), `Away OG Power edge lands at ${Math.round(Math.abs(homePowerEdge))} points over the home side.`);
        push(toneFor(awayFortressEdge >= 6, awayFortressEdge <= -6), `Away Threat vs Home Fortress differential sits at ${Math.round(awayFortressEdge)}.`);
        push(toneFor(firstStrikeEdge <= -4, firstStrikeEdge >= 8), `First-strike profile leans ${firstStrikeEdge <= 0 ? "toward the away side" : "toward the home side"} by ${Math.round(Math.abs(firstStrikeEdge))} points.`);
        push(toneFor(lockEdge <= -4, lockEdge >= 8), `Defensive lock gap runs ${Math.round(Math.abs(lockEdge))} points toward ${lockEdge <= 0 ? awayIntelligence.team : homeIntelligence.team}.`);
      } else {
        push(toneFor(homePowerEdge >= 8, homePowerEdge <= -6), `Home OG Power edge lands at ${Math.round(homePowerEdge)} points over the away side.`);
        push(toneFor(fortressEdge >= 6, fortressEdge <= -6), `Home Fortress vs Away Threat differential sits at ${Math.round(fortressEdge)}.`);
        push(toneFor(firstStrikeEdge >= 4, firstStrikeEdge <= -8), `First-strike profile leans ${firstStrikeEdge >= 0 ? "toward the home side" : "toward the away side"} by ${Math.round(Math.abs(firstStrikeEdge))} points.`);
        push(toneFor(lockEdge >= 4, lockEdge <= -8), `Defensive lock gap runs ${Math.round(Math.abs(lockEdge))} points toward ${lockEdge >= 0 ? homeIntelligence.team : awayIntelligence.team}.`);
      }
    }

    const supportCount = rows.filter((row) => row.tone === "support").length;
    const contradictionCount = rows.filter((row) => row.tone === "contradict").length;
    const score = rows.reduce((sum, row) => sum + scoreForTone(row.tone), 0);
    let alignment = "Mixed alignment";
    if (supportCount >= 3 && contradictionCount === 0) {
      alignment = "Strong alignment";
    } else if (supportCount >= 2 && contradictionCount <= 1 && score > 0) {
      alignment = "Moderate alignment";
    } else if (contradictionCount >= 2 && contradictionCount > supportCount) {
      alignment = "Contradiction warning";
    }
    return {
      family,
      pick,
      alignment,
      rows: rows.slice(0, 4),
    };
  };

  const reasonTokenLabel = (token) =>
    safeTitleLabel(
      String(token || "")
        .replace(/^H2H_/i, "H2H ")
        .replace(/^BTTS_/i, "BTTS ")
        .replace(/^OU25_/i, "OU25 ")
        .replace(/^TEAM_/i, "Team ")
        .replace(/^AWAY_/i, "Away ")
        .replace(/^HOME_/i, "Home ")
    );

  const renderFixtureDecisionVerdict = (fixture, decision) => {
    if (!decision) {
      return `
        <section class="section">
          <article class="panel">
            <h3>Decision companion</h3>
            <div class="notice">The reconciled fixture decision layer has not been published for this fixture yet.</div>
          </article>
        </section>
      `;
    }
    const stateToneMap = {
      SUPPORTED: "deploy",
      WATCHLIST: "observe",
      MIXED: "observe",
      FRAGILE: "reference",
      AVOID: "reference",
    };
    const routeAudit = routeAuditProfile(fixture, decision);
    const tone = stateToneMap[routeAudit.auditState] || "reference";
    return `
      <section class="section">
        <article class="panel fixture-ratings-verdict">
          <span class="metric-label">Route / audit companion</span>
          <h3>${escapeHtml(routeAudit.routeLabel || decision.primary_signal || marketVerdictDisplay(fixture))}</h3>
          <p class="section-copy">${escapeHtml(decision.public_safe_summary || "No reconciled public summary has been published yet.")}</p>
          ${renderEntitySurfaceTiles([
            { label: "Published route", value: routeAudit.routeActive ? `${routeAudit.routeMarket} ${routeAudit.routePick}` : "No pick", meta: safeTitleLabel(routeAudit.routeState || "context", "Context only"), tone: routeAudit.routeTone },
            { label: "Context audit", value: safeTitleLabel(routeAudit.auditState, "Pending"), meta: `${routeAudit.agreement ?? "—"}% agreement`, tone },
            { label: "Conflict level", value: routeAudit.conflictLabel, meta: routeAudit.conflictCopy, tone: routeAudit.conflictTone },
            { label: "Primary read", value: routeAudit.routeActive ? routeAudit.routeLabel : routeAudit.contextLabel, meta: routeAudit.routeActive ? "Published route view" : "Context-only view", tone: routeAudit.routeTone },
          ])}
        </article>
      </section>
    `;
  };

  const renderFixtureHeroDecisionAside = (fixture, clarity, matchedEntry) => {
    const decision = state.selectedFixtureDecisionIntelligence || null;
    const marketLine = primaryMarketLine(fixture);
    const confidenceTier = String(fixture.signal_summary?.confidence_tier || fixture.deploy_summary?.confidence_tier || "").toUpperCase();
    const verdictLabel = marketVerdictDisplay(fixture);
    if (!decision) {
      return `
        <article class="panel compact-panel compact-panel-primary">
          <span class="metric-label">Action state</span>
          <div class="metric-stack">
            <strong class="metric-value">${escapeHtml(verdictLabel)}</strong>
            <p class="muted">${escapeHtml(`${clarity.action_label} • ${confidenceBandDisplay(confidenceTier)}`)}</p>
          </div>
        </article>
        <article class="panel compact-panel">
          <span class="metric-label">Glance panel</span>
          <div class="metric-stack">
            <div class="mini-score-pair">
              <span class="metric-label">Book line</span>
              <strong>${escapeHtml(`${bookmakerLineDisplay(marketLine.odds)} • ${hasUsableOdds(marketLine.odds) ? formatImpliedProbability(marketLine.odds) : "Pricing pending"}`)}</strong>
            </div>
            <div class="mini-score-pair">
              <span class="metric-label">Edge</span>
              <strong class="edge-tone-${escapeHtml(valueEdgeTone(fixture))}">${escapeHtml(valueEdgeDisplay(fixture))}</strong>
            </div>
            <div class="mini-score-pair">
              <span class="metric-label">Relevance</span>
              <strong>${matchedEntry ? escapeHtml(matchedEntry.reasons.join(" / ")) : "Window fixture"}</strong>
            </div>
          </div>
        </article>
        <article class="panel compact-panel">
          <span class="metric-label">Read first</span>
          <ul class="feature-list compact-list">
            ${clarity.risk_points.slice(0, 2).map((point) => `<li>${escapeHtml(point)}</li>`).join("")}
          </ul>
        </article>
      `;
    }
    const routeAudit = routeAuditProfile(fixture, decision);
    const agreementRows = decisionReasonRows(decision, 4);
    const marketHighlights = decisionMarketSuitabilityItems(decision).slice(0, 3);
    return `
      <article class="panel compact-panel compact-panel-primary">
        <span class="metric-label">${escapeHtml(routeAudit.routeActive ? "Published route" : "Context only")}</span>
        <div class="metric-stack">
          <strong class="metric-value">${escapeHtml(routeAudit.routeLabel || verdictLabel)}</strong>
          <p class="muted">${escapeHtml(`${safeTitleLabel(routeAudit.routeState || "context", "Context")} • ${safeTitleLabel(routeAudit.confidence, "Pending")} confidence${routeAudit.routeOdds ? ` • ${bookmakerLineDisplay(routeAudit.routeOdds)}` : ""}`)}</p>
        </div>
      </article>
      <article class="panel compact-panel">
        <span class="metric-label">Context audit</span>
        <div class="metric-stack">
          <strong class="metric-value">${escapeHtml(safeTitleLabel(routeAudit.auditState, "Pending"))}</strong>
          <p class="muted">${escapeHtml(`${routeAudit.agreement ?? "—"}% agreement • ${routeAudit.conflictLabel}`)}</p>
        </div>
      </article>
      <article class="panel compact-panel">
        <span class="metric-label">Agreement stack</span>
        ${
          agreementRows.length
            ? `<ul class="feature-list compact-list fixture-ratings-verdict-list">
                ${agreementRows
                  .map(
                    (row) => `
                      <li class="fixture-ratings-verdict-item fixture-ratings-verdict-item-${escapeHtml(row.tone)}">
                        <strong>${escapeHtml(row.tone === "support" ? "✓" : "✗")}</strong>
                        <span>${escapeHtml(row.text)}</span>
                      </li>
                    `
                  )
                  .join("")}
              </ul>`
            : `<div class="notice">No reconciled agreement stack has been published for this fixture yet.</div>`
        }
      </article>
      <article class="panel compact-panel">
        <span class="metric-label">Market suitability</span>
        ${
          marketHighlights.length
            ? `<div class="metric-stack">
                ${marketHighlights
                  .map(
                    (market) => `
                      <div class="mini-score-pair">
                        <span class="metric-label">${escapeHtml(market.label)}</span>
                        <strong>${escapeHtml(`${market.rating}% · ${market.band}`)}</strong>
                      </div>
                    `
                  )
                  .join("")}
              </div>`
            : `<div class="notice">No published market suitability layer is available yet.</div>`
        }
        <p class="muted">${escapeHtml(decisionTopCaution(decision))}</p>
      </article>
    `;
  };

  const renderRatingsVerdictStrip = (fixture, decision, clarity) => {
    if (!decision) {
      return `
        <section class="section">
          <article class="panel fixture-ratings-verdict">
            <h3>Ratings verdict</h3>
            <div class="notice">The published reconciler verdict has not been published for this fixture yet.</div>
          </article>
        </section>
      `;
    }
    const deployMode = String(fixture.publish_class || "").toUpperCase() === "DEPLOY";
    const modeCopy = deployMode ? "deploy signal" : "observed lean";
    const routeAudit = routeAuditProfile(fixture, decision);
    const toneIcon = (tone) => (tone === "support" ? "✓" : tone === "contradict" ? "✗" : "~");
    const supportRows = (decision.supporting_layers || []).slice(0, 3).map((token) => ({ tone: "support", text: reasonTokenLabel(token) }));
    const cautionRows = (decision.caution_layers || []).slice(0, 3).map((token) => ({ tone: "contradict", text: reasonTokenLabel(token) }));
    const rows = [...supportRows, ...cautionRows].slice(0, 5);
    return `
      <section class="section">
        <article class="panel fixture-ratings-verdict">
          <span class="metric-label">Context audit</span>
          <h3>${escapeHtml(`${safeTitleLabel(routeAudit.auditState || "mixed")} with ${modeCopy}`)}</h3>
          <p class="section-copy">${escapeHtml(routeAudit.routeActive ? "The published route remains the pick layer. This audit explains supporting and caution context without changing the deployed route." : "No pick is published here. This audit explains why the fixture is context-only or watch-first.")}</p>
          ${renderEntitySurfaceTiles([
            { label: "Published route", value: routeAudit.routeActive ? routeAudit.routeLabel : "No pick", meta: routeAudit.routeActive ? `${routeAudit.routeMarket} ${routeAudit.routePick}` : safeTitleLabel(routeAudit.routeState || "context", "Context"), tone: routeAudit.routeTone },
            { label: "Audit agreement", value: `${routeAudit.agreement ?? "—"}%`, meta: safeTitleLabel(routeAudit.auditState, "Pending"), tone: scoreTone(routeAudit.agreement) },
            { label: "Conflict", value: routeAudit.conflictLabel, meta: routeAudit.conflictCopy, tone: routeAudit.conflictTone },
          ])}
          <ul class="feature-list compact-list fixture-ratings-verdict-list">
            ${rows
              .map(
                (row) => `
                  <li class="fixture-ratings-verdict-item fixture-ratings-verdict-item-${escapeHtml(row.tone)}">
                    <strong>${escapeHtml(toneIcon(row.tone))}</strong>
                    <span>${escapeHtml(row.text)}</span>
                  </li>
                `
              )
              .join("")}
          </ul>
        </article>
      </section>
    `;
  };

  const renderFixtureTeamFaceOff = (fixture, decision) => {
    const rowsSource = Array.isArray(decision?.team_faceoff_summary) ? decision.team_faceoff_summary : [];
    if (!rowsSource.length) {
      return `
        <section class="section">
          <article class="panel">
            <h3>Team Ratings</h3>
            <div class="notice">Published team face-off ratings are not available for this fixture yet.</div>
          </article>
        </section>
      `;
    }
    const rows = rowsSource
      .map((entry) => {
        const homeValue = Number(entry.home_value);
        const awayValue = Number(entry.away_value);
        return `
          <div class="fixture-faceoff-row">
            <div class="fixture-faceoff-side fixture-faceoff-side-home">
              <strong>${escapeHtml(Number.isFinite(homeValue) ? Math.round(homeValue) : "—")}</strong>
              <div class="fixture-faceoff-track fixture-faceoff-track-home" aria-hidden="true">
                <span class="fixture-faceoff-bar fixture-faceoff-bar-home" style="width:${Math.max(0, Math.min(100, Number.isFinite(homeValue) ? homeValue : 0))}%"></span>
              </div>
            </div>
            <div class="fixture-faceoff-center">
              <span class="metric-label">${escapeHtml(entry.label || safeTitleLabel(entry.metric))}</span>
            </div>
            <div class="fixture-faceoff-side fixture-faceoff-side-away">
              <div class="fixture-faceoff-track fixture-faceoff-track-away" aria-hidden="true">
                <span class="fixture-faceoff-bar fixture-faceoff-bar-away" style="width:${Math.max(0, Math.min(100, Number.isFinite(awayValue) ? awayValue : 0))}%"></span>
              </div>
              <strong>${escapeHtml(Number.isFinite(awayValue) ? Math.round(awayValue) : "—")}</strong>
            </div>
          </div>
        `;
      })
      .join("");
    return `
      <section class="section">
        <article class="panel">
          <h3>Team Ratings</h3>
          <p class="section-copy">Only the ratings most diagnostic for this market are surfaced here, so the page reads like a decision tool rather than a generic data dump.</p>
          <div class="fixture-faceoff-head">
            <div class="fixture-faceoff-head-team fixture-faceoff-head-home">
              ${badgeMarkup(fixture.home_team_logo_url, fixture.home_team, "lineup-team-badge")}
              <div>
                <span class="metric-label">Home</span>
                <strong>${escapeHtml(fixture.home_team)}</strong>
              </div>
            </div>
            <span class="fixture-faceoff-head-center">vs</span>
            <div class="fixture-faceoff-head-team fixture-faceoff-head-away">
              <div>
                <span class="metric-label">Away</span>
                <strong>${escapeHtml(fixture.away_team)}</strong>
              </div>
              ${badgeMarkup(fixture.away_team_logo_url, fixture.away_team, "lineup-team-badge")}
            </div>
          </div>
          <div class="fixture-faceoff-grid">
            ${rows}
          </div>
        </article>
      </section>
    `;
  };

  const renderFixtureProfileNarrative = (fixture, homeIntelligence, awayIntelligence) => {
    const decision = state.selectedFixtureDecisionIntelligence || null;
    if (decision?.profile_narrative || decision?.profile_tags) {
      const homeTags = Array.isArray(decision?.profile_tags?.home) ? decision.profile_tags.home.slice(0, 4) : [];
      const awayTags = Array.isArray(decision?.profile_tags?.away) ? decision.profile_tags.away.slice(0, 4) : [];
      return `
        <section class="section">
          <div class="split">
            <article class="panel">
              <h3>${escapeHtml(fixture.home_team)} profile</h3>
              <ul class="feature-list compact-list">
                ${homeTags.length ? homeTags.map((tag) => `<li>${escapeHtml(tag)}</li>`).join("") : `<li>Mixed team profile</li>`}
              </ul>
            </article>
            <article class="panel">
              <h3>${escapeHtml(fixture.away_team)} profile</h3>
              <ul class="feature-list compact-list">
                ${awayTags.length ? awayTags.map((tag) => `<li>${escapeHtml(tag)}</li>`).join("") : `<li>Mixed team profile</li>`}
              </ul>
            </article>
          </div>
          <article class="panel fixture-matchup-narrative">
            <span class="metric-label">Matchup narrative</span>
            <p>${escapeHtml(decision.profile_narrative || "No published matchup narrative yet.")}</p>
          </article>
        </section>
      `;
    }
    if (!homeIntelligence || !awayIntelligence) {
      return "";
    }
    const homeTags = (homeIntelligence.profile_tags || []).slice(0, 3);
    const awayTags = (awayIntelligence.profile_tags || []).slice(0, 3);
    const { family, pick } = fixtureSignalProfile(fixture);
    let sentence = `${fixture.home_team} bring ${homeTags[0] || "a mixed profile"} into a matchup with ${fixture.away_team}'s ${awayTags[0] || "own structural read"}.`;
    if (family === "BTTS") {
      sentence += pick === "NO" ? " The shape is looking for suppression and restricted access rather than an open two-way trade." : " The shape points toward repeatable access for both forward lines rather than one side fully suppressing the game.";
    } else if (family === "OU25") {
      sentence += pick === "UNDER25" ? " The cleaner read is control and suppression if the stronger structure holds." : " The cleaner read is a live goal environment if the early pressure converts into repeated entries.";
    } else if (family === "FTR") {
      sentence += pick === "AWAY" ? ` The structural lean comes from ${fixture.away_team}'s travelling profile against the home-side resistance.` : pick === "DRAW" ? " The matchup reads like a parity contest where control and caution matter more than one-sided dominance." : ` The structural lean comes from ${fixture.home_team}'s home-side control against the away profile.`;
    }
    return `
      <section class="section">
        <div class="split">
          <article class="panel">
            <h3>${escapeHtml(fixture.home_team)} profile</h3>
            <ul class="feature-list compact-list">
              ${homeTags.length ? homeTags.map((tag) => `<li>${escapeHtml(tag)}</li>`).join("") : `<li>Mixed team profile</li>`}
            </ul>
          </article>
          <article class="panel">
            <h3>${escapeHtml(fixture.away_team)} profile</h3>
            <ul class="feature-list compact-list">
              ${awayTags.length ? awayTags.map((tag) => `<li>${escapeHtml(tag)}</li>`).join("") : `<li>Mixed team profile</li>`}
            </ul>
          </article>
        </div>
        <article class="panel fixture-matchup-narrative">
          <span class="metric-label">Matchup narrative</span>
          <p>${escapeHtml(sentence)}</p>
        </article>
      </section>
    `;
  };

  const renderFixtureUnitBattle = (decision) => {
    const rows = Array.isArray(decision?.unit_battle_summary) ? decision.unit_battle_summary : [];
    if (!rows.length) {
      return `
        <section class="section">
          <article class="panel">
            <h3>Unit battle</h3>
            <div class="notice">Published unit-battle intelligence is not available for this fixture yet.</div>
          </article>
        </section>
      `;
    }
    return `
      <section class="section">
        <article class="panel">
          <h3>Unit battle</h3>
          <p class="section-copy">This is the structural reason layer: attack against defence, midfield against midfield, and where the shape is actually tilting.</p>
          <div class="fixture-player-matchups">
            ${rows
              .map(
                (row) => `
                  <article class="fixture-player-matchup-card">
                    <div class="fixture-player-matchup-head">
                      <div>
                        <span class="metric-label">${escapeHtml("Home side")}</span>
                        <h4>${escapeHtml(row.label || "Unit battle")}</h4>
                      </div>
                      <div class="fixture-player-matchup-delta">${escapeHtml(`${row.delta > 0 ? "+" : ""}${row.delta ?? "—"}`)}</div>
                      <div class="fixture-player-matchup-away">
                        <span class="metric-label">${escapeHtml("Away side")}</span>
                        <h4>${escapeHtml("Counter weight")}</h4>
                      </div>
                    </div>
                    <div class="fixture-player-metric-grid">
                      <div>
                        <div class="fixture-player-metric"><span>${escapeHtml(fixture.home_team)}</span><strong>${escapeHtml(row.home_value ?? "—")}%</strong></div>
                      </div>
                      <div>
                        <div class="fixture-player-metric"><span>${escapeHtml(fixture.away_team)}</span><strong>${escapeHtml(row.away_value ?? "—")}%</strong></div>
                      </div>
                    </div>
                  </article>
                `
              )
              .join("")}
          </div>
        </article>
      </section>
    `;
  };

  const renderDecisionKeyPlayerDrivers = (decision) => {
    const directDriverCount = Array.isArray(decision?.key_player_drivers) ? decision.key_player_drivers.length : 0;
    let drivers = Array.isArray(decision?.key_player_drivers) ? decision.key_player_drivers.slice(0, 6) : [];
    let driverSource = directDriverCount ? "fixture" : "squad";
    if (!drivers.length) {
      const support = state.selectedFixtureDecisionSupport || {};
      const collectSquadFallback = (squadPayload, teamLabel) => {
        const players = Array.isArray(squadPayload?.players) ? squadPayload.players : [];
        return players.slice(0, 2).map((player) => ({
          team: teamLabel,
          player: player.surname || player.name || "Profile pending",
          role: player.position_group || player.position || "Utility",
          driver_metric:
            Number(player?.ratings?.goal_threat || 0) >= Number(player?.ratings?.creative_spark || 0)
              ? "Goal Threat"
              : "Creative Spark",
          driver_value: Math.max(Number(player?.ratings?.goal_threat || 0), Number(player?.ratings?.creative_spark || 0)) || Number(player?.ratings?.og_player_power || 0),
          power: Number(player?.ratings?.og_player_power || 0),
          source: "Squad fallback",
        }));
      };
      drivers = [
        ...collectSquadFallback(support.homeSquadIntelligence, support.homeTeamIntelligence?.team || "Home"),
        ...collectSquadFallback(support.awaySquadIntelligence, support.awayTeamIntelligence?.team || "Away"),
      ].slice(0, 6);
    }
    if (!drivers.length) {
      return `
        <section class="section">
          <article class="panel">
            <h3>Key player drivers</h3>
            <div class="notice">Published player-driver intelligence is not available for this fixture yet.</div>
          </article>
        </section>
      `;
    }
    return `
      <section class="section">
          <article class="panel">
            <h3>Key player drivers</h3>
            <p class="section-copy">${escapeHtml(
              driverSource === "squad"
                ? "Lineup-specific player drivers are not published for this fixture yet, so this surface is deliberately using the strongest publish-safe squad profiles."
                : "These are the players carrying the structural edge or caution inside the fixture decision layer."
            )}</p>
            <div class="pill-row">
              <span class="chip ${driverSource === "squad" ? "chip-observe" : "chip-signal"}">${escapeHtml(driverSource === "squad" ? "Squad fallback active" : "Fixture player layer")}</span>
              <span class="chip chip-reference">${escapeHtml(driverSource === "squad" ? "Not a confirmed XI" : `${directDriverCount} direct drivers`)}</span>
            </div>
          <div class="fixture-driver-grid">
            ${drivers
              .map(
                (driver) => `
                  <article class="fixture-driver-card ${driverSource === "squad" ? "fixture-driver-card-fallback" : ""}">
                    <span class="metric-label">${escapeHtml(driver.team || "Team")}</span>
                    <h4>${escapeHtml(driver.player || "Profile pending")}</h4>
                    <p class="muted">${escapeHtml(`${safeTitleLabel(driver.role, "Utility")} · ${driver.driver_metric || "Impact"} ${driver.driver_value ?? "—"}%`)}</p>
                    <div class="pill-row">
                      <span class="chip chip-reference">${escapeHtml(`Power ${driver.power ?? "—"}%`)}</span>
                      <span class="chip chip-reference">${escapeHtml(`${driver.driver_metric || "Impact"} ${driver.driver_value ?? "—"}%`)}</span>
                      ${driver.source ? `<span class="chip chip-observe">${escapeHtml(driver.source)}</span>` : ""}
                    </div>
                  </article>
                `
              )
              .join("")}
          </div>
        </article>
      </section>
    `;
  };

  const renderDecisionKeyMismatches = (decision) => {
    const mismatches = Array.isArray(decision?.key_mismatches) ? decision.key_mismatches.slice(0, 4) : [];
    if (!mismatches.length) {
      return "";
    }
    return `
      <section class="section">
        <article class="panel">
          <h3>Key mismatches</h3>
          <ul class="feature-list compact-list">
            ${mismatches
              .map(
                (item) => `
                  <li>
                    <strong>${escapeHtml(item.summary || item.zone || "Mismatch edge")}</strong><br />
                    <span class="muted">${escapeHtml(`${item.advantage || "Advantage"} • ${item.mismatch_score ?? "—"} points`)}</span>
                  </li>
                `
              )
              .join("")}
          </ul>
        </article>
      </section>
    `;
  };

  const renderDecisionMarketSuitability = (decision) => {
    const items = decisionMarketSuitabilityItems(decision);
    if (!items.length) {
      return `
        <section class="section">
          <article class="panel">
            <h3>Market suitability</h3>
            <div class="notice">Published market-suitability intelligence is not available for this fixture yet.</div>
          </article>
        </section>
      `;
    }
    const posture = decisionMarketPosture(decision);
    const roleFor = (market) => {
      if (posture?.best === market) return "Best";
      if (posture?.secondary === market) return "Secondary";
      if (posture?.avoid === market) return "Avoid";
      if (posture?.weak === market) return "Weak";
      return "Context";
    };
    const roleClassFor = (role) => String(role || "context").toLowerCase();
    return `
      <section class="section">
        <article class="panel">
          <div class="intel-placeholder-head">
            <div>
              <span class="metric-label">Market intelligence</span>
              <h3>Ranked market posture</h3>
            </div>
            <span class="chip chip-signal">${escapeHtml(`${items.length} reads`)}</span>
          </div>
          <p class="section-copy">This is the judgement layer, not a market dump: how well the current fixture structure supports each product family.</p>
          <div class="market-posture-rail">
            ${[
              ["Best", posture?.best],
              ["Secondary", posture?.secondary],
              ["Weak", posture?.weak],
              ["Avoid", posture?.avoid],
            ]
              .map(
                ([label, market]) => `
                  <article class="market-posture-card market-posture-card-${escapeHtml(roleClassFor(label))}">
                    <span class="metric-label">${escapeHtml(label)}</span>
                    <strong>${escapeHtml(market ? `${market.label} · ${market.rating}%` : label === "Avoid" ? "None flagged" : "Pending")}</strong>
                    <p class="muted">${escapeHtml(market?.band || (label === "Avoid" ? "No hard avoid state" : "No published read yet"))}</p>
                  </article>
                `
              )
              .join("")}
          </div>
          <div class="fixture-market-suitability-grid fixture-market-suitability-grid-compact">
            ${items
              .slice(0, 6)
              .map(
                (market) => {
                  const role = roleFor(market);
                  return `
                  <article class="market-intel-card market-intel-card-${escapeHtml(roleClassFor(role))}">
                    <div class="market-intel-card-head">
                      <div>
                        <span class="signal-label">${escapeHtml(market.label)}</span>
                        <span class="signal-value">${escapeHtml(`${market?.rating ?? "—"}% · ${safeTitleLabel(market?.band, "Pending")}`)}</span>
                      </div>
                      <span class="chip ${role === "Best" || role === "Secondary" ? "chip-signal" : role === "Avoid" || role === "Weak" ? "chip-observe" : "chip-reference"}">${escapeHtml(role)}</span>
                    </div>
                    <p class="muted">${escapeHtml(market?.read || "No published read yet.")}</p>
                    <div class="market-intel-meta">
                      <span>${escapeHtml(`${market.support?.length || 0} support / ${market.cautions?.length || 0} caution`)}</span>
                    </div>
                  </article>
                `;
                }
              )
              .join("")}
          </div>
        </article>
      </section>
    `;
  };

  const renderFixtureH2HSupport = (fixture, h2hSupport = null) => {
    const decision = state.selectedFixtureDecisionIntelligence || null;
    const coverage = h2hCoverageProfile(h2hSupport, decision);
    const renderH2HTiles = (source = {}) => {
      const summaryItems = [
        { label: "Sample", value: `${source.sample_size ?? 0} matches`, meta: "Last five cap" },
        { label: "Goal environment", value: `${source.goal_environment ?? 0}%`, meta: "Historic scoring climate" },
        { label: "BTTS regime", value: `${source.btts_regime ?? 0}%`, meta: "Historic two-way scoring" },
        { label: "Over 2.5", value: `${source.over25_rate ?? 0}%`, meta: "Historic 3+ goal rate" },
        { label: "Draw rate", value: `${source.draw_rate ?? 0}%`, meta: "Historic stalemate share" },
        { label: "Booking heat", value: `${source.booking_heat ?? 0}%`, meta: "Historic card climate" },
      ];
      return renderEntitySurfaceTiles(summaryItems.map((item) => ({ ...item, tone: scoreTone(parseFloat(String(item.value))) })));
    };
    if (h2hSupport && coverage.status !== "fallback") {
      return `
        <section class="section">
          <article class="panel">
            <div class="intel-placeholder-head">
              <div>
                <span class="metric-label">H2H context</span>
                <h3>${escapeHtml(coverage.status === "historical" ? "Historical same-team-pair context" : "Published H2H support")}</h3>
              </div>
              <span class="chip ${coverage.status === "historical" ? "chip-observe" : "chip-signal"}">${escapeHtml(coverage.label)}</span>
            </div>
            <p class="section-copy">${escapeHtml(coverage.summary)}</p>
            ${renderH2HTiles(h2hSupport)}
          </article>
        </section>
      `;
    }
    if (decision?.h2h_context) {
      const context = decision.h2h_context;
      if (!context.available) {
        return `
          <section class="section">
            <article class="panel intel-placeholder-panel">
              <div class="intel-placeholder-head">
                <div>
                  <span class="metric-label">H2H context</span>
                  <h3>No recent direct matchup sample published</h3>
                </div>
                <span class="chip chip-reference">${escapeHtml(coverage.label)}</span>
              </div>
              <p class="section-copy">This layer is not used to force the decision. When a reliable direct sample is unavailable, the page keeps ratings, squad drivers, and market posture in charge.</p>
              <div class="lineup-fallback-grid">
                <article class="lineup-fallback-card">
                  <span class="metric-label">Direct H2H</span>
                  <strong>Unavailable</strong>
                  <p class="muted">No recent direct matchup sample has passed the publish-safe source for this fixture key.</p>
                </article>
                <article class="lineup-fallback-card">
                  <span class="metric-label">Decision role</span>
                  <strong>Does not lead</strong>
                  <p class="muted">Team ratings, player drivers, and market alignment remain the primary decision layers.</p>
                </article>
                <article class="lineup-fallback-card">
                  <span class="metric-label">Product state</span>
                  <strong>Handled deliberately</strong>
                  <p class="muted">The page explains the gap rather than rendering a broken or empty tab.</p>
                </article>
              </div>
            </article>
          </section>
        `;
      }
      return `
        <section class="section">
          <article class="panel">
            <h3>H2H context</h3>
            <p class="section-copy">${escapeHtml(context.summary || "Historic meeting regime is shown here as supporting context, not as the primary signal layer.")}</p>
            ${renderH2HTiles(context)}
          </article>
        </section>
      `;
    }
    if (!h2hSupport) {
      return `
        <section class="section">
          <article class="panel intel-placeholder-panel">
            <div class="intel-placeholder-head">
              <div>
                <span class="metric-label">H2H context</span>
                <h3>No recent direct matchup sample published</h3>
              </div>
              <span class="chip chip-reference">Context unavailable</span>
            </div>
            <p class="section-copy">This layer is not used to force the decision. Team ratings, squad drivers, and market posture remain primary.</p>
          </article>
        </section>
      `;
    }
    return `
      <section class="section">
        <article class="panel">
          <h3>H2H context</h3>
          <p class="section-copy">${escapeHtml(h2hSupport.summary || "Historic meeting regime is shown here as supporting context, not as the primary signal layer.")}</p>
          ${renderH2HTiles(h2hSupport)}
        </article>
      </section>
    `;
  };

  const renderFixtureOverviewPrimer = (fixture, clarity) => {
    const decision = state.selectedFixtureDecisionIntelligence || null;
    const preview = decision?.preview || null;
    const posture = decisionMarketPosture(decision);
    const watchlist = decision?.watchlist || null;
    const supportRows = decisionReasonRows(decision, 4);
    const routeAudit = routeAuditProfile(fixture, decision);

    if (!decision) {
      return `
        <section class="section">
          <div class="split">
            <article class="panel">
              <h3>Fixture primer</h3>
              <p class="section-copy">${escapeHtml(clarity.action_copy)}</p>
              <div class="notice">The reconciled fixture decision layer has not been published for this fixture yet.</div>
            </article>
          </div>
        </section>
      `;
    }

    return `
      <section class="section">
        <div class="split">
          <article class="panel">
            <span class="metric-label">Fixture primer</span>
            <h3>${escapeHtml(routeAudit.routeActive ? routeAudit.routeLabel : "No published pick")}</h3>
            <p class="section-copy">${escapeHtml(preview?.short_summary || decision.public_safe_summary || "No published public-safe summary is available yet.")}</p>
            ${renderEntitySurfaceTiles([
              { label: "Route", value: routeAudit.routeActive ? safeTitleLabel(routeAudit.routeState, "Deploy") : "No pick", meta: routeAudit.routeActive ? `${routeAudit.routeMarket} ${routeAudit.routePick}` : safeTitleLabel(routeAudit.routeState || "context", "Context only"), tone: routeAudit.routeTone },
              { label: "Audit", value: safeTitleLabel(routeAudit.auditState, "Pending"), meta: `${routeAudit.agreement ?? "—"}% agreement`, tone: routeAudit.auditTone },
              { label: "Conflict", value: routeAudit.conflictLabel, meta: routeAudit.conflictCopy, tone: routeAudit.conflictTone },
            ])}
            <ul class="feature-list compact-list fixture-ratings-verdict-list">
              ${supportRows.length
                ? supportRows
                    .map(
                      (row) => `
                        <li class="fixture-ratings-verdict-item fixture-ratings-verdict-item-${escapeHtml(row.tone)}">
                          <strong>${escapeHtml(row.tone === "support" ? "✓" : "✗")}</strong>
                          <span>${escapeHtml(row.text)}</span>
                        </li>
                      `
                    )
                    .join("")
                : `<li>No published agreement stack is available yet.</li>`}
            </ul>
          </article>
          <article class="panel">
            <span class="metric-label">Market posture</span>
            <h3>${escapeHtml(preview?.market_summary || "No published market summary yet.")}</h3>
            ${renderEntitySurfaceTiles([
              {
                label: "Best market",
                value: posture?.best ? `${posture.best.label} · ${posture.best.rating}%` : "Pending",
                meta: posture?.best?.band || "No published read yet",
                tone: posture?.best ? scoreTone(posture.best.rating) : "reference",
              },
              {
                label: "Secondary",
                value: posture?.secondary ? `${posture.secondary.label} · ${posture.secondary.rating}%` : "Pending",
                meta: posture?.secondary?.band || "No published read yet",
                tone: posture?.secondary ? scoreTone(posture.secondary.rating) : "reference",
              },
              {
                label: "Weak read",
                value: posture?.weak ? `${posture.weak.label} · ${posture.weak.rating}%` : "Pending",
                meta: posture?.weak?.band || "No published read yet",
                tone: posture?.weak ? scoreTone(posture.weak.rating) : "reference",
              },
              {
                label: "Avoid / caution",
                value: posture?.avoid ? `${posture.avoid.label} · ${posture.avoid.rating}%` : "None flagged",
                meta: posture?.avoid?.band || "No market in avoid state",
                tone: posture?.avoid ? "observe" : "reference",
              },
            ])}
            <p class="section-copy"><strong>Main caution:</strong> ${escapeHtml(preview?.caution_line || decisionTopCaution(decision))}</p>
            <a class="ghost-button" href="./fixture.html?fixture=${encodeURIComponent(fixture.fixture_key || "")}&tab=markets#fixture-tab-markets">See full market posture</a>
          </article>
        </div>
      </section>
      ${
        watchlist
          ? `
      <section class="section">
          <article class="panel">
            <h3>Watchlist posture</h3>
            ${renderEntitySurfaceTiles([
              {
                label: "Watch state",
                value: safeTitleLabel(watchlist.state || routeAudit.auditState, "Pending"),
                meta: watchlist.label || "Live confirmation layer",
                tone: decisionStateTone(watchlist.state || routeAudit.auditState),
              },
              {
                label: "Readiness",
                value: `${watchlist.readiness_score ?? routeAudit.agreement ?? "—"}%`,
                meta: "Pre-match watch posture",
                tone: scoreTone(watchlist.readiness_score ?? routeAudit.agreement),
              },
            ])}
            <p class="section-copy">${escapeHtml(watchlist.public_summary || "Pre-match is not clean enough for full deployment, but the shape is interesting enough to monitor live.")}</p>
          </article>
      </section>
      `
          : ""
      }
    `;
  };

  const renderFixtureDecisionCompanion = (fixture, clarity) => {
    const decision = state.selectedFixtureDecisionIntelligence || null;
    const bundle = state.selectedFixtureDecisionSupport || null;
    const homeTeamIntelligence = bundle?.homeTeamIntelligence || null;
    const awayTeamIntelligence = bundle?.awayTeamIntelligence || null;
    return `
      ${renderFixtureDecisionVerdict(fixture, decision)}
      ${renderRatingsVerdictStrip(fixture, decision, clarity)}
      ${renderFixtureTeamFaceOff(fixture, decision)}
      ${renderFixtureProfileNarrative(fixture, homeTeamIntelligence, awayTeamIntelligence)}
      ${renderFixtureUnitBattle(decision)}
      ${renderDecisionKeyPlayerDrivers(decision)}
      ${renderDecisionKeyMismatches(decision)}
      ${renderDecisionMarketSuitability(decision)}
      ${renderFixtureH2HSupport(fixture, bundle?.h2hSupport || null)}
    `;
  };

  const renderSquadRoleSummary = (payload) => {
    if (!payload || !Array.isArray(payload.players) || !payload.players.length) {
      return `<div class="notice">No publish-safe squad role mix is available for this team yet.</div>`;
    }
    const grouped = payload.players.reduce((acc, player) => {
      const key = safeTitleLabel(player?.position_group, "Utility");
      if (!acc[key]) {
        acc[key] = { label: key, value: 0, best: 0 };
      }
      acc[key].value += 1;
      acc[key].best = Math.max(acc[key].best, Number(player?.ratings?.og_player_power || 0));
      return acc;
    }, {});
    const items = Object.values(grouped)
      .sort((left, right) => right.best - left.best || right.value - left.value || left.label.localeCompare(right.label))
      .slice(0, 6)
      .map((item) => ({
        label: item.label,
        value: item.best,
        band: `${item.value} players`,
        meta: `${item.value} squad profiles`,
        tone: scoreTone(item.best),
      }));
    return renderScoreBreakdown(items, "No publish-safe squad role mix is available for this team yet.");
  };

  const renderSquadSnapshot = (payload) => {
    if (!payload || !Array.isArray(payload.players) || !payload.players.length) {
      return `<div class="notice">Publish-safe squad intelligence is not available for this team yet.</div>`;
    }
    const topPlayers = payload.players.slice(0, 5);
    return `
      <div class="split">
        <article class="panel">
          <h3>Squad depth snapshot</h3>
          <p class="section-copy">This keeps the player layer publish-safe: strength distribution, trust depth, and where the best squad profiles are concentrated.</p>
          ${renderSquadDepthTiles(payload)}
        </article>
        <article class="panel">
          <h3>Squad intelligence leaders</h3>
          <ul class="feature-list compact-list">
            ${renderSquadLeaderList("Power", payload?.leaders?.power)}
            ${renderSquadLeaderList("Goal threat", payload?.leaders?.goal_threat)}
            ${renderSquadLeaderList("Creative spark", payload?.leaders?.creative_spark)}
            ${renderSquadLeaderList("Discipline risk", payload?.leaders?.discipline_risk)}
          </ul>
        </article>
        <article class="panel">
          <h3>Top rated profiles</h3>
          <ul class="feature-list compact-list">
            ${topPlayers
              .map(
                (player) => `
                  <li>
                    <strong>${escapeHtml(player.surname || player.name || "Player")}</strong><br />
                    <span class="muted">${escapeHtml(`${safeTitleLabel(player.position_group, "Utility")} • ${player.ratings?.og_player_power ?? "—"}% OG Player Power`)}</span>
                  </li>
                `
              )
              .join("")}
          </ul>
        </article>
      </div>
      <div class="split">
        <article class="panel">
          <h3>Squad role mix</h3>
          <p class="section-copy">Role-group strength based on the best current publish-safe player power profiles in this squad.</p>
          ${renderSquadRoleSummary(payload)}
        </article>
        <article class="panel">
          <h3>Featured player intelligence</h3>
          <div class="card-grid card-grid-compact">
            ${topPlayers.slice(0, 3).map((player) => renderPlayerIntelligenceCard(player)).join("")}
          </div>
        </article>
      </div>
    `;
  };

  const renderEntitySubnav = (items, label) => `
    <section class="section section-tight">
      <nav class="page-subnav" aria-label="${escapeHtml(label)}">
        <div class="page-subnav-scroll">
          ${items
            .map(
              ([text, href, active]) => `
                <a class="page-subnav-link ${active ? "is-active" : ""}" href="${href}">
                  ${escapeHtml(text)}
                </a>
              `
            )
            .join("")}
        </div>
      </nav>
    </section>
  `;

  const renderDirectoryCard = ({ title, badgeUrl, badgeName, href, metaLines = [], summary = "", ctaLabel = "Open" }) => `
    <article class="panel entity-directory-card">
      <div class="entity-directory-head">
        ${badgeMarkup(badgeUrl, badgeName || title, "entity-mark")}
        <div>
          <h3>${escapeHtml(title)}</h3>
          ${summary ? `<p class="muted">${escapeHtml(summary)}</p>` : ""}
        </div>
      </div>
      <ul class="feature-list compact-list">
        ${metaLines.map((line) => `<li>${escapeHtml(line)}</li>`).join("")}
      </ul>
      <div class="cta-row">
        <a class="ghost-button" href="${href}">${escapeHtml(ctaLabel)}</a>
      </div>
    </article>
  `;

  const entityFixtureCard = (row) => {
    const deskState = publicDeskState(row);
    const notes = Array.isArray(row.context_summary?.notes) ? row.context_summary.notes.slice(0, 2) : [];
    return `
      <article class="panel entity-fixture-card entity-fixture-card-${escapeHtml(deskState.tone)}">
        <div class="fixture-stream-summary-main">
          <div class="intelligence-card-head">
            <span class="fixture-state-pill fixture-state-pill-${escapeHtml(deskState.tone)}">${escapeHtml(deskState.label)}</span>
            <span class="chip chip-market">${escapeHtml(marketFamilyLabel(row.signal_summary?.market_family))}</span>
            <span class="chip chip-reference">${escapeHtml(formatKickoffLabel(row.kickoff_time))}</span>
          </div>
          <strong class="fixture-teamline dashboard-teamline">
            ${badgeMarkup(row.home_team_logo_url, row.home_team)}
            <span class="team-name">${escapeHtml(row.home_team)}</span>
            <span class="versus">vs</span>
            ${badgeMarkup(row.away_team_logo_url, row.away_team)}
            <span class="team-name">${escapeHtml(row.away_team)}</span>
          </strong>
          <p class="fixture-stream-headline">${escapeHtml(
            row.signal_summary?.headline || row.signal_summary?.summary_text || "Fixture intelligence is available."
          )}</p>
        </div>
        ${
          notes.length
            ? `<ul class="feature-list compact-list">${notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>`
            : `<p class="muted">This row currently carries a concise public-safe read only.</p>`
        }
        <div class="cta-row">
          <a class="ghost-button" href="${fixtureDetailHref(row)}">Open fixture intelligence</a>
        </div>
      </article>
    `;
  };

  const renderEntityFixtureSection = (rows, emptyCopy) => {
    if (!rows.length) {
      return `<div class="notice">${escapeHtml(emptyCopy)}</div>`;
    }
    return `<div class="card-grid">${rows.map((row) => entityFixtureCard(row)).join("")}</div>`;
  };

  const renderEntityCompactEmpty = (copy, tone = "default") => `
    <div class="entity-empty-card ${tone === "muted" ? "entity-empty-card-muted" : ""}">
      <p class="section-copy">${escapeHtml(copy)}</p>
    </div>
  `;

  const renderTeamIntelligenceBuckets = (team) => {
    const deployRows = team.rows.filter((row) => String(row.publish_class || row.fixture_class || "").toUpperCase() === "DEPLOY");
    const observeRows = team.rows.filter((row) => String(row.publish_class || row.fixture_class || "").toUpperCase() === "OBSERVE");
    const contextRows = team.rows.filter((row) => {
      const key = String(row.publish_class || row.fixture_class || "").toUpperCase();
      return key === "CONTEXT" || key === "MONITOR";
    });
    const hasDeploy = deployRows.length > 0;
    const hasObserve = observeRows.length > 0;
    const hasContext = contextRows.length > 0;
    const adaptiveSplitClass =
      hasDeploy && hasObserve ? "split split-top" : "split split-top split-entity-adaptive";
    const primaryTitle = hasDeploy ? "Deployable team reads" : "Watch-first team reads";
    const primaryCopy = hasDeploy
      ? "These are the current-window rows where this team is part of an active deploy posture."
      : "These are the rows worth tracking, but not treating as direct deployment calls yet.";
    const primaryRows = hasDeploy ? deployRows : observeRows;
    const primaryEmptyCopy = hasDeploy
      ? "No deployable team-linked rows are visible in the current window yet."
      : "No observe-level team-linked rows are visible in the current window yet.";
    const secondaryTitle = hasDeploy ? "Watch-first team reads" : "Deployable team reads";
    const secondaryCopy = hasDeploy
      ? "These are the rows worth tracking, but not treating as direct deployment calls yet."
      : "These are the current-window rows where this team is part of an active deploy posture.";
    const secondaryRows = hasDeploy ? observeRows : deployRows;
    const secondaryEmptyCopy = hasDeploy
      ? "No observe-level team-linked rows are visible in the current window yet."
      : "No deployable team-linked rows are visible in the current window yet.";
    const primaryContent =
      primaryRows.length > 0
        ? renderEntityFixtureSection(primaryRows, primaryEmptyCopy)
        : renderEntityCompactEmpty(primaryEmptyCopy, "muted");
    const secondaryContent =
      secondaryRows.length > 0
        ? renderEntityFixtureSection(secondaryRows, secondaryEmptyCopy)
        : renderEntityCompactEmpty(secondaryEmptyCopy, "muted");
    return `
      <section class="section">
        <div class="${adaptiveSplitClass}">
          <article class="panel ${hasDeploy && !hasObserve ? "panel-primary-entity" : !hasDeploy && hasObserve ? "panel-primary-entity" : ""}">
            <h3>${primaryTitle}</h3>
            <p class="section-copy">${primaryCopy}</p>
            ${primaryContent}
          </article>
          <article class="panel ${primaryRows.length > 0 && !secondaryRows.length ? "panel-secondary-entity panel-compact-stack" : ""}">
            <h3>${secondaryTitle}</h3>
            <p class="section-copy">${secondaryCopy}</p>
            ${secondaryContent}
          </article>
        </div>
      </section>
      <section class="section">
        <article class="panel ${hasContext ? "" : "panel-compact-stack"}">
          <h3>Context and monitor layer</h3>
          <p class="section-copy">This is the softer context around the team: useful shape, caution, or slate-awareness without a direct deploy call.</p>
          ${
            hasContext
              ? renderEntityFixtureSection(contextRows, "No context or monitor rows are visible for this team in the current window.")
              : renderEntityCompactEmpty("No context or monitor rows are visible for this team in the current window.", "muted")
          }
        </article>
      </section>
    `;
  };

  const matchesView = () => {
    const rows = orderedFixtureRows();
    const visibleRows = matchesFeedRows(rows);
    const nextFixture = visibleRows[0] || rows[0] || null;
    return `
      <section class="x-matches-page">
        <header class="x-timeline-header">
          <div>
            <h1>Matches</h1>
            <p>Timeline view for fixtures, model reads, weather context, injury notes, and admin updates.</p>
          </div>
          <div class="x-next-kickoff">
            <span>Next kickoff</span>
            <strong>${escapeHtml(nextFixture ? `${teamCardName(nextFixture.home_team)} x ${teamCardName(nextFixture.away_team)}` : "No active fixture")}</strong>
            <small>${escapeHtml(nextFixture ? formatKickoffLabel(nextFixture.kickoff_time) : "Window pending")}</small>
          </div>
        </header>
        <form id="matches-search" class="x-search-bar" action="./matches.html" method="get">
          <label>
            <span>Search teams, players, fixtures</span>
            <input name="q" value="${escapeHtml(matchesSearchQuery)}" placeholder="Search team, league, market..." autocomplete="off" />
          </label>
          <button class="button" type="submit">Search</button>
          ${
            matchesSearchQuery || matchesFavouritesOnly
              ? `<a class="ghost-button" href="./matches.html">Clear</a>`
              : ""
          }
        </form>
        <div class="x-feed-stats">
          <span>${escapeHtml(visibleRows.length)} visible fixtures</span>
          <span>${escapeHtml(new Set(rows.map((row) => row.league)).size)} competitions</span>
          <span>${escapeHtml(state.runtime.matchFavourites.length)} favourites</span>
          ${matchesFavouritesOnly ? `<span>Favourite filter active</span>` : ""}
        </div>
        ${renderMatchesTimeline(rows)}
      </section>
      ${matchesBottomNav()}
    `;
  };

  const liveView = () => {
    const now = Date.now();
    const rows = orderedFixtureRows().filter((row) => {
      const ts = kickoffTimestamp(row.kickoff_time);
      return Number.isFinite(ts) && ts >= now - 4 * 60 * 60 * 1000 && ts <= now + 8 * 60 * 60 * 1000;
    });
    const fallbackRows = rows.length ? rows : orderedFixtureRows().slice(0, 12);
    return `
      <section class="hero">
        <article class="hero-main">
          <p class="hero-kicker">Live</p>
          <h1>Current and near-kickoff desk.</h1>
          <p>This is the calmer live-state surface. It focuses on fixtures in or near the current football window rather than the full board.</p>
          <div class="pill-row">
            <span class="stat-chip">Near kickoff</span>
            <span class="stat-chip">Current window</span>
            <span class="stat-chip">Reduced clutter</span>
          </div>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Live desk size</span>
            <span class="metric-value">${escapeHtml(fallbackRows.length)}</span>
          </div>
        </aside>
      </section>
      <section class="section">
        ${renderPublicFixtureGroups(fallbackRows)}
      </section>
    `;
  };

  const competitionDirectoryView = () => {
    const groups = groupItemsByLeague(orderedFixtureRows());
    return `
      <section class="hero">
        <article class="hero-main">
          <p class="hero-kicker">Competitions</p>
          <h1>League-first orientation.</h1>
          <p>Browse the active competitions first, then drop into a competition page when you want fixtures, results, standings, and current signal posture in one place.</p>
          <div class="pill-row">
            <span class="stat-chip">Standings aware</span>
            <span class="stat-chip">Current window fixtures</span>
            <span class="stat-chip">Calm context first</span>
          </div>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Active competitions</span>
            <span class="metric-value">${escapeHtml(groups.length)}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Window fixtures</span>
            <span class="metric-value">${escapeHtml(orderedFixtureRows().length)}</span>
          </div>
        </aside>
      </section>
      <section class="section">
        <div class="card-grid">
          ${groups
            .map((group) => {
              const first = group.items[0];
              return renderDirectoryCard({
                title: group.league,
                badgeUrl: first?.league_logo_url || "",
                badgeName: group.league,
                href: competitionPageHref(group.league),
                summary: "Current-window competition desk",
                metaLines: [
                  `Fixtures in window: ${group.items.length}`,
                  `Active teams: ${new Set(group.items.flatMap((row) => [row.home_team, row.away_team]).filter(Boolean)).size}`,
                  `Earliest fixture: ${formatKickoffLabel(first?.kickoff_time || "")}`,
                ],
                ctaLabel: "Open competition desk",
              });
            })
            .join("")}
        </div>
      </section>
    `;
  };

  const competitionEntityView = (competition) => {
    const competitionClassMix = collectPublishClassMix(competition.rows);
    const competitionMarketMix = collectMarketFamilyMix(competition.rows);
    const competitionTrackedRows =
      competitionClassMix.DEPLOY + competitionClassMix.OBSERVE + competitionClassMix.CONTEXT + competitionClassMix.MONITOR;
    const tabs = [
      ["Overview", competitionPageHref(competition.name, "overview"), selectedCompetitionTab === "overview"],
      ["Fixtures", competitionPageHref(competition.name, "fixtures"), selectedCompetitionTab === "fixtures"],
      ["Results", competitionPageHref(competition.name, "results"), selectedCompetitionTab === "results"],
      ["Table", competitionPageHref(competition.name, "table"), selectedCompetitionTab === "table"],
      ["Context", competitionPageHref(competition.name, "context"), selectedCompetitionTab === "context"],
    ];
    const overviewContent = `
      <section class="section">
        <div class="split">
          <article class="panel">
            <h3>Competition overview</h3>
            <ul class="feature-list compact-list">
              <li>Fixtures in current window: ${escapeHtml(competition.rows.length)}</li>
              <li>Active teams in view: ${escapeHtml(competition.teamCount)}</li>
              <li>Deploy rows: ${escapeHtml(competition.deployCount)}</li>
              <li>Observe rows: ${escapeHtml(competition.observeCount)}</li>
            </ul>
          </article>
          <article class="panel">
            <h3>Ownership of this page</h3>
            <ul class="feature-list compact-list">
              <li>Competition pages orient the slate across fixtures and standings.</li>
              <li>Full decision logic, caution framing, and Telegram-ready language stay on fixture pages.</li>
              <li>Use this desk to scan the competition, then drop into a fixture when the row deserves the full intelligence treatment.</li>
            </ul>
          </article>
        </div>
      </section>
      <section class="section">
        <div class="split">
          <article class="panel">
            <h3>Competition signal surface</h3>
            <p class="section-copy">This keeps the current-window competition mix visible: how much is deployable, how much is watch-first, and how much is context only.</p>
            ${renderEntitySurfaceTiles([
              { label: "Deploy", value: competitionClassMix.DEPLOY, meta: `${Math.round((competitionClassMix.DEPLOY / Math.max(1, competition.rows.length)) * 100)}% of rows`, tone: "deploy" },
              { label: "Observe", value: competitionClassMix.OBSERVE, meta: `${Math.round((competitionClassMix.OBSERVE / Math.max(1, competition.rows.length)) * 100)}% of rows`, tone: "observe" },
              { label: "Context / monitor", value: competitionClassMix.CONTEXT + competitionClassMix.MONITOR, meta: `${Math.round((((competitionClassMix.CONTEXT + competitionClassMix.MONITOR) / Math.max(1, competition.rows.length)) * 100))}% of rows`, tone: "reference" },
            ])}
          </article>
          <article class="panel">
            <h3>Market-family distribution</h3>
            <p class="section-copy">This shows which market families are actually driving the visible competition slice right now.</p>
            ${renderEntityBreakdown(
              competitionMarketMix.map((item) => ({
                ...item,
                tone:
                  item.label === "BTTS" ? "deploy" : item.label === "OU25" ? "observe" : "reference",
              })),
              competition.rows.length,
              "No market-family distribution is visible for this competition yet."
            )}
          </article>
        </div>
      </section>
      <section class="section">
        <div class="section-head">
          <div>
            <h2>Featured window fixtures</h2>
            <p class="section-copy">This is the current competition slice that is active in the published window.</p>
          </div>
        </div>
        ${renderEntityFixtureSection(competition.rows.slice(0, 6), "No fixtures are visible for this competition in the current window yet.")}
      </section>
    `;
    const fixturesContent = `
      <section class="section">
        <article class="panel widget-reference-shell archive-layer">
          <div class="widget-reference-head">
            <div>
              <span class="metric-label">Broader schedule</span>
              <h4>Competition fixtures archive</h4>
            </div>
          </div>
          <p class="section-copy">This is the wider league schedule from the upstream feed, so the competition page can feel like a real destination instead of only a current-window slice.</p>
          <div
            class="widget-reference-frame"
            data-role="competition-fixtures-reference"
            data-competition="${escapeHtml(competition.name)}"
            data-league-id="${escapeHtml(competition.apiLeagueId)}"
            data-season="${escapeHtml(competition.apiSeason)}"
          >
            <div class="reference-loading">Loading broader competition fixtures…</div>
          </div>
        </article>
      </section>
      <section class="section">
        <article class="panel current-window-layer">
          <div class="widget-reference-head">
            <div>
              <span class="metric-label">OG current window</span>
              <h4>Current-window fixtures</h4>
            </div>
          </div>
          <p class="section-copy">These are the fixtures this competition currently contributes to the active Odds Genius public window.</p>
          ${renderEntityFixtureSection(
            competition.fixtures.upcoming,
            "No upcoming fixtures from this competition are currently visible in the published window."
          )}
        </article>
      </section>
    `;
    const resultsContent = `
      <section class="section">
        <article class="panel widget-reference-shell archive-layer">
          <div class="widget-reference-head">
            <div>
              <span class="metric-label">Broader results</span>
              <h4>Competition results archive</h4>
            </div>
          </div>
          <p class="section-copy">This pulls a wider recent-results layer from the upstream feed so the competition page carries a real archive feel, not just today’s public slice.</p>
          <div
            class="widget-reference-frame"
            data-role="competition-results-reference"
            data-competition="${escapeHtml(competition.name)}"
            data-league-id="${escapeHtml(competition.apiLeagueId)}"
            data-season="${escapeHtml(competition.apiSeason)}"
          >
            <div class="reference-loading">Loading broader competition results…</div>
          </div>
        </article>
      </section>
      <section class="section">
        <article class="panel current-window-layer">
          <div class="widget-reference-head">
            <div>
              <span class="metric-label">OG current window</span>
              <h4>Current-window settled fixtures</h4>
            </div>
          </div>
          <p class="section-copy">These are competition fixtures in the current published window that already look settled or complete.</p>
          ${renderEntityFixtureSection(
            competition.fixtures.results,
            "No completed fixtures from this competition are currently visible in the published window."
          )}
        </article>
      </section>
    `;
    const tableContent = `
      <section class="section">
        <article class="panel widget-reference-shell">
          <div class="widget-reference-head">
            <div>
              <span class="metric-label">Standings desk</span>
              <h4>${escapeHtml(competition.name)} table</h4>
            </div>
          </div>
          <p class="section-copy">Reference standings stay here at competition level, while fixture pages keep the direct match-level read.</p>
          <div
            class="widget-reference-frame"
            data-role="competition-standings-reference"
            data-competition="${escapeHtml(competition.name)}"
            data-league-id="${escapeHtml(competition.apiLeagueId)}"
            data-season="${escapeHtml(competition.apiSeason)}"
          >
            <div class="reference-loading">Loading competition table…</div>
          </div>
        </article>
      </section>
    `;
    const contextContent = `
      <section class="section">
        <div class="split">
          <article class="panel">
            <h3>Deploy posture inside this competition</h3>
            <div class="pill-row">
              <span class="chip chip-signal">Deploy rows: ${escapeHtml(competition.deployCount)}</span>
              <span class="chip chip-observe">Observe rows: ${escapeHtml(competition.observeCount)}</span>
              <span class="chip chip-reference">Teams: ${escapeHtml(competition.teamCount)}</span>
            </div>
            <p class="section-copy">This is the competition-level read: how much of the current slate is actionable, watch-only, or context-driven.</p>
            ${renderEntitySurfaceTiles([
              { label: "Tracked rows", value: competitionTrackedRows, meta: "Deploy + observe + context + monitor", tone: "reference" },
              { label: "Lead market", value: competitionMarketMix[0]?.label || "—", meta: competitionMarketMix[0] ? `${competitionMarketMix[0].value} visible rows` : "No market data", tone: "deploy" },
              { label: "Deploy share", value: `${Math.round((competitionClassMix.DEPLOY / Math.max(1, competition.rows.length)) * 100)}%`, meta: "Of current-window competition rows", tone: "observe" },
            ])}
          </article>
          <article class="panel">
            <h3>Competition context in view</h3>
            <ul class="feature-list compact-list">
              <li>Competition pages keep the slate shape visible.</li>
              <li>Fixture pages carry the final decision framing and caution language.</li>
              <li>Tables, results, and current-window deploy posture belong here.</li>
            </ul>
          </article>
        </div>
        <div class="section-head">
          <div>
            <h2>Competition intelligence stream</h2>
            <p class="section-copy">Current public fixture cards from this competition, kept together as a league-first desk.</p>
          </div>
        </div>
        ${renderEntityFixtureSection(competition.rows, "No current public fixture cards are visible for this competition yet.")}
      </section>
    `;
    const tabContent = {
      overview: overviewContent,
      fixtures: fixturesContent,
      results: resultsContent,
      table: tableContent,
      context: contextContent,
    };
    return `
      <section class="hero">
        <article class="hero-main entity-hero entity-hero-competition">
          <div class="entity-directory-head">
            ${badgeMarkup(competition.logo, competition.name, "entity-mark entity-mark-lg entity-mark-competition")}
            <div>
              <p class="hero-kicker">Competition desk</p>
              <h1>${escapeHtml(competition.name)}</h1>
            </div>
          </div>
          <p>Competition pages own standings, slate-level context, and grouped fixtures. Fixture pages still own the final deploy read.</p>
          <div class="pill-row">
            <span class="stat-chip">Fixtures ${escapeHtml(competition.rows.length)}</span>
            <span class="stat-chip">Teams ${escapeHtml(competition.teamCount)}</span>
            <span class="stat-chip">Deploy ${escapeHtml(competition.deployCount)}</span>
          </div>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Current window</span>
            <span class="metric-value">${escapeHtml(competition.rows.length)}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Completed</span>
            <span class="metric-value">${escapeHtml(competition.fixtures.results.length)}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Upcoming</span>
            <span class="metric-value">${escapeHtml(competition.fixtures.upcoming.length)}</span>
          </div>
        </aside>
      </section>
      ${renderEntitySubnav(tabs, "Competition sections")}
      ${tabContent[selectedCompetitionTab] || overviewContent}
    `;
  };

  const competitionsView = () => {
    const competition = selectedCompetition ? collectCompetitionEntity(selectedCompetition) : null;
    if (selectedCompetition && competition) {
      return competitionEntityView(competition);
    }
    if (selectedCompetition && !competition) {
      return `
        <section class="section">
          <div class="empty-state">
            <strong>Competition not found in this window.</strong>
            <p>The selected competition is not part of the current published fixture-intelligence window yet.</p>
            <a class="button" href="./competitions.html">Back to competitions</a>
          </div>
        </section>
      `;
    }
    return competitionDirectoryView();
  };

  const teamDirectoryView = () => {
    const teamMap = new Map();
    orderedFixtureRows().forEach((row) => {
      [
        { name: row.home_team, logo: row.home_team_logo_url, fixture: row },
        { name: row.away_team, logo: row.away_team_logo_url, fixture: row },
      ].forEach((entry) => {
        if (!entry.name) {
          return;
        }
        if (!teamMap.has(entry.name)) {
          teamMap.set(entry.name, { name: entry.name, logo: entry.logo, rows: [] });
        }
        teamMap.get(entry.name).rows.push(entry.fixture);
      });
    });
    const rows = Array.from(teamMap.values())
      .sort((left, right) => right.rows.length - left.rows.length || left.name.localeCompare(right.name))
      .slice(0, 16);
    return `
      <section class="hero">
        <article class="hero-main entity-hero">
          <p class="hero-kicker">Teams</p>
          <h1>Team-led entry into the board.</h1>
          <p>Team pages own the current window team story: fixtures, results, current form, and where the model is seeing signal around that side.</p>
          <div class="pill-row">
            <span class="stat-chip">Fixtures</span>
            <span class="stat-chip">Results</span>
            <span class="stat-chip">Form and intelligence</span>
          </div>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Teams in window</span>
            <span class="metric-value">${escapeHtml(rows.length)}</span>
          </div>
        </aside>
      </section>
      <section class="section">
        <div class="card-grid">
          ${rows
            .map((team) =>
              renderDirectoryCard({
                title: team.name,
                badgeUrl: team.logo,
                badgeName: team.name,
                href: teamPageHref(team.name),
                summary: "Current-window team desk",
                metaLines: [
                  `Window mentions: ${team.rows.length}`,
                  `Competitions: ${new Set(team.rows.map((row) => row.league).filter(Boolean)).size}`,
                  `Next visible fixture: ${team.rows[0] ? `${team.rows[0].home_team} vs ${team.rows[0].away_team}` : "—"}`,
                ],
                ctaLabel: "Open team desk",
              })
            )
            .join("")}
        </div>
      </section>
    `;
  };

  const teamEntityView = (team) => {
    const teamIntelligence = state.selectedTeamIntelligence || null;
    const squadIntelligence = state.selectedTeamSquadIntelligence || null;
    const lineupSnapshot = state.selectedTeamLineupSnapshot || null;
    const teamRecentRows = team.rows.slice(0, 6);
    const teamClassMix = collectPublishClassMix(teamRecentRows);
    const teamMarketMix = collectMarketFamilyMix(teamRecentRows);
    const trackedTeamRows = teamClassMix.DEPLOY + teamClassMix.OBSERVE + teamClassMix.CONTEXT + teamClassMix.MONITOR;
    const nextFixtureRows = team.fixtures.upcoming.slice(0, 1);
    const latestResultRows = team.fixtures.results.slice(0, 1);
    const hasUpcomingFixture = nextFixtureRows.length > 0;
    const hasLatestResult = latestResultRows.length > 0;
    const tabs = [
      ["Overview", teamPageHref(team.name, "overview"), selectedTeamTab === "overview"],
      ["Fixtures", teamPageHref(team.name, "fixtures"), selectedTeamTab === "fixtures"],
      ["Results", teamPageHref(team.name, "results"), selectedTeamTab === "results"],
      ["Form", teamPageHref(team.name, "form"), selectedTeamTab === "form"],
      ["News", teamPageHref(team.name, "news"), selectedTeamTab === "news"],
      ["Intelligence", teamPageHref(team.name, "intelligence"), selectedTeamTab === "intelligence"],
    ];
    const intelligenceHeroCopy =
      teamIntelligence?.summary?.profile ||
      "Team pages own the current team story: fixtures, results, recent form, and grouped intelligence. Match verdicts stay inside fixture pages.";
    const intelligenceOverviewSection = teamIntelligence
      ? `
        <section class="section">
          <div class="split">
            <article class="panel">
              <h3>Odds Genius team intelligence</h3>
              <p class="section-copy">Publish-safe team intelligence derived from league-relative team strength, xG shape, scoring pressure, defensive resistance, and home/away balance.</p>
              ${renderEntitySurfaceTiles([
                { label: "OG Power", value: `${teamIntelligence.ratings?.og_power_rating ?? "—"}%`, meta: teamIntelligence.rating_bands?.og_power_rating || "", tone: scoreTone(teamIntelligence.ratings?.og_power_rating) },
                { label: "Attack Flow", value: `${teamIntelligence.ratings?.attack_flow_rating ?? "—"}%`, meta: teamIntelligence.rating_bands?.attack_flow_rating || "", tone: scoreTone(teamIntelligence.ratings?.attack_flow_rating) },
                { label: "Defensive Lock", value: `${teamIntelligence.ratings?.defensive_lock_rating ?? "—"}%`, meta: teamIntelligence.rating_bands?.defensive_lock_rating || "", tone: scoreTone(teamIntelligence.ratings?.defensive_lock_rating) },
                { label: "Chaos Rating", value: `${teamIntelligence.ratings?.chaos_rating ?? "—"}%`, meta: teamIntelligence.rating_bands?.chaos_rating || "", tone: scoreTone(100 - Number(teamIntelligence.ratings?.chaos_rating || 0)) },
              ])}
              <p class="section-copy"><strong>Primary strengths:</strong> ${escapeHtml((teamIntelligence.summary?.primary_strengths || []).join(", ") || "Mixed team profile")}</p>
              <p class="section-copy"><strong>Main caution:</strong> ${escapeHtml(teamIntelligence.summary?.main_caution || "No additional caution published yet.")}</p>
            </article>
            <article class="panel">
              <h3>Market intelligence</h3>
              ${renderMarketTendencyList(teamIntelligence.market_tendencies)}
              <div class="pill-row">
                ${(teamIntelligence.profile_tags || []).map((tag) => `<span class="chip chip-reference">${escapeHtml(tag)}</span>`).join("")}
              </div>
            </article>
          </div>
        </section>
      `
      : "";
    const timingProfileSection = teamIntelligence
      ? `
        <section class="section">
          <div class="split">
            <article class="panel">
              <h3>Timing profile</h3>
              <p class="section-copy">This is the publish-safe timing layer for how the team tends to control or destabilize matches.</p>
              ${renderScoreBreakdown(
                [
                  { label: "Early Threat", value: teamIntelligence.timing_profile?.early_threat, band: teamIntelligence.rating_bands?.first_strike_rating },
                  { label: "Half-Time Control", value: teamIntelligence.timing_profile?.half_time_control, band: teamIntelligence.rating_bands?.control_rating },
                  { label: "Late Surge", value: teamIntelligence.timing_profile?.late_surge, band: teamIntelligence.rating_bands?.over25_heat_rating },
                  { label: "Late Fragility", value: teamIntelligence.timing_profile?.late_fragility, band: teamIntelligence.rating_bands?.chaos_rating },
                ],
                "No timing profile has been published for this team yet."
              )}
            </article>
            ${renderProfileSurface("Home profile", teamIntelligence.home_profile)}
          </div>
          <div class="split">
            ${renderProfileSurface("Away profile", teamIntelligence.away_profile)}
            ${renderIntelligenceHeadline(
              teamIntelligence.summary?.headline || `OG Power Rating: ${teamIntelligence.ratings?.og_power_rating ?? "—"}%`,
              teamIntelligence.summary?.profile || "No summary profile has been published for this team yet.",
              scoreTone(teamIntelligence.ratings?.og_power_rating)
            )}
          </div>
        </section>
      `
      : "";
    const overviewContent = `
      ${intelligenceOverviewSection}
      ${renderTeamOverviewDrivers(squadIntelligence)}
      ${renderTeamLineupSnapshot(lineupSnapshot)}
      <section class="section">
        <div class="split">
          <article class="panel">
            <h3>Team overview</h3>
            <ul class="feature-list compact-list">
              <li>Fixtures in current window: ${escapeHtml(team.rows.length)}</li>
              <li>Deploy rows: ${escapeHtml(team.deployCount)}</li>
              <li>Observe rows: ${escapeHtml(team.observeCount)}</li>
              <li>Context / monitor rows: ${escapeHtml(team.contextCount)}</li>
            </ul>
          </article>
          <article class="panel">
            <h3>Competitions in view</h3>
            <div class="pill-row">
              ${team.relatedCompetitions
                .map(
                  (league) =>
                    `<a class="chip chip-reference" href="${competitionPageHref(league)}">${escapeHtml(league)}</a>`
                )
                .join("")}
            </div>
            <p class="section-copy">Team pages keep the broader team story visible. Final market verdicts and discipline language stay on the fixture page itself.</p>
          </article>
        </div>
      </section>
      <section class="section">
        <div class="split">
          <article class="panel">
            <h3>Recent output mix</h3>
            <p class="section-copy">This is the current-window posture around this team: how much of the latest visible slice is deployable, watch-first, or softer context.</p>
            ${renderEntitySurfaceTiles([
              { label: "Recent rows", value: teamRecentRows.length, meta: "Latest visible team-linked rows", tone: "reference" },
              { label: "Deploy share", value: `${Math.round((teamClassMix.DEPLOY / Math.max(1, teamRecentRows.length)) * 100)}%`, meta: `${teamClassMix.DEPLOY} deploy rows`, tone: "deploy" },
              { label: "Watch share", value: `${Math.round((teamClassMix.OBSERVE / Math.max(1, teamRecentRows.length)) * 100)}%`, meta: `${teamClassMix.OBSERVE} observe rows`, tone: "observe" },
            ])}
          </article>
          <article class="panel">
            <h3>Market-family mix</h3>
            <p class="section-copy">This shows which market families are appearing most often in the team’s latest visible output layer.</p>
            ${renderEntityBreakdown(
              teamMarketMix.map((item) => ({
                ...item,
                tone: item.label === "BTTS" ? "deploy" : item.label === "OU25" ? "observe" : "reference",
              })),
              teamRecentRows.length,
              "No recent market-family mix is visible for this team yet."
            )}
          </article>
        </div>
      </section>
      ${
        teamIntelligence
          ? `<section class="section">
              <div class="split">
                <article class="panel">
                  <h3>Published ratings stack</h3>
                  ${renderScoreBreakdown(
                    [
                      { label: TEAM_RATING_LABELS.og_power_rating, value: teamIntelligence.ratings?.og_power_rating, band: teamIntelligence.rating_bands?.og_power_rating },
                      { label: TEAM_RATING_LABELS.attack_flow_rating, value: teamIntelligence.ratings?.attack_flow_rating, band: teamIntelligence.rating_bands?.attack_flow_rating },
                      { label: TEAM_RATING_LABELS.defensive_lock_rating, value: teamIntelligence.ratings?.defensive_lock_rating, band: teamIntelligence.rating_bands?.defensive_lock_rating },
                      { label: TEAM_RATING_LABELS.goal_heat_rating, value: teamIntelligence.ratings?.goal_heat_rating, band: teamIntelligence.rating_bands?.goal_heat_rating },
                      { label: TEAM_RATING_LABELS.btts_pressure_rating, value: teamIntelligence.ratings?.btts_pressure_rating, band: teamIntelligence.rating_bands?.btts_pressure_rating },
                      { label: TEAM_RATING_LABELS.over25_heat_rating, value: teamIntelligence.ratings?.over25_heat_rating, band: teamIntelligence.rating_bands?.over25_heat_rating },
                      { label: TEAM_RATING_LABELS.first_strike_rating, value: teamIntelligence.ratings?.first_strike_rating, band: teamIntelligence.rating_bands?.first_strike_rating },
                      { label: TEAM_RATING_LABELS.corner_pressure_rating, value: teamIntelligence.ratings?.corner_pressure_rating, band: teamIntelligence.rating_bands?.corner_pressure_rating },
                    ],
                    "No publish-safe team ratings are available yet."
                  )}
                </article>
                <article class="panel">
                  <h3>Profile summary</h3>
                  <ul class="feature-list compact-list">
                    <li><strong>Headline</strong><br /><span class="muted">${escapeHtml(teamIntelligence.summary?.headline || "No headline yet")}</span></li>
                    <li><strong>Profile</strong><br /><span class="muted">${escapeHtml(teamIntelligence.summary?.profile || "No summary profile available.")}</span></li>
                    <li><strong>Main caution</strong><br /><span class="muted">${escapeHtml(teamIntelligence.summary?.main_caution || "No caution published yet.")}</span></li>
                  </ul>
                </article>
              </div>
            </section>`
          : ""
      }
      <section class="section">
        <div class="split split-top split-team-density">
          <article class="panel ${hasLatestResult ? "team-spotlight-panel" : "team-empty-note"}">
            <h3>Latest visible result</h3>
            <p class="section-copy">The latest settled current-window result helps anchor the desk before you move into deeper form and signal layers.</p>
            ${
              hasLatestResult
                ? renderEntityFixtureSection(latestResultRows, "No current-window settled result is visible for this team right now.")
                : `<div class="notice">No current-window settled result is visible for this team right now.</div>`
            }
          </article>
          <article class="panel team-empty-note">
            <h3>Next visible fixture</h3>
            <p class="section-copy">Keep the next team-linked match close to the top of the desk when one is in view. If the window is result-only, this panel stays compact instead of creating empty drag.</p>
            ${
              hasUpcomingFixture
                ? renderEntityFixtureSection(nextFixtureRows, "No upcoming current-window fixture is visible for this team right now.")
                : `<div class="notice">No upcoming current-window fixture is visible for this team right now.</div>`
            }
            ${
              !hasUpcomingFixture && hasLatestResult
                ? `<div class="pill-row"><span class="chip chip-reference">Result-led window</span><span class="chip chip-observe">Next match will appear here when the public window rolls forward</span></div>`
                : ""
            }
          </article>
        </div>
      </section>
      <section class="section">
        <div class="section-head">
          <div>
            <h2>Featured current-window fixtures</h2>
            <p class="section-copy">The strongest team-linked fixture cards stay close to the top of the desk.</p>
          </div>
        </div>
        ${renderEntityFixtureSection(team.rows.slice(0, 6), "No published fixtures are currently visible for this team.")}
      </section>
    `;
    const fixturesContent = `
      <section class="section">
        <div class="section-head">
          <div>
            <h2>Upcoming fixtures</h2>
            <p class="section-copy">Use the team desk to orient around what is ahead, then open the fixture page when you want the full intelligence read.</p>
          </div>
        </div>
        ${renderEntityFixtureSection(
          team.fixtures.upcoming,
          "No upcoming fixtures for this team are currently visible in the published window."
        )}
      </section>
    `;
    const resultsContent = `
      <section class="section">
        <div class="section-head">
          <div>
            <h2>Recent results in view</h2>
            <p class="section-copy">These are current-window team fixtures that already look settled or complete.</p>
          </div>
        </div>
        ${renderEntityFixtureSection(
          team.fixtures.results,
          "No completed fixtures for this team are currently visible in the published window."
        )}
      </section>
    `;
    const formContent = `
      <section class="section">
        <div class="split">
          <article
            class="panel"
            data-role="team-form-reference"
            data-team="${escapeHtml(team.name)}"
            data-team-id="${escapeHtml(team.apiTeamId)}"
          >
            <h3>Recent team rhythm</h3>
            <p class="section-copy">This is the live form layer for the selected team: recent finished results, current scoring rhythm, and near-term shape.</p>
            <div class="reference-loading">Loading recent team form…</div>
          </article>
          <article class="panel">
            <h3>Published window results</h3>
            <p class="section-copy">This keeps the live upstream recent-form reference separate from the results we already have in the current published window.</p>
            ${renderEntityFixtureSection(
              team.fixtures.results.slice(0, 4),
              "No current-window settled fixtures are visible for this team yet."
            )}
          </article>
        </div>
      </section>
      <section class="section">
        <article class="panel">
          <h3>Form ownership on team pages</h3>
          <ul class="feature-list compact-list">
            <li>Form stays team-level here rather than competition-level.</li>
            <li>The broader league rhythm still matters, but the team desk should foreground this side’s recent shape first.</li>
            <li>If a single match matters most, the fixture page still owns the direct deployment call.</li>
          </ul>
        </article>
      </section>
    `;
    const intelligenceContent = `
      <section class="section">
        <div class="split">
          <article class="panel">
            <h3>Team signal surface</h3>
            <p class="section-copy">This is the high-level mix across the latest team-linked rows before you drop into individual fixture cards.</p>
            ${renderEntitySurfaceTiles([
              { label: "Tracked rows", value: trackedTeamRows, meta: "Deploy + observe + context + monitor", tone: "reference" },
              { label: "Lead market", value: teamMarketMix[0]?.label || "—", meta: teamMarketMix[0] ? `${teamMarketMix[0].value} recent rows` : "No market data", tone: "deploy" },
              { label: "Context share", value: `${Math.round(((teamClassMix.CONTEXT + teamClassMix.MONITOR) / Math.max(1, teamRecentRows.length)) * 100)}%`, meta: `${teamClassMix.CONTEXT + teamClassMix.MONITOR} softer rows`, tone: "observe" },
            ])}
          </article>
          ${
            teamIntelligence
              ? `
                <article class="panel">
                  <h3>Published team ratings</h3>
                  ${renderScoreBreakdown(
                    [
                      { label: TEAM_RATING_LABELS.og_power_rating, value: teamIntelligence.ratings?.og_power_rating, band: teamIntelligence.rating_bands?.og_power_rating },
                      { label: TEAM_RATING_LABELS.attack_flow_rating, value: teamIntelligence.ratings?.attack_flow_rating, band: teamIntelligence.rating_bands?.attack_flow_rating },
                      { label: TEAM_RATING_LABELS.defensive_lock_rating, value: teamIntelligence.ratings?.defensive_lock_rating, band: teamIntelligence.rating_bands?.defensive_lock_rating },
                      { label: TEAM_RATING_LABELS.goal_heat_rating, value: teamIntelligence.ratings?.goal_heat_rating, band: teamIntelligence.rating_bands?.goal_heat_rating },
                      { label: TEAM_RATING_LABELS.btts_pressure_rating, value: teamIntelligence.ratings?.btts_pressure_rating, band: teamIntelligence.rating_bands?.btts_pressure_rating },
                      { label: TEAM_RATING_LABELS.over25_heat_rating, value: teamIntelligence.ratings?.over25_heat_rating, band: teamIntelligence.rating_bands?.over25_heat_rating },
                    ],
                    "No publish-safe team ratings are available yet."
                  )}
                </article>
              `
              : `
                <article class="panel">
                  <h3>Why this stays team-level</h3>
                  <ul class="feature-list compact-list">
                    <li>Team desks keep recent output mix and grouped posture visible.</li>
                    <li>Competition desks own the broader league distribution and standings layer.</li>
                    <li>Fixture pages still carry the final market verdict and discipline framing.</li>
                  </ul>
                </article>
              `
          }
        </div>
      </section>
      ${timingProfileSection}
      ${squadIntelligence ? `<section class="section">${renderSquadSnapshot(squadIntelligence)}</section>` : ""}
      ${renderTeamLineupSnapshot(lineupSnapshot)}
      ${renderTeamIntelligenceBuckets(team)}
    `;
    const tabContent = {
      overview: overviewContent,
      fixtures: fixturesContent,
      results: resultsContent,
      form: formContent,
      news: renderTeamNewsSection(team.name),
      intelligence: intelligenceContent,
    };
    return `
      <section class="hero">
        <article class="hero-main entity-hero">
          <div class="entity-directory-head">
            ${badgeMarkup(team.logo, team.name, "entity-mark entity-mark-lg")}
            <div>
              <p class="hero-kicker">Team desk</p>
              <h1>${escapeHtml(team.name)}</h1>
            </div>
          </div>
          <p>${escapeHtml(intelligenceHeroCopy)}</p>
          ${
            teamIntelligence
              ? `<div class="team-hero-intel-grid">
                  ${[
                    ["OG Power", `${teamIntelligence.ratings?.og_power_rating ?? "—"}%`, teamIntelligence.rating_bands?.og_power_rating || "Team strength"],
                    ["Attack Flow", `${teamIntelligence.ratings?.attack_flow_rating ?? "—"}%`, teamIntelligence.rating_bands?.attack_flow_rating || "Chance creation"],
                    ["Defensive Lock", `${teamIntelligence.ratings?.defensive_lock_rating ?? "—"}%`, teamIntelligence.rating_bands?.defensive_lock_rating || "Suppression"],
                    ["Squad Profiles", squadIntelligence ? String(squadIntelligence.players?.length || 0) : "Pending", squadIntelligence ? "Player layer active" : "Awaiting squad layer"],
                  ]
                    .map(
                      ([label, value, meta]) => `
                        <article class="team-hero-intel-card">
                          <span class="metric-label">${escapeHtml(label)}</span>
                          <strong>${escapeHtml(value)}</strong>
                          <p class="muted">${escapeHtml(meta)}</p>
                        </article>
                      `
                    )
                    .join("")}
                </div>`
              : ""
          }
          <div class="pill-row">
            ${
              teamIntelligence
                ? `<span class="stat-chip">${escapeHtml(teamIntelligence.summary?.headline || `OG Power Rating: ${teamIntelligence.ratings?.og_power_rating ?? "—"}%`)}</span>`
                : `<span class="stat-chip">Window fixtures ${escapeHtml(team.rows.length)}</span>`
            }
            <span class="stat-chip">Deploy ${escapeHtml(team.deployCount)}</span>
            <span class="stat-chip">Observe ${escapeHtml(team.observeCount)}</span>
            ${(teamIntelligence?.profile_tags || []).slice(0, 2).map((tag) => `<span class="stat-chip">${escapeHtml(tag)}</span>`).join("")}
          </div>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">${escapeHtml(teamIntelligence ? "OG power" : "Competitions")}</span>
            <span class="metric-value">${escapeHtml(teamIntelligence ? `${teamIntelligence.ratings?.og_power_rating ?? "—"}%` : team.relatedCompetitions.length)}</span>
          </div>
          <div class="metric">
            <span class="metric-label">${escapeHtml(teamIntelligence ? "Sample confidence" : "Completed")}</span>
            <span class="metric-value">${escapeHtml(teamIntelligence ? teamIntelligence.sample_confidence?.label || "—" : team.fixtures.results.length)}</span>
          </div>
          <div class="metric">
            <span class="metric-label">${escapeHtml(squadIntelligence ? "Squad profiles" : "Upcoming")}</span>
            <span class="metric-value">${escapeHtml(squadIntelligence ? squadIntelligence.players?.length || 0 : team.fixtures.upcoming.length)}</span>
          </div>
        </aside>
      </section>
      ${renderEntitySubnav(tabs, "Team sections")}
      ${tabContent[selectedTeamTab] || overviewContent}
    `;
  };

  const teamsView = () => {
    const team = selectedTeam ? collectTeamEntity(selectedTeam) : null;
    if (selectedTeam && team) {
      return teamEntityView(team);
    }
    if (selectedTeam && !team) {
      return `
        <section class="section">
          <div class="empty-state">
            <strong>Team not found in this window.</strong>
            <p>The selected team is not part of the current published fixture-intelligence window yet.</p>
            <a class="button" href="./teams.html">Back to teams</a>
          </div>
        </section>
      `;
    }
    return teamDirectoryView();
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
        ${state.publicPredictions.map((row, index) => predictionCard(row, true, index)).join("")}
      </div>
    </section>
  `;

  const loadingShell = (message) => `
    <div class="loading loading-shell">
      <div class="loading-copy">
        <span class="metric-label">Booting Odds Genius</span>
        <strong>${escapeHtml(message)}</strong>
      </div>
      <div class="loading-skeleton-grid" aria-hidden="true">
        <article class="loading-skeleton-card">
          <div class="loading-skeleton-pill"></div>
          <div class="loading-skeleton-line loading-skeleton-line-lg"></div>
          <div class="loading-skeleton-line loading-skeleton-line-md"></div>
          <div class="loading-skeleton-line loading-skeleton-line-sm"></div>
        </article>
        <article class="loading-skeleton-card">
          <div class="loading-skeleton-pill"></div>
          <div class="loading-skeleton-line loading-skeleton-line-md"></div>
          <div class="loading-skeleton-line loading-skeleton-line-sm"></div>
          <div class="loading-skeleton-line loading-skeleton-line-xs"></div>
        </article>
        <article class="loading-skeleton-card">
          <div class="loading-skeleton-pill"></div>
          <div class="loading-skeleton-line loading-skeleton-line-md"></div>
          <div class="loading-skeleton-line loading-skeleton-line-sm"></div>
          <div class="loading-skeleton-line loading-skeleton-line-sm"></div>
        </article>
      </div>
    </div>
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

  const resultOutcomeLabel = (status) => {
    const normalized = String(status || "").toLowerCase();
    if (normalized === "won") return "Won";
    if (normalized === "lost") return "Lost";
    if (normalized === "cashed") return "Cashed";
    if (normalized === "void") return "Void";
    return "Pending";
  };

  const resultOutcomeTone = (status) => {
    const normalized = String(status || "").toLowerCase();
    if (normalized === "won") return "won";
    if (normalized === "lost") return "lost";
    if (normalized === "cashed") return "cashed";
    if (normalized === "void") return "void";
    return "pending";
  };

  const resultScoreLabel = (item = {}) => {
    if (item.score) return item.score;
    const home = item.final_home_score;
    const away = item.final_away_score;
    if (home !== null && home !== undefined && away !== null && away !== undefined) {
      return `${home}-${away}`;
    }
    return "Score pending";
  };

  const feedPercent = (value) => (value == null || Number.isNaN(Number(value)) ? "Pending" : compactPercent(Number(value)));

  const resultSummaryCopy = (summary) => {
    if (!summary) return "No settled rows";
    const wins = Number(summary.wins || 0);
    const settled = Number(summary.settled || 0);
    return `${wins}/${settled} · ${feedPercent(summary.hit_rate)}`;
  };

  const renderResultFeedItems = (items = [], limit = 12) => {
    const visible = items.slice(0, limit);
    if (!visible.length) return `<div class="notice">No graded rows published for this window yet.</div>`;
    return `
      <div class="public-result-list">
        ${visible
          .map((item) => {
            const tone = resultOutcomeTone(item.result_status);
            const isObserve = String(item.publish_class || "").toUpperCase() === "OBSERVE" || String(item.tier || "").toUpperCase() === "OBSERVE";
            const signal = item.site_signal_alignment
              ? `${item.site_signal_alignment}${item.site_signal_state ? ` · ${item.site_signal_state}` : ""}`
              : item.decision_state || "";
            return `
              <article class="public-result-row public-result-row-${escapeHtml(tone)} ${isObserve ? "public-result-row-observe" : ""}">
                <div class="public-result-main">
                  <span class="result-status-pill result-status-pill-${escapeHtml(tone)}">${escapeHtml(resultOutcomeLabel(item.result_status))}</span>
                  <div>
                    <strong>${escapeHtml(item.home_team)} vs ${escapeHtml(item.away_team)}</strong>
                    <p class="muted">${escapeHtml(item.league || "League")} · ${escapeHtml(resultScoreLabel(item))}</p>
                  </div>
                </div>
                <div class="public-result-pick">
                  <span class="metric-label">${escapeHtml(`${item.market || "Market"} · ${item.tier || "Tier"}`)}</span>
                  <strong>${escapeHtml(item.pick || "Pick")}</strong>
                  <span class="muted">Actual ${escapeHtml(item.actual || "pending")}${item.bookie_od ? ` · Odds ${escapeHtml(item.bookie_od)}` : ""}</span>
                </div>
                <div class="public-result-signal">
                  <span class="metric-label">${isObserve ? "Watchlist" : "Model + intelligence"}</span>
                  <span>${escapeHtml(signal || (item.profit_units != null ? `${Number(item.profit_units) >= 0 ? "+" : ""}${item.profit_units}u` : isObserve ? "Observe only" : "Deploy row"))}</span>
                </div>
              </article>
            `;
          })
          .join("")}
      </div>
    `;
  };

  const liveResultsView = (feed, weekly) => {
    const windows = Array.isArray(feed.windows) ? feed.windows : [];
    const summary = feed.summary || {};
    const primaryWindow = windows[0] || {};
    const secondaryWindow = windows[1] || {};
    const windowPanels = windows
      .map((window) => {
        const deploy = window.summary?.deploy || {};
        const observe = window.summary?.observe || {};
        return `
          <article class="panel result-window-panel">
            <span class="muted">${escapeHtml(window.period_start || "")} → ${escapeHtml(window.period_end || "")}</span>
            <h3>${escapeHtml(window.title || "Results window")}</h3>
            <p>${escapeHtml(window.subtitle || "")}</p>
            <div class="result-window-metrics">
              <span><strong>${escapeHtml(resultSummaryCopy(deploy))}</strong><small>Deploy</small></span>
              <span><strong>${escapeHtml(resultSummaryCopy(observe))}</strong><small>Observe</small></span>
            </div>
          </article>
        `;
      })
      .join("");

    const marketRollupMap = (primaryWindow.by_market || []).reduce((acc, item) => {
      acc[marketLabelCanonical(item.market)] = item;
      return acc;
    }, {});
    const marketCards = ["FTR", "BTTS", "OU25", "TG1.5"]
      .map((market) => {
        const item = marketRollupMap[market] || { market, hit_rate: null, wins: 0, settled: 0, rows: 0 };
        return `
          <article class="panel market-proof-card market-proof-card--${resultsStatusTone(item.hit_rate)}">
            <span class="muted">${escapeHtml(market)}</span>
            <strong>${escapeHtml(feedPercent(item.hit_rate))}</strong>
            <span>${escapeHtml(`${item.wins || 0}/${item.settled || 0} settled`)}</span>
          </article>
        `;
      })
      .join("");

    return `
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Live Proof Feed</p>
          <h1>Wins, losses, and watchlist evidence.</h1>
          <p>
            Public-safe results from scored deploy outputs and settled provider results. Deploy rows are the paid/actionable proof layer; OBSERVE rows stay separate as research evidence.
          </p>
          <div class="pill-row">
            <span class="stat-chip">Deploy ${escapeHtml(`${summary.deploy_wins || 0}/${summary.deploy_settled || 0} · ${feedPercent(summary.deploy_hit_rate)}`)}</span>
            <span class="stat-chip">Generated ${escapeHtml(String(feed.generated_at || "").slice(0, 10))}</span>
          </div>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Combined deploy hit rate</span>
            <span class="metric-value">${escapeHtml(feedPercent(summary.deploy_hit_rate))}</span>
            <span class="muted">${escapeHtml(`${summary.deploy_wins || 0}/${summary.deploy_settled || 0} settled deploy rows`)}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Watchlist hit rate</span>
            <span class="metric-value">${escapeHtml(feedPercent(summary.observe_hit_rate))}</span>
            <span class="muted">${escapeHtml(`${summary.observe_wins || 0}/${summary.observe_settled || 0} settled observe rows`)}</span>
          </div>
        </aside>
      </section>

      <section class="section">
        <div class="results-highlight results-highlight--four">
          ${statPanel("MLS live deploy", resultSummaryCopy(primaryWindow.summary?.deploy), "2026-05-14 provider final")}
          ${statPanel("MLS OU25 standard", "8/8", "Live night deploy subset")}
          ${statPanel("Weekend deploy", resultSummaryCopy(secondaryWindow.summary?.deploy), "2026-05-09 → 2026-05-11")}
          ${statPanel("Weekend EV+", resultSummaryCopy(secondaryWindow.summary?.ev_positive), "Settled positive-EV rows")}
        </div>
      </section>

      <section class="section">
        <div class="result-window-grid">${windowPanels}</div>
      </section>

      <section class="section">
        <div class="section-head">
          <div>
            <h2>Latest live audit</h2>
            <p class="section-copy">MLS live board, including the one BTTS miss where intelligence flagged conflict pre-result.</p>
          </div>
        </div>
        ${renderResultFeedItems(primaryWindow.featured_results || primaryWindow.items || [], 16)}
      </section>

      <section class="section">
        <div class="section-head">
          <div>
            <h2>Live night by market</h2>
            <p class="section-copy">Deployable rows only. Observe rows are deliberately excluded from this market split.</p>
          </div>
        </div>
        <div class="stats-grid">${marketCards}</div>
      </section>

      <section class="section">
        <div class="section-head">
          <div>
            <h2>Weekend proof sample</h2>
            <p class="section-copy">Settled public rows from the most recent broad weekend scoring audit.</p>
          </div>
        </div>
        ${renderResultFeedItems(secondaryWindow.featured_results || secondaryWindow.items || [], 12)}
      </section>

      ${
        weekly
          ? `<section class="section">
              <div class="notice">
                Historical archive remains available underneath this feed: ${escapeHtml(weekly.wins || 0)} wins, ${escapeHtml(weekly.losses || 0)} losses from the previous published weekly proof window.
              </div>
            </section>`
          : ""
      }
    `;
  };

  const resultsView = () => {
    const weekly = state.weeklyResults;
    const archive = state.resultsArchive;
    const liveFeed = state.liveResultsFeed;
    if (!weekly && liveFeed?.windows?.length) {
      return liveResultsView(liveFeed, weekly);
    }
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
            Run python3 scripts/publish_results_proof.py after settled outcomes are available to publish weekly proof.
          </div>
        </section>
      `;
    }

    const weeklyMarketRollups = (weekly.by_market || []).reduce((acc, item) => {
      acc[marketLabelCanonical(item.market)] = item;
      return acc;
    }, {});
    const marketCards = ["FTR", "BTTS", "OU25", "TG1.5"]
      .map((market) => {
        const item = weeklyMarketRollups[market] || {
          market,
          hit_rate: null,
          settled_picks: 0,
          total_picks: 0,
        };
        return `
          <article class="panel market-proof-card market-proof-card--${resultsStatusTone(item.hit_rate)}">
            <span class="muted">${escapeHtml(market)}</span>
            <strong>${escapeHtml(item.hit_rate == null ? "Pending" : `${Math.round(item.hit_rate * 100)}%`)}</strong>
            <span>${escapeHtml(`${item.settled_picks || 0}/${item.total_picks || 0} settled`)}</span>
          </article>
        `;
      })
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

    const recentRows = [...(weekly.items || [])]
      .sort((left, right) => String(right.kickoff_time || "").localeCompare(String(left.kickoff_time || "")))
      .slice(0, 16);

    return `
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Weekly Proof</p>
          <h1>Settled board proof.</h1>
          <p>
            Public-safe weekly proof generated from published picks and final provider results. Every row is
            automatically graded as won, lost, void, or pending.
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
        <div class="section-head">
          <div>
            <h2>Recent graded picks</h2>
            <p class="section-copy">The public proof ledger. Winning selections carry a mint border, losing selections carry an orange/red border, pending and void rows stay neutral.</p>
          </div>
        </div>
        ${renderResultFeedItems(recentRows, 16)}
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

      ${
        archive
          ? `<section class="section">
              <div class="section-head">
                <div>
                  <h2>Archive context</h2>
                  <p class="section-copy">Cumulative idempotent archive across previous published proof windows.</p>
                </div>
              </div>
              <div class="results-highlight results-highlight--four">
                ${statPanel("Archive picks", archive.total_picks || 0, `${archive.period_start || ""} → ${archive.period_end || ""}`)}
                ${statPanel("Archive settled", archive.settled_picks || 0, `${archive.pending_picks || 0} pending`)}
                ${statPanel("Archive hit rate", archive.overall_hit_rate == null ? "Pending" : compactPercent(archive.overall_hit_rate), `${archive.wins || 0}/${archive.settled_picks || 0} settled`)}
                ${statPanel("Archive ROI", archive.overall_roi == null ? "Pending" : compactPercent(archive.overall_roi), `${archive.overall_profit_units || 0} units`)}
              </div>
            </section>`
          : ""
      }

      ${
        liveFeed?.windows?.length
          ? `<section class="section">
              <div class="notice">
                Older live proof feed is still available as supporting context, but the primary public proof source is now the automated weekly/archive settlement JSON.
              </div>
            </section>`
          : ""
      }

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
            Start free. Secure OG Founder Early Access before the Premium, Pro, and Pro+ ladder expands.
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
          <span class="metric-label">Launch window</span>
          <span class="metric-value">World Cup + pre-season</span>
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
          <p class="pricing-subcopy">Discounted early access to the core premium fixture intelligence while the launch system hardens.</p>
          <ul class="feature-list">
            <li>Core premium fixture intelligence.</li>
            <li>Model cards and market posture.</li>
            <li>Results archive and proof context.</li>
            <li>Founder-only discounted access while active.</li>
            <li>Protected Worker-backed access.</li>
            <li>Selected beta surfaces as they become safe to expose.</li>
          </ul>
          <div class="notice founder-guardrail">
            £20/month for life while active. Non-transferable. Founder pricing ends after the first 250 users.
          </div>
          <div class="cta-row">
            ${checkoutCta().replace("Unlock founding membership", "Secure founder access")}
          </div>
        </article>
        <article class="card pricing-card pricing-card-pro">
          <span class="pricing-tag">Premium</span>
          <div class="pricing-price">£49<span class="pricing-price-note">/month</span></div>
          <p class="pricing-subcopy">The standard paid intelligence layer once Founder closes.</p>
          <ul class="feature-list">
            <li>Model cards and fixture reads.</li>
            <li>Results archive.</li>
            <li>Market posture and pass/no-edge context.</li>
            <li>Premium route gating.</li>
          </ul>
        </article>
        <article class="card pricing-card pricing-card-pro">
          <span class="pricing-tag">Pro</span>
          <div class="pricing-price">£99<span class="pricing-price-note">/month</span></div>
          <p class="pricing-subcopy">Expanded football intelligence for users who want deeper context.</p>
          <ul class="feature-list">
            <li>Player-event beta cards.</li>
            <li>Deeper team and player intelligence.</li>
            <li>Goal-combo and TG1.5 context.</li>
            <li>Expert expandable panels.</li>
          </ul>
        </article>
        <article class="card pricing-card pricing-card-pro-plus">
          <span class="pricing-tag">Pro+</span>
          <div class="pricing-price">£500<span class="pricing-price-note">/month</span></div>
          <p class="pricing-subcopy">Audit-style workflow layer for serious operators.</p>
          <ul class="feature-list">
            <li>Advanced filters and audit dashboards.</li>
            <li>Downloadable intelligence.</li>
            <li>Operational freshness and coverage views.</li>
            <li>B2B/API path later.</li>
          </ul>
        </article>
      </div>
      <div class="pricing-band">
        <article class="pricing-band-card">
          <span class="metric-label">Founder advantage</span>
          <strong>Grandfathered pricing while active</strong>
        </article>
        <article class="pricing-band-card">
          <span class="metric-label">Expansion path</span>
          <strong>World Cup + pre-season edition of the core premium product</strong>
        </article>
      </div>
      <div class="section-head pricing-matrix-head">
        <div>
          <h2>Tier visibility contract</h2>
          <p class="section-copy">What each access level is intended to see at launch.</p>
        </div>
      </div>
      <div class="table-shell pricing-matrix-shell">
        <table class="pricing-matrix">
          <thead>
            <tr>
              <th>Feature detail</th>
              <th>Free tier</th>
              <th>OG Founder</th>
              <th>Premium</th>
              <th>Pro</th>
              <th>Pro+</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Board scope</td>
              <td>Limited public board</td>
              <td>Core premium fixture intelligence</td>
              <td>Core premium fixture intelligence</td>
              <td>Full premium plus player-event context</td>
              <td>Full premium plus audit views</td>
            </tr>
            <tr>
              <td>Proof layer</td>
              <td>Public results and methodology</td>
              <td>Results archive and market split</td>
              <td>Results archive and market split</td>
              <td>Results archive and deeper filters</td>
              <td>Audit-style result dashboards</td>
            </tr>
            <tr>
              <td>Fixture intelligence</td>
              <td>Public-safe summary only</td>
              <td>Market cards, fixture reads, posture</td>
              <td>Market cards, fixture reads, posture</td>
              <td>Team/player depth and combos</td>
              <td>Advanced filters and downloads</td>
            </tr>
            <tr>
              <td>Player events</td>
              <td>Not included</td>
              <td>Selected beta previews only</td>
              <td>Selected beta previews only</td>
              <td>Player-event intelligence cards</td>
              <td>Player-event filters and exports</td>
            </tr>
            <tr>
              <td>Account state</td>
              <td>Static public access</td>
              <td>Protected access through live Worker entitlement</td>
              <td>Protected access through live Worker entitlement</td>
              <td>Protected access through live Worker entitlement</td>
              <td>Protected access plus billing/admin workflow</td>
            </tr>
            <tr>
              <td>Future path</td>
              <td>None</td>
              <td>Discounted while active</td>
              <td>Standard paid access</td>
              <td>Advanced football intelligence</td>
              <td>B2B/API and licensing later</td>
            </tr>
          </tbody>
        </table>
      </div>
      <section class="pricing-visual-note">
        <article class="pricing-visual-card">
          <span class="metric-label">Founder access</span>
          <h2>Founder pricing now. Premium ladder next.</h2>
          <p class="section-copy">
            OG Founder sits above the free board and maps to the core premium fixture intelligence contract. Pro, Pro+, Syndicate, and B2B/API can expand later without muddying launch access.
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

  const methodologyView = () => {
    const summary = state.summary || {};
    const weekly = state.weeklyResults || {};
    const settled = Number(weekly.settled_picks || 0);
    const pending = Number(weekly.pending_picks || 0);
    const weeklyHitRate = weekly.overall_hit_rate == null ? "Pending" : compactPercent(weekly.overall_hit_rate);
    const generatedAt = weekly.generated_at || summary.generated_at || "";
    return `
      <section class="section split">
        <article class="hero-main">
          <p class="hero-kicker">Methodology</p>
          <h1>Walk-forward validation and live proof are separate.</h1>
          <p>
            Odds Genius does not re-run model logic in the browser. The website displays approved exports from the
            deployment and publishing pipeline, then settles published picks against final results.
          </p>
          <p class="section-copy">
            Walk-forward validation explains how the system behaved across historical rolling windows. Live proof shows what was actually published and then graded in public.
          </p>
          <div class="cta-row">
            <a class="button" href="./results.html">Open live proof</a>
            <a class="ghost-button" href="./pricing.html">Founder access</a>
          </div>
        </article>
        <aside class="hero-side">
          <div class="metric">
            <span class="metric-label">Weekly live hit rate</span>
            <span class="metric-value">${escapeHtml(weeklyHitRate)}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Settled / pending</span>
            <span class="metric-value">${escapeHtml(`${settled} / ${pending}`)}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Last proof update</span>
            <span class="metric-value">${escapeHtml(formatDateTime(generatedAt) || "Pending")}</span>
          </div>
        </aside>
      </section>

      <section class="section split">
        <article class="panel featured-proof-panel">
          <span class="metric-label">Walk-forward validation</span>
          <h3>Historical replay before launch claims.</h3>
          <p class="muted">
            Walk-forward tests replay the system through rolling historical windows so each window is judged on data that would have been available at that point. This is research proof, not a promise that the next fixture will behave the same way.
          </p>
          <div class="proof-strip">
            ${proofTile("Rolling windows", "139", "Historical validation estate")}
            ${proofTile("Competitions", "28", "Research coverage")}
          </div>
        </article>
        <article class="panel">
          <span class="metric-label">Live public proof</span>
          <h3>Published first, graded later.</h3>
          <p class="muted">
            Live proof only counts rows that were published to the site before settlement. Each pick becomes won, lost, void, or pending from the settlement pipeline.
          </p>
          <ul class="method-list">
            <li>Pending picks remain visible until final results are available.</li>
            <li>Void picks stay separate from wins and losses.</li>
            <li>FTR, BTTS, OU25, and TG1.5 are reported separately.</li>
          </ul>
        </article>
      </section>

      <section class="section">
        <div class="section-head">
          <div>
            <h2>What the site is allowed to show</h2>
            <p class="section-copy">The public product is an export surface. It shows approved intelligence, proof, and premium context without exposing private model files or pipeline internals.</p>
          </div>
        </div>
        <div class="card-grid">
          <article class="panel">
            <h3>Pipeline boundary</h3>
            <p class="muted">Ingest, enrichment, routing, and settlement stay in controlled scripts. The browser reads website-safe JSON only.</p>
          </article>
          <article class="panel">
            <h3>Decision stack</h3>
            <p class="muted">Signals can deploy, observe, monitor, or pass. The system is not designed to force a pick on every fixture.</p>
          </article>
          <article class="panel">
            <h3>Commercial proof</h3>
            <p class="muted">Founder access is sold on public proof, premium fixture context, and disciplined market posture, not guaranteed outcomes.</p>
          </article>
        </div>
      </section>

      <section class="section split">
        <article class="panel">
          <span class="metric-label">Market discipline</span>
          <h3>No blended proof headline.</h3>
          <p class="muted">
            Football markets behave differently, so the public stats keep match result, both-teams-to-score, match totals, and team-goal lines apart.
          </p>
          ${renderResultsMarketSplit(weekly)}
        </article>
        <article class="panel">
          <span class="metric-label">Schema boundary</span>
          <h3>Free and premium payloads are different.</h3>
          <div class="prediction-meta-grid dashboard-odds-grid">
            <div class="signal-cell signal-cell-model">
              <span class="signal-label">Public fields</span>
              <span class="signal-value">${escapeHtml(String((summary.public_fields || []).length || 0))}</span>
              <span class="muted">Free board and proof surfaces</span>
            </div>
            <div class="signal-cell signal-cell-model">
              <span class="signal-label">Premium fields</span>
              <span class="signal-value">${escapeHtml(String((summary.premium_fields || []).length || 0))}</span>
              <span class="muted">Worker-gated intelligence payload</span>
            </div>
          </div>
          <p class="footer-note">Historical performance and live settlement are informational. They are not financial advice and do not guarantee future results.</p>
        </article>
      </section>
    `;
  };

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
    const savedFollowCount =
      parsePreferenceList(notificationPreferences?.favourite_teams).length +
      parsePreferenceList(notificationPreferences?.favourite_leagues).length +
      parsePreferenceList(notificationPreferences?.favourite_markets).length +
      parsePreferenceList(notificationPreferences?.followed_fixtures).length;
    const alertCount = Array.isArray(state.runtime.accountAlerts) ? state.runtime.accountAlerts.length : 0;
    const deliveryPosture =
      notificationPreferences?.telegram_enabled && !notificationPreferences?.website_only_mode ? "Selective" : "Website-first";
    const telegramStatusLabel = telegramLinked ? "Linked" : telegramReady ? "Ready to link" : "Not linked";

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
            <section class="section section-tight">
              <nav class="page-subnav" aria-label="Account sections">
                <div class="page-subnav-scroll">
                  <a class="page-subnav-link is-active" href="#account-overview">Workspace</a>
                  <a class="page-subnav-link" href="#activity-desk">Activity</a>
                  <a class="page-subnav-link" href="#devices">Devices</a>
                  <a class="page-subnav-link" href="#preferences">Preferences</a>
                  <a class="page-subnav-link" href="#billing">Billing</a>
                  <a class="page-subnav-link" href="#help">Help</a>
                </div>
              </nav>
            </section>
            ${
              !onboardingCompleted
                ? `
                  <section class="section">
                    <article class="panel">
                      <h3>Calm setup still needs one final pass</h3>
                      <p class="muted">Your account is live, but the desk still gets better once your preset, follows, delivery posture, and decision-support settings are locked in together.</p>
                      <div class="stats-grid">
                        ${statPanel("Progress", `${onboardingSummary.completed}/${onboardingSummary.total}`, "Preset, follows, delivery, decision support, reset mode")}
                        ${statPanel("Preset", stylePresetLabel(userStylePreset), stylePresetSummary(userStylePreset))}
                        ${statPanel("Delivery", deliveryPosture, feedRoutingExplanation(userStylePreset))}
                        ${statPanel("Outcome", "Calmer account", "Less noise, clearer relevance, stronger pacing")}
                      </div>
                      <div class="cta-row">
                        <a class="button" href="./onboarding.html">Finish calm setup</a>
                        <a class="ghost-button" href="#preferences">Open preferences anyway</a>
                      </div>
                    </article>
                  </section>
                `
                : ""
            }
            <section class="section split" id="account-overview">
              <article class="panel">
                <h3>Member workspace</h3>
                <p class="muted">This is the calm top layer for your account: what posture you saved, how your desk will deliver, and what is actively shaping the product around you.</p>
                <div class="stats-grid">
                  ${statPanel("Membership", subscriptionStatus, entitled ? "Premium access is unlocked on this device." : "Membership is still settling.")}
                  ${statPanel("Preset", stylePresetLabel(userStylePreset), stylePresetSummary(userStylePreset))}
                  ${statPanel("Saved follows", String(savedFollowCount), followedSignalsConfigured ? "Teams, leagues, markets, and fixtures are shaping the desk." : "No follow environment saved yet.")}
                  ${statPanel("Alerts", String(alertCount), alertCount ? "Queued and delivered account alerts are live." : "No account alerts are stored yet.")}
                </div>
              </article>
              <article class="panel">
                <h3>Account and delivery</h3>
                <ul class="feature-list">
                  <li>Verified email: ${displayEmail ? escapeHtml(displayEmail) : "Signed in"}</li>
                  <li>Membership status: ${escapeHtml(subscriptionStatus)}</li>
                  <li>D1 profile state: ${accountState ? "Active" : state.runtime.accountStateError ? "Unavailable" : "Pending"}</li>
                  <li>Delivery posture: ${escapeHtml(deliveryPosture)}</li>
                  <li>Telegram: ${escapeHtml(telegramStatusLabel)}</li>
                </ul>
                ${
                  telegramLinked
                    ? `
                      <p class="muted">Telegram is linked as ${escapeHtml(formatTelegramIdentity(telegramLink) || "Linked account")} and can be used for stronger interruptions only.</p>
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
                          : `<p class="muted">Link Telegram when you want premium comms, elite deployment alerts, and future acca drops beyond the website shell.</p>`
                      }
                    `
                }
              </article>
            </section>
            <section class="section split" id="activity-desk">
              <article class="panel">
                <h3>Followed intelligence</h3>
                <p class="muted">A shorter preview of the fixtures currently matching your saved follows. Use the dashboard for the fuller grouped fixture workspace.</p>
                ${
                  !followedSignalsConfigured
                    ? `<div class="notice">Add followed teams, leagues, markets, or fixtures below to start shaping your personal intelligence board.</div>`
                    : followedIntelligence.length
                      ? `
                        <div class="card-grid intelligence-grid">
                          ${followedIntelligence
                            .slice(0, 4)
                            .map((entry) => intelligenceCard(entry, Boolean(notificationPreferences?.telegram_enabled)))
                            .join("")}
                        </div>
                      `
                      : `<div class="notice">Your follow settings are saved, but no current published fixtures matched this window yet. That will change as covered-fixture and context publishing expands.</div>`
                }
              </article>
              <article class="panel">
                <h3>My alerts</h3>
                <p class="muted">This is the transition layer from followed intelligence into Telegram. Lower-relevance market-only matches stay on the website instead of breaking attention.</p>
                ${renderNotice(state.runtime.alertsMessage, state.runtime.alertsMessage ? "success" : "default")}
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
            <section class="section split">
              <article class="panel" id="devices">
                <h3>Devices</h3>
                <p class="muted">This device layer is here to make magic-link access feel intentional rather than invisible.</p>
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
                    ? `<div class="card-grid">${recentSessions.slice(0, 3).map((session) => accountSessionCard(session)).join("")}</div>`
                    : `<div class="notice">No recent secondary device sessions are stored for this account yet.</div>`
                }
              </article>
              <article class="panel">
                <h3>Workspace controls</h3>
                <p class="muted">This is the compact read of what shapes your desk before you open the full preferences layer underneath.</p>
                <div class="stats-grid">
                  ${statPanel("Style", stylePresetLabel(userStylePreset), stylePresetSummary(userStylePreset))}
                  ${statPanel("Language", languageLabel(notificationPreferences?.language_preference || "en-GB"), "Language is ready for fuller localisation later.")}
                  ${statPanel("Interruptions", deliveryPosture, feedRoutingExplanation(userStylePreset))}
                  ${statPanel("Onboarding", `${onboardingSummary.completed}/${onboardingSummary.total}`, onboardingCompleted ? "Calm setup is complete." : "There are still first-run choices to lock in.")}
                </div>
              </article>
            </section>
            <section class="section">
              <article class="panel" id="preferences">
                <h3>Intelligence preferences</h3>
                <p class="muted">Open the full control layer here when you want to change the desk, not just inspect it.</p>
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
          ${dashboardComputePanel(matches)}
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
    const confidenceTier = String(fixture.signal_summary?.confidence_tier || fixture.deploy_summary?.confidence_tier || "").toUpperCase();
    const marketLine = primaryMarketLine(fixture);
    const verdictLabel = marketVerdictDisplay(fixture);
    const heroMode = fixtureTimeState(fixture.kickoff_time).tone === "scheduled" ? "editorial" : "scoreboard";
    const alternativeLine = alternativeMarketLine(fixture);
    const marketStructure = marketStructureRows(odds);
    const matchCopy = matchReasons.length
      ? `This fixture matches your saved follows through ${matchReasons.join(", ")}.`
      : "This fixture is being shown from the current intelligence window rather than a direct saved follow.";
    const relatedFixtures = state.fixtureIntelligence
      .filter((row) => row.fixture_key !== fixture.fixture_key && row.league === fixture.league)
      .slice(0, 4);
    const fixtureTabs = [
      ["prediction", "Prediction"],
      ["markets", "Markets"],
      ["lineups", "Lineups"],
      ["stats", "Stats"],
      ["table", "Table"],
      ["h2h", "H2H"],
      ["form", "Form"],
      ["news", "News"],
      ["context", "Context"],
    ];
    const requestedFixtureTab =
      selectedFixtureTab === "overview" || selectedFixtureTab === "intelligence" ? "prediction" : selectedFixtureTab;
    const activeFixtureTab = fixtureTabs.some(([key]) => key === requestedFixtureTab) ? requestedFixtureTab : "prediction";
    const followMatchLabel = matchedEntry ? matchedEntry.reasons.join(" / ") : "Not followed";
    const fixtureSummaryNotice = renderNotice(
      state.runtime.fixtureAlertMessage,
      state.runtime.fixtureAlertMessage ? "success" : "default"
    );
    const relatedFixturesMarkup = relatedFixtures.length
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
      : `<div class="notice">No related fixtures are available from the current published intelligence window.</div>`;
    const activeTabContent = (() => {
      const requiredTier = fixtureTabRequiredTier(activeFixtureTab);
      if (!hasTierAccess(requiredTier)) {
        return fixtureLockedTabContent(activeFixtureTab);
      }
      if (activeFixtureTab === "prediction") {
        return `
          ${renderFixtureOverviewPrimer(fixture, clarity)}
          ${renderFixtureTeamFaceOff(fixture, state.selectedFixtureDecisionIntelligence || null)}
          ${renderDecisionKeyPlayerDrivers(state.selectedFixtureDecisionIntelligence || null)}
        `;
      }
      if (activeFixtureTab === "lineups") {
        return `
          ${renderFixtureLineupIntelligence(state.selectedFixtureLineupIntelligence, fixture)}
        `;
      }
      if (activeFixtureTab === "markets") {
        return `
          ${renderDecisionMarketSuitability(state.selectedFixtureDecisionIntelligence || null)}
        `;
      }
      if (activeFixtureTab === "h2h") {
        return `
          ${renderFixtureH2HSupport(fixture, state.selectedFixtureDecisionSupport?.h2hSupport || null)}
        `;
      }
      if (activeFixtureTab === "table") {
        return `
          <section class="section">
            ${fixtureSummaryNotice}
            <article class="panel">
              ${fixtureStandingsWidgetMarkup(fixture)}
            </article>
          </section>
        `;
      }
      if (activeFixtureTab === "stats") {
        return `
          <section class="section">
            ${fixtureSummaryNotice}
            <div class="fixture-detail-grid">
              <article class="panel">
                <h3>Publish-safe stats</h3>
                <div class="prediction-meta-grid dashboard-odds-grid">
                  <div class="signal-cell signal-cell-model">
                    <span class="signal-label">Fixture key</span>
                    <span class="signal-value">${escapeHtml(fixture.fixture_key || fixture.fixture_id || "Pending")}</span>
                  </div>
                  <div class="signal-cell signal-cell-model">
                    <span class="signal-label">Provider fixture</span>
                    <span class="signal-value">${escapeHtml(String(fixture.api_fixture_id || "Unmapped"))}</span>
                  </div>
                  <div class="signal-cell signal-cell-model">
                    <span class="signal-label">League id</span>
                    <span class="signal-value">${escapeHtml(String(fixture.api_league_id || "Unmapped"))}</span>
                  </div>
                  <div class="signal-cell signal-cell-model">
                    <span class="signal-label">Season</span>
                    <span class="signal-value">${escapeHtml(String(fixture.api_season || "Pending"))}</span>
                  </div>
                  <div class="signal-cell signal-cell-model">
                    <span class="signal-label">Snapshot phase</span>
                    <span class="signal-value">${escapeHtml(safeTitleLabel(fixture.snapshot_phase || "Pending"))}</span>
                  </div>
                  <div class="signal-cell signal-cell-model">
                    <span class="signal-label">Coverage</span>
                    <span class="signal-value">${escapeHtml(safeTitleLabel(fixture.coverage_status || "Pending"))}</span>
                  </div>
                </div>
                <div class="fixture-stats-note">
                  <span class="metric-label">Stats tab ownership</span>
                  <p class="muted">This tab is now for data identity, freshness, and coverage. Prediction language stays in Prediction; bookmaker posture stays in Markets.</p>
                </div>
              </article>
              <article class="panel">
                <h3>Freshness metadata</h3>
                <div class="prediction-meta-grid dashboard-odds-grid">
                  ${fixtureFreshnessMeta(
                    fixture,
                    state.selectedFixtureDecisionIntelligence || null,
                    state.selectedFixtureLineupIntelligence || null,
                    state.selectedFixtureDecisionSupport?.h2hSupport || null
                  )
                    .map(
                      (entry) => `
                        <div class="signal-cell signal-cell-market">
                          <span class="signal-label">${escapeHtml(entry.label)}</span>
                          <span class="signal-value">${escapeHtml(entry.value)}</span>
                          <span class="muted">${escapeHtml(entry.note)}</span>
                        </div>
                      `
                    )
                    .join("")}
                </div>
                <div class="fixture-stats-note">
                  <span class="metric-label">Automation state</span>
                  <p class="muted">If a timestamp is unavailable, the UI shows the fallback state rather than pretending the layer is fresher than the published payload.</p>
                </div>
              </article>
            </div>
          </section>
        `;
      }
      if (activeFixtureTab === "form") {
        return `
          <section class="section">
            ${fixtureSummaryNotice}
            <article class="panel fixture-form-panel">
              <div
                class="fixture-form-shell"
                data-role="fixture-form-reference"
                data-league-id="${escapeHtml(String(fixture.api_league_id || ""))}"
                data-season="${escapeHtml(String(fixture.api_season || ""))}"
                data-home-team-id="${escapeHtml(String(fixture.api_home_team_id || "").trim() || extractTeamIdFromLogoUrl(fixture.home_team_logo_url))}"
                data-away-team-id="${escapeHtml(String(fixture.api_away_team_id || "").trim() || extractTeamIdFromLogoUrl(fixture.away_team_logo_url))}"
                data-home-team="${escapeHtml(fixture.home_team || "")}"
                data-away-team="${escapeHtml(fixture.away_team || "")}"
                data-home-logo="${escapeHtml(fixture.home_team_logo_url || "")}"
                data-away-logo="${escapeHtml(fixture.away_team_logo_url || "")}"
              >
                <div class="fixture-form-head">
                  <span class="metric-label">Form intelligence</span>
                  <h3>Team rhythm</h3>
                  <p class="muted">Live league position, recent results, and scoring rhythm for both sides. This layer explains whether the current read is moving with or against team form.</p>
                </div>
                <div class="fixture-rhythm-board">
                  <article class="fixture-rhythm-team fixture-rhythm-loading">
                    <div class="fixture-rhythm-team-head">
                      ${badgeMarkup(fixture.home_team_logo_url, fixture.home_team, "lineup-team-badge fixture-rhythm-badge")}
                      <div>
                        <span class="metric-label">Home rhythm</span>
                        <h4>${escapeHtml(fixture.home_team)}</h4>
                        <p class="muted">Preparing form reference</p>
                      </div>
                    </div>
                    <div class="fixture-rhythm-stats">
                      <span><strong>—</strong><small>Position</small></span>
                      <span><strong>—</strong><small>Points</small></span>
                      <span><strong>—</strong><small>Goal diff</small></span>
                      <span><strong>—</strong><small>Last five</small></span>
                    </div>
                    <div class="notice">Live team rhythm will appear here when the league form reference is available.</div>
                  </article>
                  <article class="fixture-rhythm-team fixture-rhythm-loading fixture-rhythm-team-away">
                    <div class="fixture-rhythm-team-head">
                      ${badgeMarkup(fixture.away_team_logo_url, fixture.away_team, "lineup-team-badge fixture-rhythm-badge")}
                      <div>
                        <span class="metric-label">Away rhythm</span>
                        <h4>${escapeHtml(fixture.away_team)}</h4>
                        <p class="muted">Preparing form reference</p>
                      </div>
                    </div>
                    <div class="fixture-rhythm-stats">
                      <span><strong>—</strong><small>Position</small></span>
                      <span><strong>—</strong><small>Points</small></span>
                      <span><strong>—</strong><small>Goal diff</small></span>
                      <span><strong>—</strong><small>Last five</small></span>
                    </div>
                    <div class="notice">Live team rhythm will appear here when the league form reference is available.</div>
                  </article>
                </div>
              </div>
            </article>
          </section>
        `;
      }
      if (activeFixtureTab === "news") {
        return renderFixtureNewsSection(fixture);
      }
      if (activeFixtureTab === "context") {
        return `
          <section class="section split">
            ${fixtureSummaryNotice}
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
              <h3>Context notes</h3>
              ${
                notes.length
                  ? `<ul class="feature-list">${notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>`
                  : `<div class="notice">No extra context notes are currently published for this fixture.</div>`
              }
            </article>
          </section>
          <section class="section">
            <article class="panel">
              <h3>Related league fixtures</h3>
              ${relatedFixturesMarkup}
            </article>
          </section>
        `;
      }
      return `
        <section class="section split">
          ${fixtureSummaryNotice}
          <article class="panel">
            <h3>${escapeHtml(clarity.meaning_title)}</h3>
            <p class="muted">${escapeHtml(clarity.meaning_copy)}</p>
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
              <span class="metric-label">Telegram relevance</span>
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
      `;
    })();

    return `
      <section class="section fixture-hero-shell fixture-hero-shell-wide">
        <article class="hero-main fixture-hero-main fixture-hero-main-${escapeHtml(heroMode)}">
          ${renderFixtureHeroScoreboard(fixture, clarity)}
        </article>
      </section>
      ${fixtureMarketCardsMarkup(fixture, state.selectedFixtureDecisionIntelligence || null)}
      ${fixturePlayerEventCardsMarkup(fixture)}
      ${
        hasTierAccess("founder")
          ? `
            ${renderFixturePredictionDeck(fixture, clarity, matchedEntry, publishClass)}
            ${renderFixtureCoverageTruthStrip(
              fixture,
              state.selectedFixtureDecisionIntelligence || null,
              state.selectedFixtureLineupIntelligence || null,
              state.selectedFixtureDecisionSupport?.h2hSupport || null
            )}
            ${renderFixtureFreshnessPanel(
              fixture,
              state.selectedFixtureDecisionIntelligence || null,
              state.selectedFixtureLineupIntelligence || null,
              state.selectedFixtureDecisionSupport?.h2hSupport || null
            )}
            ${renderFixtureWeatherContext(fixture)}
            ${renderFixtureHeroMedia(fixture)}
          `
          : fixtureContextGate()
      }
      <section class="section section-tight">
        <nav class="page-subnav" aria-label="Fixture sections">
          <div class="page-subnav-scroll">
            ${fixtureTabs
              .map(
                ([key, label]) => {
                  const locked = !hasTierAccess(fixtureTabRequiredTier(key));
                  return `
                  <a
                    id="fixture-tab-${escapeHtml(key)}"
                    class="page-subnav-link ${activeFixtureTab === key ? "is-active" : ""} ${locked ? "is-locked" : ""}"
                    href="${fixtureTabHref(key)}"
                  >
                    ${escapeHtml(label)}
                    ${locked ? `<span>Locked</span>` : ""}
                  </a>
                `;
                }
              )
              .join("")}
          </div>
        </nav>
      </section>
      ${activeTabContent}
    `;
  };

  const render = () => {
    const views = {
      home: homeView,
      matches: matchesView,
      live: liveView,
      competitions: competitionsView,
      teams: teamsView,
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
    app.innerHTML = `<div class="app-view" data-view="${escapeHtml(page)}">${view()}</div>`;
    if (page === "fixture" || page === "teams" || page === "competitions") {
      hydrateFixtureReferenceWidgets();
    }
  };

  const hydrateFixtureReferenceWidgets = async () => {
    const standingsRoots = Array.from(document.querySelectorAll("[data-role='fixture-standings-reference']"));
    const lineupRoots = Array.from(document.querySelectorAll("[data-role='fixture-lineups-reference']"));
    const scoreboardRoots = Array.from(document.querySelectorAll("[data-role='fixture-scoreboard']"));
    const formRoots = Array.from(document.querySelectorAll("[data-role='fixture-form-reference']"));
    const competitionStandingsRoots = Array.from(document.querySelectorAll("[data-role='competition-standings-reference']"));
    const competitionFixturesRoots = Array.from(document.querySelectorAll("[data-role='competition-fixtures-reference']"));
    const competitionResultsRoots = Array.from(document.querySelectorAll("[data-role='competition-results-reference']"));
    const teamFormRoots = Array.from(document.querySelectorAll("[data-role='team-form-reference']"));
    if (
      (!lineupRoots.length &&
        !standingsRoots.length &&
        !scoreboardRoots.length &&
        !formRoots.length &&
        !competitionStandingsRoots.length &&
        !competitionFixturesRoots.length &&
        !competitionResultsRoots.length &&
        !teamFormRoots.length) ||
      !workerConfigured()
    ) {
      return;
    }
    const fixtureLookupCache = new Map();
    const fixtureDetailsCache = new Map();
    const standingsCache = new Map();
    const teamFixturesCache = new Map();
    const competitionFixturesCache = new Map();
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
    const fetchFixtureDetails = async (fixtureId) => {
      const id = String(fixtureId || "").trim();
      if (!id) {
        return null;
      }
      if (!fixtureDetailsCache.has(id)) {
        fixtureDetailsCache.set(
          id,
          fetchWorkerJson(`/api/widgets/football/fixtures?id=${encodeURIComponent(id)}`, { method: "GET" })
        );
      }
      const result = await fixtureDetailsCache.get(id);
      return result?.payload?.response?.[0] || null;
    };
    const fetchStandingsRows = async (leagueId, season) => {
      const cacheKey = `${leagueId}:${season}`;
      if (!standingsCache.has(cacheKey)) {
        const params = new URLSearchParams({ league: String(leagueId || ""), season: String(season || "") });
        standingsCache.set(
          cacheKey,
          fetchWorkerJson(`/api/widgets/football/standings?${params.toString()}`, { method: "GET" })
        );
      }
      const { response, payload } = await standingsCache.get(cacheKey);
      if (!response.ok || !payload?.response?.length) {
        return [];
      }
      const groups = payload.response?.[0]?.league?.standings || [];
      return Array.isArray(groups) && groups.length ? groups[0] : [];
    };
    const fetchCompetitionFixtures = async (leagueId, season, scope) => {
      const cacheKey = `${leagueId}:${season}:${scope}`;
      if (!competitionFixturesCache.has(cacheKey)) {
        const params = new URLSearchParams({
          league: String(leagueId || ""),
          season: String(season || ""),
          [scope === "results" ? "last" : "next"]: scope === "results" ? "12" : "10",
        });
        competitionFixturesCache.set(
          cacheKey,
          fetchWorkerJson(`/api/widgets/football/fixtures?${params.toString()}`, { method: "GET" })
        );
      }
      const { response, payload } = await competitionFixturesCache.get(cacheKey);
      if (!response.ok || !Array.isArray(payload?.response)) {
        return [];
      }
      return payload.response;
    };
    const isFinishedStatus = (statusShort) => /FT|AET|PEN/i.test(String(statusShort || "").trim());
    const fetchRecentTeamFixtures = async (teamId) => {
      const id = String(teamId || "").trim();
      if (!id) {
        return [];
      }
      if (!teamFixturesCache.has(id)) {
        teamFixturesCache.set(
          id,
          fetchWorkerJson(`/api/widgets/football/fixtures?team=${encodeURIComponent(id)}&last=8`, { method: "GET" })
        );
      }
      const { response, payload } = await teamFixturesCache.get(id);
      if (!response.ok || !Array.isArray(payload?.response)) {
        return [];
      }
      return payload.response.filter((entry) => isFinishedStatus(entry?.fixture?.status?.short)).slice(0, 5);
    };
    const formOutcomeMeta = (fixtureRow, teamId) => {
      const teams = fixtureRow?.teams || {};
      const homeId = String(teams?.home?.id || "").trim();
      const isHome = homeId === String(teamId || "").trim();
      const result = formatFixtureResultChip(fixtureRow, isHome ? "home" : "away");
      const label = String(result.label || "D").toUpperCase();
      const tone = label === "WIN" || label === "W" ? "w" : label === "LOSS" || label === "L" ? "l" : "d";
      const goals = fixtureRow?.goals || {};
      const scored = Number(isHome ? goals.home : goals.away);
      const conceded = Number(isHome ? goals.away : goals.home);
      return {
        isHome,
        opponent: isHome ? teams?.away : teams?.home,
        scored: Number.isFinite(scored) ? scored : null,
        conceded: Number.isFinite(conceded) ? conceded : null,
        label: tone === "w" ? "W" : tone === "l" ? "L" : "D",
        tone,
      };
    };
    const formRhythmSummary = (fixtures, teamId) => {
      const list = Array.isArray(fixtures) ? fixtures : [];
      const summary = list.reduce(
        (acc, fixtureRow) => {
          const meta = formOutcomeMeta(fixtureRow, teamId);
          acc.played += 1;
          if (meta.tone === "w") acc.wins += 1;
          if (meta.tone === "d") acc.draws += 1;
          if (meta.tone === "l") acc.losses += 1;
          if (meta.scored !== null) acc.scored += meta.scored;
          if (meta.conceded !== null) acc.conceded += meta.conceded;
          return acc;
        },
        { played: 0, wins: 0, draws: 0, losses: 0, scored: 0, conceded: 0 }
      );
      const points = summary.wins * 3 + summary.draws;
      const scoreBalance = summary.scored - summary.conceded;
      let label = summary.played ? "Mixed rhythm" : "Form feed pending";
      if (summary.played && points >= summary.played * 2) {
        label = "Strong rhythm";
      } else if (summary.played && points <= summary.played) {
        label = "Fragile rhythm";
      }
      return { ...summary, points, scoreBalance, label };
    };
    const renderRecentResults = (fixtures, teamId) => {
      const fixtureList = Array.isArray(fixtures) ? fixtures : [];
      if (!fixtureList.length) {
        return `<div class="notice">Recent finished fixtures are not available from the upstream feed yet.</div>`;
      }
      return `
        <div class="team-form-results">
          ${fixtureList
            .map((fixtureRow) => {
              const meta = formOutcomeMeta(fixtureRow, teamId);
              const score = `${meta.scored ?? "—"}-${meta.conceded ?? "—"}`;
              const opponent = meta.opponent || {};
              return `
                <article class="team-form-card">
                  <div class="team-form-card-top">
                    <span class="form-pill form-pill-${escapeHtml(meta.tone)}">${escapeHtml(meta.label)}</span>
                    ${badgeMarkup(opponent.logo, opponent.name || "Opponent", "team-form-opponent-badge")}
                  </div>
                  <strong>${escapeHtml(score)}</strong>
                  <span class="muted">${escapeHtml(opponent.name || "Opponent")}</span>
                  <span class="muted">${escapeHtml(formatKickoffLabel(fixtureRow?.fixture?.date || ""))}</span>
                </article>
              `;
            })
            .join("")}
        </div>
      `;
    };
    const renderFormSequence = (formString, fixtures, teamId) => {
      const letters = String(formString || "")
        .split("")
        .filter(Boolean)
        .slice(0, 5);
      const fallbackLetters = (Array.isArray(fixtures) ? fixtures : [])
        .map((fixtureRow) => formOutcomeMeta(fixtureRow, teamId).label)
        .filter(Boolean)
        .slice(0, 5);
      const sequence = letters.length ? letters : fallbackLetters;
      if (!sequence.length) {
        return `<span class="muted">No current form string</span>`;
      }
      return sequence
        .map((letter) => {
          const tone = letter === "W" ? "w" : letter === "L" ? "l" : "d";
          return `<span class="form-pill form-pill-${escapeHtml(tone)}">${escapeHtml(letter)}</span>`;
        })
        .join("");
    };
    const renderFixtureRhythmTeam = (row, fixtures, teamId, side, fallbackName, fallbackLogo) => {
      const team = row?.team || {};
      const teamName = team.name || fallbackName || "Team";
      const logo = team.logo || fallbackLogo || "";
      const summary = formRhythmSummary(fixtures, teamId);
      const record = `${summary.wins}W ${summary.draws}D ${summary.losses}L`;
      return `
        <article class="fixture-rhythm-team fixture-rhythm-team-${escapeHtml(side)}">
          <div class="fixture-rhythm-team-head">
            ${badgeMarkup(logo, teamName, "lineup-team-badge fixture-rhythm-badge")}
            <div>
              <span class="metric-label">${escapeHtml(side === "home" ? "Home rhythm" : "Away rhythm")}</span>
              <h4>${escapeHtml(teamName)}</h4>
              <p class="muted">${escapeHtml(summary.label)} · last ${escapeHtml(String(summary.played || 0))}</p>
            </div>
          </div>
          <div class="fixture-rhythm-stats">
            <span><strong>${escapeHtml(row?.rank ?? "—")}</strong><small>Position</small></span>
            <span><strong>${escapeHtml(row?.points ?? "—")}</strong><small>Points</small></span>
            <span><strong>${escapeHtml(row?.goalsDiff ?? "—")}</strong><small>Goal diff</small></span>
            <span><strong>${escapeHtml(record)}</strong><small>Last five</small></span>
          </div>
          <div class="fixture-rhythm-sequence">
            ${renderFormSequence(row?.form || "", fixtures, teamId)}
          </div>
          <div class="fixture-rhythm-balance">
            <span>${escapeHtml(`${summary.scored} scored`)}</span>
            <div class="fixture-rhythm-balance-track" aria-hidden="true">
              <span style="width:${escapeHtml(String(Math.min(100, Math.max(12, summary.scored * 12))))}%"></span>
            </div>
            <span>${escapeHtml(`${summary.conceded} conceded`)}</span>
          </div>
          ${renderRecentResults(fixtures, teamId)}
        </article>
      `;
    };
    const formScoringSummary = (fixtures, teamId) => {
      const summary = (Array.isArray(fixtures) ? fixtures : []).reduce(
        (acc, fixtureRow) => {
          const teams = fixtureRow?.teams || {};
          const homeId = String(teams?.home?.id || "").trim();
          const isHome = homeId === String(teamId || "").trim();
          const goals = fixtureRow?.goals || {};
          const scored = Number(isHome ? goals.home : goals.away);
          const conceded = Number(isHome ? goals.away : goals.home);
          if (Number.isFinite(scored)) {
            acc.scored += scored;
          }
          if (Number.isFinite(conceded)) {
            acc.conceded += conceded;
          }
          return acc;
        },
        { scored: 0, conceded: 0 }
      );
      return `${summary.scored} scored • ${summary.conceded} conceded`;
    };
    const renderCompetitionArchiveCards = (fixtures, scope) => {
      const list = Array.isArray(fixtures) ? fixtures : [];
      if (!list.length) {
        return `
          <div class="archive-empty-card">
            <span class="metric-label">${scope === "results" ? "Archive unavailable" : "Schedule unavailable"}</span>
            <p class="section-copy">No ${scope === "results" ? "recent results" : "upcoming fixtures"} are available from the upstream feed yet.</p>
          </div>
        `;
      }
      return `
        <div class="card-grid archive-grid">
          ${list
            .map((fixtureRow) => {
              const teams = fixtureRow?.teams || {};
              const goals = fixtureRow?.goals || {};
              const fixtureMeta = fixtureRow?.fixture || {};
              const status = fixtureMeta?.status || {};
              const hasListedScore =
                goals.home !== null &&
                goals.home !== undefined &&
                goals.away !== null &&
                goals.away !== undefined &&
                Number.isFinite(Number(goals.home)) &&
                Number.isFinite(Number(goals.away));
              const score = hasListedScore ? `${goals.home}-${goals.away}` : "vs";
              const statusText =
                scope === "results"
                  ? String(status.short || status.long || "FT").trim() || "FT"
                  : formatKickoffLabel(fixtureMeta?.date || "");
              return `
                <article class="panel archive-fixture-card">
                  <div class="intelligence-card-head">
                    <span class="chip chip-reference">${escapeHtml(statusText)}</span>
                    ${fixtureRow?.league?.round ? `<span class="chip chip-reference">${escapeHtml(fixtureRow.league.round)}</span>` : ""}
                  </div>
                  <strong class="fixture-teamline dashboard-teamline">
                    ${badgeMarkup(teams?.home?.logo, teams?.home?.name || "Home")}
                    <span class="team-name">${escapeHtml(teams?.home?.name || "Home")}</span>
                    <span class="versus">${escapeHtml(score)}</span>
                    ${badgeMarkup(teams?.away?.logo, teams?.away?.name || "Away")}
                    <span class="team-name">${escapeHtml(teams?.away?.name || "Away")}</span>
                  </strong>
                  <p class="muted">${escapeHtml(formatKickoffLabel(fixtureMeta?.date || ""))}</p>
                </article>
              `;
            })
            .join("")}
        </div>
      `;
    };
    const teamSheetRatingLookup = buildTeamSheetRatingLookup();
    const renderLineupSquad = (players, emptyCopy, teamName = "") => {
      const list = Array.isArray(players) ? players : [];
      if (!list.length) {
        return `<div class="notice">${escapeHtml(emptyCopy)}</div>`;
      }
      return `
        <div class="lineup-player-list">
          ${list
            .map((entry) => {
              const player = entry?.player || {};
              const jersey = player.number ? `#${player.number}` : "Squad";
              const role = player.pos || player.grid || "Player";
              const ratingProfile = findTeamSheetRatingProfile(teamSheetRatingLookup, teamName, player.name || "");
              const power = playerProfilePower(ratingProfile);
              const rank = playerProfileRank(ratingProfile);
              const meta = [
                jersey,
                role,
                power !== null ? `${power}% OG` : "Rating pending",
                rank ? `Club rank ${rank}` : "",
              ].filter(Boolean);
              return `
                <article class="lineup-player-card">
                  ${renderOgRatingBadge(power, "small", power !== null ? `${player.name || "Player"} OG player rating` : "Rating pending")}
                  <div>
                    <strong>${escapeHtml(player.name || "Unnamed player")}</strong>
                    <span class="muted">${escapeHtml(meta.join(" · "))}</span>
                  </div>
                </article>
              `;
            })
            .join("")}
        </div>
      `;
    };
    await Promise.all(
      standingsRoots.map(async (root) => {
        const frame = root.querySelector(".widget-reference-frame");
        if (!frame) {
          return;
        }
        try {
          const homeTeamId = String(root.dataset.homeTeamId || "").trim();
          const awayTeamId = String(root.dataset.awayTeamId || "").trim();
          const homeName = normalizePreferenceText(root.dataset.home || "");
          const awayName = normalizePreferenceText(root.dataset.away || "");
          let leagueId = String(root.dataset.apiLeagueId || "").trim();
          let season = String(root.dataset.apiSeason || "").trim();
          if (!leagueId || !season) {
            const fixtureLookup = await resolveFixtureReference(root);
            if (fixtureLookup?.response?.ok && fixtureLookup?.payload?.ok) {
              leagueId = String(fixtureLookup.payload.league_id || leagueId).trim();
              season = String(fixtureLookup.payload.season || season).trim();
            }
          }
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
          const isFixtureTeamRow = (row) => {
            const rowTeamId = String(row?.team?.id || "").trim();
            const rowTeamName = normalizePreferenceText(row?.team?.name || "");
            return (
              (rowTeamId && (rowTeamId === homeTeamId || rowTeamId === awayTeamId)) ||
              rowTeamName === homeName ||
              rowTeamName === awayName
            );
          };
          const matchSpecificTeam = (teamId, teamName) =>
            tableRows.find((row) => {
              const rowTeamId = String(row?.team?.id || "").trim();
              const rowTeamName = normalizePreferenceText(row?.team?.name || "");
              return (teamId && rowTeamId === teamId) || rowTeamName === teamName;
            }) || null;
          const homeActiveRow = matchSpecificTeam(homeTeamId, homeName);
          const awayActiveRow = matchSpecificTeam(awayTeamId, awayName);
          const activeRows = [homeActiveRow, awayActiveRow].filter(Boolean);
          const formSequenceMarkup = (formString) =>
            String(formString || "")
              .split("")
              .filter(Boolean)
              .slice(0, 5)
              .map((letter) => `<span class="form-pill form-pill-${escapeHtml(letter.toLowerCase())}">${escapeHtml(letter)}</span>`)
              .join("") || `<span class="muted">No form string</span>`;
          const standingSpotlightCard = (row, index) => {
            const team = row?.team || {};
            const all = row?.all || {};
            const goals = all?.goals || {};
            return `
              <article class="standings-spotlight-card ${index === 1 ? "standings-spotlight-card-away" : ""}">
                <div class="standings-spotlight-team">
                  ${badgeMarkup(team.logo, team.name || "Team", "lineup-team-badge standings-team-badge-lg")}
                  <div>
                    <span class="metric-label">${index === 1 ? "Away table read" : "Home table read"}</span>
                    <h4>${escapeHtml(team.name || "Team")}</h4>
                    <p class="muted">${escapeHtml(all.played ?? "—")} played · ${escapeHtml(goals.for ?? "—")}:${escapeHtml(goals.against ?? "—")} goals</p>
                  </div>
                </div>
                <div class="standings-spotlight-stats">
                  <span><strong>${escapeHtml(row.rank ?? "—")}</strong><small>Position</small></span>
                  <span><strong>${escapeHtml(row.points ?? "—")}</strong><small>Points</small></span>
                  <span><strong>${escapeHtml(row.goalsDiff ?? "—")}</strong><small>Goal diff</small></span>
                </div>
                <div class="standings-form-sequence standings-form-sequence-large">
                  ${formSequenceMarkup(row.form)}
                </div>
              </article>
            `;
          };
          frame.innerHTML = `
            ${
              activeRows.length
                ? `<div class="standings-spotlight-grid">${activeRows.map((row, index) => standingSpotlightCard(row, index)).join("")}</div>`
                : ""
            }
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
                .map((row) => {
                  const isActiveTeam = isFixtureTeamRow(row);
                  return `
                    <div class="standings-reference-row ${isActiveTeam ? "standings-reference-row-active" : ""}">
                      <span>${escapeHtml(row.rank ?? "")}</span>
                      <strong class="standings-reference-team">
                        ${badgeMarkup(row.team?.logo, row.team?.name || "Team", "lineup-team-badge standings-team-badge")}
                        <span>${escapeHtml(row.team?.name || "")}</span>
                      </strong>
                      <span>${escapeHtml(row.all?.played ?? "")}</span>
                      <span>${escapeHtml(row.goalsDiff ?? "")}</span>
                      <span>${escapeHtml(row.points ?? "")}</span>
                      <span class="standings-form-sequence">${formSequenceMarkup(row.form)}</span>
                    </div>
                  `;
                })
                .join("")}
            </div>
          `;
        } catch (error) {
          frame.innerHTML = `<div class="notice reference-loading">${escapeHtml(
            error.message || "League table reference could not be loaded for this fixture."
          )}</div>`;
        }
      })
    );
    await Promise.all(
      scoreboardRoots.map(async (root) => {
        try {
          let fixtureId = String(root.dataset.apiFixtureId || "").trim();
          if (!fixtureId) {
            const fixtureLookup = await resolveFixtureReference(root);
            if (fixtureLookup?.response?.ok && fixtureLookup?.payload?.fixture_id) {
              fixtureId = String(fixtureLookup.payload.fixture_id || "").trim();
            }
          }
          const fixtureDetails = await fetchFixtureDetails(fixtureId);
          if (!fixtureDetails) {
            return;
          }
          const status = fixtureDetails?.fixture?.status || {};
          const goals = fixtureDetails?.goals || {};
          const hasScore = Number.isFinite(Number(goals.home)) && Number.isFinite(Number(goals.away));
          const centerLabel = hasScore ? `${goals.home} - ${goals.away}` : "vs";
          const statusLabel = String(status.short || status.long || "").trim() || fixtureTimeState(root.dataset.kickoffTime).label;
          const detailLabel =
            status.elapsed != null
              ? `${status.elapsed}'`
              : status.long || fixtureTimeState(root.dataset.kickoffTime).detail;
          const centerTone = /FT|AET|PEN|CANC|PST/i.test(statusLabel)
            ? "final"
            : /1H|2H|HT|LIVE|ET|BT/i.test(statusLabel)
              ? "live"
              : "scheduled";
          const centerNode = root.querySelector(".fixture-hero-center");
          if (centerNode) {
            centerNode.innerHTML = `
              <span class="metric-label">${escapeHtml(statusLabel)}</span>
              <strong class="fixture-hero-score">${escapeHtml(centerLabel)}</strong>
              <span class="muted">${escapeHtml(detailLabel)}</span>
            `;
          }
          const scorerNode = root.querySelector("[data-role='fixture-scorers']");
          if (scorerNode) {
            scorerNode.innerHTML = renderFixtureScorerStrip(
              collectGoalScorerRows(fixtureDetails, { home_team: root.dataset.home, away_team: root.dataset.away })
            );
          }
          const badge = root.querySelector(".fixture-status-badge");
          if (badge) {
            badge.textContent = statusLabel;
            badge.className = `fixture-status-badge fixture-status-badge-${centerTone}`;
          }
        } catch {
          return;
        }
      })
    );
    await Promise.all(
      lineupRoots.map(async (root) => {
        const frame = root.querySelector(".widget-reference-frame");
        if (!frame) {
          return;
        }
        try {
          let fixtureId = String(root.dataset.apiFixtureId || "").trim();
          let fixtureLookup = null;
          if (!fixtureId) {
            fixtureLookup = await resolveFixtureReference(root);
            if (fixtureLookup.response.ok && fixtureLookup.payload?.ok && fixtureLookup.payload.fixture_id) {
              fixtureId = String(fixtureLookup.payload.fixture_id || fixtureId).trim();
            }
          }
          if (!fixtureId) {
            throw new Error(fixtureLookup?.payload?.message || "Unable to resolve fixture lineups for this page.");
          }
          const { response, payload } = await fetchWorkerJson(
            `/api/widgets/football/fixtures/lineups?fixture=${encodeURIComponent(fixtureId)}`,
            { method: "GET" }
          );
          const rows = response.ok && Array.isArray(payload?.response) ? payload.response : [];
          if (!rows.length) {
            throw new Error("Confirmed lineups are not available from the upstream provider for this fixture yet. Team sheets usually land closer to kickoff, so keep the intelligence, table, and context layers visible in the meantime.");
          }
          frame.innerHTML = `
            <div class="lineup-reference-grid">
              ${rows
                .slice(0, 2)
                .map((team) => {
                  const teamInfo = team?.team || {};
                  const coach = team?.coach || {};
                  return `
                    <article class="panel lineup-team-panel">
                      <div class="lineup-team-head">
                        <div class="lineup-team-title">
                          ${badgeMarkup(teamInfo.logo, teamInfo.name, "lineup-team-badge")}
                          <div>
                            <span class="metric-label">Confirmed provider lineup</span>
                            <h4>${escapeHtml(teamInfo.name || "Team")}</h4>
                            <p class="muted">Formation ${escapeHtml(team?.formation || "TBC")} • Coach ${escapeHtml(coach.name || "TBC")}</p>
                          </div>
                        </div>
                      </div>
                      <div class="lineup-section">
                        <span class="metric-label">Starting XI</span>
                        ${renderLineupSquad(team?.startXI, "Starting XI not available yet.", teamInfo.name)}
                      </div>
                      <div class="lineup-section">
                        <span class="metric-label">Bench</span>
                        ${renderLineupSquad(team?.substitutes, "Substitutes list not available yet.", teamInfo.name)}
                      </div>
                    </article>
                  `;
                })
                .join("")}
            </div>
          `;
        } catch (error) {
          const rawMessage = error.message || "";
          const fallbackMessage = /failed to fetch/i.test(rawMessage)
            ? "The live lineup reference could not be reached from this local page. Use the publish-safe lineup intelligence below; if confirmed teams are unavailable, it will show the squad fallback state deliberately."
            : rawMessage || "Lineups and formations are not available for this fixture yet.";
          frame.innerHTML = `
            <div class="lineup-empty-state">
              <div class="lineup-empty-grid">
                <article class="lineup-empty-card">
                  <span class="metric-label">Confirmed provider lineup</span>
                  <p class="muted">${escapeHtml(fallbackMessage)}</p>
                </article>
              </div>
            </div>
          `;
        }
      })
    );
    await Promise.all(
      competitionStandingsRoots.map(async (root) => {
        try {
          const frame = root.querySelector(".widget-reference-frame") || root;
          const leagueId = String(root.dataset.leagueId || "").trim();
          const season = String(root.dataset.season || "").trim();
          if (!leagueId || !season) {
            throw new Error("Competition table is not available for this league yet.");
          }
          const rows = await fetchStandingsRows(leagueId, season);
          if (!rows.length) {
            throw new Error("Competition table is not available from the upstream source yet.");
          }
          frame.innerHTML = `
            <div class="standings-table">
              <div class="standings-table-head">
                <span>Pos</span>
                <span>Team</span>
                <span>P</span>
                <span>GD</span>
                <span>Pts</span>
                <span>Form</span>
              </div>
              ${rows
                .slice(0, 10)
                .map((row) => {
                  const formDots = String(row?.form || "")
                    .split("")
                    .filter(Boolean)
                    .slice(0, 5)
                    .map((letter) => {
                      const tone = letter === "W" ? "w" : letter === "L" ? "l" : "d";
                      return `<span class="form-pill form-pill-${escapeHtml(tone)}">${escapeHtml(letter)}</span>`;
                    })
                    .join("");
                  return `
                    <div class="standings-row">
                      <span>${escapeHtml(row.rank ?? "—")}</span>
                      <span class="standings-team">
                        ${badgeMarkup(row?.team?.logo, row?.team?.name || "Team")}
                        <strong>${escapeHtml(row?.team?.name || "Team")}</strong>
                      </span>
                      <span>${escapeHtml(row.all?.played ?? row.played ?? "—")}</span>
                      <span>${escapeHtml(row.goalsDiff ?? "—")}</span>
                      <span>${escapeHtml(row.points ?? "—")}</span>
                      <span class="form-sequence">${formDots || `<span class="muted">—</span>`}</span>
                    </div>
                  `;
                })
                .join("")}
            </div>
          `;
        } catch (error) {
          root.innerHTML = `<div class="notice">${escapeHtml(error.message || "Competition table unavailable.")}</div>`;
        }
      })
    );
    await Promise.all(
      competitionFixturesRoots.map(async (root) => {
        try {
          const frame = root.querySelector(".widget-reference-frame") || root;
          const leagueId = String(root.dataset.leagueId || "").trim();
          const season = String(root.dataset.season || "").trim();
          if (!leagueId || !season) {
            throw new Error("Broader competition fixtures are not available for this league yet.");
          }
          const fixtures = await fetchCompetitionFixtures(leagueId, season, "fixtures");
          frame.innerHTML = renderCompetitionArchiveCards(fixtures, "fixtures");
        } catch (error) {
          const frame = root.querySelector(".widget-reference-frame") || root;
          frame.innerHTML = `
            <div class="archive-empty-card">
              <span class="metric-label">Schedule unavailable</span>
              <p class="section-copy">${escapeHtml(error.message || "Competition fixtures unavailable.")}</p>
            </div>
          `;
        }
      })
    );
    await Promise.all(
      competitionResultsRoots.map(async (root) => {
        try {
          const frame = root.querySelector(".widget-reference-frame") || root;
          const leagueId = String(root.dataset.leagueId || "").trim();
          const season = String(root.dataset.season || "").trim();
          if (!leagueId || !season) {
            throw new Error("Broader competition results are not available for this league yet.");
          }
          const fixtures = await fetchCompetitionFixtures(leagueId, season, "results");
          frame.innerHTML = renderCompetitionArchiveCards(fixtures, "results");
        } catch (error) {
          const frame = root.querySelector(".widget-reference-frame") || root;
          frame.innerHTML = `
            <div class="archive-empty-card">
              <span class="metric-label">Archive unavailable</span>
              <p class="section-copy">${escapeHtml(error.message || "Competition results unavailable.")}</p>
            </div>
          `;
        }
      })
    );
    await Promise.all(
      teamFormRoots.map(async (root) => {
        try {
          const teamId = String(root.dataset.teamId || "").trim();
          if (!teamId) {
            throw new Error("Recent team form is not available for this side yet.");
          }
          const fixtures = await fetchRecentTeamFixtures(teamId);
          root.innerHTML = `
            <h3>Recent team rhythm</h3>
            <p class="section-copy">Live recent-results reference for this side. This gives the team page a cleaner form layer without replacing fixture-level deployment logic.</p>
            <p class="muted">${escapeHtml(formScoringSummary(fixtures, teamId))}</p>
            ${renderRecentResults(fixtures, teamId)}
          `;
        } catch (error) {
          root.innerHTML = `
            <h3>Recent team rhythm</h3>
            <div class="notice">${escapeHtml(error.message || "Recent team form is not available yet.")}</div>
          `;
        }
      })
    );
    await Promise.all(
      formRoots.map(async (root) => {
        try {
          let leagueId = String(root.dataset.leagueId || "").trim();
          let season = String(root.dataset.season || "").trim();
          if (!leagueId || !season) {
            root.innerHTML = `
              <div class="fixture-form-head">
                <span class="metric-label">Form intelligence</span>
                <h3>Team rhythm</h3>
                <p class="muted">The league form reference is not attached to this published fixture yet, so the page keeps the form layer explicit instead of pretending a live rhythm sample exists.</p>
              </div>
              <div class="fixture-rhythm-board">
                ${renderFixtureRhythmTeam(
                  null,
                  [],
                  root.dataset.homeTeamId || "",
                  "home",
                  root.dataset.homeTeam || "",
                  root.dataset.homeLogo || ""
                )}
                ${renderFixtureRhythmTeam(
                  null,
                  [],
                  root.dataset.awayTeamId || "",
                  "away",
                  root.dataset.awayTeam || "",
                  root.dataset.awayLogo || ""
                )}
              </div>
              <div class="fixture-rhythm-footer">
                <span class="chip chip-reference">Form feed pending</span>
                <span class="muted">Team ratings, player drivers, H2H context, and market posture remain available while the live form reference is missing.</span>
              </div>
            `;
            return;
          }
          const standingsRows = await fetchStandingsRows(leagueId, season);
          if (!standingsRows.length) {
            return;
          }
          const homeTeamId = String(root.dataset.homeTeamId || "").trim();
          const awayTeamId = String(root.dataset.awayTeamId || "").trim();
          const [homeFixtures, awayFixtures] = await Promise.all([
            fetchRecentTeamFixtures(homeTeamId),
            fetchRecentTeamFixtures(awayTeamId),
          ]);
          const matchByTeam = (teamId) =>
            standingsRows.find((row) => String(row?.team?.id || "").trim() === teamId) || null;
          const homeRow = matchByTeam(homeTeamId);
          const awayRow = matchByTeam(awayTeamId);
          root.innerHTML = `
            <div class="fixture-form-head">
              <span class="metric-label">Form intelligence</span>
              <h3>Team rhythm</h3>
              <p class="muted">Live league position, recent results, and scoring rhythm for both sides. This layer explains whether the current read is moving with or against team form.</p>
            </div>
            <div class="fixture-rhythm-board">
              ${renderFixtureRhythmTeam(
                homeRow,
                homeFixtures,
                homeTeamId,
                "home",
                root.dataset.homeTeam || "",
                root.dataset.homeLogo || ""
              )}
              ${renderFixtureRhythmTeam(
                awayRow,
                awayFixtures,
                awayTeamId,
                "away",
                root.dataset.awayTeam || "",
                root.dataset.awayLogo || ""
              )}
            </div>
            <div class="fixture-rhythm-footer">
              <span class="chip chip-reference">Team rhythm</span>
              <span class="muted">Recent form is supporting context only. Deploy state, market posture, and caution layers remain the decision spine.</span>
            </div>
          `;
        } catch {
          return;
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
    state.runtime.sessionAccessTier = "";
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
    state.runtime.sessionAccessTier = String(payload.access_tier || "");
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
    state.runtime.sessionAccessTier = "";
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
    const favouriteTarget = event.target.closest("[data-action='toggle-match-favourite']");
    if (favouriteTarget) {
      event.preventDefault();
      toggleMatchFavourite(favouriteTarget.dataset.fixtureKey);
      render();
      return;
    }

    const historyBackTarget = event.target.closest("[data-action='history-back']");
    if (historyBackTarget) {
      event.preventDefault();
      window.history.back();
      return;
    }

    const historyForwardTarget = event.target.closest("[data-action='history-forward']");
    if (historyForwardTarget) {
      event.preventDefault();
      window.history.forward();
      return;
    }

    const timelineExpandTarget = event.target.closest("[data-action='timeline-expand']");
    if (timelineExpandTarget) {
      event.preventDefault();
      const fixtureKey = String(timelineExpandTarget.dataset.fixtureKey || "");
      state.runtime.timelineExpandedFixture = state.runtime.timelineExpandedFixture === fixtureKey ? "" : fixtureKey;
      render();
      if (state.runtime.timelineExpandedFixture) {
        await loadTimelineFixturePayload(fixtureKey);
      }
      return;
    }

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

  document.querySelectorAll("[data-mobile-nav]").forEach((select) => {
    select.addEventListener("change", (event) => {
      const targetHref = event.target.value;
      if (targetHref) {
        window.location.href = targetHref;
      }
    });
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
    app.innerHTML = loadingShell(loadingMessage);
    syncActiveNav();
    state.runtime.premiumToken = readStoredPremiumToken();
    state.runtime.matchFavourites = readMatchFavourites();
    state.runtime.internalAdminKey = readStoredInternalAdminKey();
    state.runtime.internalOperatorId = readStoredInternalOperatorId();
    await loadAuthSession();
    await loadAccountState();
    await loadAccountSessions();
    await loadAccountAlerts();

    try {
      const [
        summary,
        publicPredictions,
        premiumPredictions,
        teamIntelligenceIndex,
        clubSquadIntelligenceIndex,
        fixtureDecisionIndex,
        fixtureLineupIndex,
        fixtureH2HIndex,
      ] = await Promise.all([
        fetchJson(`${DATA_ROOT}/publish_summary.json`),
        fetchJson(`${DATA_ROOT}/public_predictions.json`),
        premiumDemoMode ? fetchOptionalJson(`${DATA_ROOT}/premium_predictions.json`) : Promise.resolve([]),
        fetchOptionalJson(`${DATA_ROOT}/team_intelligence/team_ratings_index.json`),
        siteDataApiConfigured() ? Promise.resolve([]) : fetchOptionalJson(`${DATA_ROOT}/player_intelligence/club_squad_ratings.json`),
        fetchOptionalJson(`${DATA_ROOT}/fixture_decision_intelligence/index.json`),
        fetchOptionalJson(`${DATA_ROOT}/fixture_lineup_intelligence/index.json`),
        fetchOptionalJson(`${DATA_ROOT}/fixture_h2h_support/index.json`),
      ]);
      const weeklyResults = await fetchOptionalJson(`${DATA_ROOT}/weekly_results.json`);
      const resultsArchive = await fetchOptionalJson(`${DATA_ROOT}/results_archive.json`);
      const liveResultsFeed = await fetchOptionalJson(`${DATA_ROOT}/live_results_feed.json`);
      const fixtureIntelligenceRows = await loadFixtureIntelligenceRows();
      state.summary = summary;
      state.publicPredictions = publicPredictions;
      state.premiumPredictions = Array.isArray(premiumPredictions) ? premiumPredictions : [];
      state.weeklyResults = weeklyResults;
      state.resultsArchive = resultsArchive;
      state.liveResultsFeed = liveResultsFeed;
      state.fixtureIntelligence = fixtureIntelligenceRows;
      state.teamIntelligenceIndex = Array.isArray(teamIntelligenceIndex) ? teamIntelligenceIndex : [];
      state.clubSquadIntelligenceIndex = Array.isArray(clubSquadIntelligenceIndex) ? clubSquadIntelligenceIndex : [];
      state.fixtureDecisionIndex = Array.isArray(fixtureDecisionIndex) ? fixtureDecisionIndex : [];
      state.fixtureLineupIndex = Array.isArray(fixtureLineupIndex) ? fixtureLineupIndex : [];
      state.fixtureH2HIndex = Array.isArray(fixtureH2HIndex) ? fixtureH2HIndex : [];
      await loadSelectedTeamIntelligence();
      await loadSelectedFixtureSiteData();
      await loadSelectedFixtureStats();
      await loadSelectedFixtureExternalContent();
      await loadSelectedFixtureLineupIntelligence();
      await loadSelectedFixtureDecisionIntelligence();
      await loadProtectedPremiumPredictions();
      render();
    } catch (error) {
      renderError(error);
    }
  };

  boot();
})();
