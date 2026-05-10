import {
  buildSubscriberRecord,
  getSubscriberStateStore,
  loadSubscriberRecordByEmail,
  loadSubscriberRecordBySubscriptionId,
  persistSubscriberRecord,
} from "./subscriber_store.js";
import {
  addAccountAdminNote,
  createAccountRiskFlag,
  createAccountSession,
  completeTelegramLink,
  ensureAccountRiskState,
  getAccountDb,
  getAccountStateByUserId,
  getOpenAccountRiskFlagByType,
  getAccountRiskState,
  getAccountSessionById,
  getAccountStateByEmail,
  listAccountAdminNotesByUser,
  listAccountSessionsByUser,
  listAccountRiskFlagsByUser,
  listAuthEventsByUser,
  listDueNotificationAlerts,
  listActiveAccountSessionsByUser,
  listNotificationAlertsByUser,
  listUsersEligibleForTelegramAlerts,
  markNotificationAlertDelivered,
  markNotificationAlertFailed,
  mirrorSubscriptionFromRecord,
  recordAuthEvent,
  revokeAccountSession,
  revokeOtherAccountSessions,
  updateAccountRiskState,
  updateAccountRiskFlagStatus,
  setPrimaryAccountSession,
  touchAccountSessionSeen,
  upsertNotificationAlerts,
  updateNotificationPreferences,
} from "./account_store.js";
import { issuePremiumToken, verifyPremiumAccess } from "./auth.js";

const STRIPE_WEBHOOK_TOLERANCE_SECONDS = 300;
const PREMIUM_CACHE_TTL_SECONDS = 300;
const PREMIUM_CACHE_VERSION = "v1";
const WIDGET_FOOTBALL_PROXY_VERSION = "v1";
const WIDGET_FOOTBALL_STANDINGS_CACHE_TTL_SECONDS = 600;
const AUTH_MAGIC_LINK_TTL_SECONDS = 15 * 60;
const AUTH_SESSION_TTL_SECONDS = 7 * 24 * 60 * 60;
const AUTH_REQUEST_IP_COOLDOWN_SECONDS = 60;
const AUTH_REQUEST_EMAIL_COOLDOWN_SECONDS = 5 * 60;
const TELEGRAM_LINK_TTL_SECONDS = 10 * 60;
const AUTH_MAGIC_KEY_PREFIX = "auth_magic:";
const AUTH_RATE_LIMIT_KEY_PREFIX = "auth_rl:";
const TELEGRAM_LINK_KEY_PREFIX = "telegram_link:";
const AUTH_SESSION_COOKIE = "og_premium_session";
const TELEGRAM_WEBHOOK_SECRET_HEADER = "x-telegram-bot-api-secret-token";
const PREMIUM_ALLOWED_FIELDS = [
  "fixture_id",
  "fixture_key",
  "kickoff_time",
  "league",
  "league_logo_url",
  "league_flag_url",
  "home_team",
  "home_team_logo_url",
  "away_team",
  "away_team_logo_url",
  "market",
  "pick",
  "confidence_tier",
  "model_prob",
  "bookie_implied_prob",
  "value_edge",
  "bookie_od",
  "reason_tokens",
  "human_reason",
  "slip_role_hint",
  "safe_for_small_acca_flag",
  "safe_for_large_acca_flag",
  "correct_score_shortlist",
  "premium_tier",
  "logo_join_status",
];

const json = (payload, status = 200, extraHeaders = {}) =>
  new Response(JSON.stringify(payload, null, 2), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      ...extraHeaders,
    },
  });

const methodNotAllowed = (allowed) =>
  json(
    {
      ok: false,
      status: "method_not_allowed",
      message: `Use ${allowed} for this route.`,
    },
    405,
    { allow: allowed }
  );

const notFound = (pathname) =>
  json(
    {
      ok: false,
      status: "not_found",
      message: `No Worker route is defined for ${pathname}.`,
    },
    404
  );

const envSummary = (env) => ({
  has_site_url: Boolean(env.SITE_URL),
  has_premium_data_source: Boolean(env.PREMIUM_DATA_SOURCE),
  has_stripe_secret_key: Boolean(env.STRIPE_SECRET_KEY),
  has_stripe_webhook_secret: Boolean(env.STRIPE_WEBHOOK_SECRET),
  has_stripe_price_id: Boolean(env.STRIPE_PRICE_ID),
  has_subscriber_state_binding: Boolean(getSubscriberStateStore(env)),
  has_premium_token_secret: Boolean(env.PREMIUM_TOKEN_SECRET),
  has_auth_magic_link_secret: Boolean(env.AUTH_MAGIC_LINK_SECRET),
  has_auth_session_secret: Boolean(env.AUTH_SESSION_SECRET),
  has_resend_api_key: Boolean(env.RESEND_API_KEY),
  has_auth_email_from: Boolean(env.AUTH_EMAIL_FROM),
  has_account_db: Boolean(getAccountDb(env)),
  has_telegram_bot_token: Boolean(getTelegramBotToken(env)),
  has_telegram_bot_username: Boolean(env.TELEGRAM_BOT_USERNAME),
  has_telegram_webhook_secret: Boolean(getTelegramWebhookSecret(env)),
});

const buildCorsHeaders = (request, env) => {
  const origin = request.headers.get("origin") || "";
  const siteOrigin = (() => {
    try {
      return env.SITE_URL ? new URL(env.SITE_URL).origin : "";
    } catch {
      return "";
    }
  })();

  const allowOrigin =
    origin && (origin === siteOrigin || origin === "http://localhost:8788" || origin === "http://127.0.0.1:8788")
      ? origin
      : siteOrigin;

  if (!allowOrigin) {
    return {};
  }

  return {
    "access-control-allow-origin": allowOrigin,
    "access-control-allow-credentials": "true",
    "access-control-allow-methods": "GET,POST,OPTIONS",
    "access-control-allow-headers": "content-type,authorization",
    vary: "Origin",
  };
};

const withCors = (response, request, env) => {
  const headers = new Headers(response.headers);
  const corsHeaders = buildCorsHeaders(request, env);
  for (const [key, value] of Object.entries(corsHeaders)) {
    headers.set(key, value);
  }
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
};

const placeholder = (route, nextStep, env, extras = {}) =>
  json({
    ok: false,
    wired: false,
    route,
    message: "Placeholder route only. Real backend wiring is not implemented yet.",
    next_step: nextStep,
    required_env_vars: [
      "STRIPE_SECRET_KEY",
      "STRIPE_WEBHOOK_SECRET",
      "SITE_URL",
      "STRIPE_PRICE_ID",
      "PREMIUM_DATA_SOURCE",
    ],
    env_summary: envSummary(env),
    ...extras,
  });

const configError = (message, missing = []) =>
  json(
    {
      ok: false,
      status: "config_error",
      message,
      missing_env_vars: missing,
    },
    500
  );

const requestError = (message, details = null, status = 400) =>
  json(
    {
      ok: false,
      status: "request_error",
      message,
      details,
    },
    status
  );

const stripeError = (message, details = null, status = 502) =>
  json(
    {
      ok: false,
      status: "stripe_error",
      message,
      details,
    },
    status
  );

const unauthorizedError = (message, details = null) =>
  json(
    {
      ok: false,
      status: "unauthorized",
      message,
      details,
    },
    401
  );

const normalizeSiteUrl = (value) => String(value || "").replace(/\/+$/, "");
const isFiniteNumber = (value) => typeof value === "number" && Number.isFinite(value);
const isLikelyEmail = (value) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(value || "").trim());
const normalizeEmail = (value) => String(value || "").trim().toLowerCase();
const normalizePreferenceText = (value) =>
  String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
const parsePreferenceList = (value) =>
  Array.isArray(value)
    ? value.map((entry) => String(entry || "").trim()).filter(Boolean)
    : String(value || "")
        .split(",")
        .map((entry) => entry.trim())
        .filter(Boolean);
const marketFamilyLabel = (value) => {
  const key = String(value || "").toUpperCase();
  if (key === "FTR") return "FTR";
  if (key === "BTTS") return "BTTS";
  if (key === "OU25") return "OU25";
  return key || "INTEL";
};
const fixturePreferenceLabel = (fixture) =>
  `${String(fixture?.home_team || "Home").trim()} v ${String(fixture?.away_team || "Away").trim()}`;
const nowIso = () => new Date().toISOString();
const buildId = (prefix) => `${prefix}_${crypto.randomUUID()}`;
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();
const normalizeStylePreset = (value) => {
  const preset = String(value || "").trim().toLowerCase();
  if (["analyst", "disciplined_bettor", "tactical_reader", "researcher"].includes(preset)) {
    return preset;
  }
  return "disciplined_bettor";
};
const kickoffTimestamp = (value) => {
  const parsed = Date.parse(String(value || "").trim());
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
const isEliteFixture = (fixture) => {
  const tier = String(fixture?.confidence_tier || fixture?.premium_tier || "").toUpperCase();
  return tier === "ELITE";
};

const redirect = (location, status = 303, extraHeaders = {}) =>
  new Response(null, {
    status,
    headers: {
      location,
      ...extraHeaders,
    },
  });

const clearCookieHeader = (name) =>
  `${name}=; Path=/; HttpOnly; Secure; SameSite=None; Max-Age=0; Expires=Thu, 01 Jan 1970 00:00:00 GMT`;

const getCookieValue = (request, name) => {
  const cookieHeader = request.headers.get("cookie") || "";
  for (const fragment of cookieHeader.split(";")) {
    const [cookieName, value] = fragment.split("=", 2).map((part) => part.trim());
    if (cookieName === name && value) {
      return value;
    }
  }
  return null;
};

const getRequestIp = (request) =>
  String(request.headers.get("cf-connecting-ip") || request.headers.get("x-forwarded-for") || "")
    .split(",")[0]
    .trim();
const getUserAgentHint = (request) => String(request.headers.get("user-agent") || "").slice(0, 240);

const maskEmailHint = (email) => {
  const normalized = normalizeEmail(email);
  const [localPart, domain] = normalized.split("@");
  if (!localPart || !domain) {
    return "";
  }
  const visible = localPart.slice(0, 1) || "u";
  return `${visible}***@${domain}`;
};

const buildTelegramLinkCode = () =>
  Array.from(crypto.getRandomValues(new Uint8Array(6)))
    .map((value) => "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"[value % 32])
    .join("");

const TELEGRAM_LINK_CODE_PATTERN = /\b([A-HJ-NP-Z2-9]{6})\b/;

const getSessionSecret = (env) => String(env.AUTH_SESSION_SECRET || env.AUTH_MAGIC_LINK_SECRET || "").trim();

const getTelegramBotToken = (env) => String(env.TELEGRAM_BOT_TOKEN || "").trim();
const getTelegramWebhookSecret = (env) => String(env.TELEGRAM_WEBHOOK_SECRET || "").trim();

const bytesToBase64Url = (bytes) =>
  btoa(String.fromCharCode(...bytes))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");

const base64UrlToBytes = (value) => {
  const normalized = String(value || "").replace(/-/g, "+").replace(/_/g, "/");
  const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "=");
  try {
    return Uint8Array.from(atob(padded), (char) => char.charCodeAt(0));
  } catch {
    return null;
  }
};

const bytesToUtf8 = (bytes) => {
  try {
    return textDecoder.decode(bytes);
  } catch {
    return null;
  }
};

const buildOpaqueToken = (bytes = 32) => {
  const random = new Uint8Array(bytes);
  crypto.getRandomValues(random);
  return bytesToBase64Url(random);
};

const buildSessionCookieHeader = (token) =>
  `${AUTH_SESSION_COOKIE}=${token}; Path=/; HttpOnly; Secure; SameSite=None; Max-Age=${AUTH_SESSION_TTL_SECONDS}`;

const shouldClearSessionCookie = (status) =>
  [
    "invalid_session",
    "expired_session",
    "revoked_session",
    "session_not_found",
    "session_mismatch",
    "session_email_mismatch",
    "subscriber_state_missing",
    "inactive_subscription",
    "suspended_account",
  ].includes(String(status || ""));

const sessionFailureHeaders = (status) =>
  shouldClearSessionCookie(status) ? { "set-cookie": clearCookieHeader(AUTH_SESSION_COOKIE) } : {};

const blockedAccountHeaders = () => ({
  "set-cookie": clearCookieHeader(AUTH_SESSION_COOKIE),
});

const getInternalAdminSecret = (env) => String(env.INTERNAL_ADMIN_SECRET || "").trim();

const verifyInternalAdminRequest = (request, env) => {
  const configured = getInternalAdminSecret(env);
  if (!configured) {
    return {
      ok: false,
      status: "internal_admin_not_wired",
      message: "INTERNAL_ADMIN_SECRET is required for internal review routes.",
      code: 501,
    };
  }
  const supplied = String(
    request.headers.get("x-og-internal-admin") || request.headers.get("authorization")?.replace(/^Bearer\s+/i, "") || ""
  ).trim();
  if (!supplied || supplied !== configured) {
    return {
      ok: false,
      status: "internal_admin_unauthorized",
      message: "Internal admin authorization failed.",
      code: 401,
    };
  }
  return { ok: true };
};

const putKvJson = async (store, key, payload, options = {}) => {
  if (options.expirationTtl) {
    await store.put(key, JSON.stringify(payload), { expirationTtl: options.expirationTtl });
    return;
  }
  await store.put(key, JSON.stringify(payload));
};

const getKvJson = async (store, key) => {
  const raw = await store.get(key);
  if (!raw) {
    return null;
  }
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
};

const deleteKvKey = async (store, key) => {
  if (typeof store.delete === "function") {
    await store.delete(key);
    return;
  }
  await store.put(key, JSON.stringify({ consumed: true, consumed_at: new Date().toISOString() }), {
    expirationTtl: 60,
  });
};

const signText = async (secret, value) => {
  const key = await crypto.subtle.importKey(
    "raw",
    textEncoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signature = await crypto.subtle.sign("HMAC", key, textEncoder.encode(value));
  return new Uint8Array(signature);
};

const buildHashedOpaqueValue = async (secret, value, prefix = "hash") =>
  bytesToBase64Url(await signText(secret, `${prefix}:${String(value || "")}`));

const deriveDeviceLabel = (request) => {
  const userAgent = String(request.headers.get("user-agent") || "");
  const browser = /edg\//i.test(userAgent)
    ? "Edge"
    : /chrome\//i.test(userAgent)
      ? "Chrome"
      : /firefox\//i.test(userAgent)
        ? "Firefox"
        : /safari\//i.test(userAgent) && !/chrome\//i.test(userAgent)
          ? "Safari"
          : "Browser";
  const platform = /iphone|ipad|ios/i.test(userAgent)
    ? "iPhone"
    : /android/i.test(userAgent)
      ? "Android"
      : /mac os x|macintosh/i.test(userAgent)
        ? "Mac"
        : /windows/i.test(userAgent)
          ? "Windows"
          : /linux/i.test(userAgent)
            ? "Linux"
            : "device";
  return `${browser} on ${platform}`;
};

const buildSignedJsonToken = async (payload, secret) => {
  const payloadSegment = bytesToBase64Url(textEncoder.encode(JSON.stringify(payload)));
  const signatureBytes = await signText(secret, payloadSegment);
  return `${payloadSegment}.${bytesToBase64Url(signatureBytes)}`;
};

const parseSignedJsonToken = async (token, secret) => {
  const parts = String(token || "").split(".");
  if (parts.length !== 2 || !parts[0] || !parts[1]) {
    return { ok: false, status: "invalid_session", message: "Session token format is invalid." };
  }

  const payloadBytes = base64UrlToBytes(parts[0]);
  const signatureBytes = base64UrlToBytes(parts[1]);
  if (!payloadBytes || !signatureBytes) {
    return { ok: false, status: "invalid_session", message: "Session token could not be decoded." };
  }

  const expectedSignature = await signText(secret, parts[0]);
  if (!constantTimeEqual(expectedSignature, signatureBytes)) {
    return { ok: false, status: "invalid_session", message: "Session token signature verification failed." };
  }

  const payloadText = bytesToUtf8(payloadBytes);
  if (!payloadText) {
    return { ok: false, status: "invalid_session", message: "Session payload is unreadable." };
  }

  let payload;
  try {
    payload = JSON.parse(payloadText);
  } catch {
    return { ok: false, status: "invalid_session", message: "Session payload is not valid JSON." };
  }

  return { ok: true, payload };
};

const getSessionCookieToken = (request) => getCookieValue(request, AUTH_SESSION_COOKIE);

const hasTransitionalPremiumToken = (request) =>
  Boolean(getCookieValue(request, "og_premium_token")) ||
  /^Bearer\s+.+/i.test(request.headers.get("authorization") || "");

const checkAndRecordCooldown = async (store, key, cooldownSeconds) => {
  const now = Math.floor(Date.now() / 1000);
  const existing = await getKvJson(store, key);
  if (existing?.next_allowed_at && Number(existing.next_allowed_at) > now) {
    return { ok: false, retry_after: Number(existing.next_allowed_at) - now };
  }
  await putKvJson(
    store,
    key,
    {
      next_allowed_at: now + cooldownSeconds,
      updated_at: new Date().toISOString(),
    },
    { expirationTtl: cooldownSeconds }
  );
  return { ok: true };
};

const createMagicLinkRecord = (email, subscriberRecord) => {
  const now = Math.floor(Date.now() / 1000);
  return {
    email,
    customer_id: subscriberRecord.customer_id,
    subscription_id: subscriberRecord.subscription_id,
    issued_at: new Date(now * 1000).toISOString(),
    exp: now + AUTH_MAGIC_LINK_TTL_SECONDS,
  };
};

async function sendMagicLinkEmail(email, verifyUrl, env) {
  if (!env.RESEND_API_KEY || !env.AUTH_EMAIL_FROM) {
    return {
      ok: false,
      status: "auth_not_wired",
      message: "Transactional email delivery is not configured.",
      missing: [
        !env.RESEND_API_KEY ? "RESEND_API_KEY" : null,
        !env.AUTH_EMAIL_FROM ? "AUTH_EMAIL_FROM" : null,
      ].filter(Boolean),
    };
  }

  const response = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      authorization: `Bearer ${env.RESEND_API_KEY}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      from: env.AUTH_EMAIL_FROM,
      to: [email],
      subject: "Your Odds Genius sign-in link",
      text: `Sign in to Odds Genius: ${verifyUrl}\n\nThis link expires in 15 minutes.`,
      html: `<p>Sign in to <strong>Odds Genius</strong>.</p><p><a href="${verifyUrl}">Open your sign-in link</a></p><p>This link expires in 15 minutes.</p>`,
    }),
  });

  if (!response.ok) {
    const details = await response.text();
    return {
      ok: false,
      status: "email_send_failed",
      message: "Unable to send sign-in link right now.",
      details,
    };
  }

  return { ok: true };
}

const resolvePremiumSourceUrl = (request, env) => {
  const source = String(env.PREMIUM_DATA_SOURCE || "").trim();
  if (!source) {
    return {
      ok: false,
      status: "premium_source_missing",
      message: "PREMIUM_DATA_SOURCE is not configured.",
      recommendation: "Point PREMIUM_DATA_SOURCE at a protected JSON endpoint or same-origin asset path.",
      http_status: 500,
    };
  }

  if (/^https?:\/\//i.test(source)) {
    return { ok: true, targetUrl: source };
  }

  if (source.startsWith("/")) {
    return { ok: true, targetUrl: new URL(source, request.url).toString() };
  }

  return {
    ok: false,
    status: "premium_source_unsupported",
    message: "PREMIUM_DATA_SOURCE must currently be an absolute URL or same-origin path.",
    recommendation: "For production, use KV, R2, or a protected static asset fetch strategy rather than local filesystem paths.",
    http_status: 501,
  };
};

const buildPremiumCacheKey = (targetUrl) => {
  const cacheUrl = new URL("https://og-premium-cache.invalid/premium-board");
  cacheUrl.searchParams.set("v", PREMIUM_CACHE_VERSION);
  cacheUrl.searchParams.set("source", targetUrl);
  return new Request(cacheUrl.toString(), { method: "GET" });
};

const getPremiumCache = () => globalThis.caches?.default || null;
const getApiSportsFootballKey = (env) => String(env.API_SPORTS_FOOTBALL_KEY || "").trim();

const buildWidgetFootballCacheKey = (endpoint, params = {}) => {
  const cacheUrl = new URL(`https://og-widget-cache.invalid/${endpoint}`);
  cacheUrl.searchParams.set("v", WIDGET_FOOTBALL_PROXY_VERSION);
  Object.entries(params)
    .filter(([, value]) => value != null && value !== "")
    .sort(([left], [right]) => left.localeCompare(right))
    .forEach(([key, value]) => {
      cacheUrl.searchParams.set(key, String(value));
    });
  return new Request(cacheUrl.toString(), { method: "GET" });
};

const positiveIntegerParam = (value) => {
  const text = String(value || "").trim();
  if (!/^\d+$/.test(text)) {
    return null;
  }
  const numeric = Number(text);
  if (!Number.isInteger(numeric) || numeric <= 0) {
    return null;
  }
  return numeric;
};

const optionalPositiveIntegerParam = (value) => {
  const text = String(value || "").trim();
  if (!text) {
    return null;
  }
  return positiveIntegerParam(text);
};

const normalizeWidgetLookupText = (value) =>
  normalizePreferenceText(value)
    .replace(/\b(fc|cf|ac|sc|afc|club)\b/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const widgetTeamNamesMatch = (left, right) => {
  const a = normalizeWidgetLookupText(left);
  const b = normalizeWidgetLookupText(right);
  if (!a || !b) {
    return false;
  }
  return a === b || a.includes(b) || b.includes(a);
};

const shiftIsoDate = (dateText, deltaDays) => {
  const parsed = new Date(`${dateText}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) {
    return "";
  }
  parsed.setUTCDate(parsed.getUTCDate() + deltaDays);
  return parsed.toISOString().slice(0, 10);
};

const proxyWidgetStandingsResponse = async (request, env) => {
  const footballKey = getApiSportsFootballKey(env);
  if (!footballKey) {
    return configError("API_SPORTS_FOOTBALL_KEY is required for the widget standings prototype.", [
      "API_SPORTS_FOOTBALL_KEY",
    ]);
  }
  if (request.method !== "GET") {
    return methodNotAllowed("GET");
  }
  const requestUrl = new URL(request.url);
  const league = positiveIntegerParam(requestUrl.searchParams.get("league"));
  const season = positiveIntegerParam(requestUrl.searchParams.get("season"));
  if (!league || !season) {
    return requestError("Widget standings prototype requires numeric league and season query parameters.");
  }

  const cache = getPremiumCache();
  const cacheKey = buildWidgetFootballCacheKey("football-standings", { league, season });
  if (cache) {
    const cached = await cache.match(cacheKey);
    if (cached) {
      return new Response(cached.body, {
        status: cached.status,
        statusText: cached.statusText,
        headers: new Headers(cached.headers),
      });
    }
  }

  const candidateSeasons = Array.from(
    new Set([season, season - 1, season - 2, season - 3].filter((value) => Number.isInteger(value) && value > 0))
  );
  let selectedSeason = season;
  let selectedStatus = 502;
  let selectedStatusText = "Bad Gateway";
  let selectedBody = "";
  let selectedPayload = null;
  let selectedContentType = "application/json; charset=utf-8";

  for (const candidateSeason of candidateSeasons) {
    const upstreamUrl = new URL("https://v3.football.api-sports.io/standings");
    upstreamUrl.searchParams.set("league", String(league));
    upstreamUrl.searchParams.set("season", String(candidateSeason));
    const upstreamResponse = await fetch(upstreamUrl.toString(), {
      method: "GET",
      headers: {
        accept: "application/json",
        "x-apisports-key": footballKey,
      },
    });
    const bodyText = await upstreamResponse.text();
    selectedSeason = candidateSeason;
    selectedStatus = upstreamResponse.status;
    selectedStatusText = upstreamResponse.statusText;
    selectedBody = bodyText;
    selectedContentType = upstreamResponse.headers.get("content-type") || selectedContentType;
    if (!upstreamResponse.ok) {
      continue;
    }
    try {
      const payload = JSON.parse(bodyText);
      selectedPayload = payload;
      const responseRows = Array.isArray(payload?.response) ? payload.response : [];
      if (responseRows.length) {
        break;
      }
    } catch (error) {
      selectedPayload = null;
      break;
    }
  }

  const headers = new Headers();
  headers.set("content-type", selectedContentType);
  headers.set("x-og-widget-proxy", "football-standings");
  headers.set("x-og-widget-standings-requested-season", String(season));
  headers.set("x-og-widget-standings-selected-season", String(selectedSeason));
  headers.set("cache-control", `public, max-age=${WIDGET_FOOTBALL_STANDINGS_CACHE_TTL_SECONDS}`);

  const response = new Response(selectedBody, {
    status: selectedStatus,
    statusText: selectedStatusText,
    headers,
  });

  if (selectedStatus >= 200 && selectedStatus < 300 && cache) {
    await cache.put(cacheKey, response.clone());
  }

  return response;
};

const proxyGenericWidgetFootballResponse = async (request, env) => {
  const footballKey = getApiSportsFootballKey(env);
  if (!footballKey) {
    return configError("API_SPORTS_FOOTBALL_KEY is required for the football widget proxy.", [
      "API_SPORTS_FOOTBALL_KEY",
    ]);
  }
  if (request.method !== "GET") {
    return methodNotAllowed("GET");
  }
  const requestUrl = new URL(request.url);
  const pathname = requestUrl.pathname || "";
  const prefix = "/api/widgets/football/";
  const widgetPath = pathname.startsWith(prefix) ? pathname.slice(prefix.length) : "";
  if (!widgetPath) {
    return requestError("Football widget proxy requires an upstream endpoint path.");
  }

  const cache = getPremiumCache();
  const cacheKey = buildWidgetFootballCacheKey("football-generic", {
    path: widgetPath,
    query: requestUrl.searchParams.toString(),
  });
  if (cache) {
    const cached = await cache.match(cacheKey);
    if (cached) {
      return new Response(cached.body, {
        status: cached.status,
        statusText: cached.statusText,
        headers: new Headers(cached.headers),
      });
    }
  }

  const upstreamUrl = new URL(`https://v3.football.api-sports.io/${widgetPath}`);
  requestUrl.searchParams.forEach((value, key) => {
    upstreamUrl.searchParams.append(key, value);
  });
  const upstreamResponse = await fetch(upstreamUrl.toString(), {
    method: "GET",
    headers: {
      accept: request.headers.get("accept") || "application/json, text/plain, */*",
      "x-apisports-key": footballKey,
    },
  });
  const bodyText = await upstreamResponse.text();
  const headers = new Headers();
  headers.set("content-type", upstreamResponse.headers.get("content-type") || "application/json; charset=utf-8");
  headers.set("x-og-widget-proxy", `football-${widgetPath}`);
  headers.set("cache-control", "public, max-age=300");

  const response = new Response(bodyText, {
    status: upstreamResponse.status,
    statusText: upstreamResponse.statusText,
    headers,
  });
  if (upstreamResponse.ok && cache) {
    await cache.put(cacheKey, response.clone());
  }
  return response;
};

const lookupWidgetFixtureResponse = async (request, env) => {
  const footballKey = getApiSportsFootballKey(env);
  if (!footballKey) {
    return configError("API_SPORTS_FOOTBALL_KEY is required for the widget fixture lookup prototype.", [
      "API_SPORTS_FOOTBALL_KEY",
    ]);
  }
  if (request.method !== "GET") {
    return methodNotAllowed("GET");
  }
  const requestUrl = new URL(request.url);
  const date = String(requestUrl.searchParams.get("date") || "").trim();
  const home = String(requestUrl.searchParams.get("home") || "").trim();
  const away = String(requestUrl.searchParams.get("away") || "").trim();
  const homeTeamId = optionalPositiveIntegerParam(requestUrl.searchParams.get("home_team_id"));
  const awayTeamId = optionalPositiveIntegerParam(requestUrl.searchParams.get("away_team_id"));
  if (!date || !home || !away) {
    return requestError(
      "Widget fixture lookup requires date, home, and away query parameters."
    );
  }

  const cache = getPremiumCache();
  const cacheKey = buildWidgetFootballCacheKey("football-fixture-lookup", {
    date,
    home,
    away,
    home_team_id: homeTeamId,
    away_team_id: awayTeamId,
  });
  if (cache) {
    const cached = await cache.match(cacheKey);
    if (cached) {
      return new Response(cached.body, {
        status: cached.status,
        statusText: cached.statusText,
        headers: new Headers(cached.headers),
      });
    }
  }

  const candidateDates = Array.from(new Set([date, shiftIsoDate(date, -1), shiftIsoDate(date, 1)].filter(Boolean)));
  let matched = null;
  let upstreamFailure = null;
  for (const candidateDate of candidateDates) {
    const upstreamUrl = new URL("https://v3.football.api-sports.io/fixtures");
    upstreamUrl.searchParams.set("date", candidateDate);
    const upstreamResponse = await fetch(upstreamUrl.toString(), {
      method: "GET",
      headers: {
        accept: "application/json",
        "x-apisports-key": footballKey,
      },
    });
    let payload;
    try {
      payload = await upstreamResponse.json();
    } catch (error) {
      return requestError("Widget fixture lookup could not parse the upstream fixture response.", error.message, 502);
    }
    if (!upstreamResponse.ok) {
      upstreamFailure = {
        ok: false,
        status: "widget_fixture_lookup_upstream_failed",
        message: "The upstream fixture lookup request failed.",
        upstream_status: upstreamResponse.status,
        upstream_payload: payload,
      };
      continue;
    }
    const fixtures = Array.isArray(payload?.response) ? payload.response : [];
    matched = fixtures.find((entry) => {
      const upstreamHomeId = Number(entry?.teams?.home?.id || 0);
      const upstreamAwayId = Number(entry?.teams?.away?.id || 0);
      if (
        homeTeamId &&
        awayTeamId &&
        upstreamHomeId === homeTeamId &&
        upstreamAwayId === awayTeamId
      ) {
        return true;
      }
      const fixtureHome = entry?.teams?.home?.name || "";
      const fixtureAway = entry?.teams?.away?.name || "";
      return widgetTeamNamesMatch(fixtureHome, home) && widgetTeamNamesMatch(fixtureAway, away);
    });
    if (matched?.fixture?.id) {
      break;
    }
  }
  if (!matched?.fixture?.id && upstreamFailure) {
    return json(upstreamFailure, 502);
  }
  if (!matched?.fixture?.id) {
    return json(
      {
        ok: false,
        status: "widget_fixture_lookup_not_found",
        message: "No matching upstream fixture id was found for this fixture yet.",
      },
      404
    );
  }

  const response = json(
      {
        ok: true,
        status: "widget_fixture_lookup_ready",
        fixture_id: matched.fixture.id,
        fixture_status: matched.fixture?.status?.short || "",
        league_id: matched.league?.id || null,
        season: matched.league?.season || null,
        home_team: matched.teams?.home?.name || home,
        away_team: matched.teams?.away?.name || away,
      },
    200,
    {
      "cache-control": "public, max-age=300",
      "x-og-widget-proxy": "football-fixture-lookup",
    }
  );
  if (cache) {
    await cache.put(cacheKey, response.clone());
  }
  return response;
};

const parseStripeSignatureHeader = (headerValue) => {
  const parsed = {
    timestamp: null,
    signatures: [],
  };

  for (const part of String(headerValue || "").split(",")) {
    const [key, value] = part.split("=", 2).map((item) => item.trim());
    if (!key || !value) {
      continue;
    }
    if (key === "t") {
      parsed.timestamp = value;
    }
    if (key === "v1") {
      parsed.signatures.push(value);
    }
  }

  return parsed;
};

const hexToBytes = (hex) => {
  if (!/^[0-9a-f]+$/i.test(hex) || hex.length % 2 !== 0) {
    return null;
  }
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    bytes[i / 2] = Number.parseInt(hex.slice(i, i + 2), 16);
  }
  return bytes;
};

const constantTimeEqual = (left, right) => {
  if (!left || !right || left.length !== right.length) {
    return false;
  }
  let result = 0;
  for (let i = 0; i < left.length; i += 1) {
    result |= left[i] ^ right[i];
  }
  return result === 0;
};

const computeStripeWebhookSignature = async (secret, payload) => {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signature = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(payload));
  return new Uint8Array(signature);
};

async function verifyStripeWebhookSignature(request, env, rawBody) {
  if (!env.STRIPE_WEBHOOK_SECRET) {
    return {
      ok: false,
      response: configError("Missing required Stripe webhook environment variables.", ["STRIPE_WEBHOOK_SECRET"]),
    };
  }

  const headerValue = request.headers.get("stripe-signature");
  if (!headerValue) {
    return {
      ok: false,
      response: unauthorizedError("Missing Stripe signature header."),
    };
  }

  const parsed = parseStripeSignatureHeader(headerValue);
  if (!parsed.timestamp || !parsed.signatures.length) {
    return {
      ok: false,
      response: unauthorizedError("Stripe signature header is malformed."),
    };
  }

  const timestampSeconds = Number(parsed.timestamp);
  if (!Number.isFinite(timestampSeconds)) {
    return {
      ok: false,
      response: unauthorizedError("Stripe signature timestamp is invalid."),
    };
  }

  const ageSeconds = Math.abs(Math.floor(Date.now() / 1000) - timestampSeconds);
  if (ageSeconds > STRIPE_WEBHOOK_TOLERANCE_SECONDS) {
    return {
      ok: false,
      response: unauthorizedError("Stripe webhook timestamp is outside the allowed tolerance window."),
    };
  }

  const signedPayload = `${parsed.timestamp}.${rawBody}`;
  const expected = await computeStripeWebhookSignature(env.STRIPE_WEBHOOK_SECRET, signedPayload);
  const matched = parsed.signatures.some((candidate) => {
    const bytes = hexToBytes(candidate);
    return constantTimeEqual(expected, bytes);
  });

  if (!matched) {
    return {
      ok: false,
      response: unauthorizedError("Stripe webhook signature verification failed."),
    };
  }

  return { ok: true };
}

async function createCheckoutSession(request, env) {
  const missing = ["STRIPE_SECRET_KEY", "SITE_URL", "STRIPE_PRICE_ID"].filter((key) => !env[key]);
  if (missing.length) {
    return configError("Missing required Stripe checkout environment variables.", missing);
  }

  let payload = {};
  const contentType = request.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    try {
      payload = await request.json();
    } catch (error) {
      return requestError("Checkout request body must be valid JSON.", error.message);
    }
  }

  const siteUrl = normalizeSiteUrl(env.SITE_URL);
  if (!/^https?:\/\//.test(siteUrl)) {
    return configError("SITE_URL must be a full http or https URL.", ["SITE_URL"]);
  }

  const body = new URLSearchParams();
  body.set("mode", "subscription");
  body.set("success_url", `${siteUrl}/account.html?checkout=success`);
  body.set("cancel_url", `${siteUrl}/pricing.html?checkout=cancelled`);
  body.set("line_items[0][price]", env.STRIPE_PRICE_ID);
  body.set("line_items[0][quantity]", "1");
  body.set("allow_promotion_codes", "true");

  if (typeof payload.email === "string" && payload.email.trim()) {
    body.set("customer_email", payload.email.trim());
  }

  if (typeof payload.reference === "string" && payload.reference.trim()) {
    body.set("client_reference_id", payload.reference.trim().slice(0, 200));
  }

  const stripeResponse = await fetch("https://api.stripe.com/v1/checkout/sessions", {
    method: "POST",
    headers: {
      authorization: `Bearer ${env.STRIPE_SECRET_KEY}`,
      "content-type": "application/x-www-form-urlencoded",
    },
    body: body.toString(),
  });

  let stripePayload = null;
  try {
    stripePayload = await stripeResponse.json();
  } catch (error) {
    return stripeError("Stripe checkout response was not valid JSON.", error.message);
  }

  if (!stripeResponse.ok) {
    return stripeError(
      "Stripe checkout session creation failed.",
      stripePayload?.error?.message || stripePayload?.error || stripePayload,
      stripeResponse.status
    );
  }

  if (!stripePayload?.url) {
    return stripeError("Stripe checkout session succeeded but no redirect URL was returned.");
  }

  return json({
    ok: true,
    url: stripePayload.url,
  });
}

async function handleStripeWebhook(request, env) {
  const store = getSubscriberStateStore(env);
  if (!store) {
    return configError("Missing required subscriber state binding.", ["SUBSCRIBER_STATE"]);
  }

  const rawBody = await request.text();
  const verification = await verifyStripeWebhookSignature(request, env, rawBody);
  if (!verification.ok) {
    return verification.response;
  }

  let event;
  try {
    event = JSON.parse(rawBody);
  } catch (error) {
    return requestError("Stripe webhook body must be valid JSON.", error.message);
  }

  const supportedTypes = new Set([
    "checkout.session.completed",
    "customer.subscription.created",
    "customer.subscription.updated",
    "customer.subscription.deleted",
  ]);

  if (!supportedTypes.has(event?.type)) {
    return json({
      ok: true,
      received: true,
      ignored: true,
      event_type: event?.type || null,
      message: "Webhook received but event type is not handled by this scaffold.",
    });
  }

  const record = buildSubscriberRecord(event);
  if (!record) {
    return requestError("Webhook event could not be converted into a subscriber state record.", event?.type);
  }

  try {
    const persisted = await persistSubscriberRecord(store, record);
    const accountDb = getAccountDb(env);
    if (accountDb) {
      try {
        await mirrorSubscriptionFromRecord(accountDb, record);
      } catch (error) {
        return stripeError("Webhook persistence succeeded but D1 subscription mirroring failed.", error.message, 500);
      }
    }
    return json({
      ok: true,
      received: true,
      stored: true,
      mirrored_to_d1: Boolean(accountDb),
      event_type: event.type,
      record: {
        customer_id: record.customer_id,
        subscription_id: record.subscription_id,
        status: record.status,
        price_id: record.price_id,
        current_period_end: record.current_period_end,
        updated_at: record.updated_at,
      },
      keys: persisted,
    });
  } catch (error) {
    return stripeError("Webhook verification passed but subscriber state persistence failed.", error.message, 500);
  }
}

async function handlePremiumPredictions(request, env) {
  const access = await resolvePremiumAccess(request, env);
  if (!access.ok) {
    return json(
      {
        ok: false,
        status: access.status,
        message: access.message,
        recommendation: access.recommendation,
        route: "/api/premium/predictions",
        locked: true,
        data_note: "Premium predictions remain unavailable until verified session-backed or token-backed entitlement is live.",
      },
      401,
      sessionFailureHeaders(access.status)
    );
  }

  const loaded = await loadPremiumPredictions(request, env);
  if (!loaded.ok) {
    return json(
      {
        ok: false,
        status: loaded.status,
        message: loaded.message,
        route: "/api/premium/predictions",
        locked: true,
        recommendation: loaded.recommendation,
      },
      loaded.http_status || 500
    );
  }

  return json(
    {
      ok: true,
      generated_at: loaded.generated_at,
      subscriber_customer_id: access.customer_id,
      auth_mode: access.auth_mode || "token",
      count: loaded.count,
      predictions: loaded.rows,
    },
    200,
    {
      "x-og-premium-cache": loaded.cache_status || "bypass",
    }
  );
}

async function handlePremiumTokenIssue(request, env) {
  let payload;
  try {
    payload = await request.json();
  } catch (error) {
    return requestError("Premium token request body must be valid JSON.", error.message);
  }

  const issued = await issuePremiumToken(payload, env);
  if (!issued.ok) {
    return json(
      {
        ok: false,
        status: issued.status,
        message: issued.message,
        route: "/api/premium/token",
        warning:
          "Developer/test scaffolding only. Final token issuance should require magic-link email verification or authenticated session checks.",
      },
      401
    );
  }

  return json({
    ok: true,
    token: issued.token,
    expires_at: issued.expires_at,
    warning:
      "Developer/test scaffolding only. Final token issuance should require magic-link email verification or authenticated session checks.",
  });
}

async function recordSessionSpreadRiskIfNeeded(accountDb, accountState, request, context = {}) {
  if (!accountDb || !accountState?.user?.id) {
    return null;
  }

  const userId = accountState.user.id;
  const riskState = await ensureAccountRiskState(accountDb, userId);
  const activeSessions = await listActiveAccountSessionsByUser(accountDb, userId);
  const distinctDeviceLabels = Array.from(
    new Set(
      activeSessions
        .map((session) => String(session.device_label || "").trim())
        .filter(Boolean)
    )
  );
  const activeSessionCount = activeSessions.length;
  const deviceCount = distinctDeviceLabels.length;

  if (activeSessionCount < 3 || deviceCount < 3) {
    return riskState;
  }

  const openFlag = await getOpenAccountRiskFlagByType(accountDb, userId, "shared_access_pattern");
  if (!openFlag?.id) {
    await createAccountRiskFlag(accountDb, {
      user_id: userId,
      flag_type: "shared_access_pattern",
      severity: activeSessionCount >= 4 ? "high" : "medium",
      source: "session_heuristic",
      summary:
        activeSessionCount >= 4
          ? "Account session spread is unusually high across multiple devices."
          : "Account is active across several recent device sessions.",
      evidence: {
        active_session_count: activeSessionCount,
        distinct_device_count: deviceCount,
        device_labels: distinctDeviceLabels.slice(0, 6),
        trigger: context.trigger || "session_verify",
        session_id: context.session_id || null,
      },
    });
    await addAccountAdminNote(accountDb, {
      user_id: userId,
      note_type: "risk_note",
      visibility: "internal",
      author_id: "system:risk",
      content:
        activeSessionCount >= 4
          ? "Automatic review note: account session spread reached a high threshold across multiple device labels."
          : "Automatic review note: account session spread reached a watch threshold across multiple device labels.",
    });
  }

  await updateAccountRiskState(accountDb, userId, {
    risk_level: activeSessionCount >= 4 ? "high" : "medium",
    review_status: activeSessionCount >= 4 ? "manual_review" : "watch",
    risk_score: Math.max(Number(riskState?.risk_score || 0), activeSessionCount >= 4 ? 45 : 25),
    last_risk_event_at: nowIso(),
  });

  try {
    await recordAuthEvent(accountDb, {
      user_id: userId,
      email_normalized: accountState.user.email_normalized || "",
      event_type: "account_risk_flagged_shared_access_pattern",
      ip_hint: getRequestIp(request) || null,
      user_agent_hint: getUserAgentHint(request),
      metadata: {
        active_session_count: activeSessionCount,
        distinct_device_count: deviceCount,
        trigger: context.trigger || "session_verify",
        session_id: context.session_id || null,
      },
    });
  } catch {
    // Best-effort audit trail only.
  }

  return getAccountRiskState(accountDb, userId);
}

const summarizeInternalAccount = async (accountDb, userId) => {
  const account = await getAccountStateByUserId(accountDb, userId);
  if (!account?.user?.id) {
    return null;
  }
  const riskState = (await getAccountRiskState(accountDb, userId)) || (await ensureAccountRiskState(accountDb, userId));
  const sessions = await listAccountSessionsByUser(accountDb, userId, { limit: 12 });
  const openFlags = await listAccountRiskFlagsByUser(accountDb, userId, { status: "open", limit: 20 });
  const activeSessions = sessions.filter(
    (session) =>
      !(Number(session.is_revoked || 0) === 1 || session.revoked_at) &&
      Date.parse(String(session.expires_at || "")) > Date.now()
  );
  const deviceLabels = Array.from(
    new Set(
      activeSessions
        .map((session) => String(session.device_label || "").trim())
        .filter(Boolean)
    )
  );
  const ipHashes = Array.from(
    new Set(
      activeSessions
        .map((session) => String(session.ip_hash || "").trim())
        .filter(Boolean)
    )
  );
  const primarySession = sessions.find((session) => Number(session.is_primary || 0) === 1) || null;

  return {
    user: account.user,
    subscription: account.subscription,
    telegram_link: account.telegram_link,
    risk_state: riskState,
    session_summary: {
      active_session_count: activeSessions.length,
      recent_session_count: sessions.length,
      distinct_device_count: deviceLabels.length,
      distinct_ip_hash_count: ipHashes.length,
      primary_device_label: primarySession?.device_label || null,
    },
    open_flags_count: openFlags.length,
  };
};

const buildInternalTimeline = async (accountDb, userId) => {
  const [authEvents, sessions, flags, notes] = await Promise.all([
    listAuthEventsByUser(accountDb, userId, { limit: 40 }),
    listAccountSessionsByUser(accountDb, userId, { limit: 20 }),
    listAccountRiskFlagsByUser(accountDb, userId, { limit: 20 }),
    listAccountAdminNotesByUser(accountDb, userId, { limit: 20 }),
  ]);

  const items = [];

  for (const event of authEvents) {
    let metadata = {};
    try {
      metadata = JSON.parse(event.metadata_json || "{}");
    } catch {
      metadata = {};
    }
    items.push({
      source_type: "auth_event",
      id: event.id,
      timestamp: event.created_at,
      event_type: event.event_type,
      summary: event.event_type.replaceAll("_", " "),
      device_label: metadata.target_device_label || metadata.primary_device_label || null,
      ip_hint: event.ip_hint || null,
      user_agent_hint: event.user_agent_hint || null,
      metadata,
    });
  }

  for (const session of sessions) {
    items.push({
      source_type: "session",
      id: session.id,
      timestamp: session.last_seen_at || session.issued_at,
      event_type: Number(session.is_revoked || 0) === 1 || session.revoked_at ? "session_revoked_state" : "session_seen",
      summary:
        Number(session.is_revoked || 0) === 1 || session.revoked_at
          ? "Session revoked"
          : "Session active",
      device_label: session.device_label || null,
      ip_hint: session.ip_hash || null,
      user_agent_hint: session.user_agent_hash || null,
      metadata: {
        session_kind: session.session_kind || "browser",
        expires_at: session.expires_at || null,
        revoked_at: session.revoked_at || null,
      },
    });
  }

  for (const flag of flags) {
    let evidence = {};
    try {
      evidence = JSON.parse(flag.evidence_json || "{}");
    } catch {
      evidence = {};
    }
    items.push({
      source_type: "risk_flag",
      id: flag.id,
      timestamp: flag.opened_at,
      event_type: `risk_flag_${flag.flag_type}`,
      summary: flag.summary,
      device_label: null,
      ip_hint: null,
      user_agent_hint: null,
      metadata: {
        severity: flag.severity,
        flag_status: flag.flag_status,
        source: flag.source,
        evidence,
      },
    });
  }

  for (const note of notes) {
    items.push({
      source_type: "admin_note",
      id: note.id,
      timestamp: note.created_at,
      event_type: `admin_note_${note.note_type}`,
      summary: note.content,
      device_label: null,
      ip_hint: null,
      user_agent_hint: null,
      metadata: {
        note_type: note.note_type,
        author_id: note.author_id || null,
      },
    });
  }

  return items.sort((a, b) => String(b.timestamp || "").localeCompare(String(a.timestamp || ""))).slice(0, 80);
};

async function verifySessionAccess(request, env) {
  const sessionToken = getSessionCookieToken(request);
  if (!sessionToken) {
    return {
      ok: false,
      status: "missing_session",
      message: "Premium session cookie is missing.",
      recommendation: "Verify email to establish a premium session on this device.",
    };
  }

  const sessionSecret = getSessionSecret(env);
  if (!sessionSecret) {
    return {
      ok: false,
      status: "auth_not_wired",
      message: "Session verification secret is not configured.",
      recommendation: "Set AUTH_MAGIC_LINK_SECRET and optionally AUTH_SESSION_SECRET.",
    };
  }

  const parsed = await parseSignedJsonToken(sessionToken, sessionSecret);
  if (!parsed.ok) {
    return {
      ok: false,
      status: parsed.status,
      message: parsed.message,
      recommendation: "Verify email again to receive a fresh sign-in session.",
    };
  }

  const email = normalizeEmail(parsed.payload?.email);
  const customerId = String(parsed.payload?.customer_id || "").trim();
  const subscriptionId = String(parsed.payload?.subscription_id || "").trim();
  const exp = Number(parsed.payload?.exp);
  const sessionId = String(parsed.payload?.sid || "").trim();
  const opaqueSessionToken = String(parsed.payload?.st || "").trim();

  if (!email || !customerId || !subscriptionId || !Number.isFinite(exp)) {
    return {
      ok: false,
      status: "invalid_session",
      message: "Premium session payload is incomplete.",
      recommendation: "Verify email again to receive a fresh sign-in session.",
    };
  }

  if (exp <= Math.floor(Date.now() / 1000)) {
    return {
      ok: false,
      status: "expired_session",
      message: "Premium session has expired.",
      recommendation: "Request a fresh sign-in link to restore premium access.",
    };
  }

  const accountDb = getAccountDb(env);
  if (accountDb && sessionId && opaqueSessionToken) {
    const sessionRow = await getAccountSessionById(accountDb, sessionId);
    if (!sessionRow?.id) {
      return {
        ok: false,
        status: "session_not_found",
        message: "Tracked account session was not found.",
        recommendation: "Verify email again to establish a fresh session on this device.",
      };
    }

    if (Number(sessionRow.is_revoked || 0) === 1 || sessionRow.revoked_at) {
      return {
        ok: false,
        status: "revoked_session",
        message: "This account session has been revoked.",
        recommendation: "Verify email again to restore access on this device.",
      };
    }

    if (Date.parse(String(sessionRow.expires_at || "")) <= Date.now()) {
      await revokeAccountSession(accountDb, sessionRow.id, "expired");
      return {
        ok: false,
        status: "expired_session",
        message: "Premium session has expired.",
        recommendation: "Request a fresh sign-in link to restore premium access.",
      };
    }

    const expectedTokenHash = await buildHashedOpaqueValue(sessionSecret, opaqueSessionToken, "session");
    if (expectedTokenHash !== String(sessionRow.session_token_hash || "")) {
      return {
        ok: false,
        status: "invalid_session",
        message: "Tracked account session token did not validate.",
        recommendation: "Verify email again to receive a fresh sign-in session.",
      };
    }

    await touchAccountSessionSeen(accountDb, sessionRow.id, nowIso());

    const accountState = await getAccountStateByEmail(accountDb, email);
    const riskState = accountState?.user?.id ? await getAccountRiskState(accountDb, accountState.user.id) : null;
    const accountStatus = String(riskState?.account_status || "").trim().toLowerCase();
    if (accountStatus === "suspended") {
      return {
        ok: false,
        status: "suspended_account",
        message: "This account is currently suspended.",
        recommendation: "Contact support if you think this is an error.",
      };
    }
  }

  const store = getSubscriberStateStore(env);
  if (!store) {
    return {
      ok: false,
      status: "state_binding_missing",
      message: "Subscriber state binding is unavailable.",
      recommendation: "Bind SUBSCRIBER_STATE before enabling premium sessions.",
    };
  }

  let record;
  try {
    record = await loadSubscriberRecordBySubscriptionId(store, subscriptionId);
  } catch (error) {
    return {
      ok: false,
      status: "state_lookup_failed",
      message: "Subscriber state lookup failed.",
      recommendation: error.message,
    };
  }

  if (!record) {
    return {
      ok: false,
      status: "subscriber_state_missing",
      message: "No subscriber state record was found for this session.",
      recommendation: "Wait for Stripe webhook persistence or verify again later.",
    };
  }

  if (record.customer_id !== customerId || record.subscription_id !== subscriptionId) {
    return {
      ok: false,
      status: "session_mismatch",
      message: "Premium session does not match current subscriber state.",
      recommendation: "Verify email again to refresh the session.",
    };
  }

  if (normalizeEmail(record.email) && normalizeEmail(record.email) !== email) {
    return {
      ok: false,
      status: "session_email_mismatch",
      message: "Premium session email does not match subscriber state.",
      recommendation: "Verify email again using the subscriber email address.",
    };
  }

  if (!["active", "trialing"].includes(String(record.status || ""))) {
    return {
      ok: false,
      status: "inactive_subscription",
      message: "Subscriber state is not active for premium delivery.",
      recommendation: "Require active or trialing status before granting premium access.",
    };
  }

  return {
    ok: true,
    auth_mode: "session",
    email,
    email_hint: maskEmailHint(email),
    customer_id: record.customer_id,
    subscription_id: record.subscription_id,
    subscription_status: record.status,
    session_id: sessionId || null,
  };
}

async function resolvePremiumAccess(request, env) {
  const sessionAccess = await verifySessionAccess(request, env);
  if (sessionAccess.ok) {
    return sessionAccess;
  }

  if (hasTransitionalPremiumToken(request)) {
    const tokenAccess = await verifyPremiumAccess(request, env);
    if (tokenAccess.ok) {
      return {
        ...tokenAccess,
        auth_mode: "transitional_token",
      };
    }
    return tokenAccess;
  }

  if (sessionAccess.status && sessionAccess.status !== "missing_session") {
    return sessionAccess;
  }

  return verifyPremiumAccess(request, env);
}

async function handleMagicLinkRequest(request, env) {
  let payload;
  try {
    payload = await request.json();
  } catch (error) {
    return requestError("Magic-link request body must be valid JSON.", error.message);
  }

  const email = typeof payload?.email === "string" ? payload.email.trim().toLowerCase() : "";
  if (!email || !isLikelyEmail(email)) {
    return requestError("A valid email address is required.");
  }

  const store = getSubscriberStateStore(env);
  if (!store) {
    return configError("Missing required subscriber state binding.", ["SUBSCRIBER_STATE"]);
  }

  const missingAuthConfig = [
    !env.AUTH_MAGIC_LINK_SECRET ? "AUTH_MAGIC_LINK_SECRET" : null,
    !env.RESEND_API_KEY ? "RESEND_API_KEY" : null,
    !env.AUTH_EMAIL_FROM ? "AUTH_EMAIL_FROM" : null,
  ].filter(Boolean);
  if (missingAuthConfig.length) {
    return json(
      {
        ok: false,
        status: "auth_not_wired",
        message: "Magic-link email delivery is not configured yet.",
        route: "/api/auth/magic-link/request",
        next_step: "Configure AUTH_MAGIC_LINK_SECRET, RESEND_API_KEY, and AUTH_EMAIL_FROM before enabling public auth.",
        missing_env_vars: missingAuthConfig,
      },
      501
    );
  }

  const normalizedEmail = normalizeEmail(email);
  const requestIp = getRequestIp(request) || "unknown";
  const ipLimit = await checkAndRecordCooldown(
    store,
    `${AUTH_RATE_LIMIT_KEY_PREFIX}ip:${requestIp}`,
    AUTH_REQUEST_IP_COOLDOWN_SECONDS
  );
  if (!ipLimit.ok) {
    return json(
      {
        ok: false,
        status: "rate_limited",
        message: "Too many requests. Try again later.",
      },
      429,
      { "retry-after": String(ipLimit.retry_after || AUTH_REQUEST_IP_COOLDOWN_SECONDS) }
    );
  }

  const emailLimit = await checkAndRecordCooldown(
    store,
    `${AUTH_RATE_LIMIT_KEY_PREFIX}email:${normalizedEmail}`,
    AUTH_REQUEST_EMAIL_COOLDOWN_SECONDS
  );
  if (!emailLimit.ok) {
    return json(
      {
        ok: false,
        status: "rate_limited",
        message: "Too many requests. Try again later.",
      },
      429,
      { "retry-after": String(emailLimit.retry_after || AUTH_REQUEST_EMAIL_COOLDOWN_SECONDS) }
    );
  }

  let record = null;
  try {
    record = await loadSubscriberRecordByEmail(store, normalizedEmail);
  } catch (error) {
    return json(
      {
        ok: false,
        status: "state_lookup_failed",
        message: "Unable to prepare sign-in link right now.",
        details: error.message,
      },
      500
    );
  }

  if (record && ["active", "trialing"].includes(String(record.status || ""))) {
    const token = buildOpaqueToken();
    const magicRecord = createMagicLinkRecord(normalizedEmail, record);
    await putKvJson(store, `${AUTH_MAGIC_KEY_PREFIX}${token}`, magicRecord, {
      expirationTtl: AUTH_MAGIC_LINK_TTL_SECONDS,
    });

    const workerOrigin = new URL(request.url).origin;
    const verifyUrl = `${workerOrigin}/api/auth/magic-link/verify?token=${encodeURIComponent(token)}`;
    const sent = await sendMagicLinkEmail(normalizedEmail, verifyUrl, env);
    if (!sent.ok) {
      return json(
        {
          ok: false,
          status: sent.status,
          message: sent.message,
          route: "/api/auth/magic-link/request",
          details: sent.details || null,
          missing_env_vars: sent.missing || [],
        },
        sent.status === "auth_not_wired" ? 501 : 502
      );
    }
  }

  const accountDb = getAccountDb(env);
  if (accountDb) {
    try {
      await recordAuthEvent(accountDb, {
        email_normalized: normalizedEmail,
        event_type: "magic_link_requested",
        ip_hint: requestIp || null,
        user_agent_hint: getUserAgentHint(request),
        metadata: {
          eligible: Boolean(record && ["active", "trialing"].includes(String(record.status || ""))),
        },
      });
    } catch {
      // Best-effort audit trail only.
    }
  }

  return json({
    ok: true,
    status: "magic_link_requested",
    message: "If the address is eligible, a sign-in link has been sent.",
    route: "/api/auth/magic-link/request",
  });
}

async function handleMagicLinkVerify(request, env) {
  const url = new URL(request.url);
  const token = String(url.searchParams.get("token") || "").trim();
  const siteUrl = normalizeSiteUrl(env.SITE_URL || `${url.protocol}//${url.host}`);
  if (!token) {
    return redirect(`${siteUrl}/account.html?auth=invalid`);
  }

  const sessionSecret = getSessionSecret(env);
  if (!env.AUTH_MAGIC_LINK_SECRET || !sessionSecret) {
    return redirect(`${siteUrl}/account.html?auth=not_wired`);
  }

  const store = getSubscriberStateStore(env);
  if (!store) {
    return redirect(`${siteUrl}/account.html?auth=invalid`);
  }

  const record = await getKvJson(store, `${AUTH_MAGIC_KEY_PREFIX}${token}`);
  if (!record) {
    return redirect(`${siteUrl}/account.html?auth=invalid`);
  }

  await deleteKvKey(store, `${AUTH_MAGIC_KEY_PREFIX}${token}`);

  const email = normalizeEmail(record.email);
  const customerId = String(record.customer_id || "").trim();
  const subscriptionId = String(record.subscription_id || "").trim();
  const exp = Number(record.exp);
  if (!email || !customerId || !subscriptionId || !Number.isFinite(exp)) {
    return redirect(`${siteUrl}/account.html?auth=invalid`);
  }

  if (exp <= Math.floor(Date.now() / 1000)) {
    return redirect(`${siteUrl}/account.html?auth=expired`);
  }

  let subscriberRecord;
  try {
    subscriberRecord = await loadSubscriberRecordBySubscriptionId(store, subscriptionId);
  } catch {
    return redirect(`${siteUrl}/account.html?auth=invalid`);
  }

  if (!subscriberRecord) {
    return redirect(`${siteUrl}/account.html?auth=invalid`);
  }

  if (subscriberRecord.customer_id !== customerId || normalizeEmail(subscriberRecord.email) !== email) {
    return redirect(`${siteUrl}/account.html?auth=invalid`);
  }

  if (!["active", "trialing"].includes(String(subscriberRecord.status || ""))) {
    return redirect(`${siteUrl}/account.html?auth=inactive`);
  }

  const accountDb = getAccountDb(env);
  let trackedSessionId = "";
  let trackedSessionToken = "";
  if (accountDb) {
    try {
      const accountState = await mirrorSubscriptionFromRecord(accountDb, subscriberRecord, {
        email,
        emailVerifiedAt: new Date().toISOString(),
      });
      if (accountState?.user?.id) {
        const now = nowIso();
        trackedSessionId = buildId("sess");
        trackedSessionToken = buildOpaqueToken(32);
        const activeSessions = await listActiveAccountSessionsByUser(accountDb, accountState.user.id);
        await createAccountSession(accountDb, {
          id: trackedSessionId,
          user_id: accountState.user.id,
          session_token_hash: await buildHashedOpaqueValue(sessionSecret, trackedSessionToken, "session"),
          device_label: deriveDeviceLabel(request),
          user_agent_hash: getUserAgentHint(request)
            ? await buildHashedOpaqueValue(sessionSecret, getUserAgentHint(request), "ua")
            : null,
          ip_hash: getRequestIp(request)
            ? await buildHashedOpaqueValue(sessionSecret, getRequestIp(request), "ip")
            : null,
          session_kind: "browser",
          is_primary: activeSessions.length ? 0 : 1,
          is_revoked: 0,
          issued_at: now,
          last_seen_at: now,
          expires_at: new Date((Math.floor(Date.now() / 1000) + AUTH_SESSION_TTL_SECONDS) * 1000).toISOString(),
          revoked_at: null,
          revoke_reason: null,
          created_at: now,
          updated_at: now,
        });
        await ensureAccountRiskState(accountDb, accountState.user.id);
        await recordSessionSpreadRiskIfNeeded(accountDb, accountState, request, {
          trigger: "magic_link_verify",
          session_id: trackedSessionId,
        });
      }
      await recordAuthEvent(accountDb, {
        user_id: accountState?.user?.id || null,
        email_normalized: email,
        event_type: "magic_link_verified",
        ip_hint: getRequestIp(request) || null,
        user_agent_hint: getUserAgentHint(request),
        metadata: {
          customer_id: customerId,
          subscription_id: subscriptionId,
        },
      });
    } catch {
      // Account-state mirroring is additive only; auth should still succeed.
    }
  }

  const sessionPayload = {
    email,
    customer_id: customerId,
    subscription_id: subscriptionId,
    exp: Math.floor(Date.now() / 1000) + AUTH_SESSION_TTL_SECONDS,
    sid: trackedSessionId || undefined,
    st: trackedSessionToken || undefined,
  };
  const sessionToken = await buildSignedJsonToken(sessionPayload, sessionSecret);
  return redirect(`${siteUrl}/account.html?auth=success`, 303, {
    "set-cookie": buildSessionCookieHeader(sessionToken),
  });
}

async function handleAuthSession(request, env) {
  const sessionAccess = await verifySessionAccess(request, env);
  if (sessionAccess.ok) {
    return json({
      ok: true,
      authenticated: true,
      entitled: true,
      auth_mode: "session",
      email_hint: sessionAccess.email_hint,
      customer_id: sessionAccess.customer_id,
      subscription_id: sessionAccess.subscription_id,
      subscription_status: sessionAccess.subscription_status,
    });
  }

  if (hasTransitionalPremiumToken(request)) {
    const access = await verifyPremiumAccess(request, env);
    if (access.ok) {
      return json({
        ok: true,
        authenticated: true,
        entitled: true,
        auth_mode: "transitional_token",
        customer_id: access.customer_id,
        subscription_id: access.subscription_id,
        subscription_status: "active",
      });
    }
    return json(
      {
        ok: true,
        authenticated: false,
        entitled: false,
        status: access.status || "unauthenticated",
      },
      200,
      sessionFailureHeaders(sessionAccess.status)
    );
  }

  return json(
    {
      ok: true,
      authenticated: false,
      entitled: false,
      status: sessionAccess.status === "missing_session" ? "" : sessionAccess.status,
    },
    200,
    sessionFailureHeaders(sessionAccess.status)
  );
}

async function handleAccountState(request, env) {
  const sessionAccess = await verifySessionAccess(request, env);
  if (!sessionAccess.ok) {
    return json(
      {
        ok: false,
        authenticated: false,
        entitled: false,
        status: sessionAccess.status || "unauthenticated",
        message: sessionAccess.message || "Sign in to view account state.",
      },
      401,
      sessionFailureHeaders(sessionAccess.status)
    );
  }

  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return json({
      ok: true,
      authenticated: true,
      entitled: true,
      auth_mode: "session",
      d1_enabled: false,
      email_hint: sessionAccess.email_hint,
      customer_id: sessionAccess.customer_id,
      subscription_id: sessionAccess.subscription_id,
      subscription_status: sessionAccess.subscription_status,
    });
  }

  const accountState = await getAccountStateByEmail(accountDb, sessionAccess.email);
  return json({
    ok: true,
    authenticated: true,
    entitled: true,
    auth_mode: "session",
    d1_enabled: true,
    email_hint: sessionAccess.email_hint,
    customer_id: sessionAccess.customer_id,
    subscription_id: sessionAccess.subscription_id,
    subscription_status: sessionAccess.subscription_status,
    account: accountState,
  });
}

async function handleAccountSessionsState(request, env) {
  const sessionAccess = await verifySessionAccess(request, env);
  if (!sessionAccess.ok) {
    return json(
      {
        ok: false,
        authenticated: false,
        entitled: false,
        status: sessionAccess.status || "unauthenticated",
        message: sessionAccess.message || "Sign in to view account devices.",
      },
      401,
      sessionFailureHeaders(sessionAccess.status)
    );
  }

  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for account sessions.", ["ACCOUNT_DB"]);
  }

  const accountState = await getAccountStateByEmail(accountDb, sessionAccess.email);
  if (!accountState?.user?.id) {
    return json(
      {
        ok: false,
        status: "account_state_missing",
        message: "Unable to load account devices for this account yet.",
      },
      500
    );
  }

  const sessions = await listAccountSessionsByUser(accountDb, accountState.user.id, { limit: 8 });
  return json({
    ok: true,
    status: "account_sessions_loaded",
    current_session_id: sessionAccess.session_id || null,
    sessions: sessions.map((session) => ({
      id: session.id,
      device_label: session.device_label || "Browser session",
      session_kind: session.session_kind || "browser",
      is_current: session.id === sessionAccess.session_id,
      is_primary: Number(session.is_primary || 0) === 1,
      is_revoked: Number(session.is_revoked || 0) === 1 || Boolean(session.revoked_at),
      issued_at: session.issued_at || null,
      last_seen_at: session.last_seen_at || null,
      expires_at: session.expires_at || null,
      revoked_at: session.revoked_at || null,
      revoke_reason: session.revoke_reason || null,
    })),
  });
}

async function handleAccountSessionRevoke(request, env) {
  const sessionAccess = await verifySessionAccess(request, env);
  if (!sessionAccess.ok) {
    return json(
      {
        ok: false,
        authenticated: false,
        entitled: false,
        status: sessionAccess.status || "unauthenticated",
        message: sessionAccess.message || "Sign in to manage devices.",
      },
      401,
      sessionFailureHeaders(sessionAccess.status)
    );
  }

  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for account session actions.", ["ACCOUNT_DB"]);
  }

  let payload;
  try {
    payload = await request.json();
  } catch (error) {
    return requestError("Session revoke body must be valid JSON.", error.message);
  }

  const sessionId = String(payload?.session_id || "").trim();
  if (!sessionId) {
    return requestError("A session_id is required to revoke a device session.");
  }

  const accountState = await getAccountStateByEmail(accountDb, sessionAccess.email);
  if (!accountState?.user?.id) {
    return json(
      {
        ok: false,
        status: "account_state_missing",
        message: "Unable to load account devices for this account yet.",
      },
      500
    );
  }

  const targetSession = await getAccountSessionById(accountDb, sessionId);
  if (!targetSession?.id || targetSession.user_id !== accountState.user.id) {
    return json(
      {
        ok: false,
        status: "session_not_found",
        message: "That device session was not found for this account.",
      },
      404
    );
  }

  const revokedAt = nowIso();
  await revokeAccountSession(accountDb, targetSession.id, "user_revoked", revokedAt);
  try {
    await recordAuthEvent(accountDb, {
      user_id: accountState.user.id,
      email_normalized: sessionAccess.email,
      event_type: "account_session_revoked",
      ip_hint: getRequestIp(request) || null,
      user_agent_hint: getUserAgentHint(request),
      metadata: {
        current_session_id: sessionAccess.session_id || null,
        target_session_id: targetSession.id,
        target_device_label: targetSession.device_label || null,
      },
    });
  } catch {
    // Best-effort audit trail only.
  }

  const sessions = await listAccountSessionsByUser(accountDb, accountState.user.id, { limit: 8 });
  const responseHeaders = targetSession.id === sessionAccess.session_id
    ? { "set-cookie": clearCookieHeader(AUTH_SESSION_COOKIE) }
    : {};

  return json(
    {
      ok: true,
      status:
        targetSession.id === sessionAccess.session_id
          ? "account_current_session_revoked"
          : "account_session_revoked",
      message:
        targetSession.id === sessionAccess.session_id
          ? "This device has been signed out."
          : "Selected device session revoked.",
      revoked_session_id: targetSession.id,
      sessions: sessions.map((session) => ({
        id: session.id,
        device_label: session.device_label || "Browser session",
        session_kind: session.session_kind || "browser",
        is_current: session.id === sessionAccess.session_id,
        is_primary: Number(session.is_primary || 0) === 1,
        is_revoked: Number(session.is_revoked || 0) === 1 || Boolean(session.revoked_at),
        issued_at: session.issued_at || null,
        last_seen_at: session.last_seen_at || null,
        expires_at: session.expires_at || null,
        revoked_at: session.revoked_at || null,
        revoke_reason: session.revoke_reason || null,
      })),
    },
    200,
    responseHeaders
  );
}

async function handleAccountSessionsRevokeOthers(request, env) {
  const sessionAccess = await verifySessionAccess(request, env);
  if (!sessionAccess.ok) {
    return json(
      {
        ok: false,
        authenticated: false,
        entitled: false,
        status: sessionAccess.status || "unauthenticated",
        message: sessionAccess.message || "Sign in to manage devices.",
      },
      401,
      sessionFailureHeaders(sessionAccess.status)
    );
  }

  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for account session actions.", ["ACCOUNT_DB"]);
  }

  const accountState = await getAccountStateByEmail(accountDb, sessionAccess.email);
  if (!accountState?.user?.id) {
    return json(
      {
        ok: false,
        status: "account_state_missing",
        message: "Unable to load account devices for this account yet.",
      },
      500
    );
  }

  const revokedCount = await revokeOtherAccountSessions(
    accountDb,
    accountState.user.id,
    sessionAccess.session_id || null,
    "user_revoked_other_sessions",
    nowIso()
  );
  try {
    await recordAuthEvent(accountDb, {
      user_id: accountState.user.id,
      email_normalized: sessionAccess.email,
      event_type: "account_other_sessions_revoked",
      ip_hint: getRequestIp(request) || null,
      user_agent_hint: getUserAgentHint(request),
      metadata: {
        current_session_id: sessionAccess.session_id || null,
        revoked_count: revokedCount,
      },
    });
  } catch {
    // Best-effort audit trail only.
  }

  const sessions = await listAccountSessionsByUser(accountDb, accountState.user.id, { limit: 8 });
  return json({
    ok: true,
    status: "account_other_sessions_revoked",
    message: revokedCount
      ? "Other signed-in devices have been signed out."
      : "No other active device sessions were found.",
    revoked_count: revokedCount,
    sessions: sessions.map((session) => ({
      id: session.id,
      device_label: session.device_label || "Browser session",
      session_kind: session.session_kind || "browser",
      is_current: session.id === sessionAccess.session_id,
      is_primary: Number(session.is_primary || 0) === 1,
      is_revoked: Number(session.is_revoked || 0) === 1 || Boolean(session.revoked_at),
      issued_at: session.issued_at || null,
      last_seen_at: session.last_seen_at || null,
      expires_at: session.expires_at || null,
      revoked_at: session.revoked_at || null,
      revoke_reason: session.revoke_reason || null,
    })),
  });
}

async function handleAccountSessionMakePrimary(request, env) {
  const sessionAccess = await verifySessionAccess(request, env);
  if (!sessionAccess.ok) {
    return json(
      {
        ok: false,
        authenticated: false,
        entitled: false,
        status: sessionAccess.status || "unauthenticated",
        message: sessionAccess.message || "Sign in to manage devices.",
      },
      401,
      sessionFailureHeaders(sessionAccess.status)
    );
  }

  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for account session actions.", ["ACCOUNT_DB"]);
  }

  let payload;
  try {
    payload = await request.json();
  } catch (error) {
    return requestError("Make-primary body must be valid JSON.", error.message);
  }

  const sessionId = String(payload?.session_id || "").trim();
  if (!sessionId) {
    return requestError("A session_id is required to make a device primary.");
  }

  const accountState = await getAccountStateByEmail(accountDb, sessionAccess.email);
  if (!accountState?.user?.id) {
    return json(
      {
        ok: false,
        status: "account_state_missing",
        message: "Unable to load account devices for this account yet.",
      },
      500
    );
  }

  const targetSession = await getAccountSessionById(accountDb, sessionId);
  if (!targetSession?.id || targetSession.user_id !== accountState.user.id) {
    return json(
      {
        ok: false,
        status: "session_not_found",
        message: "That device session was not found for this account.",
      },
      404
    );
  }

  if (Number(targetSession.is_revoked || 0) === 1 || targetSession.revoked_at) {
    return json(
      {
        ok: false,
        status: "session_revoked",
        message: "Revoked device sessions cannot become primary.",
      },
      409
    );
  }

  await setPrimaryAccountSession(accountDb, accountState.user.id, targetSession.id, nowIso());
  try {
    await recordAuthEvent(accountDb, {
      user_id: accountState.user.id,
      email_normalized: sessionAccess.email,
      event_type: "account_session_primary_updated",
      ip_hint: getRequestIp(request) || null,
      user_agent_hint: getUserAgentHint(request),
      metadata: {
        current_session_id: sessionAccess.session_id || null,
        primary_session_id: targetSession.id,
        primary_device_label: targetSession.device_label || null,
      },
    });
  } catch {
    // Best-effort audit trail only.
  }

  const sessions = await listAccountSessionsByUser(accountDb, accountState.user.id, { limit: 8 });
  return json({
    ok: true,
    status: "account_session_primary_updated",
    message: "Primary device updated for this account.",
    primary_session_id: targetSession.id,
    sessions: sessions.map((session) => ({
      id: session.id,
      device_label: session.device_label || "Browser session",
      session_kind: session.session_kind || "browser",
      is_current: session.id === sessionAccess.session_id,
      is_primary: Number(session.is_primary || 0) === 1,
      is_revoked: Number(session.is_revoked || 0) === 1 || Boolean(session.revoked_at),
      issued_at: session.issued_at || null,
      last_seen_at: session.last_seen_at || null,
      expires_at: session.expires_at || null,
      revoked_at: session.revoked_at || null,
      revoke_reason: session.revoke_reason || null,
    })),
  });
}

async function handleInternalAccountLookup(request, env) {
  const adminAccess = verifyInternalAdminRequest(request, env);
  if (!adminAccess.ok) {
    return json({ ok: false, status: adminAccess.status, message: adminAccess.message }, adminAccess.code);
  }

  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for internal review routes.", ["ACCOUNT_DB"]);
  }

  const url = new URL(request.url);
  const userId = String(url.searchParams.get("user_id") || "").trim();
  const email = normalizeEmail(url.searchParams.get("email") || "");
  const account = userId
    ? await getAccountStateByUserId(accountDb, userId)
    : email
      ? await getAccountStateByEmail(accountDb, email)
      : null;

  if (!account?.user?.id) {
    return json(
      {
        ok: false,
        status: "account_not_found",
        message: "No account matched that internal lookup.",
      },
      404
    );
  }

  return json({
    ok: true,
    status: "internal_account_lookup_loaded",
    account_summary: await summarizeInternalAccount(accountDb, account.user.id),
  });
}

async function handleInternalAccountRead(request, env, userId) {
  const adminAccess = verifyInternalAdminRequest(request, env);
  if (!adminAccess.ok) {
    return json({ ok: false, status: adminAccess.status, message: adminAccess.message }, adminAccess.code);
  }

  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for internal review routes.", ["ACCOUNT_DB"]);
  }

  const summary = await summarizeInternalAccount(accountDb, userId);
  if (!summary?.user?.id) {
    return json(
      {
        ok: false,
        status: "account_not_found",
        message: "No account matched that internal account id.",
      },
      404
    );
  }

  return json({
    ok: true,
    status: "internal_account_loaded",
    account_summary: summary,
  });
}

async function handleInternalAccountFlagsRead(request, env, userId) {
  const adminAccess = verifyInternalAdminRequest(request, env);
  if (!adminAccess.ok) {
    return json({ ok: false, status: adminAccess.status, message: adminAccess.message }, adminAccess.code);
  }

  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for internal review routes.", ["ACCOUNT_DB"]);
  }

  const flags = await listAccountRiskFlagsByUser(accountDb, userId, { limit: 30 });
  return json({
    ok: true,
    status: "internal_account_flags_loaded",
    flags: flags.map((flag) => ({
      ...flag,
      evidence: (() => {
        try {
          return JSON.parse(flag.evidence_json || "{}");
        } catch {
          return {};
        }
      })(),
    })),
  });
}

async function handleInternalAccountNotesRead(request, env, userId) {
  const adminAccess = verifyInternalAdminRequest(request, env);
  if (!adminAccess.ok) {
    return json({ ok: false, status: adminAccess.status, message: adminAccess.message }, adminAccess.code);
  }

  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for internal review routes.", ["ACCOUNT_DB"]);
  }

  const notes = await listAccountAdminNotesByUser(accountDb, userId, { limit: 30 });
  return json({
    ok: true,
    status: "internal_account_notes_loaded",
    notes,
  });
}

async function handleInternalAccountNoteCreate(request, env, userId) {
  const adminAccess = verifyInternalAdminRequest(request, env);
  if (!adminAccess.ok) {
    return json({ ok: false, status: adminAccess.status, message: adminAccess.message }, adminAccess.code);
  }

  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for internal review routes.", ["ACCOUNT_DB"]);
  }

  let payload;
  try {
    payload = await request.json();
  } catch (error) {
    return requestError("Internal note body must be valid JSON.", error.message);
  }
  const actor = normalizeInternalActor(payload?.author_id);
  const actorError = validateInternalActor(actor);
  if (actorError) {
    return json({ ok: false, status: "internal_operator_identity_required", message: actorError }, 400);
  }

  const note = await addAccountAdminNote(accountDb, {
    user_id: userId,
    note_type: String(payload?.note_type || "").trim() || "support_note",
    visibility: String(payload?.visibility || "internal").trim() || "internal",
    content: String(payload?.content || "").trim(),
    author_id: actor,
  });
  if (!note?.id) {
    return requestError("A note_type and content are required to add an internal note.");
  }

  return json({
    ok: true,
    status: "internal_account_note_created",
    note,
  });
}

async function handleInternalAccountTimelineRead(request, env, userId) {
  const adminAccess = verifyInternalAdminRequest(request, env);
  if (!adminAccess.ok) {
    return json({ ok: false, status: adminAccess.status, message: adminAccess.message }, adminAccess.code);
  }

  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for internal review routes.", ["ACCOUNT_DB"]);
  }

  return json({
    ok: true,
    status: "internal_account_timeline_loaded",
    timeline: await buildInternalTimeline(accountDb, userId),
  });
}

function normalizeInternalActionReason(input) {
  return String(input || "")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeInternalActor(input) {
  return String(input || "")
    .replace(/\s+/g, " ")
    .trim();
}

function validateInternalActor(actor) {
  if (!actor) {
    return "Operator identity is required.";
  }
  if (actor.length < 5) {
    return "Operator identity must be at least 5 characters.";
  }
  if (["internal:web-shell", "internal:operator", "operator"].includes(actor.toLowerCase())) {
    return "Use a real operator identity instead of the generic shell label.";
  }
  return null;
}

function validateInternalActionReason(reason, label = "reason") {
  if (!reason) {
    return `${label} is required.`;
  }
  if (reason.length < 12) {
    return `${label} must be at least 12 characters so the review trail is meaningful.`;
  }
  return null;
}

const ALLOWED_INTERNAL_REVIEW_OUTCOMES = new Set([
  "monitor_only",
  "restrict_for_review",
  "suspend",
  "reinstate_ready",
]);

const ALLOWED_INTERNAL_REVIEW_PRESETS = new Set([
  "custom",
  "suspension_review",
  "sharing_risk",
  "billing_concern",
]);

async function handleInternalAccountRestrict(request, env, userId) {
  const adminAccess = verifyInternalAdminRequest(request, env);
  if (!adminAccess.ok) {
    return json({ ok: false, status: adminAccess.status, message: adminAccess.message }, adminAccess.code);
  }
  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for internal review routes.", ["ACCOUNT_DB"]);
  }
  let payload;
  try {
    payload = await request.json();
  } catch (error) {
    return requestError("Restrict body must be valid JSON.", error.message);
  }
  const reason = normalizeInternalActionReason(payload?.reason);
  const reasonError = validateInternalActionReason(reason, "Restriction reason");
  if (reasonError) {
    return json({ ok: false, status: "internal_restriction_reason_required", message: reasonError }, 400);
  }
  const actor = normalizeInternalActor(payload?.author_id);
  const actorError = validateInternalActor(actor);
  if (actorError) {
    return json({ ok: false, status: "internal_operator_identity_required", message: actorError }, 400);
  }

  await updateAccountRiskState(accountDb, userId, {
    account_status: "restricted",
    review_status: "restricted",
    risk_level: "high",
    risk_score: 60,
    last_reviewed_at: nowIso(),
    last_reviewed_by: actor,
    last_risk_event_at: nowIso(),
  });
  await addAccountAdminNote(accountDb, {
    user_id: userId,
    note_type: "risk_note",
    visibility: "internal",
    author_id: actor,
    content: `Restriction applied. ${reason}`,
  });
  await recordAuthEvent(accountDb, {
    user_id: userId,
    event_type: "internal_account_restricted",
    metadata: { reason, author_id: actor },
  });

  return json({
    ok: true,
    status: "internal_account_restricted",
    message: "Account moved into restricted review state.",
    account_summary: await summarizeInternalAccount(accountDb, userId),
  });
}

async function handleInternalAccountSuspend(request, env, userId) {
  const adminAccess = verifyInternalAdminRequest(request, env);
  if (!adminAccess.ok) {
    return json({ ok: false, status: adminAccess.status, message: adminAccess.message }, adminAccess.code);
  }
  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for internal review routes.", ["ACCOUNT_DB"]);
  }
  let payload;
  try {
    payload = await request.json();
  } catch (error) {
    return requestError("Suspend body must be valid JSON.", error.message);
  }
  const reason = normalizeInternalActionReason(payload?.reason);
  const reasonError = validateInternalActionReason(reason, "Suspension reason");
  if (reasonError) {
    return json({ ok: false, status: "internal_suspension_reason_required", message: reasonError }, 400);
  }
  const confirmation = String(payload?.confirmation || "").trim().toUpperCase();
  if (confirmation !== "SUSPEND") {
    return json(
      {
        ok: false,
        status: "internal_suspension_confirmation_required",
        message: "Type SUSPEND to confirm this action.",
      },
      400
    );
  }
  const actor = normalizeInternalActor(payload?.author_id);
  const actorError = validateInternalActor(actor);
  if (actorError) {
    return json({ ok: false, status: "internal_operator_identity_required", message: actorError }, 400);
  }
  const sessions = await listAccountSessionsByUser(accountDb, userId, { limit: 24 });
  const revokedAt = nowIso();
  for (const session of sessions) {
    if (!(Number(session.is_revoked || 0) === 1 || session.revoked_at)) {
      await revokeAccountSession(accountDb, session.id, "suspended_account", revokedAt);
    }
  }
  await updateAccountRiskState(accountDb, userId, {
    account_status: "suspended",
    review_status: "suspended",
    risk_level: "critical",
    risk_score: 90,
    suspended_at: revokedAt,
    suspension_reason: reason,
    last_reviewed_at: revokedAt,
    last_reviewed_by: actor,
    last_risk_event_at: revokedAt,
  });
  await addAccountAdminNote(accountDb, {
    user_id: userId,
    note_type: "risk_note",
    visibility: "internal",
    author_id: actor,
    content: `Suspension applied. ${reason}`,
  });
  await recordAuthEvent(accountDb, {
    user_id: userId,
    event_type: "internal_account_suspended",
    metadata: { reason, author_id: actor, revoked_session_count: sessions.length },
  });

  return json({
    ok: true,
    status: "internal_account_suspended",
    message: "Account suspended and active sessions revoked.",
    account_summary: await summarizeInternalAccount(accountDb, userId),
  });
}

async function handleInternalAccountReinstate(request, env, userId) {
  const adminAccess = verifyInternalAdminRequest(request, env);
  if (!adminAccess.ok) {
    return json({ ok: false, status: adminAccess.status, message: adminAccess.message }, adminAccess.code);
  }
  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for internal review routes.", ["ACCOUNT_DB"]);
  }
  let payload;
  try {
    payload = await request.json();
  } catch (error) {
    return requestError("Reinstate body must be valid JSON.", error.message);
  }
  const reason = normalizeInternalActionReason(payload?.reason);
  const reasonError = validateInternalActionReason(reason, "Reinstatement reason");
  if (reasonError) {
    return json({ ok: false, status: "internal_reinstatement_reason_required", message: reasonError }, 400);
  }
  const actor = normalizeInternalActor(payload?.author_id);
  const actorError = validateInternalActor(actor);
  if (actorError) {
    return json({ ok: false, status: "internal_operator_identity_required", message: actorError }, 400);
  }
  const now = nowIso();
  await updateAccountRiskState(accountDb, userId, {
    account_status: "active",
    review_status: "clear",
    risk_level: "low",
    risk_score: 0,
    suspended_at: null,
    suspension_reason: null,
    reinstated_at: now,
    reinstatement_reason: reason,
    last_reviewed_at: now,
    last_reviewed_by: actor,
    last_risk_event_at: now,
  });
  await addAccountAdminNote(accountDb, {
    user_id: userId,
    note_type: "reinstatement_note",
    visibility: "internal",
    author_id: actor,
    content: `Reinstatement applied. ${reason}`,
  });
  await recordAuthEvent(accountDb, {
    user_id: userId,
    event_type: "internal_account_reinstated",
    metadata: { reason, author_id: actor },
  });

  return json({
    ok: true,
    status: "internal_account_reinstated",
    message: "Account reinstated and review state cleared.",
    account_summary: await summarizeInternalAccount(accountDb, userId),
  });
}

async function handleInternalFlagStatusUpdate(request, env, userId, flagId, nextStatus) {
  const adminAccess = verifyInternalAdminRequest(request, env);
  if (!adminAccess.ok) {
    return json({ ok: false, status: adminAccess.status, message: adminAccess.message }, adminAccess.code);
  }
  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for internal review routes.", ["ACCOUNT_DB"]);
  }
  let payload;
  try {
    payload = await request.json();
  } catch (error) {
    return requestError("Flag update body must be valid JSON.", error.message);
  }
  const actor = normalizeInternalActor(payload?.author_id);
  const actorError = validateInternalActor(actor);
  if (actorError) {
    return json({ ok: false, status: "internal_operator_identity_required", message: actorError }, 400);
  }
  const note = normalizeInternalActionReason(payload?.resolution_note || payload?.reason);
  const noteError = validateInternalActionReason(
    note,
    nextStatus === "dismissed" ? "Dismissal note" : "Resolution note"
  );
  if (noteError) {
    return json(
      {
        ok: false,
        status: nextStatus === "dismissed" ? "internal_flag_dismissal_note_required" : "internal_flag_resolution_note_required",
        message: noteError,
      },
      400
    );
  }
  await updateAccountRiskFlagStatus(accountDb, flagId, {
    flag_status: nextStatus,
    resolved_at: nowIso(),
    resolved_by: actor,
    resolution_note: note,
  });
  await recordAuthEvent(accountDb, {
    user_id: userId,
    event_type: nextStatus === "dismissed" ? "internal_flag_dismissed" : "internal_flag_resolved",
    metadata: {
      flag_id: flagId,
      status: nextStatus,
      author_id: actor,
      resolution_note: note,
    },
  });
  await addAccountAdminNote(accountDb, {
    user_id: userId,
    note_type: "risk_note",
    visibility: "internal",
    author_id: actor,
    content: `${nextStatus === "dismissed" ? "Flag dismissed" : "Flag resolved"}: ${note}`,
  });
  return json({
    ok: true,
    status: nextStatus === "dismissed" ? "internal_flag_dismissed" : "internal_flag_resolved",
    flags: (await listAccountRiskFlagsByUser(accountDb, userId, { limit: 30 })).map((flag) => ({
      ...flag,
      evidence: (() => {
        try {
          return JSON.parse(flag.evidence_json || "{}");
        } catch {
          return {};
        }
      })(),
    })),
  });
}

async function handleInternalReviewOutcomeUpdate(request, env, userId) {
  const adminAccess = verifyInternalAdminRequest(request, env);
  if (!adminAccess.ok) {
    return json({ ok: false, status: adminAccess.status, message: adminAccess.message }, adminAccess.code);
  }
  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for internal review routes.", ["ACCOUNT_DB"]);
  }
  let payload;
  try {
    payload = await request.json();
  } catch (error) {
    return requestError("Review outcome body must be valid JSON.", error.message);
  }
  const actor = normalizeInternalActor(payload?.author_id);
  const actorError = validateInternalActor(actor);
  if (actorError) {
    return json({ ok: false, status: "internal_operator_identity_required", message: actorError }, 400);
  }
  const outcome = String(payload?.review_outcome || "")
    .trim()
    .toLowerCase();
  if (!ALLOWED_INTERNAL_REVIEW_OUTCOMES.has(outcome)) {
    return json(
      {
        ok: false,
        status: "internal_review_outcome_invalid",
        message: "Choose a valid operator review outcome before saving it.",
      },
      400
    );
  }
  const note = normalizeInternalActionReason(payload?.review_outcome_note);
  const noteError = validateInternalActionReason(note, "Review outcome note");
  if (noteError) {
    return json({ ok: false, status: "internal_review_outcome_note_required", message: noteError }, 400);
  }
  const preset = String(payload?.review_preset || "")
    .trim()
    .toLowerCase();
  if (!ALLOWED_INTERNAL_REVIEW_PRESETS.has(preset)) {
    return json(
      {
        ok: false,
        status: "internal_review_preset_invalid",
        message: "Choose a valid review preset before saving the operator decision.",
      },
      400
    );
  }
  const now = nowIso();
  await updateAccountRiskState(accountDb, userId, {
    last_review_outcome: outcome,
    last_review_outcome_note: note,
    last_review_outcome_at: now,
    last_review_outcome_by: actor,
    last_review_preset: preset,
    last_reviewed_at: now,
    last_reviewed_by: actor,
  });
  await addAccountAdminNote(accountDb, {
    user_id: userId,
    note_type: "review_outcome_note",
    visibility: "internal",
    author_id: actor,
    content: `Review outcome saved: ${outcome.replaceAll("_", " ")}. ${note}`,
  });
  await recordAuthEvent(accountDb, {
    user_id: userId,
    event_type: "internal_review_outcome_recorded",
    metadata: {
      review_outcome: outcome,
      review_outcome_note: note,
      review_preset: preset,
      author_id: actor,
    },
  });
  return json({
    ok: true,
    status: "internal_review_outcome_saved",
    message: "Review outcome saved to the internal account trail.",
    account_summary: await summarizeInternalAccount(accountDb, userId),
  });
}

async function handleAccountPreferencesUpdate(request, env) {
  const sessionAccess = await verifySessionAccess(request, env);
  if (!sessionAccess.ok) {
    return json(
      {
        ok: false,
        authenticated: false,
        entitled: false,
        status: sessionAccess.status || "unauthenticated",
        message: sessionAccess.message || "Sign in to update account preferences.",
      },
      401,
      sessionFailureHeaders(sessionAccess.status)
    );
  }

  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for account preferences.", ["ACCOUNT_DB"]);
  }

  let payload;
  try {
    payload = await request.json();
  } catch (error) {
    return requestError("Account preferences body must be valid JSON.", error.message);
  }

  const accountState =
    (await getAccountStateByEmail(accountDb, sessionAccess.email)) ||
    (await mirrorSubscriptionFromRecord(
      accountDb,
      {
        customer_id: sessionAccess.customer_id,
        subscription_id: sessionAccess.subscription_id,
        status: sessionAccess.subscription_status,
        email: sessionAccess.email,
      },
      {
        email: sessionAccess.email,
        emailVerifiedAt: new Date().toISOString(),
      }
    ));

  if (!accountState?.user?.id) {
    return json(
      {
        ok: false,
        status: "account_state_missing",
        message: "Unable to update preferences for this account yet.",
      },
      500
    );
  }

  await updateNotificationPreferences(accountDb, accountState.user.id, payload || {});
  const refreshed = await getAccountStateByEmail(accountDb, sessionAccess.email);

  try {
    await recordAuthEvent(accountDb, {
      user_id: accountState.user.id,
      email_normalized: sessionAccess.email,
      event_type: "notification_preferences_updated",
      ip_hint: getRequestIp(request) || null,
      user_agent_hint: getUserAgentHint(request),
      metadata: {
        alert_frequency_mode: refreshed?.notification_preferences?.alert_frequency_mode || null,
      },
    });
  } catch {
    // Best-effort audit trail only.
  }

  return json({
    ok: true,
    status: "notification_preferences_updated",
    message: "Intelligence preferences saved.",
    account: refreshed,
  });
}

async function handleAccountAlertsState(request, env) {
  const sessionAccess = await verifySessionAccess(request, env);
  if (!sessionAccess.ok) {
    return json(
      {
        ok: false,
        authenticated: false,
        entitled: false,
        status: sessionAccess.status || "unauthenticated",
        message: sessionAccess.message || "Sign in to view alert state.",
      },
      401,
      sessionFailureHeaders(sessionAccess.status)
    );
  }

  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for account alerts.", ["ACCOUNT_DB"]);
  }

  const accountState = await getAccountStateByEmail(accountDb, sessionAccess.email);
  if (!accountState?.user?.id) {
    return json(
      {
        ok: false,
        status: "account_state_missing",
        message: "Unable to load alert state for this account yet.",
      },
      500
    );
  }

  const alerts = await listNotificationAlertsByUser(accountDb, accountState.user.id, { limit: 24 });
  return json({
    ok: true,
    status: "account_alerts_loaded",
    alerts,
  });
}

async function handleAccountAlertsRefresh(request, env) {
  const sessionAccess = await verifySessionAccess(request, env);
  if (!sessionAccess.ok) {
    return json(
      {
        ok: false,
        authenticated: false,
        entitled: false,
        status: sessionAccess.status || "unauthenticated",
        message: sessionAccess.message || "Sign in to refresh alerts.",
      },
      401,
      sessionFailureHeaders(sessionAccess.status)
    );
  }

  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for account alerts.", ["ACCOUNT_DB"]);
  }

  const accountState = await getAccountStateByEmail(accountDb, sessionAccess.email);
  if (!accountState?.user?.id) {
    return json(
      {
        ok: false,
        status: "account_state_missing",
        message: "Unable to refresh alerts for this account yet.",
      },
      500
    );
  }

  const queueResult = await queueFollowedTelegramAlertsForAccount(accountDb, env, accountState);
  const alerts = await listNotificationAlertsByUser(accountDb, accountState.user.id, { limit: 24 });
  return json({
    ok: true,
    status: "account_alerts_refreshed",
    message: "Followed alerts refreshed from the current intelligence window.",
    matched_fixtures: queueResult.matched,
    queued_alerts: queueResult.queued,
    attempted_alerts: queueResult.attempted,
    alerts,
  });
}

async function handleAccountAlertsDispatch(request, env) {
  const sessionAccess = await verifySessionAccess(request, env);
  if (!sessionAccess.ok) {
    return json(
      {
        ok: false,
        authenticated: false,
        entitled: false,
        status: sessionAccess.status || "unauthenticated",
        message: sessionAccess.message || "Sign in to dispatch alerts.",
      },
      401,
      sessionFailureHeaders(sessionAccess.status)
    );
  }

  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for account alerts.", ["ACCOUNT_DB"]);
  }

  const accountState = await getAccountStateByEmail(accountDb, sessionAccess.email);
  if (!accountState?.user?.id) {
    return json(
      {
        ok: false,
        status: "account_state_missing",
        message: "Unable to dispatch alerts for this account yet.",
      },
      500
    );
  }

  await queueFollowedTelegramAlertsForAccount(accountDb, env, accountState);
  const dispatchResult = await dispatchDueTelegramAlerts(accountDb, env, {
    userId: accountState.user.id,
    userEmailMap: new Map([[accountState.user.id, accountState.user.email_normalized]]),
  });
  const alerts = await listNotificationAlertsByUser(accountDb, accountState.user.id, { limit: 24 });
  return json({
    ok: true,
    status: "account_alerts_dispatched",
    message: "Due Telegram alerts processed for this account.",
    ...dispatchResult,
    alerts,
  });
}

async function handleTelegramLinkStart(request, env) {
  const sessionAccess = await verifySessionAccess(request, env);
  if (!sessionAccess.ok) {
    return json(
      {
        ok: false,
        status: sessionAccess.status || "unauthenticated",
        message: sessionAccess.message || "Verify your email before linking Telegram.",
      },
      401,
      sessionFailureHeaders(sessionAccess.status)
    );
  }

  const accountDb = getAccountDb(env);
  const store = getSubscriberStateStore(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for Telegram linking.", ["ACCOUNT_DB"]);
  }
  if (!store) {
    return configError("SUBSCRIBER_STATE binding is required for Telegram linking.", ["SUBSCRIBER_STATE"]);
  }

  const accountState =
    (await getAccountStateByEmail(accountDb, sessionAccess.email)) ||
    (await mirrorSubscriptionFromRecord(
      accountDb,
      {
        customer_id: sessionAccess.customer_id,
        subscription_id: sessionAccess.subscription_id,
        status: sessionAccess.subscription_status,
        email: sessionAccess.email,
      },
      {
        email: sessionAccess.email,
        emailVerifiedAt: new Date().toISOString(),
      }
    ));

  if (!accountState?.user?.id) {
    return json(
      {
        ok: false,
        status: "account_state_missing",
        message: "Unable to prepare Telegram linking for this account yet.",
      },
      500
    );
  }

  const code = buildTelegramLinkCode();
  const issuedAt = new Date().toISOString();
  const expiresAt = new Date(Date.now() + TELEGRAM_LINK_TTL_SECONDS * 1000).toISOString();
  await putKvJson(
    store,
    `${TELEGRAM_LINK_KEY_PREFIX}${code}`,
    {
      user_id: accountState.user.id,
      email: accountState.user.email_normalized,
      customer_id: sessionAccess.customer_id,
      subscription_id: sessionAccess.subscription_id,
      issued_at: issuedAt,
      expires_at: expiresAt,
    },
    { expirationTtl: TELEGRAM_LINK_TTL_SECONDS }
  );

  try {
    await recordAuthEvent(accountDb, {
      user_id: accountState.user.id,
      email_normalized: accountState.user.email_normalized,
      event_type: "telegram_link_started",
      ip_hint: getRequestIp(request) || null,
      user_agent_hint: getUserAgentHint(request),
      metadata: { code_hint: code.slice(0, 4), expires_at: expiresAt },
    });
  } catch {
    // Best-effort audit trail only.
  }

  const botUsername = String(env.TELEGRAM_BOT_USERNAME || "").trim();
  const deepLinkUrl = botUsername ? `https://t.me/${botUsername}?start=oglink_${code}` : "";

  return json({
    ok: true,
    status: "telegram_link_ready",
    code,
    expires_at: expiresAt,
    expires_in_seconds: TELEGRAM_LINK_TTL_SECONDS,
    bot_username: botUsername || null,
    deep_link_url: deepLinkUrl || null,
    message: botUsername
      ? "Open the Telegram bot using the deep link to complete linking."
      : "Use this one-time code inside the Telegram bot to complete linking.",
  });
}

async function completeTelegramLinkFromCode(env, payload, context = {}) {
  const code = String(payload?.code || "").trim().toUpperCase();
  const telegramUserId = String(payload?.telegram_user_id || "").trim();
  const telegramUsername = String(payload?.telegram_username || "").trim();
  const telegramChatId = String(payload?.telegram_chat_id || "").trim();

  if (!code || !telegramUserId) {
    return {
      ok: false,
      response: requestError("Both `code` and `telegram_user_id` are required."),
    };
  }

  const accountDb = getAccountDb(env);
  const store = getSubscriberStateStore(env);
  if (!accountDb) {
    return {
      ok: false,
      response: configError("ACCOUNT_DB D1 binding is required for Telegram linking.", ["ACCOUNT_DB"]),
    };
  }
  if (!store) {
    return {
      ok: false,
      response: configError("SUBSCRIBER_STATE binding is required for Telegram linking.", ["SUBSCRIBER_STATE"]),
    };
  }

  const kvRecord = await getKvJson(store, `${TELEGRAM_LINK_KEY_PREFIX}${code}`);
  if (!kvRecord?.user_id || !kvRecord?.email) {
    return {
      ok: false,
      response: json(
        {
          ok: false,
          status: "invalid_link_code",
          message: "Telegram link code is invalid or expired.",
        },
        400
      ),
    };
  }

  await deleteKvKey(store, `${TELEGRAM_LINK_KEY_PREFIX}${code}`);

  const accountState = await completeTelegramLink(accountDb, {
    user_id: kvRecord.user_id,
    email: kvRecord.email,
    telegram_user_id: telegramUserId,
    telegram_username: telegramUsername || null,
    telegram_chat_id: telegramChatId || null,
  });

  try {
    await recordAuthEvent(accountDb, {
      user_id: kvRecord.user_id,
      email_normalized: kvRecord.email,
      event_type: "telegram_link_completed",
      ip_hint: context.ip_hint || null,
      user_agent_hint: context.user_agent_hint || null,
      metadata: {
        telegram_user_id: telegramUserId,
        telegram_username: telegramUsername || null,
      },
    });
  } catch {
    // Best-effort audit trail only.
  }

  return {
    ok: true,
    payload: {
      ok: true,
      status: "telegram_linked",
      message: "Telegram has been linked to this Odds Genius account.",
      account: accountState,
    },
  };
}

const extractTelegramLinkCode = (text) => {
  const normalized = String(text || "").trim();
  if (!normalized) {
    return "";
  }
  const startMatch = normalized.match(/\/start(?:@\w+)?\s+oglink[_-]?([A-HJ-NP-Z2-9]{6})/i);
  if (startMatch?.[1]) {
    return startMatch[1].toUpperCase();
  }
  const directMatch = normalized.match(TELEGRAM_LINK_CODE_PATTERN);
  return directMatch?.[1]?.toUpperCase() || "";
};

async function sendTelegramMessage(env, chatId, text) {
  const botToken = getTelegramBotToken(env);
  if (!botToken || !chatId || !text) {
    return false;
  }

  const response = await fetch(`https://api.telegram.org/bot${botToken}/sendMessage`, {
    method: "POST",
    headers: {
      "content-type": "application/json; charset=utf-8",
    },
    body: JSON.stringify({
      chat_id: chatId,
      text,
      disable_web_page_preview: true,
    }),
  });

  return response.ok;
}

async function loadFixtureIntelligenceArtifact(env) {
  const siteUrl = normalizeSiteUrl(env.SITE_URL);
  if (!siteUrl) {
    throw new Error("SITE_URL is required to load fixture intelligence.");
  }

  const response = await fetch(`${siteUrl}/public/data/fixture_intelligence_public.json`, {
    method: "GET",
    headers: {
      accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`Fixture intelligence artifact unavailable (${response.status}).`);
  }

  const payload = await response.json();
  return Array.isArray(payload?.fixtures) ? payload.fixtures : [];
}

function formatTelegramFixtureAlert(fixture, env) {
  const publishClass = String(fixture?.publish_class || fixture?.fixture_class || "MONITOR").toUpperCase();
  const marketFamily = String(fixture?.signal_summary?.market_family || "INTEL").toUpperCase();
  const headline =
    String(fixture?.signal_summary?.headline || fixture?.signal_summary?.summary_text || "").trim() ||
    "Fixture intelligence update is available.";
  const notes = Array.isArray(fixture?.context_summary?.notes)
    ? fixture.context_summary.notes.filter(Boolean).slice(0, 2)
    : [];
  const kickoff = String(fixture?.kickoff_time || "").trim();
  const siteUrl = normalizeSiteUrl(env.SITE_URL);
  const fixtureHref =
    siteUrl && fixture?.fixture_key
      ? `${siteUrl}/fixture.html?fixture=${encodeURIComponent(String(fixture.fixture_key))}`
      : "";

  const lines = [
    `Odds Genius ${publishClass} intelligence`,
    "",
    `${String(fixture?.home_team || "Home")} vs ${String(fixture?.away_team || "Away")}`,
    `${String(fixture?.league || "League")} | ${marketFamily}`,
  ];

  if (kickoff) {
    lines.push(`Kickoff: ${kickoff}`);
  }

  lines.push("", headline);

  for (const note of notes) {
    lines.push(`- ${String(note)}`);
  }

  if (fixtureHref) {
    lines.push("", `Open fixture view: ${fixtureHref}`);
  }

  return lines.join("\n");
}

function getFollowedFixtureMatches(accountState, fixtures) {
  const prefs = accountState?.notification_preferences || null;
  if (!prefs || !Array.isArray(fixtures) || !fixtures.length) {
    return [];
  }

  const teams = parsePreferenceList(prefs.favourite_teams).map(normalizePreferenceText);
  const leagues = parsePreferenceList(prefs.favourite_leagues).map(normalizePreferenceText);
  const markets = parsePreferenceList(prefs.favourite_markets).map((entry) =>
    normalizePreferenceText(entry).replace(/\s+/g, "")
  );
  const followedFixtures = parsePreferenceList(prefs.followed_fixtures).map(normalizePreferenceText);

  return fixtures
    .map((fixture) => {
      const reasons = [];
      const rowHome = normalizePreferenceText(fixture.home_team);
      const rowAway = normalizePreferenceText(fixture.away_team);
      const rowLeague = normalizePreferenceText(fixture.league);
      const rowFixture = normalizePreferenceText(fixturePreferenceLabel(fixture));
      const rowMarket = normalizePreferenceText(marketFamilyLabel(fixture.signal_summary?.market_family)).replace(
        /\s+/g,
        ""
      );

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
        fixture,
        reasons,
      };
    })
    .filter(Boolean);
}

function buildFollowMatchProfile(entry) {
  const reasons = Array.isArray(entry?.reasons) ? entry.reasons : [];
  const reasonSet = new Set(reasons);
  const publishClass = String(entry?.fixture?.publish_class || entry?.fixture?.fixture_class || "MONITOR").toUpperCase();
  const stylePreset = normalizeStylePreset(entry?.accountState?.notification_preferences?.user_style_preset);
  const kickoffNear = isNearKickoffWindow(entry?.fixture?.kickoff_time, 240);
  const eliteFixture = isEliteFixture(entry?.fixture);
  const hasFixture = reasonSet.has("followed fixture");
  const hasTeam = reasonSet.has("followed team");
  const hasLeague = reasonSet.has("followed league");
  const hasMarket = reasonSet.has("followed market");

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

  let telegramEligible = false;
  let autoGate = "website_only";
  if (hasFixture) {
    telegramEligible = true;
    autoGate = "direct_fixture_follow";
  } else if (hasTeam && publishClass === "DEPLOY") {
    telegramEligible = true;
    autoGate = "direct_team_deploy";
  } else if (hasTeam && publishClass === "OBSERVE" && stylePreset === "tactical_reader" && kickoffNear) {
    telegramEligible = true;
    autoGate = "direct_team_observe";
  } else if (
    publishClass === "DEPLOY" &&
    hasLeague &&
    hasMarket &&
    ((stylePreset === "disciplined_bettor" && (eliteFixture || kickoffNear)) || (stylePreset === "tactical_reader" && (eliteFixture || kickoffNear)))
  ) {
    telegramEligible = true;
    autoGate = "league_market_deploy";
  }

  if (stylePreset === "analyst" || stylePreset === "researcher") {
    if (!hasFixture && !hasTeam) {
      telegramEligible = false;
      autoGate = "website_depth_preferred";
    }
  }
  if (stylePreset === "tactical_reader" && publishClass === "CONTEXT" && (hasTeam || hasFixture) && kickoffNear) {
    telegramEligible = true;
    autoGate = "team_context_follow";
  }

  const relevanceTier =
    score >= 120 ? "critical" : score >= 80 ? "high" : score >= 45 ? "normal" : "low";

  return {
    reasons,
    score,
    relevance_tier: relevanceTier,
    telegram_eligible: telegramEligible,
    auto_gate: autoGate,
    has_fixture: hasFixture,
    has_team: hasTeam,
    has_league: hasLeague,
    has_market: hasMarket,
    publish_class: publishClass,
  };
}

function shouldQueueTelegramAlert(accountState, entry) {
  const prefs = accountState?.notification_preferences || null;
  if (!prefs || !prefs.telegram_enabled || prefs.website_only_mode) {
    return false;
  }
  const fixture = entry?.fixture || {};
  const publishClass = String(fixture.publish_class || fixture.fixture_class || "").toUpperCase();
  const profile = buildFollowMatchProfile(entry);
  if (!profile.telegram_eligible) {
    return false;
  }
  if (publishClass === "DEPLOY") {
    const confidenceTier = String(fixture.confidence_tier || fixture.premium_tier || "").toUpperCase();
    if (confidenceTier === "ELITE") {
      return Boolean(prefs.elite_alerts_enabled);
    }
    return Boolean(prefs.standard_alerts_enabled);
  }
  if (publishClass === "OBSERVE" || publishClass === "CONTEXT" || publishClass === "MONITOR") {
    return Boolean(prefs.allow_non_signal_intelligence);
  }
  return false;
}

function computeScheduledFor(accountState, fixture) {
  const prefs = accountState?.notification_preferences || null;
  const preMatchWindowMinutes = Math.max(0, Math.min(1440, Number(prefs?.pre_match_window_minutes ?? 90)));
  const kickoffRaw = String(fixture?.kickoff_time || "").trim();
  const kickoffDate = kickoffRaw ? new Date(kickoffRaw) : null;
  if (!kickoffDate || Number.isNaN(kickoffDate.getTime())) {
    return nowIso();
  }
  const scheduled = new Date(kickoffDate.getTime() - preMatchWindowMinutes * 60 * 1000);
  const nowDate = new Date();
  return scheduled.getTime() < nowDate.getTime() ? nowDate.toISOString() : scheduled.toISOString();
}

function buildNotificationAlertRecord(accountState, entry) {
  const userId = String(accountState?.user?.id || "").trim();
  const fixture = entry.fixture;
  const profile = buildFollowMatchProfile(entry);
  const publishClass = String(fixture?.publish_class || fixture?.fixture_class || "MONITOR").toUpperCase();
  const marketFamily = marketFamilyLabel(fixture?.signal_summary?.market_family);
  const kickoffTime = String(fixture?.kickoff_time || "").trim();
  const updatedAt = String(fixture?.updated_at || "").trim();
  const dedupeKey = [
    userId,
    "telegram",
    String(fixture?.fixture_key || "").trim(),
    publishClass,
    marketFamily,
    kickoffTime,
    updatedAt,
  ].join(":");
  const timestamp = nowIso();
  return {
    id: `alert_${crypto.randomUUID()}`,
    user_id: userId,
    channel: "telegram",
    alert_kind: publishClass === "DEPLOY" ? "follow_deploy" : "follow_intelligence",
    fixture_key: String(fixture?.fixture_key || "").trim(),
    fixture_id: String(fixture?.fixture_id || "").trim() || null,
    fixture_label: fixturePreferenceLabel(fixture),
    league: String(fixture?.league || "").trim() || null,
    market_family: marketFamily,
    publish_class: publishClass,
    reasons_json: JSON.stringify(entry.reasons || []),
    payload_json: JSON.stringify({
      fixture,
      reasons: entry.reasons || [],
      relevance_score: profile.score,
      relevance_tier: profile.relevance_tier,
      auto_gate: profile.auto_gate,
      telegram_eligible: profile.telegram_eligible,
    }),
    dedupe_key: dedupeKey,
    notification_priority: String(
      fixture?.follow_relevance?.notification_priority || profile.relevance_tier || "normal"
    ),
    scheduled_for: computeScheduledFor(accountState, fixture),
    status: "queued",
    created_at: timestamp,
    updated_at: timestamp,
  };
}

async function queueFollowedTelegramAlertsForAccount(accountDb, env, accountState) {
  if (!accountDb || !accountState?.user?.id) {
    return { attempted: 0, queued: 0, matched: 0 };
  }
  const fixtures = await loadFixtureIntelligenceArtifact(env);
  const matches = getFollowedFixtureMatches(accountState, fixtures).map((entry) => ({
    ...entry,
    accountState,
  }));
  const alerts = matches
    .filter((entry) => shouldQueueTelegramAlert(accountState, entry))
    .map((entry) => buildNotificationAlertRecord(accountState, entry))
    .filter((entry) => entry.fixture_key && entry.user_id);
  const result = await upsertNotificationAlerts(accountDb, alerts);
  return {
    ...result,
    matched: matches.length,
  };
}

async function dispatchDueTelegramAlerts(accountDb, env, options = {}) {
  if (!accountDb || !getTelegramBotToken(env)) {
    return { attempted: 0, delivered: 0, failed: 0 };
  }
  const dueAlerts = await listDueNotificationAlerts(accountDb, {
    nowIso: nowIso(),
    limit: Number(options.limit || 25),
    userId: options.userId || "",
  });

  let delivered = 0;
  let failed = 0;
  for (const alert of dueAlerts) {
    let payload = {};
    try {
      payload = JSON.parse(alert.payload_json || "{}");
    } catch {
      payload = {};
    }
    const fixture = payload.fixture || null;
    if (!fixture) {
      await markNotificationAlertFailed(accountDb, alert.id, "Missing fixture payload.");
      failed += 1;
      continue;
    }
    const userEmail = String(options.userEmailMap?.get?.(alert.user_id) || "").trim();
    const accountState = userEmail ? await getAccountStateByEmail(accountDb, userEmail) : null;
    const chatId = String(accountState?.telegram_link?.telegram_chat_id || "").trim();
    if (!chatId) {
      await markNotificationAlertFailed(accountDb, alert.id, "Telegram chat is no longer linked.");
      failed += 1;
      continue;
    }

    const deliveredOk = await sendTelegramMessage(env, chatId, formatTelegramFixtureAlert(fixture, env));
    if (!deliveredOk) {
      await markNotificationAlertFailed(accountDb, alert.id, "Telegram delivery failed.");
      failed += 1;
      continue;
    }

    await markNotificationAlertDelivered(accountDb, alert.id);
    delivered += 1;
  }

  return {
    attempted: dueAlerts.length,
    delivered,
    failed,
  };
}

async function handleTelegramWebhook(request, env) {
  const botToken = getTelegramBotToken(env);
  const webhookSecret = getTelegramWebhookSecret(env);
  if (!botToken) {
    return configError("TELEGRAM_BOT_TOKEN is required for Telegram bot webhook handling.", [
      "TELEGRAM_BOT_TOKEN",
    ]);
  }
  if (!webhookSecret) {
    return configError("TELEGRAM_WEBHOOK_SECRET is required for Telegram bot webhook handling.", [
      "TELEGRAM_WEBHOOK_SECRET",
    ]);
  }

  const providedSecret = String(request.headers.get(TELEGRAM_WEBHOOK_SECRET_HEADER) || "").trim();
  if (!providedSecret || providedSecret !== webhookSecret) {
    return unauthorizedError("Telegram webhook secret did not match.");
  }

  let update;
  try {
    update = await request.json();
  } catch (error) {
    return requestError("Telegram webhook body must be valid JSON.", error.message);
  }

  const message = update?.message || update?.edited_message || null;
  const chatId = message?.chat?.id ? String(message.chat.id) : "";
  const telegramUserId = message?.from?.id ? String(message.from.id) : "";
  const telegramUsername = String(message?.from?.username || "").trim();
  const text = String(message?.text || "").trim();

  if (!chatId || !telegramUserId) {
    return json({
      ok: true,
      status: "telegram_webhook_ignored",
      message: "Update did not contain a usable message payload.",
    });
  }

  const code = extractTelegramLinkCode(text);
  if (!code) {
    await sendTelegramMessage(
      env,
      chatId,
      "Send the one-time Odds Genius code from your account page, or open the Telegram deep link there to finish linking."
    );
    return json({
      ok: true,
      status: "telegram_webhook_processed",
      action: "instructions_sent",
    });
  }

  const completion = await completeTelegramLinkFromCode(
    env,
    {
      code,
      telegram_user_id: telegramUserId,
      telegram_username: telegramUsername || null,
      telegram_chat_id: chatId,
    },
    {
      ip_hint: getRequestIp(request) || "telegram_webhook",
      user_agent_hint: getUserAgentHint(request) || "telegram_webhook",
    }
  );

  if (!completion.ok) {
    const errorPayload = await completion.response.clone().json().catch(() => ({}));
    const errorText =
      errorPayload?.status === "invalid_link_code"
        ? "That link code is invalid or expired. Generate a fresh Telegram link code from your Odds Genius account page and try again."
        : "Odds Genius could not complete the Telegram link just now. Please generate a fresh code from your account page and try again.";
    await sendTelegramMessage(env, chatId, errorText);
    return json({
      ok: false,
      status: "telegram_link_failed",
      upstream_status: errorPayload?.status || "unknown_error",
    });
  }

  await sendTelegramMessage(
    env,
    chatId,
    "Telegram is now linked to your Odds Genius premium account. Future premium alerts can be delivered here."
  );

  return json({
    ok: true,
    status: "telegram_webhook_processed",
    action: "telegram_link_completed",
    account: completion.payload.account,
  });
}

async function handleTelegramTestAlert(request, env) {
  const sessionAccess = await verifySessionAccess(request, env);
  if (!sessionAccess.ok) {
    return json(
      {
        ok: false,
        status: sessionAccess.status || "unauthenticated",
        message: sessionAccess.message || "Verify your email before sending a Telegram test alert.",
      },
      401,
      sessionFailureHeaders(sessionAccess.status)
    );
  }

  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for Telegram alert testing.", ["ACCOUNT_DB"]);
  }
  if (!getTelegramBotToken(env)) {
    return configError("TELEGRAM_BOT_TOKEN is required for Telegram alert testing.", ["TELEGRAM_BOT_TOKEN"]);
  }

  const accountState = await getAccountStateByEmail(accountDb, sessionAccess.email);
  const telegramLink = accountState?.telegram_link || null;
  const chatId = String(telegramLink?.telegram_chat_id || "").trim();
  if (!telegramLink || telegramLink.link_status !== "linked" || !chatId) {
    return json(
      {
        ok: false,
        status: "telegram_not_linked",
        message: "Link Telegram from your account page before sending a test alert.",
      },
      400
    );
  }

  const messageLines = [
    "Odds Genius Telegram delivery is live.",
    "",
    "This is a test premium alert from your linked account.",
    `Membership: ${sessionAccess.subscription_status || "active"}`,
    "Future use: elite deployment alerts, premium comms, and controlled acca drops.",
  ];

  const delivered = await sendTelegramMessage(env, chatId, messageLines.join("\n"));
  if (!delivered) {
    return json(
      {
        ok: false,
        status: "telegram_delivery_failed",
        message: "Telegram test alert could not be delivered.",
      },
      502
    );
  }

  try {
    await recordAuthEvent(accountDb, {
      user_id: accountState?.user?.id || null,
      email_normalized: sessionAccess.email,
      event_type: "telegram_test_alert_sent",
      ip_hint: getRequestIp(request) || null,
      user_agent_hint: getUserAgentHint(request),
      metadata: {
        telegram_chat_id_hint: chatId.slice(-6),
      },
    });
  } catch {
    // Best-effort audit trail only.
  }

  return json({
    ok: true,
    status: "telegram_test_alert_sent",
    message: "Telegram test alert sent to your linked account.",
  });
}

async function handleTelegramFixtureAlert(request, env) {
  const sessionAccess = await verifySessionAccess(request, env);
  if (!sessionAccess.ok) {
    return json(
      {
        ok: false,
        status: sessionAccess.status || "unauthenticated",
        message: sessionAccess.message || "Verify your email before sending a fixture intelligence alert.",
      },
      401,
      sessionFailureHeaders(sessionAccess.status)
    );
  }

  const accountDb = getAccountDb(env);
  if (!accountDb) {
    return configError("ACCOUNT_DB D1 binding is required for Telegram fixture alerts.", ["ACCOUNT_DB"]);
  }
  if (!getTelegramBotToken(env)) {
    return configError("TELEGRAM_BOT_TOKEN is required for Telegram fixture alerts.", ["TELEGRAM_BOT_TOKEN"]);
  }

  let payload;
  try {
    payload = await request.json();
  } catch (error) {
    return requestError("Telegram fixture alert body must be valid JSON.", error.message);
  }

  const fixtureKey = String(payload?.fixture_key || "").trim();
  if (!fixtureKey) {
    return requestError("`fixture_key` is required for Telegram fixture alerts.");
  }

  const accountState = await getAccountStateByEmail(accountDb, sessionAccess.email);
  const telegramLink = accountState?.telegram_link || null;
  const chatId = String(telegramLink?.telegram_chat_id || "").trim();
  if (!telegramLink || telegramLink.link_status !== "linked" || !chatId) {
    return json(
      {
        ok: false,
        status: "telegram_not_linked",
        message: "Link Telegram from your account page before sending a fixture alert.",
      },
      400
    );
  }

  let fixture;
  try {
    const fixtures = await loadFixtureIntelligenceArtifact(env);
    fixture = fixtures.find((row) => String(row?.fixture_key || "").trim() === fixtureKey) || null;
  } catch (error) {
    return json(
      {
        ok: false,
        status: "fixture_intelligence_unavailable",
        message: error.message || "Published fixture intelligence could not be loaded.",
      },
      502
    );
  }

  if (!fixture) {
    return json(
      {
        ok: false,
        status: "fixture_not_found",
        message: "That fixture is not present in the current published intelligence window.",
      },
      404
    );
  }

  const delivered = await sendTelegramMessage(env, chatId, formatTelegramFixtureAlert(fixture, env));
  if (!delivered) {
    return json(
      {
        ok: false,
        status: "telegram_delivery_failed",
        message: "Fixture intelligence alert could not be delivered to Telegram.",
      },
      502
    );
  }

  try {
    await recordAuthEvent(accountDb, {
      user_id: accountState?.user?.id || null,
      email_normalized: sessionAccess.email,
      event_type: "telegram_fixture_alert_sent",
      ip_hint: getRequestIp(request) || null,
      user_agent_hint: getUserAgentHint(request),
      metadata: {
        fixture_key: fixtureKey,
        publish_class: String(fixture?.publish_class || fixture?.fixture_class || ""),
        market_family: String(fixture?.signal_summary?.market_family || ""),
        telegram_chat_id_hint: chatId.slice(-6),
      },
    });
  } catch {
    // Best-effort audit trail only.
  }

  return json({
    ok: true,
    status: "telegram_fixture_alert_sent",
    message: "Fixture intelligence alert sent to your linked Telegram account.",
    fixture_key: fixtureKey,
  });
}

async function handleScheduledAlertTick(env) {
  const accountDb = getAccountDb(env);
  if (!accountDb || !getTelegramBotToken(env)) {
    return;
  }

  const eligibleUsers = await listUsersEligibleForTelegramAlerts(accountDb);
  const userEmailMap = new Map();
  for (const user of eligibleUsers) {
    const email = String(user?.email_normalized || user?.email || "").trim().toLowerCase();
    if (!email) {
      continue;
    }
    const accountState = await getAccountStateByEmail(accountDb, email);
    if (!accountState?.user?.id) {
      continue;
    }
    userEmailMap.set(accountState.user.id, email);
    await queueFollowedTelegramAlertsForAccount(accountDb, env, accountState);
  }

  await dispatchDueTelegramAlerts(accountDb, env, {
    limit: 100,
    userEmailMap,
  });
}

async function handleTelegramLinkComplete(request, env) {
  let payload;
  try {
    payload = await request.json();
  } catch (error) {
    return requestError("Telegram link completion body must be valid JSON.", error.message);
  }

  const completion = await completeTelegramLinkFromCode(env, payload, {
    ip_hint: getRequestIp(request) || null,
    user_agent_hint: getUserAgentHint(request),
  });
  if (!completion.ok) {
    return completion.response;
  }

  return json(completion.payload);
}

async function handleLogout() {
  const headers = new Headers({
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
  });
  headers.append("set-cookie", clearCookieHeader(AUTH_SESSION_COOKIE));
  headers.append("set-cookie", clearCookieHeader("og_premium_token"));
  return new Response(
    JSON.stringify(
      {
        ok: true,
        status: "logged_out",
      },
      null,
      2
    ),
    {
      status: 200,
      headers,
    }
  );
}

const sanitizeCorrectScoreShortlist = (value) => {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .slice(0, 3)
    .map((entry) => {
      if (!entry || typeof entry !== "object") {
        return null;
      }
      const scoreline = typeof entry.scoreline === "string" ? entry.scoreline.trim() : "";
      const probability = entry.probability;
      if (!scoreline || !isFiniteNumber(probability)) {
        return null;
      }
      return {
        scoreline,
        probability,
      };
    })
    .filter(Boolean);
};

const sanitizePremiumRow = (row) => {
  if (!row || typeof row !== "object" || Array.isArray(row)) {
    return null;
  }

  const sanitized = {};
  for (const key of PREMIUM_ALLOWED_FIELDS) {
    if (!(key in row)) {
      continue;
    }

    if (key === "correct_score_shortlist") {
      sanitized[key] = sanitizeCorrectScoreShortlist(row[key]);
      continue;
    }

    if (key === "reason_tokens") {
      sanitized[key] = Array.isArray(row[key])
        ? row[key].filter((token) => typeof token === "string").map((token) => token.trim()).filter(Boolean)
        : [];
      continue;
    }

    sanitized[key] = row[key];
  }

  return sanitized;
};

const buildCachedPremiumPayload = (loaded) => ({
  generated_at: loaded.generated_at,
  count: loaded.rows.length,
  predictions: loaded.rows,
  source: loaded.source,
});

async function readCachedPremiumPayload(cacheKey) {
  const cache = getPremiumCache();
  if (!cache) {
    return { ok: true, hit: false, cache_status: "bypass" };
  }

  const response = await cache.match(cacheKey);
  if (!response) {
    return { ok: true, hit: false, cache_status: "miss" };
  }

  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    return {
      ok: false,
      status: "premium_cache_invalid_json",
      message: "Cached premium payload could not be decoded.",
      recommendation: error.message,
      http_status: 500,
    };
  }

  if (!payload || !Array.isArray(payload.predictions)) {
    return {
      ok: false,
      status: "premium_cache_shape_invalid",
      message: "Cached premium payload is missing the predictions array.",
      recommendation: "Rebuild the cached payload from the published premium source.",
      http_status: 500,
    };
  }

  return {
    ok: true,
    hit: true,
    cache_status: "hit",
    payload,
  };
}

async function writeCachedPremiumPayload(cacheKey, payload) {
  const cache = getPremiumCache();
  if (!cache) {
    return;
  }

  const response = new Response(JSON.stringify(payload), {
    status: 200,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": `public, max-age=${PREMIUM_CACHE_TTL_SECONDS}`,
    },
  });

  await cache.put(cacheKey, response);
}

async function fetchPremiumSource(request, env) {
  const resolved = resolvePremiumSourceUrl(request, env);
  if (!resolved.ok) {
    return resolved;
  }
  const { targetUrl } = resolved;

  let response;
  try {
    response = await fetch(targetUrl, { headers: { accept: "application/json" } });
  } catch (error) {
    return {
      ok: false,
      status: "premium_source_fetch_failed",
      message: "Premium source fetch failed.",
      recommendation: error.message,
      http_status: 502,
    };
  }

  if (!response.ok) {
    return {
      ok: false,
      status: "premium_source_unavailable",
      message: `Premium source responded with ${response.status}.`,
      recommendation: "Check PREMIUM_DATA_SOURCE and deployment routing.",
      http_status: 502,
    };
  }

  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    return {
      ok: false,
      status: "premium_source_invalid_json",
      message: "Premium source did not return valid JSON.",
      recommendation: error.message,
      http_status: 502,
    };
  }

  return {
    ok: true,
    payload,
    source: targetUrl,
    fetched_at: (() => {
      const lastModified = response.headers.get("last-modified");
      const parsed = lastModified ? new Date(lastModified) : null;
      return parsed && !Number.isNaN(parsed.getTime()) ? parsed.toISOString() : null;
    })(),
  };
}

async function loadPremiumPredictions(request, env) {
  const resolved = resolvePremiumSourceUrl(request, env);
  if (!resolved.ok) {
    return resolved;
  }

  const cacheKey = buildPremiumCacheKey(resolved.targetUrl);
  const cached = await readCachedPremiumPayload(cacheKey);
  if (!cached.ok) {
    return cached;
  }

  if (cached.hit) {
    return {
      ok: true,
      generated_at: typeof cached.payload.generated_at === "string" ? cached.payload.generated_at : null,
      rows: cached.payload.predictions,
      source: cached.payload.source || resolved.targetUrl,
      count: Number.isFinite(Number(cached.payload.count))
        ? Number(cached.payload.count)
        : cached.payload.predictions.length,
      cache_status: cached.cache_status,
    };
  }

  const fetched = await fetchPremiumSource(request, env);
  if (!fetched.ok) {
    return fetched;
  }

  const payload = fetched.payload;
  const rows = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.predictions)
      ? payload.predictions
      : Array.isArray(payload?.data)
        ? payload.data
        : null;

  if (!rows) {
    return {
      ok: false,
      status: "premium_source_shape_invalid",
      message: "Premium source JSON must be an array or an object with predictions/data array.",
      recommendation: "Publish premium predictions as an array or wrapped object with predictions.",
      http_status: 502,
    };
  }

  const sanitizedRows = rows.map(sanitizePremiumRow).filter(Boolean);

  const loaded = {
    ok: true,
    generated_at:
      typeof payload?.generated_at === "string" ? payload.generated_at : fetched.fetched_at || new Date().toISOString(),
    rows: sanitizedRows,
    source: fetched.source,
    count: sanitizedRows.length,
    cache_status: cached.cache_status,
  };

  await writeCachedPremiumPayload(cacheKey, buildCachedPremiumPayload(loaded));

  return loaded;
}

async function handleRequest(request, env) {
  const url = new URL(request.url);
  const { pathname } = url;
  let response;

  if (request.method === "OPTIONS") {
    return withCors(new Response(null, { status: 204 }), request, env);
  }

  if (pathname === "/health") {
    if (request.method !== "GET") {
      response = methodNotAllowed("GET");
      return withCors(response, request, env);
    }
    response = json({
      ok: true,
      service: "odds-genius-worker",
      status: "placeholder_ready",
      routes: [
        "GET /health",
        "POST /api/auth/magic-link/request",
        "GET /api/auth/magic-link/verify",
        "GET /api/auth/session",
        "POST /api/auth/logout",
        "GET /api/account/state",
        "GET /api/account/sessions",
        "POST /api/account/sessions/revoke",
        "POST /api/account/sessions/revoke-others",
        "POST /api/account/sessions/make-primary",
        "GET /internal/accounts/lookup",
        "GET /internal/accounts/:user_id",
        "GET /internal/accounts/:user_id/flags",
        "GET /internal/accounts/:user_id/notes",
        "POST /internal/accounts/:user_id/notes",
        "GET /internal/accounts/:user_id/timeline",
        "POST /internal/accounts/:user_id/restrict",
        "POST /internal/accounts/:user_id/suspend",
        "POST /internal/accounts/:user_id/reinstate",
        "POST /internal/accounts/:user_id/flags/:flag_id/resolve",
        "POST /internal/accounts/:user_id/flags/:flag_id/dismiss",
        "POST /internal/accounts/:user_id/review-outcome",
        "POST /api/account/preferences",
        "GET /api/account/alerts",
        "POST /api/account/alerts/refresh",
        "POST /api/account/alerts/dispatch",
        "POST /api/account/telegram/link/start",
        "POST /api/account/telegram/link/complete",
        "POST /api/account/telegram/test-alert",
        "POST /api/account/telegram/fixture-alert",
        "GET /api/widgets/football/standings",
        "GET /api/widgets/football/fixture-lookup",
        "POST /api/telegram/webhook",
        "POST /api/stripe/checkout",
        "POST /api/premium/token",
        "POST /api/stripe/portal",
        "POST /api/stripe/webhook",
        "GET /api/premium/predictions",
      ],
      env_summary: envSummary(env),
    });
    return withCors(response, request, env);
  }

  if (pathname === "/api/stripe/checkout") {
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await createCheckoutSession(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/auth/magic-link/request") {
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await handleMagicLinkRequest(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/auth/magic-link/verify") {
    if (request.method !== "GET") {
      response = methodNotAllowed("GET");
      return withCors(response, request, env);
    }
    response = await handleMagicLinkVerify(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/auth/session") {
    if (request.method !== "GET") {
      response = methodNotAllowed("GET");
      return withCors(response, request, env);
    }
    response = await handleAuthSession(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/auth/logout") {
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await handleLogout(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/account/state") {
    if (request.method !== "GET") {
      response = methodNotAllowed("GET");
      return withCors(response, request, env);
    }
    response = await handleAccountState(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/account/sessions") {
    if (request.method !== "GET") {
      response = methodNotAllowed("GET");
      return withCors(response, request, env);
    }
    response = await handleAccountSessionsState(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/account/sessions/revoke") {
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await handleAccountSessionRevoke(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/account/sessions/revoke-others") {
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await handleAccountSessionsRevokeOthers(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/account/sessions/make-primary") {
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await handleAccountSessionMakePrimary(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/internal/accounts/lookup") {
    if (request.method !== "GET") {
      response = methodNotAllowed("GET");
      return withCors(response, request, env);
    }
    response = await handleInternalAccountLookup(request, env);
    return withCors(response, request, env);
  }

  const internalAccountMatch = pathname.match(/^\/internal\/accounts\/([^/]+)(?:\/(flags|notes|timeline))?$/);
  if (internalAccountMatch) {
    const userId = decodeURIComponent(internalAccountMatch[1] || "").trim();
    const subroute = String(internalAccountMatch[2] || "").trim();
    if (!userId) {
      response = requestError("Internal account id is required.");
      return withCors(response, request, env);
    }
    if (!subroute) {
      if (request.method !== "GET") {
        response = methodNotAllowed("GET");
        return withCors(response, request, env);
      }
      response = await handleInternalAccountRead(request, env, userId);
      return withCors(response, request, env);
    }
    if (subroute === "flags") {
      if (request.method !== "GET") {
        response = methodNotAllowed("GET");
        return withCors(response, request, env);
      }
      response = await handleInternalAccountFlagsRead(request, env, userId);
      return withCors(response, request, env);
    }
    if (subroute === "notes") {
      if (request.method === "GET") {
        response = await handleInternalAccountNotesRead(request, env, userId);
        return withCors(response, request, env);
      }
      if (request.method === "POST") {
        response = await handleInternalAccountNoteCreate(request, env, userId);
        return withCors(response, request, env);
      }
      response = methodNotAllowed("GET,POST");
      return withCors(response, request, env);
    }
    if (subroute === "timeline") {
      if (request.method !== "GET") {
        response = methodNotAllowed("GET");
        return withCors(response, request, env);
      }
      response = await handleInternalAccountTimelineRead(request, env, userId);
      return withCors(response, request, env);
    }
  }

  const internalAccountActionMatch = pathname.match(/^\/internal\/accounts\/([^/]+)\/(restrict|suspend|reinstate)$/);
  if (internalAccountActionMatch) {
    const userId = decodeURIComponent(internalAccountActionMatch[1] || "").trim();
    const action = String(internalAccountActionMatch[2] || "").trim();
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    if (action === "restrict") {
      response = await handleInternalAccountRestrict(request, env, userId);
      return withCors(response, request, env);
    }
    if (action === "suspend") {
      response = await handleInternalAccountSuspend(request, env, userId);
      return withCors(response, request, env);
    }
    if (action === "reinstate") {
      response = await handleInternalAccountReinstate(request, env, userId);
      return withCors(response, request, env);
    }
  }

  const internalFlagActionMatch = pathname.match(/^\/internal\/accounts\/([^/]+)\/flags\/([^/]+)\/(resolve|dismiss)$/);
  if (internalFlagActionMatch) {
    const userId = decodeURIComponent(internalFlagActionMatch[1] || "").trim();
    const flagId = decodeURIComponent(internalFlagActionMatch[2] || "").trim();
    const action = String(internalFlagActionMatch[3] || "").trim();
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await handleInternalFlagStatusUpdate(
      request,
      env,
      userId,
      flagId,
      action === "dismiss" ? "dismissed" : "resolved"
    );
    return withCors(response, request, env);
  }

  const internalReviewOutcomeMatch = pathname.match(/^\/internal\/accounts\/([^/]+)\/review-outcome$/);
  if (internalReviewOutcomeMatch) {
    const userId = decodeURIComponent(internalReviewOutcomeMatch[1] || "").trim();
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await handleInternalReviewOutcomeUpdate(request, env, userId);
    return withCors(response, request, env);
  }

  if (pathname === "/api/account/preferences") {
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await handleAccountPreferencesUpdate(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/account/alerts") {
    if (request.method !== "GET") {
      response = methodNotAllowed("GET");
      return withCors(response, request, env);
    }
    response = await handleAccountAlertsState(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/account/alerts/refresh") {
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await handleAccountAlertsRefresh(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/account/alerts/dispatch") {
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await handleAccountAlertsDispatch(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/account/telegram/link/start") {
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await handleTelegramLinkStart(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/account/telegram/link/complete") {
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await handleTelegramLinkComplete(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/account/telegram/test-alert") {
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await handleTelegramTestAlert(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/account/telegram/fixture-alert") {
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await handleTelegramFixtureAlert(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/widgets/football/standings") {
    response = await proxyWidgetStandingsResponse(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/widgets/football/fixture-lookup") {
    response = await lookupWidgetFixtureResponse(request, env);
    return withCors(response, request, env);
  }

  if (pathname.startsWith("/api/widgets/football/")) {
    response = await proxyGenericWidgetFootballResponse(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/telegram/webhook") {
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await handleTelegramWebhook(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/premium/token") {
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await handlePremiumTokenIssue(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/stripe/portal") {
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = placeholder(
      pathname,
      "Verify subscriber identity and create a Stripe Customer Portal session.",
      env,
      {
        security_note: "Portal access should require authenticated subscriber context.",
      }
    );
    return withCors(response, request, env);
  }

  if (pathname === "/api/stripe/webhook") {
    if (request.method !== "POST") {
      response = methodNotAllowed("POST");
      return withCors(response, request, env);
    }
    response = await handleStripeWebhook(request, env);
    return withCors(response, request, env);
  }

  if (pathname === "/api/premium/predictions") {
    if (request.method !== "GET") {
      response = methodNotAllowed("GET");
      return withCors(response, request, env);
    }
    response = await handlePremiumPredictions(request, env);
    return withCors(response, request, env);
  }

  response = notFound(pathname);
  return withCors(response, request, env);
}

export default {
  fetch: handleRequest,
  scheduled: async (_controller, env) => {
    await handleScheduledAlertTick(env);
  },
};
