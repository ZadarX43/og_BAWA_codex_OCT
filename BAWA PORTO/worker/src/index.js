import {
  buildSubscriberRecord,
  getSubscriberStateStore,
  loadSubscriberRecordByEmail,
  loadSubscriberRecordBySubscriptionId,
  persistSubscriberRecord,
} from "./subscriber_store.js";
import { issuePremiumToken, verifyPremiumAccess } from "./auth.js";

const STRIPE_WEBHOOK_TOLERANCE_SECONDS = 300;
const PREMIUM_CACHE_TTL_SECONDS = 300;
const PREMIUM_CACHE_VERSION = "v1";
const AUTH_MAGIC_LINK_TTL_SECONDS = 15 * 60;
const AUTH_SESSION_TTL_SECONDS = 7 * 24 * 60 * 60;
const AUTH_REQUEST_IP_COOLDOWN_SECONDS = 60;
const AUTH_REQUEST_EMAIL_COOLDOWN_SECONDS = 5 * 60;
const AUTH_MAGIC_KEY_PREFIX = "auth_magic:";
const AUTH_RATE_LIMIT_KEY_PREFIX = "auth_rl:";
const AUTH_SESSION_COOKIE = "og_premium_session";
const PREMIUM_ALLOWED_FIELDS = [
  "fixture_id",
  "fixture_key",
  "kickoff_time",
  "league",
  "home_team",
  "away_team",
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
});

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
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

const redirect = (location, status = 303, extraHeaders = {}) =>
  new Response(null, {
    status,
    headers: {
      location,
      ...extraHeaders,
    },
  });

const clearCookieHeader = (name) =>
  `${name}=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0; Expires=Thu, 01 Jan 1970 00:00:00 GMT`;

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

const maskEmailHint = (email) => {
  const normalized = normalizeEmail(email);
  const [localPart, domain] = normalized.split("@");
  if (!localPart || !domain) {
    return "";
  }
  const visible = localPart.slice(0, 1) || "u";
  return `${visible}***@${domain}`;
};

const getSessionSecret = (env) => String(env.AUTH_SESSION_SECRET || env.AUTH_MAGIC_LINK_SECRET || "").trim();

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
  `${AUTH_SESSION_COOKIE}=${token}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=${AUTH_SESSION_TTL_SECONDS}`;

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
    return json({
      ok: true,
      received: true,
      stored: true,
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
      401
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

    const siteUrl = normalizeSiteUrl(env.SITE_URL || new URL(request.url).origin);
    const verifyUrl = `${siteUrl}/api/auth/magic-link/verify?token=${encodeURIComponent(token)}`;
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

  const sessionPayload = {
    email,
    customer_id: customerId,
    subscription_id: subscriptionId,
    exp: Math.floor(Date.now() / 1000) + AUTH_SESSION_TTL_SECONDS,
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
    return json({
      ok: true,
      authenticated: false,
      entitled: false,
      status: access.status || "unauthenticated",
    });
  }

  return json({
    ok: true,
    authenticated: false,
    entitled: false,
    status: sessionAccess.status === "missing_session" ? "" : sessionAccess.status,
  });
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
    generated_at: typeof payload?.generated_at === "string" ? payload.generated_at : null,
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

  if (pathname === "/health") {
    if (request.method !== "GET") {
      return methodNotAllowed("GET");
    }
    return json({
      ok: true,
      service: "odds-genius-worker",
      status: "placeholder_ready",
      routes: [
        "GET /health",
        "POST /api/auth/magic-link/request",
        "GET /api/auth/magic-link/verify",
        "GET /api/auth/session",
        "POST /api/auth/logout",
        "POST /api/stripe/checkout",
        "POST /api/premium/token",
        "POST /api/stripe/portal",
        "POST /api/stripe/webhook",
        "GET /api/premium/predictions",
      ],
      env_summary: envSummary(env),
    });
  }

  if (pathname === "/api/stripe/checkout") {
    if (request.method !== "POST") {
      return methodNotAllowed("POST");
    }
    return createCheckoutSession(request, env);
  }

  if (pathname === "/api/auth/magic-link/request") {
    if (request.method !== "POST") {
      return methodNotAllowed("POST");
    }
    return handleMagicLinkRequest(request, env);
  }

  if (pathname === "/api/auth/magic-link/verify") {
    if (request.method !== "GET") {
      return methodNotAllowed("GET");
    }
    return handleMagicLinkVerify(request, env);
  }

  if (pathname === "/api/auth/session") {
    if (request.method !== "GET") {
      return methodNotAllowed("GET");
    }
    return handleAuthSession(request, env);
  }

  if (pathname === "/api/auth/logout") {
    if (request.method !== "POST") {
      return methodNotAllowed("POST");
    }
    return handleLogout(request, env);
  }

  if (pathname === "/api/premium/token") {
    if (request.method !== "POST") {
      return methodNotAllowed("POST");
    }
    return handlePremiumTokenIssue(request, env);
  }

  if (pathname === "/api/stripe/portal") {
    if (request.method !== "POST") {
      return methodNotAllowed("POST");
    }
    return placeholder(
      pathname,
      "Verify subscriber identity and create a Stripe Customer Portal session.",
      env,
      {
        security_note: "Portal access should require authenticated subscriber context.",
      }
    );
  }

  if (pathname === "/api/stripe/webhook") {
    if (request.method !== "POST") {
      return methodNotAllowed("POST");
    }
    return handleStripeWebhook(request, env);
  }

  if (pathname === "/api/premium/predictions") {
    if (request.method !== "GET") {
      return methodNotAllowed("GET");
    }
    return handlePremiumPredictions(request, env);
  }

  return notFound(pathname);
}

export default {
  fetch: handleRequest,
};
