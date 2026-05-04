import {
  buildSubscriberRecord,
  getSubscriberStateStore,
  persistSubscriberRecord,
} from "./subscriber_store.js";
import { issuePremiumToken, verifyPremiumAccess } from "./auth.js";

const STRIPE_WEBHOOK_TOLERANCE_SECONDS = 300;
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
  const access = await verifyPremiumAccess(request, env);
  if (!access.ok) {
    return json(
      {
        ok: false,
        status: access.status,
        message: access.message,
        recommendation: access.recommendation,
        route: "/api/premium/predictions",
        locked: true,
        data_note: "Premium predictions remain unavailable until verified token-based entitlement is live.",
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
      count: loaded.rows.length,
      predictions: loaded.rows,
    },
    200
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

async function fetchPremiumSource(request, env) {
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

  let targetUrl;
  if (/^https?:\/\//i.test(source)) {
    targetUrl = source;
  } else if (source.startsWith("/")) {
    targetUrl = new URL(source, request.url).toString();
  } else {
    return {
      ok: false,
      status: "premium_source_unsupported",
      message: "PREMIUM_DATA_SOURCE must currently be an absolute URL or same-origin path.",
      recommendation: "For production, use KV, R2, or a protected static asset fetch strategy rather than local filesystem paths.",
      http_status: 501,
    };
  }

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

  return {
    ok: true,
    generated_at: typeof payload?.generated_at === "string" ? payload.generated_at : null,
    rows: sanitizedRows,
    source: fetched.source,
  };
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
