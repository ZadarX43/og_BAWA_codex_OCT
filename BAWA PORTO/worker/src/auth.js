import {
  getSubscriberStateStore,
  loadSubscriberRecordBySubscriptionId,
} from "./subscriber_store.js";

const ACTIVE_STATUSES = new Set(["active", "trialing"]);
const PREMIUM_TOKEN_TTL_SECONDS = 7 * 24 * 60 * 60;

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

const extractBearerToken = (request) => {
  const authorization = request.headers.get("authorization") || "";
  const match = authorization.match(/^Bearer\s+(.+)$/i);
  return match ? match[1].trim() : null;
};

const extractCookieToken = (request) => {
  const cookieHeader = request.headers.get("cookie") || "";
  for (const fragment of cookieHeader.split(";")) {
    const [name, value] = fragment.split("=", 2).map((part) => part.trim());
    if (name === "og_premium_token" && value) {
      return value;
    }
  }
  return null;
};

const getPremiumToken = (request) => extractBearerToken(request) || extractCookieToken(request);

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

const parseToken = (token) => {
  const parts = String(token || "").split(".");
  if (parts.length !== 2 || !parts[0] || !parts[1]) {
    return { ok: false, status: "malformed_token", message: "Premium token format is invalid." };
  }

  const payloadBytes = base64UrlToBytes(parts[0]);
  const signatureBytes = base64UrlToBytes(parts[1]);
  if (!payloadBytes || !signatureBytes) {
    return { ok: false, status: "malformed_token", message: "Premium token could not be decoded." };
  }

  const payloadText = bytesToUtf8(payloadBytes);
  if (!payloadText) {
    return { ok: false, status: "malformed_token", message: "Premium token payload is unreadable." };
  }

  let payload;
  try {
    payload = JSON.parse(payloadText);
  } catch {
    return { ok: false, status: "malformed_token", message: "Premium token payload is not valid JSON." };
  }

  return {
    ok: true,
    payloadSegment: parts[0],
    payload,
    signatureBytes,
  };
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

const signPayloadSegment = async (secret, payloadSegment) => {
  const key = await crypto.subtle.importKey(
    "raw",
    textEncoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );

  const signature = await crypto.subtle.sign("HMAC", key, textEncoder.encode(payloadSegment));
  return new Uint8Array(signature);
};

const buildTokenPayloadSegment = (payload) => bytesToBase64Url(textEncoder.encode(JSON.stringify(payload)));

const validatePayloadShape = (payload) => {
  if (!payload || typeof payload !== "object") {
    return { ok: false, status: "invalid_payload", message: "Premium token payload must be an object." };
  }

  const customerId = typeof payload.customer_id === "string" ? payload.customer_id.trim() : "";
  const subscriptionId = typeof payload.subscription_id === "string" ? payload.subscription_id.trim() : "";
  const exp = Number(payload.exp);

  if (!customerId || !subscriptionId) {
    return {
      ok: false,
      status: "invalid_payload",
      message: "Premium token payload must include customer_id and subscription_id.",
    };
  }

  if (!Number.isFinite(exp)) {
    return {
      ok: false,
      status: "invalid_payload",
      message: "Premium token payload must include a numeric exp claim.",
    };
  }

  return {
    ok: true,
    customerId,
    subscriptionId,
    exp,
  };
};

export async function verifyPremiumAccess(request, env) {
  const token = getPremiumToken(request);
  if (!token) {
    return {
      ok: false,
      status: "missing_token",
      message: "Premium access token is required.",
      recommendation: "Send a signed token in Authorization or og_premium_token cookie form.",
    };
  }

  if (!env.PREMIUM_TOKEN_SECRET) {
    return {
      ok: false,
      status: "auth_not_wired",
      message: "Premium token verification secret is not configured.",
      recommendation: "Set PREMIUM_TOKEN_SECRET in Worker environment settings before enabling access.",
    };
  }

  const store = getSubscriberStateStore(env);
  if (!store) {
    return {
      ok: false,
      status: "state_binding_missing",
      message: "Subscriber state binding is unavailable.",
      recommendation: "Bind SUBSCRIBER_STATE before enabling premium delivery.",
    };
  }

  const parsed = parseToken(token);
  if (!parsed.ok) {
    return {
      ok: false,
      status: parsed.status,
      message: parsed.message,
      recommendation: "Use the v1 signed premium token format.",
    };
  }

  const expectedSignature = await signPayloadSegment(env.PREMIUM_TOKEN_SECRET, parsed.payloadSegment);
  if (!constantTimeEqual(expectedSignature, parsed.signatureBytes)) {
    return {
      ok: false,
      status: "invalid_signature",
      message: "Premium token signature verification failed.",
      recommendation: "Issue a fresh token signed with PREMIUM_TOKEN_SECRET.",
    };
  }

  const shape = validatePayloadShape(parsed.payload);
  if (!shape.ok) {
    return {
      ok: false,
      status: shape.status,
      message: shape.message,
      recommendation: "Issue a token with customer_id, subscription_id, and exp.",
    };
  }

  const nowSeconds = Math.floor(Date.now() / 1000);
  if (shape.exp <= nowSeconds) {
    return {
      ok: false,
      status: "expired_token",
      message: "Premium token has expired.",
      recommendation: "Issue a new premium access token.",
    };
  }

  let record;
  try {
    record = await loadSubscriberRecordBySubscriptionId(store, shape.subscriptionId);
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
      message: "No subscriber state record was found for this subscription.",
      recommendation: "Wait for Stripe webhook persistence or reissue the token against a valid subscription.",
    };
  }

  if (record.customer_id !== shape.customerId) {
    return {
      ok: false,
      status: "customer_mismatch",
      message: "Premium token customer_id does not match subscriber state.",
      recommendation: "Issue a token for the correct subscriber record.",
    };
  }

  if (record.subscription_id !== shape.subscriptionId) {
    return {
      ok: false,
      status: "subscription_mismatch",
      message: "Premium token subscription_id does not match subscriber state.",
      recommendation: "Issue a token for the correct subscription.",
    };
  }

  if (!ACTIVE_STATUSES.has(record.status)) {
    return {
      ok: false,
      status: "inactive_subscription",
      message: "Subscriber state is not active for premium delivery.",
      recommendation: "Require active or trialing status before granting premium access.",
    };
  }

  return {
    ok: true,
    customer_id: record.customer_id,
    subscription_id: record.subscription_id,
    price_id: record.price_id || null,
    access_tier: record.access_tier || null,
  };
}

export async function issuePremiumToken(payload, env) {
  const customerId = typeof payload?.customer_id === "string" ? payload.customer_id.trim() : "";
  const subscriptionId = typeof payload?.subscription_id === "string" ? payload.subscription_id.trim() : "";

  if (!customerId || !subscriptionId) {
    return {
      ok: false,
      status: "invalid_request",
      message: "customer_id and subscription_id are required.",
    };
  }

  if (!env.PREMIUM_TOKEN_SECRET) {
    return {
      ok: false,
      status: "auth_not_wired",
      message: "Premium token secret is not configured.",
    };
  }

  const store = getSubscriberStateStore(env);
  if (!store) {
    return {
      ok: false,
      status: "state_binding_missing",
      message: "Subscriber state binding is unavailable.",
    };
  }

  let record;
  try {
    record = await loadSubscriberRecordBySubscriptionId(store, subscriptionId);
  } catch (error) {
    return {
      ok: false,
      status: "state_lookup_failed",
      message: error.message,
    };
  }

  if (!record) {
    return {
      ok: false,
      status: "subscriber_state_missing",
      message: "No subscriber state record was found for this subscription.",
    };
  }

  if (record.customer_id !== customerId) {
    return {
      ok: false,
      status: "customer_mismatch",
      message: "customer_id does not match subscriber state.",
    };
  }

  if (record.subscription_id !== subscriptionId) {
    return {
      ok: false,
      status: "subscription_mismatch",
      message: "subscription_id does not match subscriber state.",
    };
  }

  if (!ACTIVE_STATUSES.has(record.status)) {
    return {
      ok: false,
      status: "inactive_subscription",
      message: "Subscriber state is not active for token issuance.",
    };
  }

  const exp = Math.floor(Date.now() / 1000) + PREMIUM_TOKEN_TTL_SECONDS;
  const tokenPayload = {
    customer_id: customerId,
    subscription_id: subscriptionId,
    exp,
  };
  const payloadSegment = buildTokenPayloadSegment(tokenPayload);
  const signatureBytes = await signPayloadSegment(env.PREMIUM_TOKEN_SECRET, payloadSegment);
  const token = `${payloadSegment}.${bytesToBase64Url(signatureBytes)}`;

  return {
    ok: true,
    token,
    expires_at: new Date(exp * 1000).toISOString(),
    customer_id: customerId,
    subscription_id: subscriptionId,
  };
}
