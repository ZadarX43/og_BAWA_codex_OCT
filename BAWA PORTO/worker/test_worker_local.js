import assert from "node:assert/strict";

import worker from "./src/index.js";

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

class MockKVStore {
  constructor() {
    this.map = new Map();
  }

  async get(key) {
    return this.map.has(key) ? this.map.get(key) : null;
  }

  async put(key, value) {
    this.map.set(key, value);
  }

  async delete(key) {
    this.map.delete(key);
  }

  async list({ prefix = "", cursor, limit = 100 } = {}) {
    const keys = Array.from(this.map.keys())
      .filter((key) => key.startsWith(prefix))
      .sort();
    const start = cursor ? Number(cursor) : 0;
    const slice = keys.slice(start, start + limit);
    const next = start + slice.length;
    return {
      keys: slice.map((name) => ({ name })),
      list_complete: next >= keys.length,
      cursor: next >= keys.length ? undefined : String(next),
    };
  }
}

class MockCacheStore {
  constructor() {
    this.map = new Map();
  }

  async match(request) {
    const key = typeof request === "string" ? request : request.url;
    const entry = this.map.get(key);
    if (!entry) {
      return undefined;
    }
    return new Response(entry.body, {
      status: entry.status,
      headers: entry.headers,
    });
  }

  async put(request, response) {
    const key = typeof request === "string" ? request : request.url;
    const body = await response.text();
    this.map.set(key, {
      body,
      status: response.status,
      headers: Object.fromEntries(response.headers.entries()),
    });
  }
}

const buildSubscriberRecord = (overrides = {}) => ({
  customer_id: "cus_test_active",
  subscription_id: "sub_test_active",
  status: "active",
  price_id: "price_test_founding",
  current_period_end: "2026-05-31T00:00:00.000Z",
  updated_at: "2026-05-04T00:00:00.000Z",
  ...overrides,
});

const premiumSourcePayload = {
  generated_at: "2026-05-04T18:00:00.000Z",
  predictions: [
    {
      fixture_id: "fixture_a",
      fixture_key: "fixture_key_a",
      kickoff_time: "2026-05-10T15:00:00Z",
      league: "Premier League",
      home_team: "Team A",
      away_team: "Team B",
      market: "FTR",
      pick: "HOME",
      confidence_tier: "ELITE",
      model_prob: 0.61,
      bookie_implied_prob: 0.44,
      value_edge: 0.17,
      bookie_od: 2.25,
      reason_tokens: ["DEPLOYABLE", "MARKET_FTR", "TIER_ELITE"],
      human_reason: "Strong home-result signal with enough support to stay live.",
      slip_role_hint: "anchor",
      safe_for_small_acca_flag: true,
      safe_for_large_acca_flag: false,
      correct_score_shortlist: [
        { scoreline: "1-0", probability: 0.18 },
        { scoreline: "2-0", probability: 0.11 },
      ],
      premium_tier: "ELITE",
      gate_detail: "should_not_leak",
      model_path: "/Users/secret/model.cbm",
    },
  ],
};

const installMockCache = () => {
  const originalCaches = globalThis.caches;
  const store = new MockCacheStore();
  globalThis.caches = {
    default: store,
  };
  return {
    clear: () => {
      store.map.clear();
    },
    restore: () => {
      globalThis.caches = originalCaches;
    },
  };
};

const createEnv = () => {
  const store = new MockKVStore();
  return {
    PREMIUM_TOKEN_SECRET: "test_premium_token_secret",
    AUTH_MAGIC_LINK_SECRET: "test_auth_magic_link_secret",
    AUTH_SESSION_SECRET: "test_auth_session_secret",
    RESEND_API_KEY: "test_resend_api_key",
    AUTH_EMAIL_FROM: "Odds Genius <auth@oddsgenius.test>",
    PREMIUM_DATA_SOURCE: "/premium-source.json",
    SITE_URL: "http://localhost",
    SUBSCRIBER_STATE: store,
  };
};

const writeSubscriberRecord = async (env, record) => {
  await env.SUBSCRIBER_STATE.put(`subscription:${record.subscription_id}`, JSON.stringify(record, null, 2));
  await env.SUBSCRIBER_STATE.put(`customer:${record.customer_id}`, JSON.stringify(record, null, 2));
};

const jsonRequest = (url, method, body, headers = {}) =>
  new Request(url, {
    method,
    headers: {
      "content-type": "application/json",
      ...headers,
    },
    body: body == null ? undefined : JSON.stringify(body),
  });

const makeGetRequest = (url, headers = {}) =>
  new Request(url, {
    method: "GET",
    headers,
  });

const installMockFetch = () => {
  const originalFetch = globalThis.fetch;
  const counters = {
    premiumSourceFetches: 0,
    resendSendFetches: 0,
  };
  const sentEmails = [];

  globalThis.fetch = async (input, init) => {
    const url = typeof input === "string" ? input : input.url;
    if (url === "http://localhost/premium-source.json") {
      counters.premiumSourceFetches += 1;
      return new Response(JSON.stringify(premiumSourcePayload), {
        status: 200,
        headers: {
          "content-type": "application/json; charset=utf-8",
        },
      });
    }

    if (url === "https://api.resend.com/emails") {
      counters.resendSendFetches += 1;
      sentEmails.push(JSON.parse(init?.body || "{}"));
      return new Response(JSON.stringify({ id: "email_test_123" }), {
        status: 200,
        headers: {
          "content-type": "application/json; charset=utf-8",
        },
      });
    }

    if (url.includes("api.stripe.com")) {
      throw new Error("Stripe should not be called during local Worker harness tests.");
    }

    return originalFetch(input, init);
  };

  return {
    counters,
    sentEmails,
    restore: () => {
      globalThis.fetch = originalFetch;
    },
  };
};

const issueTokenThroughRoute = async (env, body) => {
  const response = await worker.fetch(
    jsonRequest("http://localhost/api/premium/token", "POST", body),
    env
  );
  const payload = await response.json();
  assert.equal(response.status, 200, `expected token route 200, got ${response.status}: ${JSON.stringify(payload)}`);
  assert.equal(payload.ok, true);
  assert.ok(payload.token);
  assert.ok(payload.expires_at);
  return payload.token;
};

const assertPremiumRowAllowlist = (row) => {
  const keys = Object.keys(row).sort();
  for (const key of keys) {
    assert.ok(
      PREMIUM_ALLOWED_FIELDS.includes(key),
      `protected premium response leaked non-allowlisted field: ${key}`
    );
  }
};

const testProtectedRouteSuccess = async () => {
  const env = createEnv();
  await writeSubscriberRecord(env, buildSubscriberRecord());
  const token = await issueTokenThroughRoute(env, {
    customer_id: "cus_test_active",
    subscription_id: "sub_test_active",
  });

  const response = await worker.fetch(
    makeGetRequest("http://localhost/api/premium/predictions", {
      authorization: `Bearer ${token}`,
    }),
    env
  );
  const payload = await response.json();

  assert.equal(response.status, 200);
  assert.equal(payload.ok, true);
  assert.equal(payload.subscriber_customer_id, "cus_test_active");
  assert.equal(payload.generated_at, premiumSourcePayload.generated_at);
  assert.equal(payload.count, 1);
  assert.equal(Array.isArray(payload.predictions), true);
  assert.equal(payload.predictions.length, 1);
  assertPremiumRowAllowlist(payload.predictions[0]);
  assert.equal("gate_detail" in payload.predictions[0], false);
  assert.equal("model_path" in payload.predictions[0], false);
};

const testPremiumRouteCachesSharedPayload = async (counters) => {
  const env = createEnv();
  await writeSubscriberRecord(env, buildSubscriberRecord());
  const token = await issueTokenThroughRoute(env, {
    customer_id: "cus_test_active",
    subscription_id: "sub_test_active",
  });

  const first = await worker.fetch(
    makeGetRequest("http://localhost/api/premium/predictions", {
      authorization: `Bearer ${token}`,
    }),
    env
  );
  assert.equal(first.status, 200);
  assert.equal(first.headers.get("x-og-premium-cache"), "miss");

  const second = await worker.fetch(
    makeGetRequest("http://localhost/api/premium/predictions", {
      authorization: `Bearer ${token}`,
    }),
    env
  );
  assert.equal(second.status, 200);
  assert.equal(second.headers.get("x-og-premium-cache"), "hit");
  assert.equal(counters.premiumSourceFetches, 1);
};

const buildExpiredToken = async (env) => {
  const payload = {
    customer_id: "cus_test_active",
    subscription_id: "sub_test_active",
    exp: Math.floor(Date.now() / 1000) - 60,
  };
  const payloadSegment = Buffer.from(JSON.stringify(payload), "utf8")
    .toString("base64url");
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(env.PREMIUM_TOKEN_SECRET),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signature = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(payloadSegment));
  const signatureSegment = Buffer.from(new Uint8Array(signature)).toString("base64url");
  return `${payloadSegment}.${signatureSegment}`;
};

const testMissingToken = async () => {
  const env = createEnv();
  await writeSubscriberRecord(env, buildSubscriberRecord());

  const response = await worker.fetch(makeGetRequest("http://localhost/api/premium/predictions"), env);
  const payload = await response.json();

  assert.equal(response.status, 401);
  assert.equal(payload.ok, false);
  assert.equal(payload.status, "missing_token");
};

const testExpiredToken = async () => {
  const env = createEnv();
  await writeSubscriberRecord(env, buildSubscriberRecord());
  const token = await buildExpiredToken(env);

  const response = await worker.fetch(
    makeGetRequest("http://localhost/api/premium/predictions", {
      authorization: `Bearer ${token}`,
    }),
    env
  );
  const payload = await response.json();

  assert.equal(response.status, 401);
  assert.equal(payload.ok, false);
  assert.equal(payload.status, "expired_token");
};

const testInactiveSubscriber = async () => {
  const env = createEnv();
  await writeSubscriberRecord(
    env,
    buildSubscriberRecord({
      customer_id: "cus_test_inactive",
      subscription_id: "sub_test_inactive",
      status: "canceled",
    })
  );
  const token = await issueTokenThroughRoute(env, {
    customer_id: "cus_test_inactive",
    subscription_id: "sub_test_inactive",
  }).catch(() => null);

  assert.equal(token, null, "inactive subscriber should not receive a premium token");

  const expiredLikePayload = {
    customer_id: "cus_test_inactive",
    subscription_id: "sub_test_inactive",
  };
  const tokenResponse = await worker.fetch(
    jsonRequest("http://localhost/api/premium/token", "POST", expiredLikePayload),
    env
  );
  const tokenPayload = await tokenResponse.json();
  assert.equal(tokenResponse.status, 401);
  assert.equal(tokenPayload.status, "inactive_subscription");

  const activeEnv = createEnv();
  await writeSubscriberRecord(activeEnv, buildSubscriberRecord());
  const activeToken = await issueTokenThroughRoute(activeEnv, {
    customer_id: "cus_test_active",
    subscription_id: "sub_test_active",
  });
  await writeSubscriberRecord(
    activeEnv,
    buildSubscriberRecord({
      status: "past_due",
    })
  );

  const response = await worker.fetch(
    makeGetRequest("http://localhost/api/premium/predictions", {
      authorization: `Bearer ${activeToken}`,
    }),
    activeEnv
  );
  const payload = await response.json();

  assert.equal(response.status, 401);
  assert.equal(payload.ok, false);
  assert.equal(payload.status, "inactive_subscription");
};

const testMagicLinkRequestValidation = async (fetchHarness) => {
  const env = createEnv();
  await writeSubscriberRecord(
    env,
    buildSubscriberRecord({
      email: "member@example.com",
    })
  );

  const invalidResponse = await worker.fetch(
    jsonRequest("http://localhost/api/auth/magic-link/request", "POST", {
      email: "not-an-email",
    }),
    env
  );
  const invalidPayload = await invalidResponse.json();
  assert.equal(invalidResponse.status, 400);
  assert.equal(invalidPayload.status, "request_error");

  const validResponse = await worker.fetch(
    jsonRequest("http://localhost/api/auth/magic-link/request", "POST", {
      email: "member@example.com",
    }),
    env
  );
  const validPayload = await validResponse.json();
  assert.equal(validResponse.status, 200);
  assert.equal(validPayload.status, "magic_link_requested");
  assert.equal(fetchHarness.counters.resendSendFetches, 1);
  assert.equal(fetchHarness.sentEmails.length, 1);
  assert.match(fetchHarness.sentEmails[0].html, /api\/auth\/magic-link\/verify\?token=/);
};

const testAuthSessionSkeleton = async () => {
  const env = createEnv();
  const response = await worker.fetch(makeGetRequest("http://localhost/api/auth/session"), env);
  const payload = await response.json();
  assert.equal(response.status, 200);
  assert.equal(payload.ok, true);
  assert.equal(payload.authenticated, false);
  assert.equal(payload.entitled, false);
};

const extractCookieValue = (setCookieHeader, name) => {
  const match = String(setCookieHeader || "").match(new RegExp(`${name}=([^;]+)`));
  return match ? match[1] : "";
};

const testMagicLinkVerifyAndSessionFlow = async (fetchHarness) => {
  const env = createEnv();
  await writeSubscriberRecord(
    env,
    buildSubscriberRecord({
      email: "member@example.com",
    })
  );

  const missingTokenResponse = await worker.fetch(
    makeGetRequest("http://localhost/api/auth/magic-link/verify"),
    env
  );
  assert.equal(missingTokenResponse.status, 303);
  assert.equal(
    missingTokenResponse.headers.get("location"),
    "http://localhost/account.html?auth=invalid"
  );

  const requestResponse = await worker.fetch(
    jsonRequest("http://localhost/api/auth/magic-link/request", "POST", {
      email: "member@example.com",
    }),
    env
  );
  assert.equal(requestResponse.status, 200);
  const emailBody = fetchHarness.sentEmails.at(-1);
  const verifyMatch = String(emailBody?.html || "").match(/verify\?token=([^"&]+)/);
  assert.ok(verifyMatch?.[1], "expected magic-link token in email body");
  const token = decodeURIComponent(verifyMatch[1]);

  const tokenResponse = await worker.fetch(
    makeGetRequest(`http://localhost/api/auth/magic-link/verify?token=${encodeURIComponent(token)}`),
    env
  );
  assert.equal(tokenResponse.status, 303);
  assert.equal(
    tokenResponse.headers.get("location"),
    "http://localhost/account.html?auth=success"
  );
  const sessionCookie = extractCookieValue(tokenResponse.headers.get("set-cookie"), "og_premium_session");
  assert.ok(sessionCookie, "expected premium session cookie after verify");

  const sessionResponse = await worker.fetch(
    makeGetRequest("http://localhost/api/auth/session", {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const sessionPayload = await sessionResponse.json();
  assert.equal(sessionResponse.status, 200);
  assert.equal(sessionPayload.authenticated, true);
  assert.equal(sessionPayload.entitled, true);
  assert.equal(sessionPayload.auth_mode, "session");
  assert.equal(sessionPayload.subscription_status, "active");

  const premiumResponse = await worker.fetch(
    makeGetRequest("http://localhost/api/premium/predictions", {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const premiumPayload = await premiumResponse.json();
  assert.equal(premiumResponse.status, 200);
  assert.equal(premiumPayload.ok, true);
  assert.equal(premiumPayload.auth_mode, "session");
};

const testLogoutSkeleton = async () => {
  const env = createEnv();
  const response = await worker.fetch(
    jsonRequest("http://localhost/api/auth/logout", "POST", null),
    env
  );
  const payload = await response.json();
  assert.equal(response.status, 200);
  assert.equal(payload.status, "logged_out");
};

const main = async () => {
  const cacheHarness = installMockCache();
  const fetchHarness = installMockFetch();
  try {
    await testPremiumRouteCachesSharedPayload(fetchHarness.counters);
    cacheHarness.clear();
    fetchHarness.counters.premiumSourceFetches = 0;
    await testProtectedRouteSuccess();
    cacheHarness.clear();
    await testMissingToken();
    await testExpiredToken();
    await testInactiveSubscriber();
    await testMagicLinkRequestValidation(fetchHarness);
    await testAuthSessionSkeleton();
    await testMagicLinkVerifyAndSessionFlow(fetchHarness);
    await testLogoutSkeleton();
    console.log("Worker local harness passed.");
    console.log("- success route with valid token: passed");
    console.log("- premium payload cache hit/miss path: passed");
    console.log("- missing token returns 401: passed");
    console.log("- expired token returns 401: passed");
    console.log("- inactive subscriber returns 401: passed");
    console.log("- magic-link request flow: passed");
    console.log("- auth session flow: passed");
    console.log("- magic-link verify + session premium flow: passed");
    console.log("- logout skeleton: passed");
  } finally {
    fetchHarness.restore();
    cacheHarness.restore();
  }
};

await main();
