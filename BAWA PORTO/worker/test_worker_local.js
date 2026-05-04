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

const createEnv = () => {
  const store = new MockKVStore();
  return {
    PREMIUM_TOKEN_SECRET: "test_premium_token_secret",
    PREMIUM_DATA_SOURCE: "/premium-source.json",
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

  globalThis.fetch = async (input, init) => {
    const url = typeof input === "string" ? input : input.url;
    if (url === "http://localhost/premium-source.json") {
      return new Response(JSON.stringify(premiumSourcePayload), {
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

  return () => {
    globalThis.fetch = originalFetch;
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

const main = async () => {
  const restoreFetch = installMockFetch();
  try {
    await testProtectedRouteSuccess();
    await testMissingToken();
    await testExpiredToken();
    await testInactiveSubscriber();
    console.log("Worker local harness passed.");
    console.log("- success route with valid token: passed");
    console.log("- missing token returns 401: passed");
    console.log("- expired token returns 401: passed");
    console.log("- inactive subscriber returns 401: passed");
  } finally {
    restoreFetch();
  }
};

await main();
