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

class MockD1 {
  constructor() {
    this.users = [];
    this.subscriptions = [];
    this.telegramLinks = [];
    this.notificationPreferences = [];
    this.authEvents = [];
  }

  prepare() {
    throw new Error("MockD1.prepare should not be called directly; tests use __ogCall.");
  }

  async __ogCall(op, args) {
    switch (op) {
      case "get_user_by_email":
        return this.users.find((row) => row.email_normalized === args.email_normalized) || null;
      case "insert_user":
        this.users.push({ ...args });
        return { success: true };
      case "update_user_by_email": {
        const row = this.users.find((item) => item.id === args.id);
        if (row) {
          Object.assign(row, {
            email: args.email,
            email_normalized: args.email_normalized,
            email_verified_at: args.email_verified_at,
            updated_at: args.updated_at,
            account_status: args.account_status,
          });
        }
        return { success: true };
      }
      case "get_user_by_subscription_identity": {
        const sub = this.subscriptions.find(
          (row) =>
            row.stripe_customer_id === args.stripe_customer_id ||
            row.stripe_subscription_id === args.stripe_subscription_id
        );
        return sub ? this.users.find((row) => row.id === sub.user_id) || null : null;
      }
      case "get_subscription_by_identity":
        return (
          this.subscriptions.find(
            (row) =>
              row.stripe_customer_id === args.stripe_customer_id ||
              row.stripe_subscription_id === args.stripe_subscription_id
          ) || null
        );
      case "upsert_subscription": {
        const existing = this.subscriptions.find(
          (row) => row.stripe_subscription_id === args.stripe_subscription_id
        );
        if (existing) {
          Object.assign(existing, args);
        } else {
          this.subscriptions.push({ ...args });
        }
        return { success: true };
      }
      case "ensure_notification_preferences": {
        const existing = this.notificationPreferences.find((row) => row.user_id === args.user_id);
        if (!existing) {
          this.notificationPreferences.push({ ...args });
        }
        return { success: true };
      }
      case "update_notification_preferences":
        this.notificationPreferences = this.notificationPreferences.map((row) =>
          row.user_id === args.user_id ? { ...row, ...args } : row
        );
        return { success: true };
      case "insert_auth_event":
        this.authEvents.push({ ...args });
        return { success: true };
      case "get_subscription_by_user":
        return this.subscriptions.find((row) => row.user_id === args.user_id) || null;
      case "get_telegram_link_by_user":
        return (
          [...this.telegramLinks]
            .sort((a, b) => String(b.updated_at).localeCompare(String(a.updated_at)))
            .find((row) => row.user_id === args.user_id) || null
        );
      case "get_notification_preferences":
        return this.notificationPreferences.find((row) => row.user_id === args.user_id) || null;
      case "revoke_telegram_links_for_user":
        this.telegramLinks = this.telegramLinks.map((row) =>
          row.user_id === args.user_id && row.link_status !== "revoked"
            ? { ...row, link_status: "revoked", revoked_at: args.revoked_at, updated_at: args.updated_at }
            : row
        );
        return { success: true };
      case "revoke_telegram_links_for_telegram_user":
        this.telegramLinks = this.telegramLinks.map((row) =>
          row.telegram_user_id === args.telegram_user_id && row.link_status !== "revoked"
            ? { ...row, link_status: "revoked", revoked_at: args.revoked_at, updated_at: args.updated_at }
            : row
        );
        return { success: true };
      case "insert_telegram_link":
        this.telegramLinks.push({ ...args });
        return { success: true };
      case "enable_telegram_notifications":
        this.notificationPreferences = this.notificationPreferences.map((row) =>
          row.user_id === args.user_id ? { ...row, telegram_enabled: 1, updated_at: args.updated_at } : row
        );
        return { success: true };
      default:
        throw new Error(`Unhandled MockD1 op: ${op}`);
    }
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

const fixtureIntelligencePayload = {
  generated_at: "2026-05-09T14:11:03+00:00",
  fixtures: [
    {
      fixture_id: "2026_05_09_Luzern_Servette",
      fixture_key: "2026_05_09_Luzern_Servette",
      fixture_class: "OBSERVE",
      publish_class: "OBSERVE",
      coverage_status: "covered",
      kickoff_time: "2026-05-09 19:30:00",
      league: "Swiss Super League",
      home_team: "Luzern",
      away_team: "Servette",
      signal_summary: {
        market_family: "BTTS",
        headline: "Observed BTTS lean based on attacking shape, but not enough stability for deployment.",
        summary_text: "Observed BTTS lean based on attacking shape, but not enough stability for deployment.",
      },
      context_summary: {
        notes: ["Team-intelligence caution remains active around this fixture."],
      },
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
    TELEGRAM_BOT_TOKEN: "telegram_bot_test_token",
    TELEGRAM_WEBHOOK_SECRET: "telegram_webhook_secret_test",
    PREMIUM_DATA_SOURCE: "/premium-source.json",
    SITE_URL: "http://localhost",
    SUBSCRIBER_STATE: store,
    ACCOUNT_DB: new MockD1(),
    TELEGRAM_BOT_USERNAME: "oddsgeniusbot",
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
    telegramSendFetches: 0,
  };
  const sentEmails = [];
  const sentTelegramMessages = [];

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

    if (url === "http://localhost/public/data/fixture_intelligence_public.json") {
      return new Response(JSON.stringify(fixtureIntelligencePayload), {
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

    if (url === "https://api.telegram.org/bottelegram_bot_test_token/sendMessage") {
      counters.telegramSendFetches += 1;
      sentTelegramMessages.push(JSON.parse(init?.body || "{}"));
      return new Response(JSON.stringify({ ok: true, result: { message_id: 1 } }), {
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
    sentTelegramMessages,
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

const testAccountStateAndTelegramLinkFlow = async (fetchHarness) => {
  const env = createEnv();
  await writeSubscriberRecord(
    env,
    buildSubscriberRecord({
      email: "member@example.com",
    })
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
  const sessionCookie = extractCookieValue(tokenResponse.headers.get("set-cookie"), "og_premium_session");
  assert.ok(sessionCookie, "expected premium session cookie after verify");

  const accountStateResponse = await worker.fetch(
    makeGetRequest("http://localhost/api/account/state", {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const accountStatePayload = await accountStateResponse.json();
  assert.equal(accountStateResponse.status, 200);
  assert.equal(accountStatePayload.ok, true);
  assert.equal(accountStatePayload.d1_enabled, true);
  assert.equal(accountStatePayload.account.user.email_normalized, "member@example.com");
  assert.equal(accountStatePayload.account.subscription.subscription_status, "active");

  const startResponse = await worker.fetch(
    jsonRequest("http://localhost/api/account/telegram/link/start", "POST", null, {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const startPayload = await startResponse.json();
  assert.equal(startResponse.status, 200);
  assert.equal(startPayload.status, "telegram_link_ready");
  assert.ok(startPayload.code);
  assert.match(startPayload.deep_link_url, /t\.me\/oddsgeniusbot\?start=oglink_/);

  const completeResponse = await worker.fetch(
    jsonRequest("http://localhost/api/account/telegram/link/complete", "POST", {
      code: startPayload.code,
      telegram_user_id: "tg_user_123",
      telegram_username: "ogfounder",
      telegram_chat_id: "chat_456",
    }),
    env
  );
  const completePayload = await completeResponse.json();
  assert.equal(completeResponse.status, 200);
  assert.equal(completePayload.status, "telegram_linked");
  assert.equal(completePayload.account.telegram_link.telegram_user_id, "tg_user_123");
  assert.equal(completePayload.account.notification_preferences.telegram_enabled, 1);
};

const testTelegramWebhookCompletesLinkFlow = async (fetchHarness) => {
  const env = createEnv();
  await writeSubscriberRecord(
    env,
    buildSubscriberRecord({
      email: "member@example.com",
    })
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
  const sessionCookie = extractCookieValue(tokenResponse.headers.get("set-cookie"), "og_premium_session");
  assert.ok(sessionCookie, "expected premium session cookie after verify");

  const startResponse = await worker.fetch(
    jsonRequest("http://localhost/api/account/telegram/link/start", "POST", null, {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const startPayload = await startResponse.json();
  assert.equal(startResponse.status, 200);
  assert.ok(startPayload.code);

  const webhookResponse = await worker.fetch(
    jsonRequest(
      "http://localhost/api/telegram/webhook",
      "POST",
      {
        update_id: 123456,
        message: {
          message_id: 42,
          text: `/start oglink_${startPayload.code}`,
          chat: { id: 99887766, type: "private" },
          from: { id: 123123123, username: "ogfounder" },
        },
      },
      {
        [ "x-telegram-bot-api-secret-token" ]: env.TELEGRAM_WEBHOOK_SECRET,
      }
    ),
    env
  );
  const webhookPayload = await webhookResponse.json();
  assert.equal(webhookResponse.status, 200);
  assert.equal(webhookPayload.status, "telegram_webhook_processed");
  assert.equal(webhookPayload.action, "telegram_link_completed");
  assert.equal(webhookPayload.account.telegram_link.telegram_user_id, "123123123");
  assert.equal(webhookPayload.account.notification_preferences.telegram_enabled, 1);
  assert.equal(fetchHarness.counters.telegramSendFetches >= 1, true);
  assert.match(fetchHarness.sentTelegramMessages.at(-1)?.text || "", /now linked/i);
};

const testTelegramTestAlertRoute = async (fetchHarness) => {
  const env = createEnv();
  await writeSubscriberRecord(
    env,
    buildSubscriberRecord({
      email: "member@example.com",
    })
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
  const sessionCookie = extractCookieValue(tokenResponse.headers.get("set-cookie"), "og_premium_session");
  assert.ok(sessionCookie, "expected premium session cookie after verify");

  const startResponse = await worker.fetch(
    jsonRequest("http://localhost/api/account/telegram/link/start", "POST", null, {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const startPayload = await startResponse.json();
  assert.equal(startResponse.status, 200);
  assert.ok(startPayload.code);

  const completeResponse = await worker.fetch(
    jsonRequest("http://localhost/api/account/telegram/link/complete", "POST", {
      code: startPayload.code,
      telegram_user_id: "tg_user_123",
      telegram_username: "ogfounder",
      telegram_chat_id: "chat_456",
    }),
    env
  );
  assert.equal(completeResponse.status, 200);

  const beforeCount = fetchHarness.counters.telegramSendFetches;
  const alertResponse = await worker.fetch(
    jsonRequest("http://localhost/api/account/telegram/test-alert", "POST", null, {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const alertPayload = await alertResponse.json();
  assert.equal(alertResponse.status, 200);
  assert.equal(alertPayload.status, "telegram_test_alert_sent");
  assert.equal(fetchHarness.counters.telegramSendFetches, beforeCount + 1);
  assert.match(fetchHarness.sentTelegramMessages.at(-1)?.text || "", /test premium alert/i);
};

const testTelegramFixtureAlertRoute = async (fetchHarness) => {
  const env = createEnv();
  await writeSubscriberRecord(
    env,
    buildSubscriberRecord({
      email: "member@example.com",
    })
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
  const sessionCookie = extractCookieValue(tokenResponse.headers.get("set-cookie"), "og_premium_session");
  assert.ok(sessionCookie, "expected premium session cookie after verify");

  const startResponse = await worker.fetch(
    jsonRequest("http://localhost/api/account/telegram/link/start", "POST", null, {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const startPayload = await startResponse.json();
  assert.equal(startResponse.status, 200);
  assert.ok(startPayload.code);

  const completeResponse = await worker.fetch(
    jsonRequest("http://localhost/api/account/telegram/link/complete", "POST", {
      code: startPayload.code,
      telegram_user_id: "tg_user_123",
      telegram_username: "ogfounder",
      telegram_chat_id: "chat_456",
    }),
    env
  );
  assert.equal(completeResponse.status, 200);

  const beforeCount = fetchHarness.counters.telegramSendFetches;
  const alertResponse = await worker.fetch(
    jsonRequest(
      "http://localhost/api/account/telegram/fixture-alert",
      "POST",
      {
        fixture_key: "2026_05_09_Luzern_Servette",
      },
      {
        cookie: `og_premium_session=${sessionCookie}`,
      }
    ),
    env
  );
  const alertPayload = await alertResponse.json();
  assert.equal(alertResponse.status, 200);
  assert.equal(alertPayload.status, "telegram_fixture_alert_sent");
  assert.equal(fetchHarness.counters.telegramSendFetches, beforeCount + 1);
  assert.match(fetchHarness.sentTelegramMessages.at(-1)?.text || "", /Luzern vs Servette/i);
  assert.match(fetchHarness.sentTelegramMessages.at(-1)?.text || "", /fixture\.html\?fixture=2026_05_09_Luzern_Servette/i);
};

const testAccountPreferencesUpdate = async (fetchHarness) => {
  const env = createEnv();
  await writeSubscriberRecord(
    env,
    buildSubscriberRecord({
      email: "member@example.com",
    })
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
  const sessionCookie = extractCookieValue(tokenResponse.headers.get("set-cookie"), "og_premium_session");
  assert.ok(sessionCookie, "expected premium session cookie after verify");

  const prefsResponse = await worker.fetch(
    jsonRequest(
      "http://localhost/api/account/preferences",
      "POST",
      {
        telegram_enabled: true,
        email_enabled: true,
        elite_alerts_enabled: true,
        standard_alerts_enabled: false,
        acca_alerts_enabled: true,
        correct_score_alerts_enabled: true,
        injury_alerts_enabled: true,
        weather_alerts_enabled: true,
        market_movement_alerts_enabled: false,
        volatility_alerts_enabled: true,
        team_news_alerts_enabled: true,
        daily_digest_enabled: true,
        results_digest_enabled: true,
        weekend_slate_digest_enabled: true,
        website_only_mode: false,
        allow_non_signal_intelligence: true,
        alert_frequency_mode: "immediate",
        pre_match_window_minutes: 120,
        favourite_teams: "Arsenal, Porto",
        favourite_leagues: "Premier League, Champions League",
        favourite_markets: "BTTS, OU25",
        followed_fixtures: "Arsenal v Chelsea, Porto v Benfica",
      },
      {
        cookie: `og_premium_session=${sessionCookie}`,
      }
    ),
    env
  );
  const prefsPayload = await prefsResponse.json();
  assert.equal(prefsResponse.status, 200);
  assert.equal(prefsPayload.status, "notification_preferences_updated");
  assert.equal(prefsPayload.account.notification_preferences.alert_frequency_mode, "immediate");
  assert.equal(prefsPayload.account.notification_preferences.standard_alerts_enabled, 0);
  assert.deepEqual(prefsPayload.account.notification_preferences.favourite_teams, ["Arsenal", "Porto"]);
  assert.deepEqual(prefsPayload.account.notification_preferences.favourite_markets, ["BTTS", "OU25"]);
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
    await testAccountStateAndTelegramLinkFlow(fetchHarness);
    await testTelegramWebhookCompletesLinkFlow(fetchHarness);
    await testTelegramTestAlertRoute(fetchHarness);
    await testTelegramFixtureAlertRoute(fetchHarness);
    await testAccountPreferencesUpdate(fetchHarness);
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
    console.log("- D1-backed account state + Telegram link flow: passed");
    console.log("- Telegram bot webhook completion flow: passed");
    console.log("- Telegram test alert route: passed");
    console.log("- Telegram fixture alert route: passed");
    console.log("- Account preferences update route: passed");
    console.log("- logout skeleton: passed");
  } finally {
    fetchHarness.restore();
    cacheHarness.restore();
  }
};

await main();
