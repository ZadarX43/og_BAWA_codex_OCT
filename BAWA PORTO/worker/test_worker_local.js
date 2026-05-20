import assert from "node:assert/strict";

import worker from "./src/index.js";

const FOUNDER_ALLOWED_FIELDS = [
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
const PRO_ALLOWED_FIELDS = [
  ...FOUNDER_ALLOWED_FIELDS,
  "player_event_signals",
  "team_intelligence",
];
const PRO_PLUS_ALLOWED_FIELDS = [
  ...PRO_ALLOWED_FIELDS,
  "audit_summary",
  "downloadable_payload",
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

class MockR2Bucket {
  constructor() {
    this.map = new Map();
  }

  async get(key) {
    if (!this.map.has(key)) {
      return null;
    }
    const value = this.map.get(key);
    return {
      async text() {
        return value;
      },
    };
  }

  async put(key, value) {
    this.map.set(key, typeof value === "string" ? value : JSON.stringify(value));
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
    this.notificationAlerts = [];
    this.accountSessions = [];
    this.accountRiskStates = [];
    this.accountRiskFlags = [];
    this.accountAdminNotes = [];
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
      case "get_user_by_id":
        return this.users.find((row) => row.id === args.id) || null;
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
      case "get_account_risk_state":
        return this.accountRiskStates.find((row) => row.user_id === args.user_id) || null;
      case "insert_account_risk_state":
        this.accountRiskStates.push({ ...args });
        return { success: true };
      case "update_account_risk_state":
        this.accountRiskStates = this.accountRiskStates.map((row) =>
          row.user_id === args.user_id ? { ...row, ...args } : row
        );
        return { success: true };
      case "get_open_account_risk_flag_by_type":
        return (
          this.accountRiskFlags.find(
            (row) =>
              row.user_id === args.user_id &&
              row.flag_type === args.flag_type &&
              row.flag_status === "open"
          ) || null
        );
      case "insert_account_risk_flag":
        this.accountRiskFlags.push({ ...args });
        return { success: true };
      case "update_account_risk_flag_status":
        this.accountRiskFlags = this.accountRiskFlags.map((row) =>
          row.id === args.id
            ? {
                ...row,
                flag_status: args.flag_status,
                resolved_at: args.resolved_at,
                resolved_by: args.resolved_by,
                resolution_note: args.resolution_note,
                updated_at: args.updated_at,
              }
            : row
        );
        return { success: true };
      case "insert_account_admin_note":
        this.accountAdminNotes.push({ ...args });
        return { success: true };
      case "list_account_risk_flags_by_user":
        return this.accountRiskFlags
          .filter((row) => row.user_id === args.user_id && (!args.status || row.flag_status === args.status))
          .sort((a, b) => String(b.opened_at).localeCompare(String(a.opened_at)))
          .slice(0, args.limit);
      case "list_account_admin_notes_by_user":
        return this.accountAdminNotes
          .filter((row) => row.user_id === args.user_id)
          .sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))
          .slice(0, args.limit);
      case "list_auth_events_by_user":
        return this.authEvents
          .filter((row) => row.user_id === args.user_id)
          .sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))
          .slice(0, args.limit);
      case "list_active_account_sessions_by_user":
        return this.accountSessions
          .filter((row) => row.user_id === args.user_id && !row.is_revoked && String(row.expires_at) > new Date().toISOString())
          .sort((a, b) => String(a.issued_at).localeCompare(String(b.issued_at)));
      case "list_account_sessions_by_user":
        return this.accountSessions
          .filter((row) => row.user_id === args.user_id)
          .sort((a, b) =>
            String(b.last_seen_at || b.issued_at).localeCompare(String(a.last_seen_at || a.issued_at))
          )
          .slice(0, args.limit);
      case "get_account_session_by_id":
        return this.accountSessions.find((row) => row.id === args.id) || null;
      case "insert_account_session":
        this.accountSessions.push({ ...args });
        return { success: true };
      case "touch_account_session_seen":
        this.accountSessions = this.accountSessions.map((row) =>
          row.id === args.id ? { ...row, last_seen_at: args.last_seen_at, updated_at: args.updated_at } : row
        );
        return { success: true };
      case "revoke_account_session":
        this.accountSessions = this.accountSessions.map((row) =>
          row.id === args.id
            ? { ...row, is_revoked: 1, revoked_at: args.revoked_at, revoke_reason: args.revoke_reason, updated_at: args.updated_at }
            : row
        );
        return { success: true };
      case "revoke_other_account_sessions": {
        let changes = 0;
        this.accountSessions = this.accountSessions.map((row) => {
          if (
            row.user_id === args.user_id &&
            row.id !== args.current_session_id &&
            !row.is_revoked
          ) {
            changes += 1;
            return {
              ...row,
              is_revoked: 1,
              revoked_at: args.revoked_at,
              revoke_reason: args.revoke_reason,
              updated_at: args.updated_at,
            };
          }
          return row;
        });
        return changes;
      }
      case "set_primary_account_session":
        this.accountSessions = this.accountSessions.map((row) =>
          row.user_id === args.user_id
            ? {
                ...row,
                is_primary: row.id === args.session_id ? 1 : 0,
                updated_at: args.updated_at,
              }
            : row
        );
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
      case "list_users_eligible_for_telegram_alerts":
        return this.users
          .map((user) => {
            const prefs = this.notificationPreferences.find((row) => row.user_id === user.id);
            const link = this.telegramLinks.find(
              (row) => row.user_id === user.id && row.link_status === "linked" && row.telegram_chat_id
            );
            const sub = this.subscriptions.find((row) => row.user_id === user.id);
            if (!prefs || !link) {
              return null;
            }
            if (!prefs.telegram_enabled || prefs.website_only_mode) {
              return null;
            }
            if (sub && !["active", "trialing"].includes(String(sub.subscription_status || ""))) {
              return null;
            }
            return {
              id: user.id,
              email: user.email,
              email_normalized: user.email_normalized,
              telegram_chat_id: link.telegram_chat_id,
            };
          })
          .filter(Boolean);
      case "upsert_notification_alert": {
        const existing = this.notificationAlerts.find((row) => row.dedupe_key === args.dedupe_key);
        if (existing) {
          if (existing.status !== "delivered") {
            Object.assign(existing, {
              fixture_id: args.fixture_id,
              fixture_label: args.fixture_label,
              league: args.league,
              market_family: args.market_family,
              publish_class: args.publish_class,
              reasons_json: args.reasons_json,
              payload_json: args.payload_json,
              notification_priority: args.notification_priority,
              scheduled_for: args.scheduled_for,
              last_error: null,
              updated_at: args.updated_at,
            });
          }
        } else {
          this.notificationAlerts.push({ ...args });
        }
        return { success: true };
      }
      case "list_notification_alerts_by_user":
        return this.notificationAlerts
          .filter((row) => row.user_id === args.user_id)
          .sort((a, b) => String(a.scheduled_for).localeCompare(String(b.scheduled_for)))
          .slice(0, args.limit);
      case "list_due_notification_alerts":
        return this.notificationAlerts
          .filter((row) => {
            if (row.channel !== "telegram" || row.status !== "queued") {
              return false;
            }
            if (args.user_id && row.user_id !== args.user_id) {
              return false;
            }
            return String(row.scheduled_for) <= String(args.now_iso);
          })
          .sort((a, b) => String(a.scheduled_for).localeCompare(String(b.scheduled_for)))
          .slice(0, args.limit);
      case "mark_notification_alert_delivered":
        this.notificationAlerts = this.notificationAlerts.map((row) =>
          row.id === args.id
            ? { ...row, status: "delivered", delivered_at: args.delivered_at, last_error: null, updated_at: args.updated_at }
            : row
        );
        return { success: true };
      case "mark_notification_alert_failed":
        this.notificationAlerts = this.notificationAlerts.map((row) =>
          row.id === args.id
            ? { ...row, status: "failed", last_error: args.last_error, updated_at: args.updated_at }
            : row
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
      player_event_signals: [{ player: "Team A Forward", market: "shots", probability: 0.58 }],
      team_intelligence: { home_press: "strong" },
      audit_summary: { calibration_bucket: "walk_forward_green" },
      downloadable_payload: { export_id: "downloadable_should_only_reach_pro_plus" },
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
    {
      fixture_id: "2026_05_09_Atlanta_United_FC_LA_Galaxy",
      fixture_key: "2026_05_09_Atlanta_United_FC_LA_Galaxy",
      fixture_class: "DEPLOY",
      publish_class: "DEPLOY",
      coverage_status: "covered",
      kickoff_time: "2099-05-09 19:30:00",
      league: "USA MLS",
      home_team: "Atlanta United FC",
      away_team: "LA Galaxy",
      confidence_tier: "ELITE",
      premium_tier: "ELITE",
      signal_summary: {
        market_family: "BTTS",
        headline: "Elite BTTS deployment remained live on Yes.",
        summary_text: "Elite BTTS deployment remained live on Yes.",
      },
      context_summary: {
        notes: ["Home-side scoring context remains active around this fixture."],
      },
    },
  ],
};

const r2FixturePayload = {
  schema: "fixture_page_payload_v2",
  fixture_key: "fixture_r2",
  fixture: {
    fixture_key: "fixture_r2",
    league: "Test League",
    home_team: "R2 Home",
    away_team: "R2 Away",
  },
  decision: {
    fixture_key: "fixture_r2",
    context_signal: "R2 Home Win",
  },
  lineup: {
    predicted_lineups: true,
  },
  h2h: {
    sample_size: 4,
  },
  stats: {
    team_stats: [{ team_name: "R2 Home", shots: 13 }],
    player_stats: [{ player_name: "R2 Forward", shots: 3 }],
    match_events: [{ minute: 12, event_type: "shot" }],
    lineup_slots: [{ player_name: "R2 Forward", is_starting_xi: true }],
    market_intelligence: [{ market_key: "FTR", rank_role: "best" }],
    player_event_shortlists: [{ player_name: "R2 Forward", event_family: "shots", shortlist_rank: 1 }],
  },
  context: {
    media: [{ type: "youtube_embed", title: "Preview" }],
    news_signals: [{ type: "news_signal", title: "Team news" }],
    weather_signals: [{ type: "weather_context", summary: "Calm" }],
    space_weather_signals: [],
    sentiment_signals: [],
  },
  fixture_brain: {
    player_event_cards: [{ event_family: "shots", title: "Shots shortlist" }],
  },
};

const r2TeamPayload = {
  schema: "team_page_payload_v1",
  competition_key: "test_league",
  team_slug: "r2_home",
  team: { team: "R2 Home", team_slug: "r2_home" },
  squad: { club: "R2 Home", players: 20 },
  lineup_snapshot: { team: "R2 Home", shape: "4-3-3" },
  premium: {
    players: [{ player_name: "R2 Forward", rating_power: 88 }],
    recent_team_stats: [{ fixture_key: "fixture_r2", shots: 13 }],
    recent_lineup_slots: [{ player_name: "R2 Forward", is_starting_xi: true }],
    player_event_shortlists: [{ player_name: "R2 Forward", event_family: "shots" }],
  },
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
    INTERNAL_ADMIN_SECRET: "test_internal_admin_secret",
    RESEND_API_KEY: "test_resend_api_key",
    AUTH_EMAIL_FROM: "Odds Genius <auth@oddsgenius.test>",
    TELEGRAM_BOT_TOKEN: "telegram_bot_test_token",
    TELEGRAM_WEBHOOK_SECRET: "telegram_webhook_secret_test",
    API_SPORTS_FOOTBALL_KEY: "api_sports_test_key",
    PREMIUM_DATA_SOURCE: "/premium-source.json",
    SITE_URL: "http://localhost",
    SUBSCRIBER_STATE: store,
    ACCOUNT_DB: new MockD1(),
    SITE_PAYLOADS: new MockR2Bucket(),
    TELEGRAM_BOT_USERNAME: "oddsgeniusbot",
    STRIPE_SECRET_KEY: "sk_test_worker_harness",
    STRIPE_WEBHOOK_SECRET: "whsec_test_worker_harness",
    STRIPE_PRICE_ID: "price_test_founding",
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
    widgetStandingsFetches: 0,
    widgetFixtureLookupFetches: 0,
    stripeCheckoutFetches: 0,
    stripePortalFetches: 0,
    stripeSubscriptionFetches: 0,
  };
  const sentEmails = [];
  const sentTelegramMessages = [];
  const stripeCheckoutRequests = [];
  const stripePortalRequests = [];

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

    if (url === "https://v3.football.api-sports.io/standings?league=140&season=2025") {
      counters.widgetStandingsFetches += 1;
      return new Response(
        JSON.stringify({
          get: "standings",
          parameters: { league: "140", season: "2025" },
          response: [
            {
              league: {
                id: 140,
                name: "La Liga",
                season: 2025,
                standings: [[{ rank: 1, team: { id: 529, name: "Barcelona" }, points: 88 }]],
              },
            },
          ],
        }),
        {
          status: 200,
          headers: {
            "content-type": "application/json; charset=utf-8",
          },
        }
      );
    }

    if (url === "https://v3.football.api-sports.io/fixtures?date=2026-05-10") {
      counters.widgetFixtureLookupFetches += 1;
      return new Response(
        JSON.stringify({
          get: "fixtures",
          parameters: { date: "2026-05-10" },
          response: [
            {
              fixture: { id: 120001, status: { short: "NS" } },
              league: { id: 140, season: 2025 },
              teams: {
                home: { id: 529, name: "FC Barcelona" },
                away: { id: 541, name: "Real Madrid" },
              },
            },
          ],
        }),
        {
          status: 200,
          headers: {
            "content-type": "application/json; charset=utf-8",
          },
        }
      );
    }

    if (url === "https://api.stripe.com/v1/checkout/sessions") {
      counters.stripeCheckoutFetches += 1;
      stripeCheckoutRequests.push({
        headers: Object.fromEntries(new Headers(init?.headers || {}).entries()),
        body: Object.fromEntries(new URLSearchParams(String(init?.body || "")).entries()),
      });
      return new Response(
        JSON.stringify({
          id: "cs_test_worker_checkout",
          url: "https://checkout.stripe.com/c/pay/cs_test_worker_checkout",
        }),
        {
          status: 200,
          headers: {
            "content-type": "application/json; charset=utf-8",
          },
        }
      );
    }

    if (url === "https://api.stripe.com/v1/billing_portal/sessions") {
      counters.stripePortalFetches += 1;
      stripePortalRequests.push({
        headers: Object.fromEntries(new Headers(init?.headers || {}).entries()),
        body: Object.fromEntries(new URLSearchParams(String(init?.body || "")).entries()),
      });
      return new Response(
        JSON.stringify({
          id: "bps_test_worker_portal",
          url: "https://billing.stripe.com/p/session/test_worker_portal",
        }),
        {
          status: 200,
          headers: {
            "content-type": "application/json; charset=utf-8",
          },
        }
      );
    }

    if (url === "https://api.stripe.com/v1/subscriptions/sub_test_checkout_active") {
      counters.stripeSubscriptionFetches += 1;
      return new Response(
        JSON.stringify({
          id: "sub_test_checkout_active",
          customer: "cus_test_checkout_active",
          status: "active",
          current_period_end: 1780185600,
          items: {
            data: [
              {
                price: {
                  id: "price_test_founding",
                },
              },
            ],
          },
        }),
        {
          status: 200,
          headers: {
            "content-type": "application/json; charset=utf-8",
          },
        }
      );
    }

    if (url.includes("api.stripe.com")) {
      throw new Error(`Unexpected Stripe call during local Worker harness tests: ${url}`);
    }

    return originalFetch(input, init);
  };

  return {
    counters,
    sentEmails,
    sentTelegramMessages,
    stripeCheckoutRequests,
    stripePortalRequests,
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

const assertPremiumRowAllowlist = (row, allowedFields = FOUNDER_ALLOWED_FIELDS) => {
  const keys = Object.keys(row).sort();
  for (const key of keys) {
    assert.ok(
      allowedFields.includes(key),
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
  assert.equal(payload.access_tier, "founder");
  assert.equal(payload.field_policy.access_tier, "founder");
  assert.equal(payload.count, 1);
  assert.equal(Array.isArray(payload.predictions), true);
  assert.equal(payload.predictions.length, 1);
  assertPremiumRowAllowlist(payload.predictions[0]);
  assert.equal("gate_detail" in payload.predictions[0], false);
  assert.equal("model_path" in payload.predictions[0], false);
  assert.equal("player_event_signals" in payload.predictions[0], false);
  assert.equal("audit_summary" in payload.predictions[0], false);
};

const testPremiumTierAllowlistBoundaries = async () => {
  const proEnv = createEnv();
  proEnv.STRIPE_PRO_PRICE_IDS = "price_test_pro";
  await writeSubscriberRecord(
    proEnv,
    buildSubscriberRecord({
      price_id: "price_test_pro",
    })
  );
  const proToken = await issueTokenThroughRoute(proEnv, {
    customer_id: "cus_test_active",
    subscription_id: "sub_test_active",
  });
  const proResponse = await worker.fetch(
    makeGetRequest("http://localhost/api/premium/predictions", {
      authorization: `Bearer ${proToken}`,
    }),
    proEnv
  );
  const proPayload = await proResponse.json();
  assert.equal(proResponse.status, 200);
  assert.equal(proPayload.access_tier, "pro");
  assertPremiumRowAllowlist(proPayload.predictions[0], PRO_ALLOWED_FIELDS);
  assert.equal(Array.isArray(proPayload.predictions[0].player_event_signals), true);
  assert.equal(Boolean(proPayload.predictions[0].team_intelligence), true);
  assert.equal("audit_summary" in proPayload.predictions[0], false);
  assert.equal("downloadable_payload" in proPayload.predictions[0], false);

  const proPlusEnv = createEnv();
  proPlusEnv.STRIPE_PRO_PLUS_PRICE_IDS = "price_test_pro_plus";
  await writeSubscriberRecord(
    proPlusEnv,
    buildSubscriberRecord({
      price_id: "price_test_pro_plus",
    })
  );
  const proPlusToken = await issueTokenThroughRoute(proPlusEnv, {
    customer_id: "cus_test_active",
    subscription_id: "sub_test_active",
  });
  const proPlusResponse = await worker.fetch(
    makeGetRequest("http://localhost/api/premium/predictions", {
      authorization: `Bearer ${proPlusToken}`,
    }),
    proPlusEnv
  );
  const proPlusPayload = await proPlusResponse.json();
  assert.equal(proPlusResponse.status, 200);
  assert.equal(proPlusPayload.access_tier, "pro_plus");
  assertPremiumRowAllowlist(proPlusPayload.predictions[0], PRO_PLUS_ALLOWED_FIELDS);
  assert.equal(Boolean(proPlusPayload.predictions[0].audit_summary), true);
  assert.equal(Boolean(proPlusPayload.predictions[0].downloadable_payload), true);
  assert.equal("gate_detail" in proPlusPayload.predictions[0], false);
  assert.equal("model_path" in proPlusPayload.predictions[0], false);
};

const testSitePayloadRoutesReadThroughR2 = async () => {
  const env = createEnv();
  env.STRIPE_PRO_PRICE_IDS = "price_test_pro";
  await env.SITE_PAYLOADS.put(
    "site-data/v1/payloads/fixtures/fixture_r2.json",
    JSON.stringify(r2FixturePayload)
  );
  await env.SITE_PAYLOADS.put(
    "site-data/v1/payloads/teams/test_league/r2_home.json",
    JSON.stringify(r2TeamPayload)
  );
  await writeSubscriberRecord(
    env,
    buildSubscriberRecord({
      price_id: "price_test_pro",
    })
  );
  const token = await issueTokenThroughRoute(env, {
    customer_id: "cus_test_active",
    subscription_id: "sub_test_active",
  });

  const detailResponse = await worker.fetch(makeGetRequest("http://localhost/api/site/fixtures/fixture_r2"), env);
  const detailPayload = await detailResponse.json();
  assert.equal(detailResponse.status, 200);
  assert.equal(detailPayload.meta.payload_source, "r2");
  assert.equal(detailPayload.data.fixture.home_team, "R2 Home");
  assert.equal(detailPayload.data.fixture_brain.player_event_cards.length, 1);

  const contextResponse = await worker.fetch(makeGetRequest("http://localhost/api/site/fixtures/fixture_r2/context"), env);
  const contextPayload = await contextResponse.json();
  assert.equal(contextResponse.status, 200);
  assert.equal(contextPayload.meta.payload_source, "r2");
  assert.equal(contextPayload.data.weather_signals.length, 1);

  const statsResponse = await worker.fetch(
    makeGetRequest("http://localhost/api/site/fixtures/fixture_r2/stats", {
      authorization: `Bearer ${token}`,
    }),
    env
  );
  const statsPayload = await statsResponse.json();
  assert.equal(statsResponse.status, 200);
  assert.equal(statsPayload.access_tier, "pro");
  assert.equal(statsPayload.meta.payload_source, "r2");
  assert.equal(statsPayload.data.player_event_shortlists.length, 1);

  const teamResponse = await worker.fetch(makeGetRequest("http://localhost/api/site/teams/test_league/r2_home"), env);
  const teamPayload = await teamResponse.json();
  assert.equal(teamResponse.status, 200);
  assert.equal(teamPayload.meta.payload_source, "r2");
  assert.equal(teamPayload.data.lineup_snapshot.shape, "4-3-3");

  const teamPremiumResponse = await worker.fetch(
    makeGetRequest("http://localhost/api/site/teams/test_league/r2_home/premium", {
      authorization: `Bearer ${token}`,
    }),
    env
  );
  const teamPremiumPayload = await teamPremiumResponse.json();
  assert.equal(teamPremiumResponse.status, 200);
  assert.equal(teamPremiumPayload.meta.payload_source, "r2");
  assert.equal(teamPremiumPayload.data.player_event_shortlists.length, 1);
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

const buildStripeSignatureHeader = async (env, rawBody) => {
  const timestamp = Math.floor(Date.now() / 1000);
  const signedPayload = `${timestamp}.${rawBody}`;
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(env.STRIPE_WEBHOOK_SECRET),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signature = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(signedPayload));
  return `t=${timestamp},v1=${Buffer.from(new Uint8Array(signature)).toString("hex")}`;
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

const establishMemberSessionCookie = async (fetchHarness, env, email = "member@example.com") => {
  await writeSubscriberRecord(
    env,
    buildSubscriberRecord({
      email,
    })
  );

  const requestResponse = await worker.fetch(
    jsonRequest("http://localhost/api/auth/magic-link/request", "POST", {
      email,
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
  const sessionCookie = extractCookieValue(tokenResponse.headers.get("set-cookie"), "og_premium_session");
  assert.ok(sessionCookie, "expected premium session cookie after verify");
  return sessionCookie;
};

const testStripeCheckoutSmoke = async (fetchHarness) => {
  const missingEnv = createEnv();
  delete missingEnv.STRIPE_SECRET_KEY;
  const missingResponse = await worker.fetch(
    jsonRequest("http://localhost/api/stripe/checkout", "POST", {
      email: "member@example.com",
    }),
    missingEnv
  );
  const missingPayload = await missingResponse.json();
  assert.equal(missingResponse.status, 500);
  assert.equal(missingPayload.status, "config_error");
  assert.deepEqual(missingPayload.missing_env_vars, ["STRIPE_SECRET_KEY"]);

  const env = createEnv();
  const response = await worker.fetch(
    jsonRequest("http://localhost/api/stripe/checkout", "POST", {
      email: " member@example.com ",
      reference: "founder-smoke",
    }),
    env
  );
  const payload = await response.json();
  assert.equal(response.status, 200);
  assert.equal(payload.ok, true);
  assert.equal(payload.url, "https://checkout.stripe.com/c/pay/cs_test_worker_checkout");
  assert.equal(fetchHarness.counters.stripeCheckoutFetches >= 1, true);

  const stripeRequest = fetchHarness.stripeCheckoutRequests.at(-1);
  assert.equal(stripeRequest.headers.authorization, "Bearer sk_test_worker_harness");
  assert.equal(stripeRequest.body.mode, "subscription");
  assert.equal(stripeRequest.body.success_url, "http://localhost/account.html?checkout=success");
  assert.equal(stripeRequest.body.cancel_url, "http://localhost/pricing.html?checkout=cancelled");
  assert.equal(stripeRequest.body["line_items[0][price]"], "price_test_founding");
  assert.equal(stripeRequest.body["line_items[0][quantity]"], "1");
  assert.equal(stripeRequest.body.allow_promotion_codes, "true");
  assert.equal(stripeRequest.body.customer_email, "member@example.com");
  assert.equal(stripeRequest.body.client_reference_id, "founder-smoke");
};

const testStripeCheckoutWebhookEnrichesSubscriptionState = async (fetchHarness) => {
  const env = createEnv();
  const event = {
    id: "evt_test_checkout_completed",
    type: "checkout.session.completed",
    created: Math.floor(Date.now() / 1000),
    data: {
      object: {
        id: "cs_test_checkout_active",
        customer: "cus_test_checkout_active",
        subscription: "sub_test_checkout_active",
        customer_details: {
          email: "checkout-active@example.com",
        },
      },
    },
  };
  const rawBody = JSON.stringify(event);
  const signature = await buildStripeSignatureHeader(env, rawBody);
  const response = await worker.fetch(
    new Request("http://localhost/api/stripe/webhook", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "stripe-signature": signature,
      },
      body: rawBody,
    }),
    env
  );
  const payload = await response.json();
  assert.equal(response.status, 200);
  assert.equal(payload.ok, true);
  assert.equal(payload.event_type, "checkout.session.completed");
  assert.equal(payload.record.status, "active");
  assert.equal(payload.record.price_id, "price_test_founding");
  assert.equal(payload.record.current_period_end, "2026-05-31T00:00:00.000Z");
  assert.equal(fetchHarness.counters.stripeSubscriptionFetches >= 1, true);

  const subscriptionRaw = await env.SUBSCRIBER_STATE.get("subscription:sub_test_checkout_active");
  const emailRaw = await env.SUBSCRIBER_STATE.get("email:checkout-active@example.com");
  assert.ok(subscriptionRaw, "expected enriched subscription record in KV");
  assert.ok(emailRaw, "expected enriched email lookup record in KV");
  assert.equal(JSON.parse(subscriptionRaw).status, "active");
  assert.equal(JSON.parse(emailRaw).customer_id, "cus_test_checkout_active");
  assert.equal(env.ACCOUNT_DB.subscriptions.at(-1)?.subscription_status, "active");
};

const testStripePortalSmoke = async (fetchHarness) => {
  const lockedEnv = createEnv();
  const lockedResponse = await worker.fetch(
    jsonRequest("http://localhost/api/stripe/portal", "POST", null),
    lockedEnv
  );
  const lockedPayload = await lockedResponse.json();
  assert.equal(lockedResponse.status, 401);
  assert.equal(lockedPayload.ok, false);
  assert.equal(lockedPayload.locked, true);
  assert.equal(lockedPayload.status, "missing_session");

  const env = createEnv();
  const sessionCookie = await establishMemberSessionCookie(fetchHarness, env);
  const response = await worker.fetch(
    jsonRequest("http://localhost/api/stripe/portal", "POST", null, {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const payload = await response.json();
  assert.equal(response.status, 200);
  assert.equal(payload.ok, true);
  assert.equal(payload.url, "https://billing.stripe.com/p/session/test_worker_portal");
  assert.equal(fetchHarness.counters.stripePortalFetches >= 1, true);

  const stripeRequest = fetchHarness.stripePortalRequests.at(-1);
  assert.equal(stripeRequest.headers.authorization, "Bearer sk_test_worker_harness");
  assert.equal(stripeRequest.body.customer, "cus_test_active");
  assert.equal(stripeRequest.body.return_url, "http://localhost/account.html?portal=return");
};

const testSessionRestoreAndPaymentIssueStates = async (fetchHarness) => {
  const env = createEnv();
  const sessionCookie = await establishMemberSessionCookie(fetchHarness, env);
  const restoredResponse = await worker.fetch(
    makeGetRequest("http://localhost/api/auth/session", {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const restoredPayload = await restoredResponse.json();
  assert.equal(restoredResponse.status, 200);
  assert.equal(restoredPayload.authenticated, true);
  assert.equal(restoredPayload.entitled, true);
  assert.equal(restoredPayload.subscription_status, "active");

  await writeSubscriberRecord(
    env,
    buildSubscriberRecord({
      email: "member@example.com",
      status: "past_due",
    })
  );
  const paymentIssueResponse = await worker.fetch(
    makeGetRequest("http://localhost/api/auth/session", {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const paymentIssuePayload = await paymentIssueResponse.json();
  assert.equal(paymentIssueResponse.status, 200);
  assert.equal(paymentIssuePayload.authenticated, false);
  assert.equal(paymentIssuePayload.entitled, false);
  assert.equal(paymentIssuePayload.status, "inactive_subscription");

  const premiumResponse = await worker.fetch(
    makeGetRequest("http://localhost/api/premium/predictions", {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const premiumPayload = await premiumResponse.json();
  assert.equal(premiumResponse.status, 401);
  assert.equal(premiumPayload.locked, true);
  assert.equal(premiumPayload.status, "inactive_subscription");
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
  assert.equal(env.ACCOUNT_DB.accountSessions.length, 1);
  assert.equal(env.ACCOUNT_DB.accountSessions[0].session_kind, "browser");
  assert.equal(env.ACCOUNT_DB.accountRiskStates.length, 1);
  assert.equal(env.ACCOUNT_DB.accountRiskStates[0].risk_level, "low");
  assert.equal(env.ACCOUNT_DB.accountRiskFlags.length, 0);

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

  env.ACCOUNT_DB.accountSessions = [];
  const missingTrackedSessionResponse = await worker.fetch(
    makeGetRequest("http://localhost/api/auth/session", {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const missingTrackedSessionPayload = await missingTrackedSessionResponse.json();
  assert.equal(missingTrackedSessionResponse.status, 200);
  assert.equal(missingTrackedSessionPayload.authenticated, false);
  assert.equal(missingTrackedSessionPayload.status, "session_not_found");
  assert.match(String(missingTrackedSessionResponse.headers.get("set-cookie") || ""), /og_premium_session=;/);
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

const testAccountSessionsState = async (fetchHarness) => {
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

  const primarySession = env.ACCOUNT_DB.accountSessions[0];
  env.ACCOUNT_DB.accountSessions.push({
    ...primarySession,
    id: "sess_old_browser",
    device_label: "Chrome on Mac",
    is_primary: 0,
    is_revoked: 0,
    issued_at: "2026-05-01T09:00:00.000Z",
    last_seen_at: "2026-05-08T09:00:00.000Z",
    expires_at: "2026-05-20T09:00:00.000Z",
    created_at: "2026-05-01T09:00:00.000Z",
    updated_at: "2026-05-08T09:00:00.000Z",
  });

  const sessionsResponse = await worker.fetch(
    makeGetRequest("http://localhost/api/account/sessions", {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const sessionsPayload = await sessionsResponse.json();
  assert.equal(sessionsResponse.status, 200);
  assert.equal(sessionsPayload.ok, true);
  assert.equal(sessionsPayload.status, "account_sessions_loaded");
  assert.equal(Array.isArray(sessionsPayload.sessions), true);
  assert.equal(sessionsPayload.sessions.length, 2);
  assert.equal(sessionsPayload.sessions[0].device_label, primarySession.device_label);
  assert.equal(sessionsPayload.sessions[0].is_current, true);
  assert.equal(sessionsPayload.sessions[0].is_primary, true);
  assert.equal(sessionsPayload.sessions[1].device_label, "Chrome on Mac");
  assert.equal(sessionsPayload.sessions[1].is_current, false);
};

const testAccountSessionActions = async (fetchHarness) => {
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

  const currentSession = env.ACCOUNT_DB.accountSessions[0];
  env.ACCOUNT_DB.accountSessions.push(
    {
      ...currentSession,
      id: "sess_other_alpha",
      device_label: "Safari on iPhone",
      is_primary: 0,
      is_revoked: 0,
      issued_at: "2026-05-02T09:00:00.000Z",
      last_seen_at: "2026-05-08T09:00:00.000Z",
      expires_at: "2099-05-20T09:00:00.000Z",
      created_at: "2026-05-02T09:00:00.000Z",
      updated_at: "2026-05-08T09:00:00.000Z",
    },
    {
      ...currentSession,
      id: "sess_other_beta",
      device_label: "Chrome on Mac",
      is_primary: 0,
      is_revoked: 0,
      issued_at: "2026-05-03T09:00:00.000Z",
      last_seen_at: "2026-05-09T09:00:00.000Z",
      expires_at: "2099-05-20T09:00:00.000Z",
      created_at: "2026-05-03T09:00:00.000Z",
      updated_at: "2026-05-09T09:00:00.000Z",
    }
  );

  const makePrimaryResponse = await worker.fetch(
    jsonRequest(
      "http://localhost/api/account/sessions/make-primary",
      "POST",
      { session_id: "sess_other_beta" },
      { cookie: `og_premium_session=${sessionCookie}` }
    ),
    env
  );
  const makePrimaryPayload = await makePrimaryResponse.json();
  assert.equal(makePrimaryResponse.status, 200);
  assert.equal(makePrimaryPayload.status, "account_session_primary_updated");
  assert.equal(makePrimaryPayload.primary_session_id, "sess_other_beta");
  assert.equal(env.ACCOUNT_DB.accountSessions.find((row) => row.id === "sess_other_beta")?.is_primary, 1);
  assert.equal(env.ACCOUNT_DB.accountSessions.find((row) => row.id === currentSession.id)?.is_primary, 0);

  const revokeOneResponse = await worker.fetch(
    jsonRequest(
      "http://localhost/api/account/sessions/revoke",
      "POST",
      { session_id: "sess_other_alpha" },
      { cookie: `og_premium_session=${sessionCookie}` }
    ),
    env
  );
  const revokeOnePayload = await revokeOneResponse.json();
  assert.equal(revokeOneResponse.status, 200);
  assert.equal(revokeOnePayload.status, "account_session_revoked");
  assert.equal(env.ACCOUNT_DB.accountSessions.find((row) => row.id === "sess_other_alpha")?.is_revoked, 1);

  const revokeOthersResponse = await worker.fetch(
    jsonRequest(
      "http://localhost/api/account/sessions/revoke-others",
      "POST",
      null,
      { cookie: `og_premium_session=${sessionCookie}` }
    ),
    env
  );
  const revokeOthersPayload = await revokeOthersResponse.json();
  assert.equal(revokeOthersResponse.status, 200);
  assert.equal(revokeOthersPayload.status, "account_other_sessions_revoked");
  assert.equal(revokeOthersPayload.revoked_count, 1);
  assert.equal(env.ACCOUNT_DB.accountSessions.find((row) => row.id === "sess_other_beta")?.is_revoked, 1);
  assert.equal(env.ACCOUNT_DB.accountSessions.find((row) => row.id === currentSession.id)?.is_revoked, 0);

  const revokeCurrentResponse = await worker.fetch(
    jsonRequest(
      "http://localhost/api/account/sessions/revoke",
      "POST",
      { session_id: currentSession.id },
      { cookie: `og_premium_session=${sessionCookie}` }
    ),
    env
  );
  const revokeCurrentPayload = await revokeCurrentResponse.json();
  assert.equal(revokeCurrentResponse.status, 200);
  assert.equal(revokeCurrentPayload.status, "account_current_session_revoked");
  assert.match(String(revokeCurrentResponse.headers.get("set-cookie") || ""), /og_premium_session=;/);
  assert.equal(env.ACCOUNT_DB.accountSessions.find((row) => row.id === currentSession.id)?.is_revoked, 1);
};

const testAccountRiskFlaggingFromSessionSpread = async (fetchHarness) => {
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

  const firstVerifyResponse = await worker.fetch(
    makeGetRequest(`http://localhost/api/auth/magic-link/verify?token=${encodeURIComponent(token)}`),
    env
  );
  assert.equal(firstVerifyResponse.status, 303);
  const primarySession = env.ACCOUNT_DB.accountSessions[0];
  const userId = primarySession.user_id;

  env.ACCOUNT_DB.accountSessions.push(
    {
      ...primarySession,
      id: "sess_risk_alpha",
      device_label: "Safari on iPhone",
      is_primary: 0,
      issued_at: "2026-05-02T09:00:00.000Z",
      last_seen_at: "2026-05-08T09:00:00.000Z",
      expires_at: "2099-05-20T09:00:00.000Z",
      created_at: "2026-05-02T09:00:00.000Z",
      updated_at: "2026-05-08T09:00:00.000Z",
    },
    {
      ...primarySession,
      id: "sess_risk_beta",
      device_label: "Chrome on Windows",
      is_primary: 0,
      issued_at: "2026-05-03T09:00:00.000Z",
      last_seen_at: "2026-05-09T09:00:00.000Z",
      expires_at: "2099-05-20T09:00:00.000Z",
      created_at: "2026-05-03T09:00:00.000Z",
      updated_at: "2026-05-09T09:00:00.000Z",
    }
  );

  await env.SUBSCRIBER_STATE.delete("auth_rl:ip:unknown");
  await env.SUBSCRIBER_STATE.delete("auth_rl:email:member@example.com");

  const secondRequestResponse = await worker.fetch(
    jsonRequest("http://localhost/api/auth/magic-link/request", "POST", {
      email: "member@example.com",
    }),
    env
  );
  assert.equal(secondRequestResponse.status, 200);
  const secondEmailBody = fetchHarness.sentEmails.at(-1);
  const secondVerifyMatch = String(secondEmailBody?.html || "").match(/verify\?token=([^"&]+)/);
  assert.ok(secondVerifyMatch?.[1], "expected second magic-link token in email body");
  const secondToken = decodeURIComponent(secondVerifyMatch[1]);

  const secondVerifyResponse = await worker.fetch(
    makeGetRequest(`http://localhost/api/auth/magic-link/verify?token=${encodeURIComponent(secondToken)}`, {
      "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0",
    }),
    env
  );
  assert.equal(secondVerifyResponse.status, 303);

  const riskState = env.ACCOUNT_DB.accountRiskStates.find((row) => row.user_id === userId);
  assert.ok(riskState);
  assert.equal(riskState.risk_level, "high");
  assert.equal(riskState.review_status, "manual_review");
  assert.ok(Number(riskState.risk_score) >= 45);
  assert.equal(env.ACCOUNT_DB.accountRiskFlags.length, 1);
  assert.equal(env.ACCOUNT_DB.accountRiskFlags[0].flag_type, "shared_access_pattern");
  assert.equal(env.ACCOUNT_DB.accountAdminNotes.length, 1);
  assert.equal(env.ACCOUNT_DB.accountAdminNotes[0].note_type, "risk_note");
};

const testInternalAccountReviewReadApis = async (fetchHarness) => {
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

  const verifyResponse = await worker.fetch(
    makeGetRequest(`http://localhost/api/auth/magic-link/verify?token=${encodeURIComponent(token)}`),
    env
  );
  assert.equal(verifyResponse.status, 303);
  const userId = env.ACCOUNT_DB.accountSessions[0].user_id;

  await worker.fetch(
    jsonRequest(
      `http://localhost/internal/accounts/${encodeURIComponent(userId)}/notes`,
      "POST",
      {
        note_type: "support_note",
        content: "Customer confirmed recent travel between devices.",
        author_id: "internal:test",
      },
      {
        "x-og-internal-admin": env.INTERNAL_ADMIN_SECRET,
      }
    ),
    env
  );

  const summaryResponse = await worker.fetch(
    makeGetRequest(`http://localhost/internal/accounts/${encodeURIComponent(userId)}`, {
      "x-og-internal-admin": env.INTERNAL_ADMIN_SECRET,
    }),
    env
  );
  const summaryPayload = await summaryResponse.json();
  assert.equal(summaryResponse.status, 200);
  assert.equal(summaryPayload.status, "internal_account_loaded");
  assert.equal(summaryPayload.account_summary.user.id, userId);

  const lookupResponse = await worker.fetch(
    makeGetRequest(`http://localhost/internal/accounts/lookup?email=${encodeURIComponent("member@example.com")}`, {
      "x-og-internal-admin": env.INTERNAL_ADMIN_SECRET,
    }),
    env
  );
  const lookupPayload = await lookupResponse.json();
  assert.equal(lookupResponse.status, 200);
  assert.equal(lookupPayload.account_summary.user.id, userId);

  const notesResponse = await worker.fetch(
    makeGetRequest(`http://localhost/internal/accounts/${encodeURIComponent(userId)}/notes`, {
      "x-og-internal-admin": env.INTERNAL_ADMIN_SECRET,
    }),
    env
  );
  const notesPayload = await notesResponse.json();
  assert.equal(notesResponse.status, 200);
  assert.equal(notesPayload.status, "internal_account_notes_loaded");
  assert.equal(notesPayload.notes.length >= 1, true);

  const flagsResponse = await worker.fetch(
    makeGetRequest(`http://localhost/internal/accounts/${encodeURIComponent(userId)}/flags`, {
      "x-og-internal-admin": env.INTERNAL_ADMIN_SECRET,
    }),
    env
  );
  const flagsPayload = await flagsResponse.json();
  assert.equal(flagsResponse.status, 200);
  assert.equal(flagsPayload.status, "internal_account_flags_loaded");

  const timelineResponse = await worker.fetch(
    makeGetRequest(`http://localhost/internal/accounts/${encodeURIComponent(userId)}/timeline`, {
      "x-og-internal-admin": env.INTERNAL_ADMIN_SECRET,
    }),
    env
  );
  const timelinePayload = await timelineResponse.json();
  assert.equal(timelineResponse.status, 200);
  assert.equal(timelinePayload.status, "internal_account_timeline_loaded");
  assert.equal(Array.isArray(timelinePayload.timeline), true);
  assert.equal(timelinePayload.timeline.length >= 1, true);

  const unauthorizedResponse = await worker.fetch(
    makeGetRequest(`http://localhost/internal/accounts/${encodeURIComponent(userId)}`),
    env
  );
  const unauthorizedPayload = await unauthorizedResponse.json();
  assert.equal(unauthorizedResponse.status, 401);
  assert.equal(unauthorizedPayload.status, "internal_admin_unauthorized");
};

const testInternalAccountReviewActions = async (fetchHarness) => {
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

  const verifyResponse = await worker.fetch(
    makeGetRequest(`http://localhost/api/auth/magic-link/verify?token=${encodeURIComponent(token)}`),
    env
  );
  assert.equal(verifyResponse.status, 303);
  const userId = env.ACCOUNT_DB.accountSessions[0].user_id;
  env.ACCOUNT_DB.accountRiskFlags.push({
    id: "riskflag_test_resolve",
    user_id: userId,
    flag_type: "manual_support_concern",
    severity: "medium",
    flag_status: "open",
    source: "support_manual",
    summary: "Support requested a manual review.",
    evidence_json: "{}",
    opened_at: "2026-05-10T10:00:00.000Z",
    resolved_at: null,
    resolved_by: null,
    resolution_note: null,
    created_at: "2026-05-10T10:00:00.000Z",
    updated_at: "2026-05-10T10:00:00.000Z",
  });
  env.ACCOUNT_DB.accountRiskFlags.push({
    id: "riskflag_test_dismiss",
    user_id: userId,
    flag_type: "false_positive_check",
    severity: "low",
    flag_status: "open",
    source: "support_manual",
    summary: "Manual dismissal path test.",
    evidence_json: "{}",
    opened_at: "2026-05-10T10:05:00.000Z",
    resolved_at: null,
    resolved_by: null,
    resolution_note: null,
    created_at: "2026-05-10T10:05:00.000Z",
    updated_at: "2026-05-10T10:05:00.000Z",
  });

  const flag = await worker.fetch(
    makeGetRequest(`http://localhost/internal/accounts/${encodeURIComponent(userId)}/flags`, {
      "x-og-internal-admin": env.INTERNAL_ADMIN_SECRET,
    }),
    env
  );
  const flagPayload = await flag.json();
  assert.equal(flag.status, 200);
  assert.equal(flagPayload.status, "internal_account_flags_loaded");
  const targetFlagId = flagPayload.flags.find((row) => row.id === "riskflag_test_resolve")?.id;
  assert.equal(targetFlagId, "riskflag_test_resolve");

  const restrictResponse = await worker.fetch(
    jsonRequest(
      `http://localhost/internal/accounts/${encodeURIComponent(userId)}/restrict`,
      "POST",
      { reason: "Manual restriction for review", author_id: "internal:test" },
      { "x-og-internal-admin": env.INTERNAL_ADMIN_SECRET }
    ),
    env
  );
  const restrictPayload = await restrictResponse.json();
  assert.equal(restrictResponse.status, 200);
  assert.equal(restrictPayload.status, "internal_account_restricted");
  assert.equal(restrictPayload.account_summary.risk_state.account_status, "restricted");

  const badRestrictResponse = await worker.fetch(
    jsonRequest(
      `http://localhost/internal/accounts/${encodeURIComponent(userId)}/restrict`,
      "POST",
      { reason: "too short", author_id: "internal:test" },
      { "x-og-internal-admin": env.INTERNAL_ADMIN_SECRET }
    ),
    env
  );
  const badRestrictPayload = await badRestrictResponse.json();
  assert.equal(badRestrictResponse.status, 400);
  assert.equal(badRestrictPayload.status, "internal_restriction_reason_required");

  const resolveResponse = await worker.fetch(
    jsonRequest(
      `http://localhost/internal/accounts/${encodeURIComponent(userId)}/flags/${encodeURIComponent(targetFlagId)}/resolve`,
      "POST",
      { resolution_note: "Reviewed and understood", author_id: "internal:test" },
      { "x-og-internal-admin": env.INTERNAL_ADMIN_SECRET }
    ),
    env
  );
  const resolvePayload = await resolveResponse.json();
  assert.equal(resolveResponse.status, 200);
  assert.equal(resolvePayload.status, "internal_flag_resolved");
  assert.equal(resolvePayload.flags.find((row) => row.id === targetFlagId)?.flag_status, "resolved");

  const dismissResponse = await worker.fetch(
    jsonRequest(
      `http://localhost/internal/accounts/${encodeURIComponent(userId)}/flags/riskflag_test_dismiss/dismiss`,
      "POST",
      { resolution_note: "False positive", author_id: "internal:test" },
      { "x-og-internal-admin": env.INTERNAL_ADMIN_SECRET }
    ),
    env
  );
  const dismissPayload = await dismissResponse.json();
  assert.equal(dismissResponse.status, 200);
  assert.equal(dismissPayload.status, "internal_flag_dismissed");
  assert.equal(
    dismissPayload.flags.find((row) => row.id === "riskflag_test_dismiss")?.flag_status,
    "dismissed"
  );

  const suspendResponse = await worker.fetch(
    jsonRequest(
      `http://localhost/internal/accounts/${encodeURIComponent(userId)}/suspend`,
      "POST",
      {
        reason: "Confirmed misuse after manual review.",
        confirmation: "SUSPEND",
        author_id: "internal:test",
      },
      { "x-og-internal-admin": env.INTERNAL_ADMIN_SECRET }
    ),
    env
  );
  const suspendPayload = await suspendResponse.json();
  assert.equal(suspendResponse.status, 200);
  assert.equal(suspendPayload.status, "internal_account_suspended");
  assert.equal(suspendPayload.account_summary.risk_state.account_status, "suspended");
  assert.equal(env.ACCOUNT_DB.accountSessions.every((row) => Number(row.is_revoked || 0) === 1), true);

  const suspendedSessionCookie = extractCookieValue(verifyResponse.headers.get("set-cookie"), "og_premium_session");
  const suspendedSessionResponse = await worker.fetch(
    makeGetRequest("http://localhost/api/auth/session", {
      cookie: `og_premium_session=${suspendedSessionCookie}`,
    }),
    env
  );
  const suspendedSessionPayload = await suspendedSessionResponse.json();
  assert.equal(suspendedSessionResponse.status, 200);
  assert.equal(suspendedSessionPayload.authenticated, false);
  assert.equal(suspendedSessionPayload.status, "revoked_session");

  const badSuspendResponse = await worker.fetch(
    jsonRequest(
      `http://localhost/internal/accounts/${encodeURIComponent(userId)}/suspend`,
      "POST",
      { reason: "Confirmed misuse after manual review.", author_id: "internal:test" },
      { "x-og-internal-admin": env.INTERNAL_ADMIN_SECRET }
    ),
    env
  );
  const badSuspendPayload = await badSuspendResponse.json();
  assert.equal(badSuspendResponse.status, 400);
  assert.equal(badSuspendPayload.status, "internal_suspension_confirmation_required");

  const reinstateResponse = await worker.fetch(
    jsonRequest(
      `http://localhost/internal/accounts/${encodeURIComponent(userId)}/reinstate`,
      "POST",
      { reason: "Support verified ownership", author_id: "internal:test" },
      { "x-og-internal-admin": env.INTERNAL_ADMIN_SECRET }
    ),
    env
  );
  const reinstatePayload = await reinstateResponse.json();
  assert.equal(reinstateResponse.status, 200);
  assert.equal(reinstatePayload.status, "internal_account_reinstated");
  assert.equal(reinstatePayload.account_summary.risk_state.account_status, "active");

  const reviewOutcomeResponse = await worker.fetch(
    jsonRequest(
      `http://localhost/internal/accounts/${encodeURIComponent(userId)}/review-outcome`,
      "POST",
      {
        review_outcome: "reinstate_ready",
        review_outcome_note: "Ownership and payment standing now support reinstatement.",
        review_preset: "billing_concern",
        author_id: "internal:test",
      },
      { "x-og-internal-admin": env.INTERNAL_ADMIN_SECRET }
    ),
    env
  );
  const reviewOutcomePayload = await reviewOutcomeResponse.json();
  assert.equal(reviewOutcomeResponse.status, 200);
  assert.equal(reviewOutcomePayload.status, "internal_review_outcome_saved");
  assert.equal(reviewOutcomePayload.account_summary.risk_state.last_review_outcome, "reinstate_ready");
  assert.equal(reviewOutcomePayload.account_summary.risk_state.last_review_preset, "billing_concern");
  assert.equal(
    reviewOutcomePayload.account_summary.risk_state.last_review_outcome_note,
    "Ownership and payment standing now support reinstatement."
  );

  const badReviewOutcomeResponse = await worker.fetch(
    jsonRequest(
      `http://localhost/internal/accounts/${encodeURIComponent(userId)}/review-outcome`,
      "POST",
      {
        review_outcome: "auto",
        review_outcome_note: "too short",
        review_preset: "custom",
        author_id: "internal:test",
      },
      { "x-og-internal-admin": env.INTERNAL_ADMIN_SECRET }
    ),
    env
  );
  const badReviewOutcomePayload = await badReviewOutcomeResponse.json();
  assert.equal(badReviewOutcomeResponse.status, 400);
  assert.equal(badReviewOutcomePayload.status, "internal_review_outcome_invalid");
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
        user_style_preset: "analyst",
        decision_companion_enabled: true,
        reset_mode_enabled: true,
        language_preference: "en-GB",
        complete_calm_setup: true,
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
  assert.equal(prefsPayload.account.notification_preferences.user_style_preset, "analyst");
  assert.equal(prefsPayload.account.notification_preferences.language_preference, "en-GB");
  assert.equal(Boolean(prefsPayload.account.notification_preferences.calm_onboarding_completed_at), true);
  assert.deepEqual(prefsPayload.account.notification_preferences.favourite_teams, ["Arsenal", "Porto"]);
  assert.deepEqual(prefsPayload.account.notification_preferences.favourite_markets, ["BTTS", "OU25"]);
};

const testAccountAlertsQueueAndDispatch = async (fetchHarness) => {
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

  const prefsResponse = await worker.fetch(
    jsonRequest(
      "http://localhost/api/account/preferences",
      "POST",
      {
        telegram_enabled: true,
        website_only_mode: false,
        allow_non_signal_intelligence: true,
        favourite_teams: "Luzern",
        favourite_leagues: "Swiss Super League",
        favourite_markets: "BTTS",
        followed_fixtures: "Luzern v Servette",
      },
      {
        cookie: `og_premium_session=${sessionCookie}`,
      }
    ),
    env
  );
  assert.equal(prefsResponse.status, 200);

  const refreshResponse = await worker.fetch(
    jsonRequest("http://localhost/api/account/alerts/refresh", "POST", null, {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const refreshPayload = await refreshResponse.json();
  assert.equal(refreshResponse.status, 200);
  assert.equal(refreshPayload.status, "account_alerts_refreshed");
  assert.equal(refreshPayload.matched_fixtures >= 1, true);
  assert.equal(refreshPayload.queued_alerts >= 1, true);

  const alertsStateResponse = await worker.fetch(
    makeGetRequest("http://localhost/api/account/alerts", {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const alertsStatePayload = await alertsStateResponse.json();
  assert.equal(alertsStateResponse.status, 200);
  assert.equal(alertsStatePayload.status, "account_alerts_loaded");
  assert.equal(Array.isArray(alertsStatePayload.alerts), true);
  assert.equal(alertsStatePayload.alerts.length >= 1, true);

  env.ACCOUNT_DB.notificationAlerts = env.ACCOUNT_DB.notificationAlerts.map((alert) => ({
    ...alert,
    scheduled_for: "2000-01-01T00:00:00.000Z",
  }));

  const beforeCount = fetchHarness.counters.telegramSendFetches;
  const dispatchResponse = await worker.fetch(
    jsonRequest("http://localhost/api/account/alerts/dispatch", "POST", null, {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const dispatchPayload = await dispatchResponse.json();
  assert.equal(dispatchResponse.status, 200);
  assert.equal(dispatchPayload.status, "account_alerts_dispatched");
  assert.equal(dispatchPayload.delivered >= 1, true);
  assert.equal(fetchHarness.counters.telegramSendFetches >= beforeCount + 1, true);
  assert.match(fetchHarness.sentTelegramMessages.at(-1)?.text || "", /Luzern vs Servette/i);
};

const testMarketOnlyObserveDoesNotAutoQueue = async (fetchHarness) => {
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

  const prefsResponse = await worker.fetch(
    jsonRequest(
      "http://localhost/api/account/preferences",
      "POST",
      {
        telegram_enabled: true,
        website_only_mode: false,
        allow_non_signal_intelligence: true,
        favourite_markets: "BTTS",
      },
      {
        cookie: `og_premium_session=${sessionCookie}`,
      }
    ),
    env
  );
  assert.equal(prefsResponse.status, 200);

  const refreshResponse = await worker.fetch(
    jsonRequest("http://localhost/api/account/alerts/refresh", "POST", null, {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const refreshPayload = await refreshResponse.json();
  assert.equal(refreshResponse.status, 200);
  assert.equal(refreshPayload.matched_fixtures >= 1, true);
  assert.equal(refreshPayload.queued_alerts, 0);
};

const testWidgetStandingsProxy = async (fetchHarness) => {
  const env = createEnv();
  const firstResponse = await worker.fetch(
    makeGetRequest("http://localhost/api/widgets/football/standings?league=140&season=2025"),
    env
  );
  const firstPayload = await firstResponse.json();
  assert.equal(firstResponse.status, 200);
  assert.equal(firstPayload.response?.[0]?.league?.id, 140);
  assert.equal(fetchHarness.counters.widgetStandingsFetches, 1);

  const secondResponse = await worker.fetch(
    makeGetRequest("http://localhost/api/widgets/football/standings?league=140&season=2025"),
    env
  );
  const secondPayload = await secondResponse.json();
  assert.equal(secondResponse.status, 200);
  assert.equal(secondPayload.response?.[0]?.league?.season, 2025);
  assert.equal(fetchHarness.counters.widgetStandingsFetches, 1);
};

const testWidgetFixtureLookupProxy = async (fetchHarness) => {
  const env = createEnv();
  const firstResponse = await worker.fetch(
    makeGetRequest(
      "http://localhost/api/widgets/football/fixture-lookup?date=2026-05-10&home=Barcelona&away=Real%20Madrid&home_team_id=529&away_team_id=541"
    ),
    env
  );
  const firstPayload = await firstResponse.json();
  assert.equal(firstResponse.status, 200);
  assert.equal(firstPayload.fixture_id, 120001);
  assert.equal(fetchHarness.counters.widgetFixtureLookupFetches, 1);

  const secondResponse = await worker.fetch(
    makeGetRequest(
      "http://localhost/api/widgets/football/fixture-lookup?date=2026-05-10&home=Barcelona&away=Real%20Madrid&home_team_id=529&away_team_id=541"
    ),
    env
  );
  const secondPayload = await secondResponse.json();
  assert.equal(secondResponse.status, 200);
  assert.equal(secondPayload.league_id, 140);
  assert.equal(fetchHarness.counters.widgetFixtureLookupFetches, 1);
};

const testAnalystLeagueMarketDeployStaysWebsiteOnly = async (fetchHarness) => {
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

  const prefsResponse = await worker.fetch(
    jsonRequest(
      "http://localhost/api/account/preferences",
      "POST",
      {
        telegram_enabled: true,
        website_only_mode: false,
        allow_non_signal_intelligence: true,
        user_style_preset: "analyst",
        favourite_leagues: "USA MLS",
        favourite_markets: "BTTS",
      },
      {
        cookie: `og_premium_session=${sessionCookie}`,
      }
    ),
    env
  );
  assert.equal(prefsResponse.status, 200);

  const refreshResponse = await worker.fetch(
    jsonRequest("http://localhost/api/account/alerts/refresh", "POST", null, {
      cookie: `og_premium_session=${sessionCookie}`,
    }),
    env
  );
  const refreshPayload = await refreshResponse.json();
  assert.equal(refreshResponse.status, 200);
  assert.equal(refreshPayload.matched_fixtures >= 1, true);
  assert.equal(refreshPayload.queued_alerts, 0);
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
    await testPremiumTierAllowlistBoundaries();
    cacheHarness.clear();
    await testSitePayloadRoutesReadThroughR2();
    cacheHarness.clear();
    await testMissingToken();
    await testExpiredToken();
    await testInactiveSubscriber();
    await testMagicLinkRequestValidation(fetchHarness);
    await testAuthSessionSkeleton();
    await testMagicLinkVerifyAndSessionFlow(fetchHarness);
    await testAccountStateAndTelegramLinkFlow(fetchHarness);
    await testAccountSessionsState(fetchHarness);
    await testAccountSessionActions(fetchHarness);
    await testAccountRiskFlaggingFromSessionSpread(fetchHarness);
    await testInternalAccountReviewReadApis(fetchHarness);
    await testInternalAccountReviewActions(fetchHarness);
    await testTelegramWebhookCompletesLinkFlow(fetchHarness);
    await testTelegramTestAlertRoute(fetchHarness);
    await testTelegramFixtureAlertRoute(fetchHarness);
    await testAccountPreferencesUpdate(fetchHarness);
    await testAccountAlertsQueueAndDispatch(fetchHarness);
    await testWidgetStandingsProxy(fetchHarness);
    await testWidgetFixtureLookupProxy(fetchHarness);
    await testMarketOnlyObserveDoesNotAutoQueue(fetchHarness);
    await testAnalystLeagueMarketDeployStaysWebsiteOnly(fetchHarness);
    await testStripeCheckoutSmoke(fetchHarness);
    await testStripeCheckoutWebhookEnrichesSubscriptionState(fetchHarness);
    await testStripePortalSmoke(fetchHarness);
    await testSessionRestoreAndPaymentIssueStates(fetchHarness);
    await testLogoutSkeleton();
    console.log("Worker local harness passed.");
    console.log("- success route with valid token: passed");
    console.log("- premium tier payload allowlists: passed");
    console.log("- site payload R2 read-through routes: passed");
    console.log("- premium payload cache hit/miss path: passed");
    console.log("- missing token returns 401: passed");
    console.log("- expired token returns 401: passed");
    console.log("- inactive subscriber returns 401: passed");
    console.log("- magic-link request flow: passed");
    console.log("- auth session flow: passed");
    console.log("- magic-link verify + session premium flow: passed");
    console.log("- D1-backed account state + Telegram link flow: passed");
    console.log("- account devices session surface: passed");
    console.log("- account device actions: passed");
    console.log("- account risk state and flag recording: passed");
    console.log("- internal account review read APIs: passed");
    console.log("- internal account review actions: passed");
    console.log("- Telegram bot webhook completion flow: passed");
    console.log("- Telegram test alert route: passed");
    console.log("- Telegram fixture alert route: passed");
    console.log("- Account preferences update route: passed");
    console.log("- account alerts queue + dispatch routes: passed");
    console.log("- widget standings proxy cache path: passed");
    console.log("- widget fixture lookup proxy cache path: passed");
    console.log("- market-only observe suppression: passed");
    console.log("- analyst league+market deploy stays website-only: passed");
    console.log("- Stripe checkout smoke path: passed");
    console.log("- Stripe checkout webhook subscription enrichment: passed");
    console.log("- Stripe billing portal smoke path: passed");
    console.log("- session restore + payment issue states: passed");
    console.log("- logout skeleton: passed");
  } finally {
    fetchHarness.restore();
    cacheHarness.restore();
  }
};

await main();
