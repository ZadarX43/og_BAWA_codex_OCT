const isoNow = () => new Date().toISOString();
const normalizeEmail = (value) => String(value || "").trim().toLowerCase();
const toJson = (value) => JSON.stringify(value ?? null);
const toIntFlag = (value, fallback = 0) => (value ? 1 : fallback ? 1 : 0);
const fromJson = (value, fallback) => {
  if (!value) {
    return fallback;
  }
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
};

const buildId = (prefix) => `${prefix}_${crypto.randomUUID()}`;
const normalizeStringList = (value, limit = 24) => {
  const input = Array.isArray(value)
    ? value
    : String(value || "")
        .split(",")
        .map((item) => item.trim());
  return Array.from(
    new Set(
      input
        .map((item) => String(item || "").trim())
        .filter(Boolean)
        .slice(0, limit)
    )
  );
};
const normalizeQuietHours = (value) => {
  if (!value || typeof value !== "object") {
    return null;
  }
  const startHour = Math.max(0, Math.min(23, Number(value.start_hour ?? value.startHour)));
  const endHour = Math.max(0, Math.min(23, Number(value.end_hour ?? value.endHour)));
  if (!Number.isFinite(startHour) || !Number.isFinite(endHour)) {
    return null;
  }
  return { start_hour: startHour, end_hour: endHour };
};
const buildDefaultNotificationPreferences = (userId, existing = {}) => {
  const now = isoNow();
  return {
    id: existing.id || buildId("pref"),
    user_id: userId,
    email_enabled: existing.email_enabled ?? 1,
    telegram_enabled: existing.telegram_enabled ?? 0,
    elite_alerts_enabled: existing.elite_alerts_enabled ?? 1,
    standard_alerts_enabled: existing.standard_alerts_enabled ?? 1,
    acca_alerts_enabled: existing.acca_alerts_enabled ?? 0,
    correct_score_alerts_enabled: existing.correct_score_alerts_enabled ?? 0,
    injury_alerts_enabled: existing.injury_alerts_enabled ?? 1,
    weather_alerts_enabled: existing.weather_alerts_enabled ?? 1,
    market_movement_alerts_enabled: existing.market_movement_alerts_enabled ?? 1,
    volatility_alerts_enabled: existing.volatility_alerts_enabled ?? 1,
    team_news_alerts_enabled: existing.team_news_alerts_enabled ?? 1,
    daily_digest_enabled: existing.daily_digest_enabled ?? 1,
    results_digest_enabled: existing.results_digest_enabled ?? 1,
    weekend_slate_digest_enabled: existing.weekend_slate_digest_enabled ?? 1,
    website_only_mode: existing.website_only_mode ?? 0,
    allow_non_signal_intelligence: existing.allow_non_signal_intelligence ?? 1,
    alert_frequency_mode: existing.alert_frequency_mode || "mixed",
    pre_match_window_minutes: existing.pre_match_window_minutes ?? 90,
    user_style_preset: existing.user_style_preset || "disciplined_bettor",
    decision_companion_enabled: existing.decision_companion_enabled ?? 1,
    reset_mode_enabled: existing.reset_mode_enabled ?? 1,
    calm_onboarding_completed_at: existing.calm_onboarding_completed_at || null,
    language_preference: existing.language_preference || "en-GB",
    favourite_markets_json: toJson(fromJson(existing.favourite_markets_json, [])),
    favourite_leagues_json: toJson(fromJson(existing.favourite_leagues_json, [])),
    favourite_teams_json: toJson(fromJson(existing.favourite_teams_json, [])),
    followed_fixtures_json: toJson(fromJson(existing.followed_fixtures_json, [])),
    quiet_hours_json: toJson(fromJson(existing.quiet_hours_json, null)),
    updated_at: existing.updated_at || now,
  };
};

const callMaybeMock = async (db, op, args, sqlRunner) => {
  if (db && typeof db.__ogCall === "function") {
    return db.__ogCall(op, args);
  }
  return sqlRunner();
};

export const getAccountDb = (env) => {
  const db = env.ACCOUNT_DB;
  if (!db || typeof db.prepare !== "function") {
    return null;
  }
  return db;
};

const first = async (db, sql, binds = []) => db.prepare(sql).bind(...binds).first();
const run = async (db, sql, binds = []) => db.prepare(sql).bind(...binds).run();
const all = async (db, sql, binds = []) => {
  const result = await db.prepare(sql).bind(...binds).all();
  return Array.isArray(result?.results) ? result.results : [];
};

export async function upsertUserByEmail(db, email, options = {}) {
  if (!db) {
    return null;
  }
  const normalizedEmail = normalizeEmail(email);
  if (!normalizedEmail) {
    return null;
  }

  const now = isoNow();
  const existing = await callMaybeMock(
    db,
    "get_user_by_email",
    { email_normalized: normalizedEmail },
    () =>
      first(
        db,
        `-- og:get_user_by_email
        SELECT id, email, email_normalized, email_verified_at, created_at, updated_at, account_status
        FROM users
        WHERE email_normalized = ?1
        LIMIT 1`,
        [normalizedEmail]
      )
  );

  const verifiedAt = options.emailVerifiedAt || existing?.email_verified_at || null;
  const accountStatus = options.accountStatus || existing?.account_status || "active";
  const displayEmail = String(email || existing?.email || normalizedEmail).trim();

  if (existing?.id) {
    await callMaybeMock(
      db,
      "update_user_by_email",
      {
        id: existing.id,
        email: displayEmail,
        email_normalized: normalizedEmail,
        email_verified_at: verifiedAt,
        updated_at: now,
        account_status: accountStatus,
      },
      () =>
        run(
          db,
          `-- og:update_user_by_email
          UPDATE users
          SET email = ?2,
              email_verified_at = ?3,
              updated_at = ?4,
              account_status = ?5
          WHERE id = ?1`,
          [existing.id, displayEmail, verifiedAt, now, accountStatus]
        )
    );
  } else {
    await callMaybeMock(
      db,
      "insert_user",
      {
        id: buildId("user"),
        email: displayEmail,
        email_normalized: normalizedEmail,
        email_verified_at: verifiedAt,
        created_at: now,
        updated_at: now,
        account_status: accountStatus,
      },
      async () => {
        const id = buildId("user");
        await run(
          db,
          `-- og:insert_user
          INSERT INTO users (id, email, email_normalized, email_verified_at, created_at, updated_at, account_status)
          VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)`,
          [id, displayEmail, normalizedEmail, verifiedAt, now, now, accountStatus]
        );
      }
    );
  }

  return callMaybeMock(
    db,
    "get_user_by_email",
    { email_normalized: normalizedEmail },
    () =>
      first(
        db,
        `-- og:get_user_by_email
        SELECT id, email, email_normalized, email_verified_at, created_at, updated_at, account_status
        FROM users
        WHERE email_normalized = ?1
        LIMIT 1`,
        [normalizedEmail]
      )
  );
}

async function loadUserBySubscriptionIdentity(db, customerId, subscriptionId) {
  if (!db || (!customerId && !subscriptionId)) {
    return null;
  }
  return callMaybeMock(
    db,
    "get_user_by_subscription_identity",
    { stripe_customer_id: customerId || "", stripe_subscription_id: subscriptionId || "" },
    () =>
      first(
        db,
        `-- og:get_user_by_subscription_identity
        SELECT u.id, u.email, u.email_normalized, u.email_verified_at, u.created_at, u.updated_at, u.account_status
        FROM subscriptions s
        JOIN users u ON u.id = s.user_id
        WHERE s.stripe_customer_id = ?1 OR s.stripe_subscription_id = ?2
        ORDER BY s.updated_at DESC
        LIMIT 1`,
        [customerId || "", subscriptionId || ""]
      )
  );
}

export async function mirrorSubscriptionFromRecord(db, record, options = {}) {
  if (!db || !record?.customer_id) {
    return null;
  }

  let user = null;
  const email = normalizeEmail(options.email || record.email || "");
  if (email) {
    user = await upsertUserByEmail(db, email, {
      emailVerifiedAt: options.emailVerifiedAt || null,
      accountStatus: "active",
    });
  }
  if (!user) {
    user = await loadUserBySubscriptionIdentity(db, record.customer_id, record.subscription_id);
  }
  if (!user?.id) {
    return null;
  }

  const now = isoNow();
  const existing = await callMaybeMock(
    db,
    "get_subscription_by_identity",
    { stripe_customer_id: record.customer_id, stripe_subscription_id: record.subscription_id || "" },
    () =>
      first(
        db,
        `-- og:get_subscription_by_identity
        SELECT id, user_id, stripe_customer_id, stripe_subscription_id, subscription_status, price_id, current_period_end, created_at, updated_at
        FROM subscriptions
        WHERE stripe_customer_id = ?1 OR stripe_subscription_id = ?2
        ORDER BY updated_at DESC
        LIMIT 1`,
        [record.customer_id, record.subscription_id || ""]
      )
  );

  const payload = {
    id: existing?.id || buildId("sub"),
    user_id: user.id,
    stripe_customer_id: record.customer_id,
    stripe_subscription_id: record.subscription_id || `${record.customer_id}:pending`,
    subscription_status: String(record.status || "").trim() || "unknown",
    price_id: record.price_id || null,
    current_period_end: record.current_period_end || null,
    created_at: existing?.created_at || now,
    updated_at: now,
  };

  await callMaybeMock(
    db,
    "upsert_subscription",
    payload,
    () =>
      run(
        db,
        `-- og:upsert_subscription
        INSERT INTO subscriptions (
          id, user_id, stripe_customer_id, stripe_subscription_id, subscription_status, price_id, current_period_end, created_at, updated_at
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        ON CONFLICT(stripe_subscription_id) DO UPDATE SET
          user_id = excluded.user_id,
          stripe_customer_id = excluded.stripe_customer_id,
          subscription_status = excluded.subscription_status,
          price_id = excluded.price_id,
          current_period_end = excluded.current_period_end,
          updated_at = excluded.updated_at`,
        [
          payload.id,
          payload.user_id,
          payload.stripe_customer_id,
          payload.stripe_subscription_id,
          payload.subscription_status,
          payload.price_id,
          payload.current_period_end,
          payload.created_at,
          payload.updated_at,
        ]
      )
  );

  await ensureNotificationPreferences(db, user.id);
  return getAccountStateByEmail(db, user.email_normalized);
}

export async function ensureNotificationPreferences(db, userId) {
  if (!db || !userId) {
    return null;
  }
  const defaults = buildDefaultNotificationPreferences(userId);
  await callMaybeMock(
    db,
    "ensure_notification_preferences",
    defaults,
    () =>
      run(
        db,
        `-- og:ensure_notification_preferences
        INSERT INTO notification_preferences (
          id, user_id, email_enabled, telegram_enabled, elite_alerts_enabled, standard_alerts_enabled,
          acca_alerts_enabled, correct_score_alerts_enabled, injury_alerts_enabled, weather_alerts_enabled,
          market_movement_alerts_enabled, volatility_alerts_enabled, team_news_alerts_enabled, daily_digest_enabled,
          results_digest_enabled, weekend_slate_digest_enabled, website_only_mode, allow_non_signal_intelligence,
          alert_frequency_mode, pre_match_window_minutes, user_style_preset, decision_companion_enabled,
          reset_mode_enabled, calm_onboarding_completed_at, language_preference, favourite_markets_json,
          favourite_leagues_json, favourite_teams_json, followed_fixtures_json, quiet_hours_json, updated_at
        )
        VALUES (
          ?1, ?2, ?3, ?4, ?5, ?6,
          ?7, ?8, ?9, ?10,
          ?11, ?12, ?13, ?14,
          ?15, ?16, ?17, ?18,
          ?19, ?20, ?21, ?22,
          ?23, ?24, ?25, ?26,
          ?27, ?28, ?29, ?30,
          ?31
        )
        ON CONFLICT(user_id) DO NOTHING`,
        [
          defaults.id,
          defaults.user_id,
          defaults.email_enabled,
          defaults.telegram_enabled,
          defaults.elite_alerts_enabled,
          defaults.standard_alerts_enabled,
          defaults.acca_alerts_enabled,
          defaults.correct_score_alerts_enabled,
          defaults.injury_alerts_enabled,
          defaults.weather_alerts_enabled,
          defaults.market_movement_alerts_enabled,
          defaults.volatility_alerts_enabled,
          defaults.team_news_alerts_enabled,
          defaults.daily_digest_enabled,
          defaults.results_digest_enabled,
          defaults.weekend_slate_digest_enabled,
          defaults.website_only_mode,
          defaults.allow_non_signal_intelligence,
          defaults.alert_frequency_mode,
          defaults.pre_match_window_minutes,
          defaults.user_style_preset,
          defaults.decision_companion_enabled,
          defaults.reset_mode_enabled,
          defaults.calm_onboarding_completed_at,
          defaults.language_preference,
          defaults.favourite_markets_json,
          defaults.favourite_leagues_json,
          defaults.favourite_teams_json,
          defaults.followed_fixtures_json,
          defaults.quiet_hours_json,
          defaults.updated_at,
        ]
      )
  );
}

export async function recordAuthEvent(db, event) {
  if (!db || !event?.event_type) {
    return;
  }
  const payload = {
    id: buildId("auth"),
    user_id: event.user_id || null,
    email_normalized: normalizeEmail(event.email_normalized || ""),
    event_type: event.event_type,
    ip_hint: event.ip_hint || null,
    user_agent_hint: event.user_agent_hint || null,
    created_at: event.created_at || isoNow(),
    metadata_json: toJson(event.metadata || {}),
  };
  await callMaybeMock(
    db,
    "insert_auth_event",
    payload,
    () =>
      run(
        db,
        `-- og:insert_auth_event
        INSERT INTO auth_events (
          id, user_id, email_normalized, event_type, ip_hint, user_agent_hint, created_at, metadata_json
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)`,
        [
          payload.id,
          payload.user_id,
          payload.email_normalized || null,
          payload.event_type,
          payload.ip_hint,
          payload.user_agent_hint,
          payload.created_at,
          payload.metadata_json,
        ]
      )
  );
}

export async function getAccountStateByEmail(db, email) {
  if (!db) {
    return null;
  }
  const normalizedEmail = normalizeEmail(email);
  if (!normalizedEmail) {
    return null;
  }

  const user = await callMaybeMock(
    db,
    "get_user_by_email",
    { email_normalized: normalizedEmail },
    () =>
      first(
        db,
        `-- og:get_user_by_email
        SELECT id, email, email_normalized, email_verified_at, created_at, updated_at, account_status
        FROM users
        WHERE email_normalized = ?1
        LIMIT 1`,
        [normalizedEmail]
      )
  );
  if (!user?.id) {
    return null;
  }

  const subscription = await callMaybeMock(
    db,
    "get_subscription_by_user",
    { user_id: user.id },
    () =>
      first(
        db,
        `-- og:get_subscription_by_user
        SELECT id, user_id, stripe_customer_id, stripe_subscription_id, subscription_status, price_id, current_period_end, created_at, updated_at
        FROM subscriptions
        WHERE user_id = ?1
        ORDER BY updated_at DESC
        LIMIT 1`,
        [user.id]
      )
  );
  const telegramLink = await callMaybeMock(
    db,
    "get_telegram_link_by_user",
    { user_id: user.id },
    () =>
      first(
        db,
        `-- og:get_telegram_link_by_user
        SELECT id, user_id, telegram_user_id, telegram_username, telegram_chat_id, link_status, linked_at, revoked_at, created_at, updated_at
        FROM telegram_links
        WHERE user_id = ?1
        ORDER BY CASE link_status WHEN 'linked' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END, updated_at DESC
        LIMIT 1`,
        [user.id]
      )
  );
  const notificationPreferences = await callMaybeMock(
    db,
    "get_notification_preferences",
    { user_id: user.id },
    () =>
      first(
        db,
        `-- og:get_notification_preferences
        SELECT id, user_id, email_enabled, telegram_enabled, elite_alerts_enabled, standard_alerts_enabled,
               acca_alerts_enabled, correct_score_alerts_enabled, injury_alerts_enabled, weather_alerts_enabled,
               market_movement_alerts_enabled, volatility_alerts_enabled, team_news_alerts_enabled,
               daily_digest_enabled, results_digest_enabled, weekend_slate_digest_enabled, website_only_mode,
               allow_non_signal_intelligence, alert_frequency_mode, pre_match_window_minutes, user_style_preset,
               decision_companion_enabled, reset_mode_enabled, calm_onboarding_completed_at, language_preference,
               favourite_markets_json, favourite_leagues_json, favourite_teams_json, followed_fixtures_json,
               quiet_hours_json, updated_at
        FROM notification_preferences
        WHERE user_id = ?1
        LIMIT 1`,
        [user.id]
      )
  );

  return {
    user,
    subscription,
    telegram_link: telegramLink
      ? {
          ...telegramLink,
          telegram_chat_id: telegramLink.telegram_chat_id || null,
        }
      : null,
    notification_preferences: notificationPreferences
      ? {
          ...notificationPreferences,
          standard_alerts_enabled: notificationPreferences.standard_alerts_enabled ?? 1,
          correct_score_alerts_enabled: notificationPreferences.correct_score_alerts_enabled ?? 0,
          injury_alerts_enabled: notificationPreferences.injury_alerts_enabled ?? 1,
          weather_alerts_enabled: notificationPreferences.weather_alerts_enabled ?? 1,
          market_movement_alerts_enabled: notificationPreferences.market_movement_alerts_enabled ?? 1,
          volatility_alerts_enabled: notificationPreferences.volatility_alerts_enabled ?? 1,
          team_news_alerts_enabled: notificationPreferences.team_news_alerts_enabled ?? 1,
          daily_digest_enabled: notificationPreferences.daily_digest_enabled ?? 1,
          weekend_slate_digest_enabled: notificationPreferences.weekend_slate_digest_enabled ?? 1,
          website_only_mode: notificationPreferences.website_only_mode ?? 0,
          allow_non_signal_intelligence: notificationPreferences.allow_non_signal_intelligence ?? 1,
          alert_frequency_mode: notificationPreferences.alert_frequency_mode || "mixed",
          pre_match_window_minutes: notificationPreferences.pre_match_window_minutes ?? 90,
          user_style_preset: notificationPreferences.user_style_preset || "disciplined_bettor",
          decision_companion_enabled: notificationPreferences.decision_companion_enabled ?? 1,
          reset_mode_enabled: notificationPreferences.reset_mode_enabled ?? 1,
          calm_onboarding_completed_at: notificationPreferences.calm_onboarding_completed_at || null,
          language_preference: notificationPreferences.language_preference || "en-GB",
          favourite_markets: fromJson(notificationPreferences.favourite_markets_json, []),
          favourite_leagues: fromJson(notificationPreferences.favourite_leagues_json, []),
          favourite_teams: fromJson(notificationPreferences.favourite_teams_json, []),
          followed_fixtures: fromJson(notificationPreferences.followed_fixtures_json, []),
          quiet_hours: fromJson(notificationPreferences.quiet_hours_json, null),
        }
      : null,
  };
}

export async function updateNotificationPreferences(db, userId, input = {}) {
  if (!db || !userId) {
    return null;
  }

  const existing = await callMaybeMock(
    db,
    "get_notification_preferences",
    { user_id: userId },
    () =>
      first(
        db,
        `-- og:get_notification_preferences
        SELECT id, user_id, email_enabled, telegram_enabled, elite_alerts_enabled, standard_alerts_enabled,
               acca_alerts_enabled, correct_score_alerts_enabled, injury_alerts_enabled, weather_alerts_enabled,
               market_movement_alerts_enabled, volatility_alerts_enabled, team_news_alerts_enabled,
               daily_digest_enabled, results_digest_enabled, weekend_slate_digest_enabled, website_only_mode,
               allow_non_signal_intelligence, alert_frequency_mode, pre_match_window_minutes, user_style_preset,
               decision_companion_enabled, reset_mode_enabled, calm_onboarding_completed_at, language_preference,
               favourite_markets_json, favourite_leagues_json, favourite_teams_json, followed_fixtures_json,
               quiet_hours_json, updated_at
        FROM notification_preferences
        WHERE user_id = ?1
        LIMIT 1`,
        [userId]
      )
  );

  const base = buildDefaultNotificationPreferences(userId, existing || {});
  const stylePreset = ["analyst", "disciplined_bettor", "tactical_reader", "researcher"].includes(
    String(input.user_style_preset || "")
  )
    ? String(input.user_style_preset)
    : base.user_style_preset;
  const frequencyMode = ["immediate", "digest_only", "mixed"].includes(String(input.alert_frequency_mode || ""))
    ? String(input.alert_frequency_mode)
    : base.alert_frequency_mode;
  const languagePreference = ["en-GB", "en-US", "pt-PT", "es-ES"].includes(String(input.language_preference || ""))
    ? String(input.language_preference)
    : base.language_preference;
  const preMatchWindow = Math.max(0, Math.min(1440, Number(input.pre_match_window_minutes ?? base.pre_match_window_minutes)));
  const quietHours = normalizeQuietHours(input.quiet_hours ?? fromJson(base.quiet_hours_json, null));
  const calmOnboardingCompletedAt =
    typeof input.calm_onboarding_completed_at === "string" && input.calm_onboarding_completed_at.trim()
      ? input.calm_onboarding_completed_at.trim()
      : input.complete_calm_setup
        ? base.calm_onboarding_completed_at || isoNow()
        : base.calm_onboarding_completed_at || null;
  const payload = {
    ...base,
    email_enabled: toIntFlag(input.email_enabled ?? base.email_enabled),
    telegram_enabled: toIntFlag(input.telegram_enabled ?? base.telegram_enabled),
    elite_alerts_enabled: toIntFlag(input.elite_alerts_enabled ?? base.elite_alerts_enabled),
    standard_alerts_enabled: toIntFlag(input.standard_alerts_enabled ?? base.standard_alerts_enabled),
    acca_alerts_enabled: toIntFlag(input.acca_alerts_enabled ?? base.acca_alerts_enabled),
    correct_score_alerts_enabled: toIntFlag(input.correct_score_alerts_enabled ?? base.correct_score_alerts_enabled),
    injury_alerts_enabled: toIntFlag(input.injury_alerts_enabled ?? base.injury_alerts_enabled),
    weather_alerts_enabled: toIntFlag(input.weather_alerts_enabled ?? base.weather_alerts_enabled),
    market_movement_alerts_enabled: toIntFlag(
      input.market_movement_alerts_enabled ?? base.market_movement_alerts_enabled
    ),
    volatility_alerts_enabled: toIntFlag(input.volatility_alerts_enabled ?? base.volatility_alerts_enabled),
    team_news_alerts_enabled: toIntFlag(input.team_news_alerts_enabled ?? base.team_news_alerts_enabled),
    daily_digest_enabled: toIntFlag(input.daily_digest_enabled ?? base.daily_digest_enabled),
    results_digest_enabled: toIntFlag(input.results_digest_enabled ?? base.results_digest_enabled),
    weekend_slate_digest_enabled: toIntFlag(
      input.weekend_slate_digest_enabled ?? base.weekend_slate_digest_enabled
    ),
    website_only_mode: toIntFlag(input.website_only_mode ?? base.website_only_mode),
    allow_non_signal_intelligence: toIntFlag(
      input.allow_non_signal_intelligence ?? base.allow_non_signal_intelligence
    ),
    alert_frequency_mode: frequencyMode,
    pre_match_window_minutes: Number.isFinite(preMatchWindow) ? preMatchWindow : base.pre_match_window_minutes,
    user_style_preset: stylePreset,
    decision_companion_enabled: toIntFlag(input.decision_companion_enabled ?? base.decision_companion_enabled),
    reset_mode_enabled: toIntFlag(input.reset_mode_enabled ?? base.reset_mode_enabled),
    calm_onboarding_completed_at: calmOnboardingCompletedAt,
    language_preference: languagePreference,
    favourite_markets_json: toJson(normalizeStringList(input.favourite_markets ?? fromJson(base.favourite_markets_json, []))),
    favourite_leagues_json: toJson(normalizeStringList(input.favourite_leagues ?? fromJson(base.favourite_leagues_json, []))),
    favourite_teams_json: toJson(normalizeStringList(input.favourite_teams ?? fromJson(base.favourite_teams_json, []))),
    followed_fixtures_json: toJson(
      normalizeStringList(input.followed_fixtures ?? fromJson(base.followed_fixtures_json, []), 40)
    ),
    quiet_hours_json: toJson(quietHours),
    updated_at: isoNow(),
  };

  await callMaybeMock(
    db,
    "update_notification_preferences",
    payload,
    () =>
      run(
        db,
        `-- og:update_notification_preferences
        UPDATE notification_preferences
        SET email_enabled = ?2,
            telegram_enabled = ?3,
            elite_alerts_enabled = ?4,
            standard_alerts_enabled = ?5,
            acca_alerts_enabled = ?6,
            correct_score_alerts_enabled = ?7,
            injury_alerts_enabled = ?8,
            weather_alerts_enabled = ?9,
            market_movement_alerts_enabled = ?10,
            volatility_alerts_enabled = ?11,
            team_news_alerts_enabled = ?12,
            daily_digest_enabled = ?13,
            results_digest_enabled = ?14,
            weekend_slate_digest_enabled = ?15,
            website_only_mode = ?16,
            allow_non_signal_intelligence = ?17,
            alert_frequency_mode = ?18,
            pre_match_window_minutes = ?19,
            user_style_preset = ?20,
            decision_companion_enabled = ?21,
            reset_mode_enabled = ?22,
            calm_onboarding_completed_at = ?23,
            language_preference = ?24,
            favourite_markets_json = ?25,
            favourite_leagues_json = ?26,
            favourite_teams_json = ?27,
            followed_fixtures_json = ?28,
            quiet_hours_json = ?29,
            updated_at = ?30
        WHERE user_id = ?1`,
        [
          payload.user_id,
          payload.email_enabled,
          payload.telegram_enabled,
          payload.elite_alerts_enabled,
          payload.standard_alerts_enabled,
          payload.acca_alerts_enabled,
          payload.correct_score_alerts_enabled,
          payload.injury_alerts_enabled,
          payload.weather_alerts_enabled,
          payload.market_movement_alerts_enabled,
          payload.volatility_alerts_enabled,
          payload.team_news_alerts_enabled,
          payload.daily_digest_enabled,
          payload.results_digest_enabled,
          payload.weekend_slate_digest_enabled,
          payload.website_only_mode,
          payload.allow_non_signal_intelligence,
          payload.alert_frequency_mode,
          payload.pre_match_window_minutes,
          payload.user_style_preset,
          payload.decision_companion_enabled,
          payload.reset_mode_enabled,
          payload.calm_onboarding_completed_at,
          payload.language_preference,
          payload.favourite_markets_json,
          payload.favourite_leagues_json,
          payload.favourite_teams_json,
          payload.followed_fixtures_json,
          payload.quiet_hours_json,
          payload.updated_at,
        ]
      )
  );

  return payload;
}

export async function completeTelegramLink(db, payload) {
  if (!db || !payload?.user_id || !payload?.telegram_user_id) {
    return null;
  }
  const now = isoNow();
  await callMaybeMock(
    db,
    "revoke_telegram_links_for_user",
    { user_id: payload.user_id, revoked_at: now, updated_at: now },
    () =>
      run(
        db,
        `-- og:revoke_telegram_links_for_user
        UPDATE telegram_links
        SET link_status = 'revoked', revoked_at = ?2, updated_at = ?3
        WHERE user_id = ?1 AND link_status != 'revoked'`,
        [payload.user_id, now, now]
      )
  );
  await callMaybeMock(
    db,
    "revoke_telegram_links_for_telegram_user",
    { telegram_user_id: payload.telegram_user_id, revoked_at: now, updated_at: now },
    () =>
      run(
        db,
        `-- og:revoke_telegram_links_for_telegram_user
        UPDATE telegram_links
        SET link_status = 'revoked', revoked_at = ?2, updated_at = ?3
        WHERE telegram_user_id = ?1 AND link_status != 'revoked'`,
        [String(payload.telegram_user_id), now, now]
      )
  );

  const linkId = buildId("tg");
  await callMaybeMock(
    db,
    "insert_telegram_link",
    {
      id: linkId,
      user_id: payload.user_id,
      telegram_user_id: String(payload.telegram_user_id),
      telegram_username: payload.telegram_username || null,
      telegram_chat_id: payload.telegram_chat_id || null,
      link_status: "linked",
      linked_at: now,
      revoked_at: null,
      created_at: now,
      updated_at: now,
    },
    () =>
      run(
        db,
        `-- og:insert_telegram_link
        INSERT INTO telegram_links (
          id, user_id, telegram_user_id, telegram_username, telegram_chat_id, link_status, linked_at, revoked_at, created_at, updated_at
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, ?8, ?9)`,
        [
          linkId,
          payload.user_id,
          String(payload.telegram_user_id),
          payload.telegram_username || null,
          payload.telegram_chat_id || null,
          "linked",
          now,
          now,
          now,
        ]
      )
  );

  await callMaybeMock(
    db,
    "enable_telegram_notifications",
    { user_id: payload.user_id, updated_at: now },
    () =>
      run(
        db,
        `-- og:enable_telegram_notifications
        UPDATE notification_preferences
        SET telegram_enabled = 1, updated_at = ?2
        WHERE user_id = ?1`,
        [payload.user_id, now]
      )
  );

  return getAccountStateByEmail(db, payload.email || "");
}

export async function listUsersEligibleForTelegramAlerts(db) {
  if (!db) {
    return [];
  }
  return callMaybeMock(
    db,
    "list_users_eligible_for_telegram_alerts",
    {},
    () =>
      all(
        db,
        `-- og:list_users_eligible_for_telegram_alerts
        SELECT DISTINCT
          u.id,
          u.email,
          u.email_normalized,
          t.telegram_chat_id
        FROM users u
        JOIN notification_preferences p ON p.user_id = u.id
        JOIN telegram_links t ON t.user_id = u.id
        LEFT JOIN subscriptions s ON s.user_id = u.id
        WHERE p.telegram_enabled = 1
          AND p.website_only_mode = 0
          AND t.link_status = 'linked'
          AND COALESCE(t.telegram_chat_id, '') != ''
          AND (
            s.subscription_status IS NULL OR
            s.subscription_status IN ('active', 'trialing')
          )
        ORDER BY u.email_normalized ASC`
      )
  );
}

export async function upsertNotificationAlerts(db, alerts = []) {
  if (!db || !Array.isArray(alerts) || !alerts.length) {
    return { attempted: 0, queued: 0 };
  }

  let queued = 0;
  for (const alert of alerts) {
    if (!alert?.user_id || !alert?.dedupe_key || !alert?.payload_json) {
      continue;
    }
    let touched = false;
    await callMaybeMock(
      db,
      "upsert_notification_alert",
      alert,
      async () => {
        const existing = await first(
          db,
          `-- og:get_notification_alert_by_dedupe
          SELECT id, status, delivered_at
          FROM notification_alerts
          WHERE dedupe_key = ?1
          LIMIT 1`,
          [alert.dedupe_key]
        );

        if (existing?.id) {
          if (existing.status === "delivered") {
            return;
          }
          await run(
            db,
            `-- og:update_notification_alert
            UPDATE notification_alerts
            SET fixture_id = ?2,
                fixture_label = ?3,
                league = ?4,
                market_family = ?5,
                publish_class = ?6,
                reasons_json = ?7,
                payload_json = ?8,
                notification_priority = ?9,
                scheduled_for = ?10,
                last_error = NULL,
                updated_at = ?11
            WHERE dedupe_key = ?1`,
            [
              alert.dedupe_key,
              alert.fixture_id || null,
              alert.fixture_label,
              alert.league || null,
              alert.market_family || null,
              alert.publish_class || null,
              alert.reasons_json || null,
              alert.payload_json,
              alert.notification_priority || "normal",
              alert.scheduled_for,
              alert.updated_at,
            ]
          );
          touched = true;
          return;
        }

        await run(
          db,
          `-- og:insert_notification_alert
          INSERT INTO notification_alerts (
            id, user_id, channel, alert_kind, fixture_key, fixture_id, fixture_label,
            league, market_family, publish_class, reasons_json, payload_json, dedupe_key,
            notification_priority, scheduled_for, status, delivered_at, last_error, created_at, updated_at
          )
          VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7,
            ?8, ?9, ?10, ?11, ?12, ?13,
            ?14, ?15, ?16, NULL, NULL, ?17, ?18
          )`,
          [
            alert.id,
            alert.user_id,
            alert.channel,
            alert.alert_kind,
            alert.fixture_key,
            alert.fixture_id || null,
            alert.fixture_label,
            alert.league || null,
            alert.market_family || null,
            alert.publish_class || null,
            alert.reasons_json || null,
            alert.payload_json,
            alert.dedupe_key,
            alert.notification_priority || "normal",
            alert.scheduled_for,
            alert.status || "queued",
            alert.created_at,
            alert.updated_at,
          ]
        );
        touched = true;
      }
    );
    if (touched || (db && typeof db.__ogCall === "function")) {
      queued += 1;
    }
  }

  return {
    attempted: alerts.length,
    queued,
  };
}

export async function listNotificationAlertsByUser(db, userId, options = {}) {
  if (!db || !userId) {
    return [];
  }
  const limit = Math.max(1, Math.min(200, Number(options.limit || 50)));
  return callMaybeMock(
    db,
    "list_notification_alerts_by_user",
    { user_id: userId, limit },
    () =>
      all(
        db,
        `-- og:list_notification_alerts_by_user
        SELECT id, user_id, channel, alert_kind, fixture_key, fixture_id, fixture_label,
               league, market_family, publish_class, reasons_json, payload_json, dedupe_key,
               notification_priority, scheduled_for, status, delivered_at, last_error, created_at, updated_at
        FROM notification_alerts
        WHERE user_id = ?1
        ORDER BY scheduled_for ASC, created_at DESC
        LIMIT ?2`,
        [userId, limit]
      )
  );
}

export async function listDueNotificationAlerts(db, options = {}) {
  if (!db) {
    return [];
  }
  const limit = Math.max(1, Math.min(200, Number(options.limit || 25)));
  const nowIso = String(options.nowIso || isoNow());
  const userId = String(options.userId || "").trim();
  return callMaybeMock(
    db,
    "list_due_notification_alerts",
    { now_iso: nowIso, limit, user_id: userId },
    () =>
      all(
        db,
        `-- og:list_due_notification_alerts
        SELECT id, user_id, channel, alert_kind, fixture_key, fixture_id, fixture_label,
               league, market_family, publish_class, reasons_json, payload_json, dedupe_key,
               notification_priority, scheduled_for, status, delivered_at, last_error, created_at, updated_at
        FROM notification_alerts
        WHERE channel = 'telegram'
          AND status = 'queued'
          AND scheduled_for <= ?1
          AND (?2 = '' OR user_id = ?2)
        ORDER BY scheduled_for ASC, created_at ASC
        LIMIT ?3`,
        [nowIso, userId, limit]
      )
  );
}

export async function markNotificationAlertDelivered(db, alertId) {
  if (!db || !alertId) {
    return;
  }
  const now = isoNow();
  await callMaybeMock(
    db,
    "mark_notification_alert_delivered",
    { id: alertId, delivered_at: now, updated_at: now },
    () =>
      run(
        db,
        `-- og:mark_notification_alert_delivered
        UPDATE notification_alerts
        SET status = 'delivered',
            delivered_at = ?2,
            last_error = NULL,
            updated_at = ?3
        WHERE id = ?1`,
        [alertId, now, now]
      )
  );
}

export async function markNotificationAlertFailed(db, alertId, errorMessage) {
  if (!db || !alertId) {
    return;
  }
  const now = isoNow();
  await callMaybeMock(
    db,
    "mark_notification_alert_failed",
    { id: alertId, last_error: String(errorMessage || "").slice(0, 500), updated_at: now },
    () =>
      run(
        db,
        `-- og:mark_notification_alert_failed
        UPDATE notification_alerts
        SET status = 'failed',
            last_error = ?2,
            updated_at = ?3
        WHERE id = ?1`,
        [alertId, String(errorMessage || "").slice(0, 500), now]
      )
  );
}
