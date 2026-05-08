const isoNow = () => new Date().toISOString();
const normalizeEmail = (value) => String(value || "").trim().toLowerCase();
const toJson = (value) => JSON.stringify(value ?? null);
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
  const now = isoNow();
  await callMaybeMock(
    db,
    "ensure_notification_preferences",
    {
      id: buildId("pref"),
      user_id: userId,
      email_enabled: 1,
      telegram_enabled: 0,
      elite_alerts_enabled: 1,
      acca_alerts_enabled: 0,
      results_digest_enabled: 1,
      favourite_markets_json: toJson([]),
      favourite_leagues_json: toJson([]),
      updated_at: now,
    },
    () =>
      run(
        db,
        `-- og:ensure_notification_preferences
        INSERT INTO notification_preferences (
          id, user_id, email_enabled, telegram_enabled, elite_alerts_enabled, acca_alerts_enabled, results_digest_enabled,
          favourite_markets_json, favourite_leagues_json, updated_at
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
        ON CONFLICT(user_id) DO NOTHING`,
        [buildId("pref"), userId, 1, 0, 1, 0, 1, toJson([]), toJson([]), now]
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
        SELECT id, user_id, email_enabled, telegram_enabled, elite_alerts_enabled, acca_alerts_enabled, results_digest_enabled,
               favourite_markets_json, favourite_leagues_json, updated_at
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
          favourite_markets: fromJson(notificationPreferences.favourite_markets_json, []),
          favourite_leagues: fromJson(notificationPreferences.favourite_leagues_json, []),
        }
      : null,
  };
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
