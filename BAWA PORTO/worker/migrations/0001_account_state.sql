PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  email TEXT NOT NULL,
  email_normalized TEXT NOT NULL UNIQUE,
  email_verified_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  account_status TEXT NOT NULL CHECK (account_status IN ('active', 'disabled', 'pending'))
);

CREATE TABLE IF NOT EXISTS subscriptions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  stripe_customer_id TEXT NOT NULL UNIQUE,
  stripe_subscription_id TEXT NOT NULL UNIQUE,
  subscription_status TEXT NOT NULL,
  price_id TEXT,
  current_period_end TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS telegram_links (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  telegram_user_id TEXT NOT NULL UNIQUE,
  telegram_username TEXT,
  telegram_chat_id TEXT,
  link_status TEXT NOT NULL CHECK (link_status IN ('pending', 'linked', 'revoked')),
  linked_at TEXT,
  revoked_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS notification_preferences (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL UNIQUE,
  email_enabled INTEGER NOT NULL DEFAULT 1,
  telegram_enabled INTEGER NOT NULL DEFAULT 0,
  elite_alerts_enabled INTEGER NOT NULL DEFAULT 1,
  acca_alerts_enabled INTEGER NOT NULL DEFAULT 0,
  results_digest_enabled INTEGER NOT NULL DEFAULT 1,
  favourite_markets_json TEXT,
  favourite_leagues_json TEXT,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS auth_events (
  id TEXT PRIMARY KEY,
  user_id TEXT,
  email_normalized TEXT,
  event_type TEXT NOT NULL,
  ip_hint TEXT,
  user_agent_hint TEXT,
  created_at TEXT NOT NULL,
  metadata_json TEXT,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_users_email_normalized
  ON users(email_normalized);

CREATE INDEX IF NOT EXISTS idx_subscriptions_user_id
  ON subscriptions(user_id);

CREATE INDEX IF NOT EXISTS idx_subscriptions_customer_id
  ON subscriptions(stripe_customer_id);

CREATE INDEX IF NOT EXISTS idx_subscriptions_subscription_id
  ON subscriptions(stripe_subscription_id);

CREATE INDEX IF NOT EXISTS idx_telegram_links_user_id
  ON telegram_links(user_id);

CREATE INDEX IF NOT EXISTS idx_telegram_links_status
  ON telegram_links(link_status);

CREATE INDEX IF NOT EXISTS idx_auth_events_user_id
  ON auth_events(user_id);

CREATE INDEX IF NOT EXISTS idx_auth_events_email_normalized
  ON auth_events(email_normalized);

CREATE INDEX IF NOT EXISTS idx_auth_events_created_at
  ON auth_events(created_at);
