PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS notification_alerts (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  channel TEXT NOT NULL,
  alert_kind TEXT NOT NULL,
  fixture_key TEXT NOT NULL,
  fixture_id TEXT,
  fixture_label TEXT NOT NULL,
  league TEXT,
  market_family TEXT,
  publish_class TEXT,
  reasons_json TEXT,
  payload_json TEXT NOT NULL,
  dedupe_key TEXT NOT NULL UNIQUE,
  notification_priority TEXT NOT NULL DEFAULT 'normal',
  scheduled_for TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  delivered_at TEXT,
  last_error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_notification_alerts_user_status
  ON notification_alerts(user_id, status, scheduled_for);

CREATE INDEX IF NOT EXISTS idx_notification_alerts_due
  ON notification_alerts(channel, status, scheduled_for);
