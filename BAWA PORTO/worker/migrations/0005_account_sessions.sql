PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS account_sessions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  session_token_hash TEXT NOT NULL UNIQUE,
  device_label TEXT,
  user_agent_hash TEXT,
  ip_hash TEXT,
  session_kind TEXT NOT NULL DEFAULT 'browser',
  is_primary INTEGER NOT NULL DEFAULT 0,
  is_revoked INTEGER NOT NULL DEFAULT 0,
  issued_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  revoked_at TEXT,
  revoke_reason TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_account_sessions_user_id
  ON account_sessions(user_id);

CREATE INDEX IF NOT EXISTS idx_account_sessions_user_revoked
  ON account_sessions(user_id, is_revoked);

CREATE INDEX IF NOT EXISTS idx_account_sessions_user_primary
  ON account_sessions(user_id, is_primary);

CREATE INDEX IF NOT EXISTS idx_account_sessions_expires_at
  ON account_sessions(expires_at);
