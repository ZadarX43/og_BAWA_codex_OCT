PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS account_risk_state (
  user_id TEXT PRIMARY KEY,
  account_status TEXT NOT NULL DEFAULT 'active',
  risk_level TEXT NOT NULL DEFAULT 'low',
  risk_score INTEGER NOT NULL DEFAULT 0,
  review_status TEXT NOT NULL DEFAULT 'clear',
  last_risk_event_at TEXT,
  last_reviewed_at TEXT,
  last_reviewed_by TEXT,
  suspended_at TEXT,
  suspension_reason TEXT,
  reinstated_at TEXT,
  reinstatement_reason TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS account_risk_flags (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  flag_type TEXT NOT NULL,
  severity TEXT NOT NULL,
  flag_status TEXT NOT NULL DEFAULT 'open',
  source TEXT NOT NULL,
  summary TEXT NOT NULL,
  evidence_json TEXT,
  opened_at TEXT NOT NULL,
  resolved_at TEXT,
  resolved_by TEXT,
  resolution_note TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_account_risk_flags_user_id
  ON account_risk_flags(user_id);

CREATE INDEX IF NOT EXISTS idx_account_risk_flags_user_status
  ON account_risk_flags(user_id, flag_status);

CREATE INDEX IF NOT EXISTS idx_account_risk_flags_type
  ON account_risk_flags(flag_type);

CREATE TABLE IF NOT EXISTS account_admin_notes (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  note_type TEXT NOT NULL,
  visibility TEXT NOT NULL DEFAULT 'internal',
  content TEXT NOT NULL,
  author_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_account_admin_notes_user_id
  ON account_admin_notes(user_id);
