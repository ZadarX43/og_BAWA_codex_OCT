CREATE TABLE IF NOT EXISTS fpl_workspace_state (
  user_id TEXT PRIMARY KEY,
  ruleset_name TEXT NOT NULL DEFAULT 'official_fpl_2026_27',
  season TEXT NOT NULL DEFAULT '2026/27',
  gameweek INTEGER,
  manager_id TEXT,
  strategy_mode TEXT NOT NULL DEFAULT 'balanced',
  bank_tenths INTEGER NOT NULL DEFAULT 0,
  free_transfers INTEGER NOT NULL DEFAULT 1,
  chip_intent TEXT NOT NULL DEFAULT 'NONE',
  squad_json TEXT NOT NULL DEFAULT '[]',
  saved_plans_json TEXT NOT NULL DEFAULT '[]',
  saved_drafts_json TEXT NOT NULL DEFAULT '[]',
  transfer_history_json TEXT NOT NULL DEFAULT '[]',
  watchlist_json TEXT NOT NULL DEFAULT '[]',
  bench_shortlist_json TEXT NOT NULL DEFAULT '[]',
  locked_targets_json TEXT NOT NULL DEFAULT '[]',
  ignored_json TEXT NOT NULL DEFAULT '[]',
  updated_at TEXT NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_fpl_workspace_state_updated
  ON fpl_workspace_state(updated_at);
