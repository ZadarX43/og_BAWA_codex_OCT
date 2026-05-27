ALTER TABLE fpl_workspace_state
  ADD COLUMN imported_players_json TEXT NOT NULL DEFAULT '[]';
