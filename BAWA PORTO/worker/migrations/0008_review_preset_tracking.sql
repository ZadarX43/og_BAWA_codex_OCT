PRAGMA foreign_keys = ON;

ALTER TABLE account_risk_state
  ADD COLUMN last_review_preset TEXT;
