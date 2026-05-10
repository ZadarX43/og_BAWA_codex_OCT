PRAGMA foreign_keys = ON;

ALTER TABLE account_risk_state
  ADD COLUMN last_review_outcome TEXT;

ALTER TABLE account_risk_state
  ADD COLUMN last_review_outcome_note TEXT;

ALTER TABLE account_risk_state
  ADD COLUMN last_review_outcome_at TEXT;

ALTER TABLE account_risk_state
  ADD COLUMN last_review_outcome_by TEXT;
