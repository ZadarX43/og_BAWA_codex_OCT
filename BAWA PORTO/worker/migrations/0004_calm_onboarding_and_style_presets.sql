ALTER TABLE notification_preferences ADD COLUMN user_style_preset TEXT NOT NULL DEFAULT 'disciplined_bettor';
ALTER TABLE notification_preferences ADD COLUMN decision_companion_enabled INTEGER NOT NULL DEFAULT 1;
ALTER TABLE notification_preferences ADD COLUMN reset_mode_enabled INTEGER NOT NULL DEFAULT 1;
ALTER TABLE notification_preferences ADD COLUMN calm_onboarding_completed_at TEXT;
ALTER TABLE notification_preferences ADD COLUMN language_preference TEXT NOT NULL DEFAULT 'en-GB';
