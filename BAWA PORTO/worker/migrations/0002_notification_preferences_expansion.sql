PRAGMA foreign_keys = ON;

ALTER TABLE notification_preferences ADD COLUMN standard_alerts_enabled INTEGER NOT NULL DEFAULT 1;
ALTER TABLE notification_preferences ADD COLUMN correct_score_alerts_enabled INTEGER NOT NULL DEFAULT 0;
ALTER TABLE notification_preferences ADD COLUMN injury_alerts_enabled INTEGER NOT NULL DEFAULT 1;
ALTER TABLE notification_preferences ADD COLUMN weather_alerts_enabled INTEGER NOT NULL DEFAULT 1;
ALTER TABLE notification_preferences ADD COLUMN market_movement_alerts_enabled INTEGER NOT NULL DEFAULT 1;
ALTER TABLE notification_preferences ADD COLUMN volatility_alerts_enabled INTEGER NOT NULL DEFAULT 1;
ALTER TABLE notification_preferences ADD COLUMN team_news_alerts_enabled INTEGER NOT NULL DEFAULT 1;
ALTER TABLE notification_preferences ADD COLUMN daily_digest_enabled INTEGER NOT NULL DEFAULT 1;
ALTER TABLE notification_preferences ADD COLUMN weekend_slate_digest_enabled INTEGER NOT NULL DEFAULT 1;
ALTER TABLE notification_preferences ADD COLUMN website_only_mode INTEGER NOT NULL DEFAULT 0;
ALTER TABLE notification_preferences ADD COLUMN allow_non_signal_intelligence INTEGER NOT NULL DEFAULT 1;
ALTER TABLE notification_preferences ADD COLUMN alert_frequency_mode TEXT NOT NULL DEFAULT 'mixed';
ALTER TABLE notification_preferences ADD COLUMN pre_match_window_minutes INTEGER NOT NULL DEFAULT 90;
ALTER TABLE notification_preferences ADD COLUMN favourite_teams_json TEXT;
ALTER TABLE notification_preferences ADD COLUMN followed_fixtures_json TEXT;
ALTER TABLE notification_preferences ADD COLUMN quiet_hours_json TEXT;
