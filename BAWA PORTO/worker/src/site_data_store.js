const parsePayload = (row, field = "payload_json") => {
  if (!row || !row[field]) {
    return null;
  }
  return JSON.parse(row[field]);
};

const first = async (statement) => statement.first();

const all = async (statement) => {
  const result = await statement.all();
  return Array.isArray(result?.results) ? result.results : [];
};

const DEFAULT_SITE_PAYLOAD_PREFIX = "site-data/v1";

const emptyFixtureStatsPayload = () => ({
  team_stats: [],
  player_stats: [],
  match_events: [],
  lineup_slots: [],
  market_intelligence: [],
  player_event_shortlists: [],
});

const emptyFixtureContextPayload = () => ({
  media: [],
  news_signals: [],
  weather_signals: [],
  space_weather_signals: [],
  sentiment_signals: [],
});

const sitePayloadObjectKey = (relativePath, prefix = DEFAULT_SITE_PAYLOAD_PREFIX) =>
  `${String(prefix || DEFAULT_SITE_PAYLOAD_PREFIX).replace(/^\/+|\/+$/g, "")}/${relativePath.replace(/^\/+/, "")}`;

const readJsonObject = async (bucket, objectKey) => {
  if (!bucket?.get) {
    return null;
  }
  const object = await bucket.get(objectKey);
  if (!object) {
    return null;
  }
  const text = await object.text();
  return JSON.parse(text);
};

export const getFixturePublishPayload = async (bucket, fixtureKey, { prefix = DEFAULT_SITE_PAYLOAD_PREFIX } = {}) => {
  const safeFixtureKey = String(fixtureKey || "").trim();
  if (!safeFixtureKey) {
    return null;
  }
  const object_key = sitePayloadObjectKey(`payloads/fixtures/${safeFixtureKey}.json`, prefix);
  const payload = await readJsonObject(bucket, object_key);
  return payload ? { payload, object_key } : null;
};

export const getTeamPublishPayload = async (
  bucket,
  competitionKey,
  teamSlug,
  { prefix = DEFAULT_SITE_PAYLOAD_PREFIX } = {}
) => {
  const safeCompetitionKey = String(competitionKey || "").trim();
  const safeTeamSlug = String(teamSlug || "").trim();
  if (!safeCompetitionKey || !safeTeamSlug) {
    return null;
  }
  const object_key = sitePayloadObjectKey(`payloads/teams/${safeCompetitionKey}/${safeTeamSlug}.json`, prefix);
  const payload = await readJsonObject(bucket, object_key);
  return payload ? { payload, object_key } : null;
};

export const fixtureDetailFromPublishPayload = (payload) => ({
  fixture: payload?.fixture || null,
  decision: payload?.decision || null,
  lineup: payload?.lineup || null,
  h2h: payload?.h2h || null,
  fixture_brain: payload?.fixture_brain || null,
});

export const fixtureStatsFromPublishPayload = (payload) => {
  const stats = payload?.stats && typeof payload.stats === "object" ? payload.stats : {};
  return {
    ...emptyFixtureStatsPayload(),
    ...stats,
  };
};

export const fixtureContextFromPublishPayload = (payload) => {
  const context = payload?.context && typeof payload.context === "object" ? payload.context : {};
  return {
    ...emptyFixtureContextPayload(),
    ...context,
  };
};

export const teamDetailFromPublishPayload = (payload) => ({
  team: payload?.team || null,
  squad: payload?.squad || null,
  lineup_snapshot: payload?.lineup_snapshot || null,
});

export const teamPremiumFromPublishPayload = (payload) => {
  const premium = payload?.premium && typeof payload.premium === "object" ? payload.premium : {};
  return {
    players: premium.players || [],
    recent_team_stats: premium.recent_team_stats || [],
    recent_lineup_slots: premium.recent_lineup_slots || [],
    player_event_shortlists: premium.player_event_shortlists || [],
  };
};

const normalizeTeamKey = (value) =>
  String(value || "")
    .trim()
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");

const trimTeamSuffixes = (value) =>
  normalizeTeamKey(value)
    .replace(/_(fc|sc|cf|afc|kv|kvc|va)$/g, "")
    .replace(/^(fc|sc|cf|afc)_/g, "");

const sameTeamKey = (left, right) => {
  const a = trimTeamSuffixes(left);
  const b = trimTeamSuffixes(right);
  return Boolean(a && b && (a === b || a.startsWith(`${b}_`) || b.startsWith(`${a}_`)));
};

export const getFixtureDetail = async (db, fixtureKey) => {
  const [fixture, decision, lineup, h2h] = await Promise.all([
    first(db.prepare("SELECT payload_json FROM fixtures WHERE fixture_key = ?").bind(fixtureKey)),
    first(db.prepare("SELECT payload_json FROM fixture_decisions WHERE fixture_key = ?").bind(fixtureKey)),
    first(db.prepare("SELECT payload_json FROM fixture_lineups WHERE fixture_key = ?").bind(fixtureKey)),
    first(db.prepare("SELECT payload_json FROM fixture_h2h WHERE fixture_key = ?").bind(fixtureKey)),
  ]);

  return {
    fixture: parsePayload(fixture),
    decision: parsePayload(decision),
    lineup: parsePayload(lineup),
    h2h: parsePayload(h2h),
  };
};

export const getTeamDetail = async (db, competitionKey, teamSlug) => {
  const [team, squad] = await Promise.all([
    first(
      db
        .prepare(
          `
          SELECT payload_json
          FROM team_intelligence
          WHERE competition_key = ? AND team_slug = ?
          ORDER BY season DESC
          LIMIT 1
          `
        )
        .bind(competitionKey, teamSlug)
    ),
    first(
      db
        .prepare(
          `
          SELECT payload_json
          FROM club_squads
          WHERE competition_key = ? AND club_slug = ?
          ORDER BY season DESC
          LIMIT 1
          `
        )
        .bind(competitionKey, teamSlug)
    ),
  ]);

  let lineupSnapshot = await first(
    db
      .prepare(
        `
        SELECT team_key, team, payload_json
        FROM team_lineup_snapshots
        WHERE competition_key = ? AND team_key = ?
        LIMIT 1
        `
      )
      .bind(competitionKey, teamSlug)
  );

  if (!lineupSnapshot) {
    const teamPayload = parsePayload(team);
    const squadPayload = parsePayload(squad);
    const candidates = [teamSlug, teamPayload?.team, squadPayload?.club].filter(Boolean);
    const snapshots = await all(
      db
        .prepare(
          `
          SELECT team_key, team, payload_json
          FROM team_lineup_snapshots
          WHERE competition_key = ?
          `
        )
        .bind(competitionKey)
    );
    lineupSnapshot =
      snapshots.find((snapshot) =>
        candidates.some((candidate) => sameTeamKey(candidate, snapshot.team_key) || sameTeamKey(candidate, snapshot.team))
      ) || null;
  }

  return {
    team: parsePayload(team),
    squad: parsePayload(squad),
    lineup_snapshot: parsePayload(lineupSnapshot),
  };
};

export const getFixtureStats = async (db, fixtureKey) => {
  const cached = await first(db.prepare("SELECT payload_json FROM site_fixture_stats_payloads WHERE fixture_key = ?").bind(fixtureKey));
  const cachedPayload = parsePayload(cached);
  if (cachedPayload) {
    return cachedPayload;
  }

  const [teamStats, matchEvents, playerStats, lineupSlots, marketIntelligence, playerEventShortlists] = await Promise.all([
    all(
      db
        .prepare(
          `
          SELECT payload_json
          FROM site_team_match_stats
          WHERE fixture_key = ?
          ORDER BY is_home DESC, team_name
          `
        )
        .bind(fixtureKey)
    ),
    all(
      db
        .prepare(
          `
          SELECT payload_json
          FROM site_match_events
          WHERE fixture_key = ?
          ORDER BY minute, extra_minute, event_id
          `
        )
        .bind(fixtureKey)
    ),
    all(
      db
        .prepare(
          `
          SELECT payload_json
          FROM site_player_match_stats
          WHERE fixture_key = ?
          ORDER BY is_home DESC, started_flag DESC, minutes DESC, rating DESC, player_key
          `
        )
        .bind(fixtureKey)
    ),
    all(
      db
        .prepare(
          `
          SELECT payload_json
          FROM site_lineup_slots
          WHERE fixture_key = ?
          ORDER BY is_home DESC, is_starting_xi DESC, broad_position, slot_code, player_name
          `
        )
        .bind(fixtureKey)
    ),
    all(
      db
        .prepare(
          `
          SELECT payload_json
          FROM site_fixture_market_intelligence
          WHERE fixture_key = ?
          ORDER BY
            CASE rank_role
              WHEN 'best' THEN 1
              WHEN 'secondary' THEN 2
              WHEN 'weak' THEN 3
              WHEN 'avoid' THEN 4
              ELSE 5
            END,
            alignment_score DESC,
            rating DESC,
            market_key
          `
        )
        .bind(fixtureKey)
    ),
    all(
      db
        .prepare(
          `
          SELECT payload_json
          FROM site_player_event_shortlists
          WHERE fixture_key = ?
          ORDER BY event_family, event_key, shortlist_rank, team_name, player_name
          `
        )
        .bind(fixtureKey)
    ),
  ]);

  return {
    team_stats: teamStats.map((row) => parsePayload(row)).filter(Boolean),
    player_stats: playerStats.map((row) => parsePayload(row)).filter(Boolean),
    match_events: matchEvents.map((row) => parsePayload(row)).filter(Boolean),
    lineup_slots: lineupSlots.map((row) => parsePayload(row)).filter(Boolean),
    market_intelligence: marketIntelligence.map((row) => parsePayload(row)).filter(Boolean),
    player_event_shortlists: playerEventShortlists.map((row) => parsePayload(row)).filter(Boolean),
  };
};

export const getFixtureContext = async (db, fixtureKey) => {
  const cached = await first(db.prepare("SELECT payload_json FROM site_fixture_context_payloads WHERE fixture_key = ?").bind(fixtureKey));
  const cachedPayload = parsePayload(cached);
  if (cachedPayload) {
    return cachedPayload;
  }

  const rows = await all(
    db
      .prepare(
        `
        SELECT payload_json
        FROM site_fixture_external_content
        WHERE fixture_key = ?
        ORDER BY priority, row_id
        `
      )
      .bind(fixtureKey)
  );
  const items = rows.map((row) => parsePayload(row)).filter(Boolean);
  return {
    media: items.filter((item) => item.type === "youtube_embed"),
    news_signals: items.filter((item) => item.type === "rss_headline_link" || item.type === "news_signal"),
    weather_signals: items.filter((item) => item.type === "weather_context" || item.type === "weather_signal"),
    space_weather_signals: items.filter((item) => item.type === "environmental_volatility"),
    sentiment_signals: items.filter((item) => item.type === "sentiment_signal"),
  };
};

export const getTeamPremiumData = async (db, competitionKey, teamSlug, { limit = 20 } = {}) => {
  const safeLimit = Math.max(1, Math.min(Number(limit) || 20, 80));
  if (safeLimit === 20) {
    const cached = await first(
      db
        .prepare(
          `
          SELECT payload_json
          FROM site_team_premium_payloads
          WHERE competition_key = ? AND team_slug = ?
          LIMIT 1
          `
        )
        .bind(competitionKey, teamSlug)
    );
    const cachedPayload = parsePayload(cached);
    if (cachedPayload) {
      return cachedPayload;
    }
  }

  const [players, teamStats, lineupSlots, playerEventShortlists] = await Promise.all([
    all(
      db
        .prepare(
          `
          SELECT payload_json
          FROM site_player_identity_map
          WHERE competition_key = ? AND club_slug = ?
          ORDER BY rating_power DESC, rank_club ASC, name
          LIMIT ?
          `
        )
        .bind(competitionKey, teamSlug, safeLimit)
    ),
    all(
      db
        .prepare(
          `
          SELECT payload_json
          FROM site_team_match_stats
          WHERE team_slug = ?
          ORDER BY fixture_key DESC
          LIMIT ?
          `
        )
        .bind(teamSlug, safeLimit)
    ),
    all(
      db
        .prepare(
          `
          SELECT payload_json
          FROM site_lineup_slots
          WHERE team_slug = ?
          ORDER BY fixture_key DESC, is_starting_xi DESC, broad_position, slot_code, player_name
          LIMIT ?
          `
        )
        .bind(teamSlug, safeLimit * 2)
    ),
    all(
      db
        .prepare(
          `
          SELECT payload_json
          FROM site_player_event_shortlists
          WHERE team_slug = ?
          ORDER BY fixture_key DESC, event_family, shortlist_rank
          LIMIT ?
          `
        )
        .bind(teamSlug, safeLimit)
    ),
  ]);

  return {
    players: players.map((row) => parsePayload(row)).filter(Boolean),
    recent_team_stats: teamStats.map((row) => parsePayload(row)).filter(Boolean),
    recent_lineup_slots: lineupSlots.map((row) => parsePayload(row)).filter(Boolean),
    player_event_shortlists: playerEventShortlists.map((row) => parsePayload(row)).filter(Boolean),
  };
};

export const getCurrentFixtures = async (db, { leagueKey = "", limit = 80, includePast = false, fromIso = "" } = {}) => {
  const safeLimit = Math.max(1, Math.min(Number(limit) || 80, 200));
  const params = [];
  const activeClause = includePast ? "" : "pre_kickoff_eligible = 1";
  const fromClause = !includePast && fromIso ? "COALESCE(fixture_kickoff_at, kickoff_time) >= ?" : "";
  if (fromClause) {
    params.push(fromIso);
  }
  const leagueClause = leagueKey ? "league_key = ?" : "";
  if (leagueClause) {
    params.push(leagueKey);
  }
  const whereClause = [activeClause, fromClause, leagueClause].filter(Boolean).join(" AND ");
  const sql = `
            SELECT payload_json
            FROM fixtures
            ${whereClause ? `WHERE ${whereClause}` : ""}
            ORDER BY kickoff_time, league, home_team
            LIMIT ?
            `;
  const statement = db.prepare(sql);
  const rows = await all(statement.bind(...params, safeLimit));

  return rows.map((row) => parsePayload(row)).filter(Boolean);
};
