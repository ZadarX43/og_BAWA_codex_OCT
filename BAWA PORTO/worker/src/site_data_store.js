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
  const [teamStats, playerStats, lineupSlots] = await Promise.all([
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
  ]);

  return {
    team_stats: teamStats.map((row) => parsePayload(row)).filter(Boolean),
    player_stats: playerStats.map((row) => parsePayload(row)).filter(Boolean),
    lineup_slots: lineupSlots.map((row) => parsePayload(row)).filter(Boolean),
  };
};

export const getTeamPremiumData = async (db, competitionKey, teamSlug, { limit = 20 } = {}) => {
  const safeLimit = Math.max(1, Math.min(Number(limit) || 20, 80));
  const [players, teamStats, lineupSlots] = await Promise.all([
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
  ]);

  return {
    players: players.map((row) => parsePayload(row)).filter(Boolean),
    recent_team_stats: teamStats.map((row) => parsePayload(row)).filter(Boolean),
    recent_lineup_slots: lineupSlots.map((row) => parsePayload(row)).filter(Boolean),
  };
};

export const getCurrentFixtures = async (db, { leagueKey = "", limit = 80 } = {}) => {
  const safeLimit = Math.max(1, Math.min(Number(limit) || 80, 200));
  const rows = leagueKey
    ? await all(
        db
        .prepare(
          `
            SELECT payload_json
            FROM fixtures
            WHERE league_key = ?
            ORDER BY kickoff_time, home_team
            LIMIT ?
            `
          )
          .bind(leagueKey, safeLimit)
      )
    : await all(
        db
        .prepare(
          `
            SELECT payload_json
            FROM fixtures
            ORDER BY kickoff_time, league, home_team
            LIMIT ?
            `
          )
          .bind(safeLimit)
      );

  return rows.map((row) => parsePayload(row)).filter(Boolean);
};
