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
