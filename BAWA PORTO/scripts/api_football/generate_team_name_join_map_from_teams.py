from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from .team_name_map import base_normalize_team_name


ROOT = Path(__file__).resolve().parents[2]
TEAMS_DIR = ROOT / "Teams"
MATCHES_MERGED_DIR = ROOT / "Matches" / "__merged__"
NORMALIZED_DIR = ROOT / "data_sources" / "api_football" / "normalized"
MANIFEST_PATHS = [
    ROOT / "configs" / "hybrid_league_manifest.json",
    ROOT / "configs" / "validated_backtest_19_manifest.json",
]
GENERATED_MAP_CSV = ROOT / "configs" / "team_name_join_map.generated.csv"
REPORTS_DIR = ROOT / "reports" / "api_football"
GENERATION_REPORT_CSV = REPORTS_DIR / "team_name_join_map_generation_report.csv"
UNRESOLVED_REPORT_CSV = REPORTS_DIR / "team_name_join_map_unresolved_from_teams.csv"

LEADING_CLUB_TOKENS = {
    "fc", "cf", "sc", "ac", "as", "cd", "ud", "rcd", "rc", "afc", "bsc", "sv",
    "vfl", "vfb", "vfr", "fk", "kf", "nk", "sk", "if", "ifk", "bk", "jk", "cs",
    "cfr", "ssc", "ss", "krc", "ks", "kks", "dac", "gd", "csu",
}
TRAILING_CLUB_TOKENS = {
    "fc", "cf", "sc", "ac", "club", "utd", "united", "city", "town", "albion",
    "athletic", "hotspur", "wanderers", "rovers", "county", "foot", "calcio",
}
ALIAS_STOPWORDS = {
    "and", "de", "da", "do", "del", "der", "di", "of", "the", "club",
}
NORMALIZED_TEXT_VARIANTS = {
    "kbenhavn": ["copenhagen", "fc copenhagen"],
    "copenhagen": ["kbenhavn", "fc kbenhavn"],
    "munchen": ["munich"],
    "munich": ["munchen"],
    "monchengladbach": ["m gladbach"],
    "m gladbach": ["monchengladbach"],
    "buducnost": ["buducnost podgorica"],
    "maritimo": ["cs maritimo"],
    "pacos": ["pacos de ferreira"],
    "pacos ferreira": ["pacos de ferreira"],
    "basaksehir": ["istanbul basaksehir"],
    "farul": ["farul constanta"],
    "constanta": ["constanta"],
    "shakhtyor": ["shakhter", "shakhter soligorsk"],
    "shakhter": ["shakhtyor", "shakhtyor soligorsk"],
    "tallinna": ["tallinn"],
    "tallinn": ["tallinna"],
    "crvena zvezda": ["red star"],
    "red star": ["crvena zvezda"],
    "fehervar": ["videoton"],
    "videoton": ["fehervar"],
}
SPECIAL_NORMALIZED_ALIASES = {
    "afc bournemouth": ["bournemouth"],
    "brighton and hove albion": ["brighton"],
    "ipswich town": ["ipswich"],
    "leicester city": ["leicester"],
    "newcastle united": ["newcastle"],
    "tottenham hotspur": ["tottenham"],
    "west ham united": ["west ham"],
    "wolverhampton wanderers": ["wolves"],
    "bayern munchen": ["bayern munich"],
    "borussia m gladbach": ["borussia monchengladbach"],
    "hertha bsc": ["hertha berlin"],
    "darmstadt 98": ["sv darmstadt 98"],
    "bochum": ["vfl bochum"],
    "clermont foot 63": ["clermont foot"],
    "troyes": ["estac troyes"],
    "gd chaves": ["chaves"],
    "cs maritimo": ["maritimo"],
    "pacos de ferreira": ["pacos ferreira"],
    "royal antwerp fc": ["antwerp"],
    "royal antwerp": ["antwerp"],
    "apoel": ["apoel nicosia"],
    "tallinna fc flora": ["flora tallinn"],
    "tallinna fc levadia": ["fc levadia tallinn", "levadia tallinn"],
    "krc genk": ["genk"],
    "olympique marseille": ["marseille"],
    "viktoria plzen": ["plzen"],
    "hjk": ["hjk helsinki"],
    "ssc farul": ["farul constanta"],
    "ssc farul constanta": ["farul constanta"],
    "shkupi": ["shkupi 1927"],
    "zalgiris": ["fk zalgiris vilnius"],
    "isloch": ["fc isloch minsk r"],
    "paksi se": ["paks"],
    "partizani tirana": ["partizani"],
    "puskas": ["puskas academy"],
    "pyunik": ["pyunik yerevan"],
    "radnicki kragujevac": ["radnicki 1923"],
    "sheriff": ["sheriff tiraspol"],
    "cs u craiova": ["universitatea craiova"],
    "universitatea cluj": ["universitatea cluj"],
    "dac 1904 dunajska streda": ["dunajska streda"],
    "banants": ["urartu", "fc urartu"],
    "artsakh": ["noah", "fc noah"],
    "red star belgrade": ["crvena zvezda", "fk crvena zvezda"],
    "kbenhavn": ["fc copenhagen", "copenhagen"],
    "fc banants": ["urartu", "fc urartu"],
    "fc artsakh": ["noah", "fc noah"],
    "kryvbas kr": ["hirnyk"],
    "pilkington xxx": ["pilkington"],
    "carlisle": ["carlisle united"],
    "southend": ["southend united"],
    "peterborough": ["peterborough united"],
    "lincoln": ["lincoln city"],
    "bradford": ["bradford city"],
    "plymouth": ["plymouth argyle"],
    "coventry": ["coventry city"],
    "birmingham": ["birmingham city"],
}


def _slugify(name: str) -> str:
    s = str(name).strip().lower()
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch)
        else:
            keep.append(" ")
    return "_".join("".join(keep).split())


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        raw = str(value or "").strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        out.append(raw)
    return out


def _alias_variants_from_tokens(tokens: list[str]) -> list[str]:
    if not tokens:
        return []

    out: list[str] = []

    def add_variant(parts: list[str]) -> None:
        phrase = " ".join([part for part in parts if part]).strip()
        if phrase:
            out.append(phrase)

    add_variant(tokens)

    while tokens and tokens[0] in LEADING_CLUB_TOKENS:
        tokens = tokens[1:]
    while tokens and tokens[-1] in TRAILING_CLUB_TOKENS:
        tokens = tokens[:-1]
    tokens = [token for token in tokens if token not in ALIAS_STOPWORDS]
    if not tokens:
        return out

    add_variant(tokens)
    if len(tokens[0]) >= 3:
        add_variant(tokens[:1])
    if len(tokens[-1]) >= 3:
        add_variant(tokens[-1:])
    if len(tokens) >= 2:
        add_variant(tokens[:2])
        add_variant(tokens[-2:])
        add_variant([tokens[-1], tokens[0]])
    if len(tokens) >= 3:
        add_variant([tokens[0], tokens[-1]])

    acronym = "".join(token[0] for token in tokens if token)
    if len(acronym) >= 3:
        out.append(acronym)

    return out


def _expand_normalized_aliases(norm: str) -> list[str]:
    if not norm:
        return []

    out: list[str] = [norm]
    out.extend(_alias_variants_from_tokens(norm.split()))

    for source, replacements in NORMALIZED_TEXT_VARIANTS.items():
        current = list(out)
        for value in current:
            if source in value:
                for replacement in replacements:
                    out.append(value.replace(source, replacement))

    current = list(out)
    for value in current:
        out.extend(SPECIAL_NORMALIZED_ALIASES.get(value, []))

    return _dedupe_keep_order(out)


def _build_name_forms(name: str) -> list[str]:
    raw = str(name or "").strip()
    if not raw:
        return []

    aliases = {
        raw,
        raw.lower(),
        _slugify(raw),
    }

    cleaned = raw
    replacements = [
        ("Football Club", "FC"),
        ("Fútbol", ""),
        ("de Fútbol", ""),
        ("Club Atlético de", ""),
        ("Club Atlético", ""),
        ("Club de Fútbol", ""),
        ("Club de Futbol", ""),
        ("Club", ""),
        ("CF", ""),
        ("FC", ""),
        ("UD", ""),
        ("CD", ""),
        ("RCD", ""),
        ("Real Club Deportivo", ""),
        ("Real Club", ""),
        ("Balompié", ""),
        ("Balompie", ""),
        ("Atlético", "Atletico"),
        ("Deportivo", ""),
    ]
    for old, new in replacements:
        cleaned = cleaned.replace(old, new)
    cleaned = " ".join(cleaned.split())
    if cleaned:
        aliases.add(cleaned)
        aliases.add(cleaned.lower())
        aliases.add(_slugify(cleaned))

    normalized_aliases: list[str] = []
    for alias in aliases:
        normalized_aliases.extend(_expand_normalized_aliases(base_normalize_team_name(alias)))

    aliases.update(normalized_aliases)
    return [a for a in _dedupe_keep_order(aliases) if str(a).strip()]


def _load_manifest_map() -> dict[str, str]:
    tag_to_league: dict[str, str] = {}
    for path in MANIFEST_PATHS:
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        for row in rows:
            tag = str(row.get("tag") or "").strip()
            league = str(row.get("league") or "").strip()
            if tag and league:
                tag_to_league.setdefault(tag, league)
    return tag_to_league


def _iter_fixture_files(tag: str) -> Iterable[Path]:
    yield from sorted(NORMALIZED_DIR.glob(f"fixtures_master__{tag}__*.csv"))


def _read_team_rows(team_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(team_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in {"team_name", "common_name", "season", "country"})
        except ValueError:
            continue
        if "team_name" not in df.columns:
            continue
        df["__source_file"] = csv_path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["team_name", "common_name", "season", "country", "__source_file"])
    out = pd.concat(frames, ignore_index=True)
    out["team_name"] = out["team_name"].astype(str).str.strip()
    out["common_name"] = out.get("common_name", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    out = out[out["team_name"].astype(bool)].copy()
    return out.drop_duplicates(subset=["team_name", "common_name"], keep="first").reset_index(drop=True)


def _read_all_team_rows() -> pd.DataFrame:
    frames = []
    for team_dir in sorted(TEAMS_DIR.iterdir()):
        if not team_dir.is_dir():
            continue
        df = _read_team_rows(team_dir)
        if df.empty:
            continue
        df["__source_league"] = team_dir.name
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["team_name", "common_name", "season", "country", "__source_file", "__source_league"])
    return pd.concat(frames, ignore_index=True)


def _read_match_names(tag: str) -> dict[str, set[str]]:
    merged_path = MATCHES_MERGED_DIR / f"{tag}__merged.csv"
    if not merged_path.exists():
        return {}
    df = pd.read_csv(merged_path, usecols=["home_team_name", "away_team_name"])
    names = pd.concat([df["home_team_name"], df["away_team_name"]], ignore_index=True).dropna().astype(str).str.strip()
    out: dict[str, set[str]] = defaultdict(set)
    for name in names:
        out[base_normalize_team_name(name)].add(name)
    return dict(out)


def _choose_display_name(values: set[str]) -> str:
    return sorted(values, key=lambda x: (len(str(x)), str(x)))[0]


def _resolve_team_canonical(team_name: str, common_name: str, match_name_map: dict[str, set[str]]) -> tuple[str | None, str]:
    candidate_scores: dict[str, int] = {}
    for source_rank, value in enumerate((team_name, common_name)):
        for alias_rank, alias in enumerate(_build_name_forms(value)):
            norm = base_normalize_team_name(alias)
            if not norm or norm not in match_name_map:
                continue
            score = (len(norm.replace(" ", "")) * 10) + (len(norm.split()) * 15) + max(0, 50 - alias_rank) + (40 if source_rank == 1 else 0)
            candidate_scores[norm] = max(candidate_scores.get(norm, 0), score)

    ranked_norms = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked_norms) == 1:
        canonical_norm = ranked_norms[0][0]
        return _choose_display_name(match_name_map[canonical_norm]), "resolved_from_match_name"
    if len(ranked_norms) > 1:
        top_norm, top_score = ranked_norms[0]
        second_score = ranked_norms[1][1]
        if top_score >= second_score + 10:
            return _choose_display_name(match_name_map[top_norm]), "resolved_from_match_name"
        return None, "ambiguous_match_name_candidates"
    return None, "no_match_name_candidate"


def _build_fs_alias_registry(tag: str, league: str) -> tuple[dict[str, str], list[dict[str, object]]]:
    match_name_map = _read_match_names(tag)
    local_team_rows = _read_team_rows(TEAMS_DIR / league)
    all_team_rows = _read_all_team_rows()

    if local_team_rows.empty:
        team_rows = all_team_rows.copy()
    else:
        other_rows = all_team_rows[all_team_rows.get("__source_league", "").astype(str) != league].copy()
        team_rows = pd.concat([local_team_rows, other_rows], ignore_index=True)

    alias_to_targets: dict[str, set[str]] = defaultdict(set)
    canonical_alias_to_targets: dict[str, set[str]] = defaultdict(set)
    team_resolution_rows: list[dict[str, object]] = []

    for _, row in team_rows.iterrows():
        team_name = str(row.get("team_name") or "").strip()
        common_name = str(row.get("common_name") or "").strip()
        fs_name, resolution = _resolve_team_canonical(team_name, common_name, match_name_map)

        team_resolution_rows.append({
            "tag": tag,
            "league": league,
            "team_name": team_name,
            "common_name": common_name,
            "resolved_fs_name": fs_name,
            "resolution_status": resolution,
            "source_league": row.get("__source_league", league),
            "source_file": row.get("__source_file"),
        })

        if not fs_name:
            continue

        fs_norm = base_normalize_team_name(fs_name)
        if fs_norm:
            alias_to_targets[fs_norm].add(fs_name)
            canonical_alias_to_targets[fs_norm].add(fs_name)

        for value in (team_name, common_name):
            for alias in _build_name_forms(value):
                norm = base_normalize_team_name(alias)
                if norm:
                    alias_to_targets[norm].add(fs_name)

    resolved_aliases = {
        alias: next(iter(targets))
        for alias, targets in alias_to_targets.items()
        if len(targets) == 1
    }
    for alias, targets in canonical_alias_to_targets.items():
        if len(targets) == 1:
            resolved_aliases[alias] = next(iter(targets))
    return resolved_aliases, team_resolution_rows


def _load_manual_api_names() -> set[tuple[str, str]]:
    manual_path = ROOT / "configs" / "team_name_join_map.csv"
    if not manual_path.exists():
        return set()
    df = pd.read_csv(manual_path)
    if df.empty:
        return set()
    approved = df[df.get("approval_status", pd.Series(dtype=object)).astype(str).str.upper().eq("APPROVED")].copy()
    out = set()
    for _, row in approved.iterrows():
        tag = str(row.get("tag", "*") or "*").strip()
        api_name = str(row.get("api_team_name") or "").strip()
        if tag and api_name:
            out.add((tag, api_name))
    return out


def _build_generated_rows_for_tag(tag: str, league: str, manual_api_names: set[tuple[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fs_alias_registry, team_resolution_rows = _build_fs_alias_registry(tag, league)
    match_name_map = _read_match_names(tag)

    generated_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()

    canonical_self_names = sorted({_choose_display_name(values) for values in match_name_map.values() if values})
    for fs_name in canonical_self_names:
        generated_rows.append({
            "tag": tag,
            "api_team_name": fs_name,
            "fs_team_name": fs_name,
            "approval_status": "APPROVED",
            "notes": f"AUTO_CANONICAL_SELF:{league}",
        })
        seen_pairs.add((tag, fs_name))

    for fixture_path in _iter_fixture_files(tag):
        season = fixture_path.stem.split("__")[-1]
        fx = pd.read_csv(fixture_path, usecols=["home_team_name", "away_team_name"])
        api_names = pd.concat([fx["home_team_name"], fx["away_team_name"]], ignore_index=True).dropna().astype(str).str.strip().unique()
        for api_name in sorted(api_names):
            if not api_name:
                continue
            if (tag, api_name) in manual_api_names or ("*", api_name) in manual_api_names:
                continue
            if (tag, api_name) in seen_pairs:
                continue
            seen_pairs.add((tag, api_name))

            candidate_scores: dict[str, int] = {}
            candidate_aliases = set()
            for rank, alias in enumerate(_build_name_forms(api_name)):
                norm = base_normalize_team_name(alias)
                if not norm:
                    continue
                candidate_aliases.add(norm)
                target = fs_alias_registry.get(norm)
                if target:
                    score = (len(norm.replace(" ", "")) * 10) + (len(norm.split()) * 15) + max(0, 50 - rank)
                    candidate_scores[target] = max(candidate_scores.get(target, 0), score)

            ranked_targets = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
            top_target = ranked_targets[0][0] if ranked_targets else None
            top_score = ranked_targets[0][1] if ranked_targets else -1
            second_score = ranked_targets[1][1] if len(ranked_targets) > 1 else -1

            if top_target and (len(ranked_targets) == 1 or top_score >= second_score + 10):
                fs_name = top_target
                generated_rows.append({
                    "tag": tag,
                    "api_team_name": api_name,
                    "fs_team_name": fs_name,
                    "approval_status": "APPROVED",
                    "notes": f"AUTO_FROM_TEAMS:{league}:{season}",
                })
            else:
                unresolved_rows.append({
                    "tag": tag,
                    "league": league,
                    "season": season,
                    "api_team_name": api_name,
                    "candidate_target_count": len(candidate_scores),
                    "candidate_fs_names": " | ".join(name for name, _ in ranked_targets),
                    "api_alias_norms": " | ".join(sorted(candidate_aliases)),
                    "resolution_status": "ambiguous_api_alias_match" if candidate_scores else "no_api_alias_match",
                })

    unresolved_rows.extend(team_resolution_rows)
    return generated_rows, unresolved_rows


def generate_join_map(tags: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    tag_to_league = _load_manifest_map()
    if tags:
        tag_to_league = {tag: tag_to_league[tag] for tag in tags if tag in tag_to_league}

    manual_api_names = _load_manual_api_names()
    generated_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []

    for tag, league in sorted(tag_to_league.items()):
        team_dir = TEAMS_DIR / league
        if not team_dir.exists():
            unresolved_rows.append({
                "tag": tag,
                "league": league,
                "season": None,
                "api_team_name": None,
                "candidate_target_count": None,
                "candidate_fs_names": None,
                "api_alias_norms": None,
                "resolution_status": "missing_team_directory",
            })
            continue
        rows, unresolved = _build_generated_rows_for_tag(tag, league, manual_api_names)
        generated_rows.extend(rows)
        unresolved_rows.extend(unresolved)

    generated_df = pd.DataFrame(generated_rows).drop_duplicates(subset=["tag", "api_team_name"], keep="first")
    if not generated_df.empty:
        generated_df = generated_df.sort_values(["tag", "api_team_name", "fs_team_name"]).reset_index(drop=True)

    unresolved_df = pd.DataFrame(unresolved_rows)
    if not unresolved_df.empty:
        unresolved_df = unresolved_df.sort_values(["tag", "resolution_status", "api_team_name"], na_position="last").reset_index(drop=True)

    return generated_df, unresolved_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate team_name_join_map entries from /Teams and match-side canonical names.")
    parser.add_argument("--tags", default="", help="Comma-separated tag subset to generate.")
    parser.add_argument("--output-csv", default=str(GENERATED_MAP_CSV))
    parser.add_argument("--report-csv", default=str(GENERATION_REPORT_CSV))
    parser.add_argument("--unresolved-csv", default=str(UNRESOLVED_REPORT_CSV))
    args = parser.parse_args()

    tags = [part.strip() for part in str(args.tags or "").split(",") if part.strip()]
    generated_df, unresolved_df = generate_join_map(tags=tags or None)

    output_csv = Path(args.output_csv)
    report_csv = Path(args.report_csv)
    unresolved_csv = Path(args.unresolved_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    unresolved_csv.parent.mkdir(parents=True, exist_ok=True)

    generated_df.to_csv(output_csv, index=False)
    unresolved_df.to_csv(unresolved_csv, index=False)

    summary_rows = []
    if not unresolved_df.empty:
        status_counts = (
            unresolved_df.groupby(["tag", "resolution_status"], dropna=False)
            .size()
            .reset_index(name="row_count")
        )
        summary_rows.extend(status_counts.to_dict("records"))
    report_df = pd.DataFrame(summary_rows)
    report_df.to_csv(report_csv, index=False)

    print(f"WROTE: {output_csv} rows={len(generated_df)}")
    print(f"WROTE: {report_csv} rows={len(report_df)}")
    print(f"WROTE: {unresolved_csv} rows={len(unresolved_df)}")
    if not generated_df.empty:
        by_tag = generated_df.groupby("tag").size().to_dict()
        print(f"generated_by_tag={by_tag}")


if __name__ == "__main__":
    main()
