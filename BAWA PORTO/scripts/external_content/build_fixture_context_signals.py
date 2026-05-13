#!/usr/bin/env python3
"""Build publish-safe fixture context signals from RSS, weather, and environment notes.

This is a website-content exporter. It does not change prediction or deploy logic.
RSS items are stored as headline/link/source signals only; article bodies are not
republished.
"""

from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import html
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path("frontend/public/data")
DEMO_FIXTURE_KEY = "2026_05_10_FC_Barcelona_Real_Madrid"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def text_key(value: Any) -> str:
    text = normalize_text(value).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def parse_date(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    try:
        parsed = email.utils.parsedate_to_datetime(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc).isoformat()
    except Exception:
        return text


def fetch_text(url: str, timeout: int = 20) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "OddsGeniusFixtureContext/1.0 (+https://oddsgenius.local)",
            "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_rss_items(xml_text: str, provider: str, source_id: str, limit: int = 80) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    items = root.findall(".//item")
    if not items:
        items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
    parsed: list[dict[str, Any]] = []
    for item in items[:limit]:
        title = normalize_text(item.findtext("title") or item.findtext("{http://www.w3.org/2005/Atom}title"))
        link = normalize_text(item.findtext("link"))
        if not link:
            atom_link = item.find("{http://www.w3.org/2005/Atom}link")
            link = normalize_text(atom_link.get("href") if atom_link is not None else "")
        description = normalize_text(
            item.findtext("description") or item.findtext("summary") or item.findtext("{http://www.w3.org/2005/Atom}summary")
        )
        published = parse_date(
            item.findtext("pubDate")
            or item.findtext("published")
            or item.findtext("updated")
            or item.findtext("{http://www.w3.org/2005/Atom}updated")
        )
        if not title or not link:
            continue
        parsed.append(
            {
                "title": html.unescape(title),
                "link": link,
                "description": html.unescape(description),
                "published_at": published,
                "provider": provider,
                "source_id": source_id,
            }
        )
    return parsed


def load_fixtures(data_root: Path) -> list[dict[str, Any]]:
    payload = read_json(data_root / "fixture_intelligence_public.json", {})
    return payload.get("fixtures", []) if isinstance(payload, dict) else []


def fixture_terms(fixture: dict[str, Any]) -> set[str]:
    terms = {
        text_key(fixture.get("home_team")),
        text_key(fixture.get("away_team")),
        text_key(fixture.get("league")),
    }
    for team in (fixture.get("home_team"), fixture.get("away_team")):
        parts = [part for part in text_key(team).split() if len(part) >= 4]
        terms.update(parts)
    return {term for term in terms if term}


def match_rss_to_fixtures(
    fixtures: list[dict[str, Any]],
    items: list[dict[str, Any]],
    *,
    per_fixture_limit: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    term_map = {fixture.get("fixture_key"): fixture_terms(fixture) for fixture in fixtures if fixture.get("fixture_key")}
    for item in items:
      haystack = text_key(f"{item.get('title')} {item.get('description')}")
      for fixture_key, terms in term_map.items():
          if not fixture_key:
              continue
          score = sum(1 for term in terms if term and term in haystack)
          if score <= 0:
              continue
          signal = {
              "content_id": f"rss_{source_safe_id(item.get('provider'))}_{abs(hash((fixture_key, item.get('link'))))}",
              "type": "rss_headline_link",
              "source_id": item.get("source_id") or "publisher_rss",
              "provider": item.get("provider") or "RSS",
              "title": item.get("title") or "",
              "summary": build_news_interpretation(item, score),
              "source_url": item.get("link") or "",
              "published_at": item.get("published_at") or "",
              "usage_mode": "rss_headline_link",
              "rights_note": "Headline/link/source signal only. Full article remains with the publisher.",
              "priority": max(1, 10 - score),
              "relevance_score": score,
          }
          out.setdefault(str(fixture_key), []).append(signal)
    for fixture_key, signals in out.items():
        signals.sort(key=lambda item: (item.get("priority", 99), item.get("published_at", "")), reverse=False)
        out[fixture_key] = signals[:per_fixture_limit]
    return out


def source_safe_id(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "rss").lower()).strip("_") or "rss"


def build_news_interpretation(item: dict[str, Any], score: int) -> str:
    if score >= 2:
        return "Source-linked team context. Review for lineup, injury, or manager-comment impact before using the fixture read."
    return "Source-linked football context. Treat as orientation unless it is confirmed by lineup or squad intelligence."


def demo_weather_signal() -> dict[str, Any]:
    return {
        "content_id": "weather_barcelona_2026_05_10",
        "type": "weather_context",
        "source_id": "weather_overlay_research",
        "provider": "Odds Genius weather overlay",
        "heading": "Weather Forecast",
        "summary": "Mild, mostly dry Barcelona conditions. Weather does not add a major drag to the fixture read.",
        "venue": "Camp Nou, Barcelona",
        "forecast_window": "2026-05-09T12:00:00Z/2026-05-10T12:00:00Z",
        "condition": "fair",
        "badge": "sun-cloud",
        "label": "Mild / Mostly Dry",
        "temperature_c": 17.0,
        "precip_mm": 0.2,
        "wind_kmh": 12.0,
        "wind_gust_kmh": 24.0,
        "cloud_cover_pct": 45,
        "severity_score": 0,
        "interpretation": [
            "No meaningful weather suppression for pace or shot quality.",
            "Low wind means crossing and long passing are not materially downgraded.",
            "This remains a soft context layer and does not force the decision."
        ],
        "usage_mode": "weather_context",
        "rights_note": "Demo website context derived from the internal weather-overlay field shape. Production provider terms must be checked before bulk use.",
        "priority": 1
    }


def demo_space_weather_signal() -> dict[str, Any]:
    return {
        "content_id": "space_weather_2026_05_10",
        "type": "environmental_volatility",
        "source_id": "noaa_swpc",
        "provider": "NOAA SWPC",
        "heading": "Space Weather",
        "summary": "No material environmental volatility flag is applied. This remains an experimental monitor only.",
        "alert_level": "Monitor",
        "forecast_window": "2026-05-10T00:00:00Z/2026-05-11T00:00:00Z",
        "interpretation": [
            "No football-market adjustment is made from this layer.",
            "Tracked as experimental environmental context, not a prediction driver."
        ],
        "usage_mode": "environmental_volatility",
        "rights_note": "Public environmental context. Do not present as deterministic football causation.",
        "priority": 2
    }


def merge_fixture_payload(data_root: Path, fixture_key: str, updates: dict[str, list[dict[str, Any]]]) -> None:
    path = data_root / "external_content" / "fixture_media" / f"{fixture_key}.json"
    payload = read_json(path, {"fixture_key": fixture_key, "media": [], "news_signals": [], "weather_signals": [], "space_weather_signals": [], "sentiment_signals": []})
    payload["fixture_key"] = fixture_key
    payload["updated_at"] = dt.datetime.now(dt.timezone.utc).date().isoformat()
    for key, items in updates.items():
        existing = payload.get(key)
        if not isinstance(existing, list):
            existing = []
        by_id = {str(item.get("content_id") or index): item for index, item in enumerate(existing)}
        for item in items:
            by_id[str(item.get("content_id") or len(by_id))] = item
        payload[key] = sorted(by_id.values(), key=lambda item: int(item.get("priority") or 99))
    for key in ("media", "news_signals", "weather_signals", "space_weather_signals", "sentiment_signals"):
        payload.setdefault(key, [])
    write_json(path, payload)


def refresh_fixture_media_index(data_root: Path) -> None:
    root = data_root / "external_content" / "fixture_media"
    items = []
    for path in sorted(root.glob("*.json")):
        if path.name == "index.json":
            continue
        payload = read_json(path, {})
        fixture_key = payload.get("fixture_key") or path.stem
        media_count = len(payload.get("media") or [])
        context_count = sum(len(payload.get(key) or []) for key in ("news_signals", "weather_signals", "space_weather_signals", "sentiment_signals"))
        items.append(
            {
                "fixture_key": fixture_key,
                "media_count": media_count,
                "context_signal_count": context_count,
                "primary_type": (payload.get("media") or [{}])[0].get("type", "") if media_count else "",
                "source_id": (payload.get("media") or [{}])[0].get("source_id", "") if media_count else "",
            }
        )
    write_json(
        root / "index.json",
        {
            "updated_at": dt.datetime.now(dt.timezone.utc).date().isoformat(),
            "items": items,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--rss-url", action="append", default=[], help="RSS/Atom URL to ingest as headline/link signals.")
    parser.add_argument("--rss-provider", action="append", default=[], help="Provider label for each --rss-url.")
    parser.add_argument("--rss-source-id", action="append", default=[], help="Source id for each --rss-url.")
    parser.add_argument("--demo-barca", action="store_true", help="Seed Barcelona vs Real Madrid with weather and space-weather demo context.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixtures = load_fixtures(args.data_root)
    fixture_updates: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for index, url in enumerate(args.rss_url):
        provider = args.rss_provider[index] if index < len(args.rss_provider) else "Publisher RSS"
        source_id = args.rss_source_id[index] if index < len(args.rss_source_id) else "publisher_rss"
        items = parse_rss_items(fetch_text(url), provider, source_id)
        for fixture_key, signals in match_rss_to_fixtures(fixtures, items).items():
            fixture_updates.setdefault(fixture_key, {}).setdefault("news_signals", []).extend(signals)

    if args.demo_barca:
        fixture_updates.setdefault(DEMO_FIXTURE_KEY, {}).setdefault("weather_signals", []).append(demo_weather_signal())
        fixture_updates.setdefault(DEMO_FIXTURE_KEY, {}).setdefault("space_weather_signals", []).append(demo_space_weather_signal())

    for fixture_key, updates in fixture_updates.items():
        merge_fixture_payload(args.data_root, fixture_key, updates)
    refresh_fixture_media_index(args.data_root)

    print(json.dumps({"fixtures_updated": len(fixture_updates), "fixture_keys": sorted(fixture_updates)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
