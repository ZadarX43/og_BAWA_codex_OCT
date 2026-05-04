#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
ASSETS = FRONTEND / "assets"
DATA = FRONTEND / "public" / "data"

HTML_FILES = [
    FRONTEND / "index.html",
    FRONTEND / "predictions.html",
    FRONTEND / "premium.html",
    FRONTEND / "results.html",
    FRONTEND / "pricing.html",
    FRONTEND / "methodology.html",
    FRONTEND / "account.html",
]

REQUIRED_FILES = [
    ASSETS / "styles.css",
    ASSETS / "app.js",
    DATA / "public_predictions.json",
    DATA / "premium_predictions.json",
    DATA / "publish_summary.json",
]

FORBIDDEN_TERMS = [
    "threshold",
    "thr",
    "gate",
    "veto",
    "lambda",
    "p00",
    "meta",
    "support",
    "raw",
    "model_path",
    "bundle",
    "feature",
    "xg",
    "h2h",
    "streak",
    "power_diff",
    "draw_risk",
    "draw_chaos",
    "policy",
    "branch",
    "state",
    "source_path",
    "api",
    "secret",
]


def check_exists(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing file: {path.relative_to(ROOT)}")


def check_html_assets(path: Path, errors: list[str]) -> None:
    text = path.read_text(encoding="utf-8")

    hrefs = re.findall(r'href="([^"]+)"', text)
    srcs = re.findall(r'src="([^"]+)"', text)
    refs = hrefs + srcs

    for ref in refs:
        if ref.startswith("http://") or ref.startswith("https://") or ref.startswith("data:"):
            continue
        if ref.startswith("#"):
            continue
        target = (path.parent / ref).resolve()
        try:
            target.relative_to(ROOT)
        except ValueError:
            errors.append(f"{path.name}: asset reference escapes repo root: {ref}")
            continue
        if not target.exists():
            errors.append(f"{path.name}: missing referenced asset: {ref}")

    visible_text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    visible_text = re.sub(r"<style\b[^>]*>.*?</style>", " ", visible_text, flags=re.IGNORECASE | re.DOTALL)
    visible_text = re.sub(r"<[^>]+>", " ", visible_text)
    lowered = visible_text.lower()
    for term in FORBIDDEN_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            errors.append(f"{path.name}: contains forbidden private/internal term `{term}`")


def check_json(path: Path, errors: list[str]) -> None:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{path.relative_to(ROOT)}: invalid JSON: {exc}")


def main() -> int:
    errors: list[str] = []

    for path in HTML_FILES:
        check_exists(path, errors)
    for path in REQUIRED_FILES:
        check_exists(path, errors)

    if errors:
        print("Frontend static smoke failed.")
        for error in errors:
            print(f"- {error}")
        return 1

    for path in HTML_FILES:
        check_html_assets(path, errors)

    for path in DATA.glob("*.json"):
        check_json(path, errors)

    if errors:
        print("Frontend static smoke failed.")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Frontend static smoke passed.")
    print(f"- HTML files checked: {len(HTML_FILES)}")
    print(f"- Required asset/data files checked: {len(REQUIRED_FILES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
