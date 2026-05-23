from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Dict, Any

import joblib
import pandas as pd

from prediction_overlay import MODEL_DIR


def _league_tag(league: str) -> str:
    return str(league).strip().replace(" ", "_")


def ftr_accuracy_report(leagues: List[str], model_dir: Path | None = None) -> pd.DataFrame:
    model_dir = model_dir or Path(MODEL_DIR)
    rows: List[Dict[str, Any]] = []

    for league in leagues:
        tag = _league_tag(league)
        path = model_dir / tag / "ftr_v2.pkl"
        if not path.exists():
            print(f"⚠️ {league}: ftr_v2.pkl missing at {path}")
            rows.append({
                "league": league,
                "ftr_v2_present": False,
                "ftr_val_accuracy": None,
            })
            continue

        try:
            bundle = joblib.load(path)
        except Exception as e:
            print(f"⚠️ {league}: could not load ftr_v2 bundle: {e}")
            rows.append({
                "league": league,
                "ftr_v2_present": True,
                "ftr_val_accuracy": None,
            })
            continue

        val_acc = bundle.get("val_accuracy", None)
        rows.append({
            "league": league,
            "ftr_v2_present": True,
            "ftr_val_accuracy": float(val_acc) if val_acc is not None else None,
            "n_train": bundle.get("n_train", None),
            "n_val": bundle.get("n_val", None),
        })

    df = pd.DataFrame(rows)
    print("\nFTR v2 validation accuracy per league:")
    print(df.to_string(index=False))

    out_dir = Path("predictions_output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "FTR_v2_accuracy_report.csv"
    df.to_csv(out_path, index=False)
    print(f"\n📁 Wrote FTR accuracy CSV → {out_path}")
    return df


if __name__ == "__main__":
    # same default investor list as train_investor_leagues_v2.py
    investor_default = [
        "Champions League",
        "Europa League",
        "Europa Conference League",
        "Germany Bundesliga",
        "France Ligue 1",
        "Italy Serie A",
        "Spain La Liga",
        "England Premier League",
        "Portugal Liga",
        "USA MLS",
    ]
    ftr_accuracy_report(investor_default)