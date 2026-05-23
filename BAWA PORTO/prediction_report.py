# prediction_report.py
import numpy as np
from sklearn.metrics import accuracy_score

def generate_prediction_report(df, league_name):
    """
    Print BTTS and Over 2.5 prediction performance summary,
    including volatility-based prediction flips and accuracy.
    """
    print("\n===== Prediction Overlay Report =====")
    print(f"{league_name}")

    # Create actual labels if missing
    if 'BTTS' not in df.columns:
        df['BTTS'] = ((df['home_team_goal_count'] > 0) & (df['away_team_goal_count'] > 0)).astype(int)
    if 'Over25' not in df.columns:
        df['Over25'] = ((df['home_team_goal_count'] + df['away_team_goal_count']) > 2).astype(int)

    # Accuracy scores
    btts_acc = accuracy_score(df['BTTS'], df['btts_pred']) if 'btts_pred' in df.columns else np.nan
    over_acc = accuracy_score(df['Over25'], df['over25_pred']) if 'over25_pred' in df.columns else np.nan

    # Volatility flips
    btts_flips = df['btts_changed'].sum() if 'btts_changed' in df.columns else 0
    btts_correct = ((df['btts_changed']) & (df['btts_pred'] == df['BTTS'])).sum() if 'btts_changed' in df.columns else 0

    over_flips = df['over25_changed'].sum() if 'over25_changed' in df.columns else 0
    over_correct = ((df['over25_changed']) & (df['over25_pred'] == df['Over25'])).sum() if 'over25_changed' in df.columns else 0

    # Print results
    print(f"BTTS Accuracy:               {btts_acc * 100:.2f}%")
    print(f"Over 2.5 Accuracy:           {over_acc * 100:.2f}%")
    print(f"BTTS Flips via Volatility:   {btts_flips} flips | {btts_correct} correct ({(btts_correct / max(btts_flips,1)) * 100:.2f}%)")
    print(f"Over2.5 Flips via Volatility:{over_flips} flips | {over_correct} correct ({(over_correct / max(over_flips,1)) * 100:.2f}%)")
    print("--------------------------------------")
