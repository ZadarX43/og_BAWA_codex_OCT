import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# == ADVANCED FEATURE ENGINEERING ==
def add_advanced_features(df):
    if all(col in df.columns for col in ['home_team_shots_on_target', 'home_team_shots', 'Home Team Pre-Match xG']):
        df['home_offensive_index'] = (df['home_team_shots_on_target'] / (df['home_team_shots'] + 1)) * df['Home Team Pre-Match xG']
    else:
        df['home_offensive_index'] = 0

    if all(col in df.columns for col in ['away_team_shots_on_target', 'away_team_shots', 'Away Team Pre-Match xG']):
        df['away_offensive_index'] = (df['away_team_shots_on_target'] / (df['away_team_shots'] + 1)) * df['Away Team Pre-Match xG']
    else:
        df['away_offensive_index'] = 0

    df['goal_possession_interaction'] = df.get('goal_count_diff', 0) * df.get('possession_diff', 0)
    return df

# == VOLATILITY OVERLAY MODULE ==
def apply_volatility_stack(df):
    def calc_emotional_index(row):
        reds = row.get('home_team_red_cards', 0) + row.get('away_team_red_cards', 0)
        imbalance = abs(
            row.get('home_team_yellow_cards', 0) + 2 * row.get('home_team_red_cards', 0) -
            (row.get('away_team_yellow_cards', 0) + 2 * row.get('away_team_red_cards', 0))
        )
        early_cards = row.get('home_team_first_half_cards', 0) + row.get('away_team_first_half_cards', 0)
        if reds >= 1 or imbalance >= 3 or early_cards >= 4: return 1.0
        if early_cards == 0 and reds == 0: return -0.5
        return 0.0

    def possession_volatility(row):
        diff = row.get('home_team_possession', 50) - row.get('away_team_possession', 50)
        xg_diff = row.get('xg_diff', 0)
        if abs(diff) >= 20 and abs(xg_diff) <= 0.3: return 1.0
        if abs(diff) < 5: return -0.3
        return 0.0

    def set_piece_pressure(row):
        corners = row.get('home_team_corner_count', 0) + row.get('away_team_corner_count', 0)
        fouls = row.get('home_team_fouls', 0) + row.get('away_team_fouls', 0)
        avg = (corners + fouls) / 2
        if avg >= 30: return 1.0
        if avg <= 10: return -0.5
        return 0.0

    def momentum_trend(row):
        gap = row.get('weighted_home_ppg', 0) - row.get('weighted_away_ppg', 0)
        if abs(gap) < 0.1: return 0.0
        if abs(gap) < 0.5: return 0.5 if gap > 0 else -0.5
        return 1.0 if gap > 0 else -1.0

    def is_derby(row):
        derbies = [
            ("Arsenal", "Tottenham"), ("Man United", "Man City"),
            ("Barcelona", "Real Madrid"), ("AC Milan", "Inter"),
            ("Roma", "Lazio"), ("Dortmund", "Bayern Munich"),
            ("Benfica", "Porto"), ("Sporting CP", "Benfica"),
            ("NY Red Bulls", "NYCFC"), ("LA Galaxy", "LAFC")
        ]
        home = row.get('home_team_name', '')
        away = row.get('away_team_name', '')
        return 1.0 if any((home == a and away == b) or (home == b and away == a) for a, b in derbies) else 0.0

    def volatility_modifiers(row):
        v_sum = (
            row['emotional_volatility_score'] +
            row['possession_dissonance_score'] +
            row['set_piece_pressure_index'] +
            row['momentum_score'] +
            row['is_derby_match']
        )
        draw_mod = 1 if v_sum <= -1 else (-1 if v_sum >= 3 else 0)
        btts_mod = 1 if v_sum >= 2 else 0
        over_mod = 1 if v_sum >= 2 else (-1 if v_sum <= -2 else 0)
        return pd.Series({
            'draw_overlay_modifier': draw_mod,
            'btts_overlay_modifier': btts_mod,
            'over25_overlay_modifier': over_mod
        })

    df['emotional_volatility_score'] = df.apply(calc_emotional_index, axis=1)
    df['possession_dissonance_score'] = df.apply(possession_volatility, axis=1)
    df['set_piece_pressure_index'] = df.apply(set_piece_pressure, axis=1)
    df['momentum_score'] = df.apply(momentum_trend, axis=1)
    df['is_derby_match'] = df.apply(is_derby, axis=1)

    mod_df = df.apply(volatility_modifiers, axis=1)
    df = pd.concat([df, mod_df], axis=1)
    return df

# === File Paths ===
complete_data_dir = "/Users/hughwade/Documents/December 24/PredictionReports/PredictionComplete"
model_save_dir = "/Users/hughwade/Documents/December 24/ModelStore"
os.makedirs(model_save_dir, exist_ok=True)

leagues = [
    "Champions_League",
    "England_Premier_League",
    "Europa_Conference",
    "Europa_League",
    "Germany_Bundesliga",
    "Italy_Serie_A",
    "Portugal_Liga",
    "Spain_La_Liga",
    "USA_MLS"
]

def train_and_save_models(df, league_name):
    if df.empty or 'home_team_goal_count' not in df.columns:
        print(f"⚠️ Skipping {league_name}: data invalid or empty.")
        return

    df = add_advanced_features(df)
    df = apply_volatility_stack(df)

    # ===== Over 2.5 Model =====
    df['Over25'] = ((df['home_team_goal_count'] + df['away_team_goal_count']) > 2).astype(int)
    over25_features = [
        'goal_count_diff', 'xg_conversion_gap',
        'finishing_efficiency_home', 'finishing_efficiency_away',
        'momentum_score', 'possession_dissonance_score',
        'Pre-Match PPG (Home)', 'Pre-Match PPG (Away)',
        'average_goals_per_match_pre_match'
    ]
    X_over = df[over25_features].fillna(0)
    y_over = df['Over25']
    Xo_train, Xo_test, yo_train, yo_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)
    over_model = LogisticRegression(solver='liblinear')
    over_model.fit(Xo_train, yo_train)
    over_acc = accuracy_score(yo_test, over_model.predict(Xo_test))
    joblib.dump(over_model, os.path.join(model_save_dir, f"{league_name}_over25_model.pkl"))
    print(f"✅ [{league_name}] Over 2.5 Accuracy: {over_acc:.2%}")

    # ===== BTTS Model =====
    df['BTTS'] = ((df['home_team_goal_count'] > 0) & (df['away_team_goal_count'] > 0)).astype(int)
    btts_features = [
        'goal_count_diff', 'possession_diff', 'home_ppg', 'away_ppg',
        'home_recent_form', 'away_recent_form',
        'Pre-Match PPG (Home)', 'Pre-Match PPG (Away)',
        'btts_percentage_pre_match'
    ]
    X_btts = df[btts_features].fillna(0)
    y_btts = df['BTTS']
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_btts, y_btts, test_size=0.2, random_state=42)
    btts_model = LogisticRegression(solver='liblinear')
    btts_model.fit(Xb_train, yb_train)
    btts_acc = accuracy_score(yb_test, btts_model.predict(Xb_test))
    joblib.dump(btts_model, os.path.join(model_save_dir, f"{league_name}_btts_model.pkl"))
    print(f"✅ [{league_name}] BTTS Accuracy: {btts_acc:.2%}")

def run_training_pipeline():
    for league in leagues:
        path = os.path.join(complete_data_dir, f"{league}_complete.csv")
        if not os.path.exists(path):
            print(f"❌ {path} does not exist.")
            continue
        df = pd.read_csv(path)
        print(f"\n=== Training for {league} ===")
        train_and_save_models(df, league)

if __name__ == "__main__":
    run_training_pipeline()
