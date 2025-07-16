import pandas as pd
import numpy as np

# === Paths ===
HIST_PATH = "Data/features_enhanced.csv"
UPCOMING_PATH = "Data/upcoming_fixtures.csv"
OUT_PATH = "Data/fixtures_with_features.csv"

# === Load Historical Data ===
hist = pd.read_csv(HIST_PATH)
# Clean columns for easier merging
hist['HomeTeam'] = hist['HomeTeam'].str.strip()
hist['AwayTeam'] = hist['AwayTeam'].str.strip()

# === Load Upcoming Fixtures ===
fixtures = pd.read_csv(UPCOMING_PATH)
fixtures['HomeTeam_original'] = fixtures['HomeTeam_original'].str.strip()
fixtures['AwayTeam_original'] = fixtures['AwayTeam_original'].str.strip()

# Rename for consistency
fixtures = fixtures.rename(columns={
    "HomeTeam_original": "HomeTeam",
    "AwayTeam_original": "AwayTeam"
})

# === Odds columns to expect ===
odds_cols = ['B365H', 'B365D', 'B365A'] # Add others if your model needs them

# Make sure these exist
for col in odds_cols:
    if col not in fixtures.columns:
        fixtures[col] = np.nan  # Fill with NaN, you will need to fill manually

# === Stat columns to generate ===
feature_cols = [c for c in hist.columns if c not in ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTR", "Result", "Referee"] + odds_cols]

# Compute recent average stats for each team (use last N games)
N = 10
def team_recent_avg(df, team, home=True):
    # Home or Away games
    if home:
        team_games = df[df['HomeTeam'] == team].sort_values("Date", ascending=False).head(N)
        prefix = 'Home'
    else:
        team_games = df[df['AwayTeam'] == team].sort_values("Date", ascending=False).head(N)
        prefix = 'Away'
    # Find columns starting with prefix (e.g. Home_GF, Away_GF, etc.)
    stat_cols = [c for c in feature_cols if c.startswith(prefix)]
    avgs = team_games[stat_cols].mean()
    return avgs

# === Build features for each fixture ===
rows = []
for idx, row in fixtures.iterrows():
    fixture = row.to_dict()
    home = fixture["HomeTeam"]
    away = fixture["AwayTeam"]
    
    # Home team features
    home_feats = team_recent_avg(hist, home, home=True)
    for k, v in home_feats.items():
        fixture[k] = v
    # Away team features
    away_feats = team_recent_avg(hist, away, home=False)
    for k, v in away_feats.items():
        fixture[k] = v
    
    # Odds (already present or needs filling)
    for col in odds_cols:
        fixture[col] = row.get(col, np.nan)
    
    # Add to output
    rows.append(fixture)

df_out = pd.DataFrame(rows)

# === Output CSV ===
df_out.to_csv(OUT_PATH, index=False)
print(f"✅ Features for fixtures saved to {OUT_PATH}")
print("🟢 Please open this file and fill in missing odds manually if needed.")
