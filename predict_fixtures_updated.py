import pandas as pd
import joblib
import os

# Load fixture file
fixtures = pd.read_csv("fixtures.csv")

# Standardize column names
fixtures.rename(columns={
    "Home Team": "HomeTeam",
    "Away Team": "AwayTeam"
}, inplace=True)

# Team name normalization mapping
TEAM_NAME_MAP = {
    "man united": "Man United",
    "man utd": "Man United",
    "manchester united": "Man United",
    "man city": "Man City",
    "manchester city": "Man City",
    "spurs": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "wolverhampton": "Wolves",
    "wolves": "Wolves",
    "nottingham forest": "Nott'm Forest",
    "notts forest": "Nott'm Forest",
    "nottm forest": "Nott'm Forest",
    "sheffield wednesday": "Sheffield Weds",
    "sheff weds": "Sheffield Weds",
    "sheffield united": "Sheffield United",
    "sheff utd": "Sheffield United",
    # Add any more variations needed here
}

def clean_team_name(name):
    return TEAM_NAME_MAP.get(name.strip().lower(), name.strip())

# Apply cleaning
fixtures["HomeTeam"] = fixtures["HomeTeam"].apply(clean_team_name)
fixtures["AwayTeam"] = fixtures["AwayTeam"].apply(clean_team_name)

# Load encoders and model
le_home = joblib.load("Model/le_home.pkl")
le_away = joblib.load("Model/le_away.pkl")
model = joblib.load("Model/model.pkl")

# Show all known teams
print("Teams in le_home:", list(le_home.classes_))
print("Teams in le_away:", list(le_away.classes_))

# Drop unknown teams
valid_mask = fixtures["HomeTeam"].isin(le_home.classes_) & fixtures["AwayTeam"].isin(le_away.classes_)
valid_fixtures = fixtures.loc[valid_mask].copy()

# Warn about dropped fixtures
dropped = len(fixtures) - len(valid_fixtures)
if dropped > 0:
    print(f"Dropped {dropped} fixture(s) with unknown teams.")

# Keep original names
valid_fixtures["HomeTeam_original"] = valid_fixtures["HomeTeam"]
valid_fixtures["AwayTeam_original"] = valid_fixtures["AwayTeam"]

# Encode team names
valid_fixtures["HomeTeam_encoded"] = le_home.transform(valid_fixtures["HomeTeam"])
valid_fixtures["AwayTeam_encoded"] = le_away.transform(valid_fixtures["AwayTeam"])

# Create feature matrix
X_pred = valid_fixtures[["HomeTeam_encoded", "AwayTeam_encoded"]]

# Make predictions
valid_fixtures["Predicted Result"] = model.predict(X_pred)

# Save predictions
cols_to_save = ["Date", "HomeTeam_encoded", "AwayTeam_encoded", "HomeTeam_original", "AwayTeam_original", "Predicted Result"]
valid_fixtures[cols_to_save].to_csv("predicted_fixtures.csv", index=False)

print("[OK] Predictions saved to predicted_fixtures.csv")



