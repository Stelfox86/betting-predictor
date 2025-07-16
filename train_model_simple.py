import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load historical data
df = pd.read_csv("Data/features_enhanced.csv")
df = df.dropna(subset=["FTR"])

# We'll use only these features for future matches
X = df[["HomeTeam", "AwayTeam"]]
y = df["FTR"].map({1: "H", 0: "D", -1: "A"})

# Encode teams
le_home = LabelEncoder().fit(X["HomeTeam"])
le_away = LabelEncoder().fit(X["AwayTeam"])
le_result = LabelEncoder().fit(y)

X_enc = pd.DataFrame({
    "home_enc": le_home.transform(X["HomeTeam"]),
    "away_enc": le_away.transform(X["AwayTeam"])
})
y_enc = le_result.transform(y)

# Train simple model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_enc, y_enc)

bundle = {
    "model": model,
    "home_encoder": le_home,
    "away_encoder": le_away,
    "result_encoder": le_result
}
os.makedirs("Model", exist_ok=True)
joblib.dump(bundle, "Model/model_bundle_simple.pkl")
print("✅ Simple model trained and saved.")



















