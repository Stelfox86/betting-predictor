# train_model_enhanced.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- Load data
df = pd.read_csv("Data/features_enhanced.csv")
df = df.dropna(subset=["FTR"])

# --- Encode target
df["Result"] = df["FTR"].map({1: "H", 0: "D", -1: "A"})
y = df["Result"]

# --- Encode teams
le_home = LabelEncoder().fit(df["HomeTeam"])
le_away = LabelEncoder().fit(df["AwayTeam"])
le_result = LabelEncoder().fit(y)

X = pd.DataFrame({
    "home_enc": le_home.transform(df["HomeTeam"]),
    "away_enc": le_away.transform(df["AwayTeam"])
})

y_enc = le_result.transform(y)

# --- Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_enc)

# --- Save bundle
bundle = {
    "model": model,
    "home_encoder": le_home,
    "away_encoder": le_away,
    "result_encoder": le_result
}
os.makedirs("Model", exist_ok=True)
joblib.dump(bundle, "Model/model_simple.pkl")

print("✅ Training complete!")


















