import pandas as pd
import joblib

# Load model
bundle = joblib.load("Model/model_bundle_simple.pkl")
model = bundle["model"]
le_home = bundle["home_encoder"]
le_away = bundle["away_encoder"]
le_result = bundle["result_encoder"]

# Load fixtures
fixtures = pd.read_csv("Data/upcoming_fixtures.csv")
fixtures = fixtures.rename(columns={
    "HomeTeam_original": "HomeTeam",
    "AwayTeam_original": "AwayTeam"
})

# Encode teams
fixtures["home_enc"] = le_home.transform(fixtures["HomeTeam"])
fixtures["away_enc"] = le_away.transform(fixtures["AwayTeam"])

X_pred = fixtures[["home_enc", "away_enc"]]
preds = model.predict(X_pred)
preds_label = le_result.inverse_transform(preds)

fixtures["Prediction"] = preds_label

# Save output (keep only necessary columns)
out = fixtures[["Date", "HomeTeam", "AwayTeam", "Prediction"]]
out.to_csv("Data/predictions.csv", index=False)
print("✅ Predictions saved to Data/predictions.csv")
































