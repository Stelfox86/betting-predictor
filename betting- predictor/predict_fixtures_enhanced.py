# predict_fixtures_enhanced.py
import pandas as pd
import joblib

# --- Paths
MODEL_PATH = "Model/model_simple.pkl"
FIXTURES_PATH = "Data/upcoming_fixtures.csv"
PREDICTIONS_OUT = "Data/predictions.csv"

# --- Load model bundle
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
le_home = bundle["home_encoder"]
le_away = bundle["away_encoder"]
le_result = bundle["result_encoder"]

# --- Load fixtures
fixtures = pd.read_csv(FIXTURES_PATH)
# Handle possible column names
home_col = next((col for col in fixtures.columns if "HomeTeam" in col), None)
away_col = next((col for col in fixtures.columns if "AwayTeam" in col), None)
if not home_col or not away_col:
    raise ValueError("Fixtures file must have HomeTeam_original and AwayTeam_original columns!")

# --- Encode teams, handle unknowns gracefully
fixtures["Home_encoded"] = fixtures[home_col].apply(lambda x: le_home.transform([x])[0] if x in le_home.classes_ else -1)
fixtures["Away_encoded"] = fixtures[away_col].apply(lambda x: le_away.transform([x])[0] if x in le_away.classes_ else -1)

# Only keep fixtures with known teams
mask = (fixtures["Home_encoded"] >= 0) & (fixtures["Away_encoded"] >= 0)
fixtures_valid = fixtures[mask]

X_pred = fixtures_valid[["Home_encoded", "Away_encoded"]].rename(
    columns={"Home_encoded": "home_enc", "Away_encoded": "away_enc"}
)

# --- Predict
preds = model.predict(X_pred)
pred_labels = le_result.inverse_transform(preds)

# --- Save predictions with all info
out_df = fixtures_valid.copy()
out_df["Prediction"] = pred_labels

# Use original names for output
out_df.rename(columns={home_col: "HomeTeam_original", away_col: "AwayTeam_original"}, inplace=True)
cols_out = ["Date", "HomeTeam_original", "AwayTeam_original", "Prediction"]
out_df[cols_out].to_csv(PREDICTIONS_OUT, index=False)
print(f"✅ Predictions saved for {len(out_df)} fixtures to {PREDICTIONS_OUT}")

# Warn if any fixtures were skipped due to unknown teams
if (~mask).sum():
    print(f"⚠️  Skipped {(~mask).sum()} fixture(s) with unknown team names!")







































