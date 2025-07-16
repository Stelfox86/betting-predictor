import joblib

# Load your individual model and encoders
model = joblib.load("Model/model.pkl")  # <- Your actual trained model
le_home = joblib.load("Model/le_home.pkl")
le_away = joblib.load("Model/le_away.pkl")

# Save as a bundle
bundle = {
    "model": model,
    "home_encoder": le_home,
    "away_encoder": le_away
}

# Save the full bundle
joblib.dump(bundle, "Model/model_bundle.pkl")
print("✅ Model bundle saved as Model/model_bundle.pkl")

