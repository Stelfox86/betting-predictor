import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

# Load the enhanced dataset
df = pd.read_csv("features_enhanced.csv")

# Encode target labels if not already encoded
if df["FTR"].dtype == object:
    le_result = LabelEncoder()
    df["FTR"] = le_result.fit_transform(df["FTR"])
    joblib.dump(le_result, "Model/le_result.pkl")

# Encode team names
le_home = LabelEncoder()
le_away = LabelEncoder()
df["HomeTeam_encoded"] = le_home.fit_transform(df["HomeTeam"])
df["AwayTeam_encoded"] = le_away.fit_transform(df["AwayTeam"])

# Save encoders
joblib.dump(le_home, "Model/le_home.pkl")
joblib.dump(le_away, "Model/le_away.pkl")

# Define feature columns
feature_cols = [
    "HomeTeam_encoded", "AwayTeam_encoded",
    "HomeAvgGoalsScored", "AwayAvgGoalsScored",
    "HomeForm", "AwayForm",
    "HomeShotsOnTarget", "AwayShotsOnTarget",
    "HomeDefStrength", "AwayDefStrength"
]

# Prepare feature matrix and target
X = df[feature_cols]
y = df["FTR"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "Model/model.pkl")
print("Model and encoders saved to 'Model/' folder.")

