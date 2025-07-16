import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Paths
DATA_PATH = 'Data/E0.csv'
MODEL_PATH = 'Model/model.pkl'
ENC_HOME_PATH = 'Model/le_home.pkl'
ENC_AWAY_PATH = 'Model/le_away.pkl'

# Load historical match data
df = pd.read_csv(DATA_PATH)

# Drop rows with missing essential values
df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR'])

# Encode team names
le_home = LabelEncoder()
le_away = LabelEncoder()
df['HomeTeam_encoded'] = le_home.fit_transform(df['HomeTeam'])
df['AwayTeam_encoded'] = le_away.fit_transform(df['AwayTeam'])

# Features and target
X = df[['HomeTeam_encoded', 'AwayTeam_encoded']]
y = df['FTR']  # Full-Time Result (H/D/A)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, MODEL_PATH)
joblib.dump(le_home, ENC_HOME_PATH)
joblib.dump(le_away, ENC_AWAY_PATH)

print("✅ Model trained and saved.")

