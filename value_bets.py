
import pandas as pd
import joblib

# Load data
df = pd.read_csv('Data/E0.csv')
preds = pd.read_csv('predictions.csv')

# Load encoders
le_home = joblib.load('Model/le_home.pkl')
le_away = joblib.load('Model/le_away.pkl')

# Encode teams in df to match preds
df = df.copy()
df['HomeTeam_enc'] = le_home.transform(df['HomeTeam'])
df['AwayTeam_enc'] = le_away.transform(df['AwayTeam'])

# Merge using encoded values
preds['HomeTeam'] = preds['HomeTeam'].astype(int)
preds['AwayTeam'] = preds['AwayTeam'].astype(int)
merged = pd.merge(
    preds,
    df,
    left_on=['Date', 'HomeTeam', 'AwayTeam'],
    right_on=['Date', 'HomeTeam_enc', 'AwayTeam_enc'],
    how='left'
)

# Calculate implied probabilities from Bet365 odds
merged['Imp_H'] = 1 / merged['B365H']
merged['Imp_D'] = 1 / merged['B365D']
merged['Imp_A'] = 1 / merged['B365A']
total = merged['Imp_H'] + merged['Imp_D'] + merged['Imp_A']
merged['Imp_H'] /= total
merged['Imp_D'] /= total
merged['Imp_A'] /= total

# Map model prediction to bookie implied probability
def get_model_vs_bookie(row):
    if row['Prediction'] == 'H':
        return row['Imp_H']
    elif row['Prediction'] == 'D':
        return row['Imp_D']
    elif row['Prediction'] == 'A':
        return row['Imp_A']
    return None

merged['Bookie_Prob'] = merged.apply(get_model_vs_bookie, axis=1)

# Identify value bets
merged['ValueBet'] = merged['Bookie_Prob'] < 0.35

# Save final output
value_bets = merged[['Date', 'HomeTeam_original', 'AwayTeam_original', 'Prediction',
                     'B365H', 'B365D', 'B365A', 'Bookie_Prob', 'ValueBet']]
value_bets.to_csv('value_bets.csv', index=False)

print("Value bets saved to value_bets.csv")
