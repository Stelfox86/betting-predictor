
import pandas as pd

# Load value bets (already includes odds and predictions)
bets = pd.read_csv('value_bets.csv')
results = pd.read_csv('Data/E0.csv')

# Merge on team names and date to bring in actual match result
merged = pd.merge(
    bets,
    results[['Date', 'HomeTeam', 'AwayTeam', 'FTR']],
    left_on=['Date', 'HomeTeam_original', 'AwayTeam_original'],
    right_on=['Date', 'HomeTeam', 'AwayTeam'],
    how='left'
)

# Rename actual result
merged['ActualResult'] = merged['FTR']
merged['BetWon'] = merged['Prediction'] == merged['ActualResult']

# Get odds from bets file (already present)
def get_odds(row):
    if row['Prediction'] == 'H':
        return row['B365H']
    elif row['Prediction'] == 'D':
        return row['B365D']
    elif row['Prediction'] == 'A':
        return row['B365A']
    return 0.0

merged['Odds'] = merged.apply(get_odds, axis=1)

# Calculate profit based on £10 stake
stake = 10
merged['Profit'] = merged.apply(lambda row: round((row['Odds'] * stake) - stake, 2) if row['BetWon'] else -stake, axis=1)

# Save result
final = merged[['Date', 'HomeTeam_original', 'AwayTeam_original', 'Prediction', 'ActualResult', 'BetWon', 'Odds', 'Profit']]
final.to_csv('bet_results.csv', index=False)

# Print summary
total_bets = len(final)
wins = final['BetWon'].sum()
losses = total_bets - wins
profit = final['Profit'].sum()

print("Results saved to bet_results.csv")
print(f"📊 Bets: {total_bets}, Wins: {wins}, Losses: {losses}, Profit: £{profit:.2f}")
