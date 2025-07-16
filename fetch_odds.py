import requests
import pandas as pd

API_KEY = 'ec42172f66edf0cbd02e02180982d5bb'
SPORT = 'soccer_epl'
REGION = 'uk'
MARKET = 'h2h'

url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/'
params = {
    'apiKey': API_KEY,
    'regions': REGION,
    'markets': MARKET,
    'oddsFormat': 'decimal'
}

response = requests.get(url, params=params)
if response.status_code != 200:
    print("Error fetching odds:", response.text)
    exit()

odds_json = response.json()
rows = []
for event in odds_json:
    home = event['home_team']
    away = event['away_team']
    commence_time = pd.to_datetime(event['commence_time']).strftime("%d/%m/%Y %H:%M")
    for bookmaker in event['bookmakers']:
        if bookmaker['key'] == 'bet365':  # Prefer Bet365 if available
            markets = bookmaker['markets']
            for market in markets:
                row = {
                    'Date': commence_time,
                    'HomeTeam': home,
                    'AwayTeam': away,
                    'Bet365_Home': None,
                    'Bet365_Draw': None,
                    'Bet365_Away': None,
                }
                for outcome in market['outcomes']:
                    if outcome['name'] == home:
                        row['Bet365_Home'] = outcome['price']
                    elif outcome['name'] == 'Draw':
                        row['Bet365_Draw'] = outcome['price']
                    elif outcome['name'] == away:
                        row['Bet365_Away'] = outcome['price']
                rows.append(row)
            break

odds_df = pd.DataFrame(rows)
print(odds_df)
