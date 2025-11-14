import requests
import pandas as pd
import numpy as np
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import os
import math

# API Configuration
API_KEY = "krRqEYZn0h4HQRNxo1PU/cPid/NMi9Ki1FJb6ajILhCdt9/ScLi1FNaCssp6GH+A"
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Function to fetch data from CFBD API
def get_data(endpoint, params=None):
    if params is None:
        params = {}
    response = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=HEADERS)
    response.raise_for_status()
    return response.json()

# Step 1: Determine current week for 2025 season
current_date = date(2025, 11, 14)
calendar = get_data("/calendar", {"year": 2025})
current_week = None
for week in calendar:
    first_game = date.fromisoformat(week['firstGameStart'][:10])
    last_game = date.fromisoformat(week['lastGameStart'][:10])
    if first_game <= current_date <= last_game:
        current_week = week['week']
        break
if current_week is None:
    raise ValueError("Could not determine current week.")

print(f"Current week: {current_week}")

# Step 2: Collect historical data for training (2015-2024)
historical_years = list(range(2015, 2025))
all_games = []
all_lines = []
all_advanced = []
all_weather = []
all_sp = []
all_talent = []
all_wepa = []
all_fpi = []
all_elo = []

for year in historical_years:
    # Games
    games = get_data("/games", {"year": year, "seasonType": "regular"})
    all_games.extend(games)
    
    # Lines
    lines = get_data("/lines", {"year": year})
    all_lines.extend(lines)
    
    # Advanced game stats
    advanced = get_data("/stats/game/advanced", {"year": year})
    all_advanced.extend(advanced)
    
    # Weather
    weather = get_data("/games/weather", {"year": year})
    all_weather.extend(weather)
    
    # SP+ ratings (season-level)
    sp = get_data("/ratings/sp", {"year": year})
    all_sp.extend(sp)
    
    # Talent
    talent = get_data("/talent", {"year": year})
    all_talent.extend(talent)
    
    # Opponent-adjusted metrics (WEPA)
    wepa = get_data("/wepa/team/season", {"year": year})
    all_wepa.extend(wepa)
    
    # FPI ratings
    fpi = get_data("/ratings/fpi", {"year": year})
    all_fpi.extend(fpi)
    
    # Elo ratings (season-level, but we'll use per-game from games)
    elo = get_data("/ratings/elo", {"year": year})
    all_elo.extend(elo)

# Convert to DataFrames
df_games = pd.DataFrame(all_games)
df_lines = pd.DataFrame(all_lines)
df_advanced = pd.DataFrame(all_advanced)
df_weather = pd.DataFrame(all_weather)
df_sp = pd.DataFrame(all_sp)
df_talent = pd.DataFrame(all_talent)
df_wepa = pd.DataFrame(all_wepa)
df_fpi = pd.DataFrame(all_fpi)
df_elo = pd.DataFrame(all_elo)

# Expand WEPA dicts
df_wepa['epa_total'] = df_wepa['epa'].apply(lambda x: x.get('total', np.nan) if isinstance(x, dict) else np.nan)
df_wepa['epaAllowed_total'] = df_wepa['epaAllowed'].apply(lambda x: x.get('total', np.nan) if isinstance(x, dict) else np.nan)
df_wepa['successRate_total'] = df_wepa['successRate'].apply(lambda x: x.get('total', np.nan) if isinstance(x, dict) else np.nan)
df_wepa['successRateAllowed_total'] = df_wepa['successRateAllowed'].apply(lambda x: x.get('total', np.nan) if isinstance(x, dict) else np.nan)

# Step 3: Data Cleaning and Merging
# Merge games and lines on 'id'
df_games = df_games.merge(df_lines[['id', 'lines']], on='id', how='left')

# Extract betting lines (use first provider for simplicity; prefer 'consensus' if available)
def extract_line(row):
    lines = row['lines']
    if isinstance(lines, float) and math.isnan(lines) or not lines:
        return pd.Series({'spread': np.nan, 'formattedSpread': '', 'overUnder': np.nan, 'homeMoneyline': np.nan, 'awayMoneyline': np.nan})
    # Prefer 'consensus' provider
    consensus = next((line for line in lines if line['provider'].lower() == 'consensus'), lines[0])
    return pd.Series({
        'spread': consensus.get('spread'),
        'formattedSpread': consensus.get('formattedSpread', ''),
        'overUnder': consensus.get('overUnder'),
        'homeMoneyline': consensus.get('homeMoneyline'),
        'awayMoneyline': consensus.get('awayMoneyline')
    })

df_games = df_games.join(df_games.apply(extract_line, axis=1))

# Parse spread to numeric (spread is typically home spread; negative if home favorite)
df_games['spread'] = pd.to_numeric(df_games['spread'], errors='coerce')

# Add targets
df_games['actual_margin'] = df_games['homePoints'] - df_games['awayPoints']
df_games['actual_total'] = df_games['homePoints'] + df_games['awayPoints']

# Merge weather
df_games = df_games.merge(df_weather[['id', 'temperature', 'windSpeed', 'precipitation']], left_on='id', right_on='id', how='left')

# Merge SP+ and talent (use previous year to avoid lookahead bias)
df_sp['year'] += 1  # Shift to next year for predictive use
df_talent['year'] += 1
df_games = df_games.merge(df_sp[['year', 'team', 'rating']], left_on=['season', 'homeTeam'], right_on=['year', 'team'], suffixes=('', '_home_sp'), how='left')
df_games = df_games.merge(df_sp[['year', 'team', 'rating']], left_on=['season', 'awayTeam'], right_on=['year', 'team'], suffixes=('', '_away_sp'), how='left')
df_games['home_sp'] = df_games['rating']
df_games['away_sp'] = df_games['rating_away_sp']
df_games.drop(['year_home_sp', 'team_home_sp', 'rating', 'rating_away_sp', 'year_away_sp', 'team_away_sp'], axis=1, inplace=True, errors='ignore')

df_games = df_games.merge(df_talent[['year', 'team', 'talent']], left_on=['season', 'homeTeam'], right_on=['year', 'team'], suffixes=('', '_home_talent'), how='left')
df_games = df_games.merge(df_talent[['year', 'team', 'talent']], left_on=['season', 'awayTeam'], right_on=['year', 'team'], suffixes=('', '_away_talent'), how='left')
df_games['home_talent'] = df_games['talent']
df_games['away_talent'] = df_games['talent_away_talent']
df_games.drop(['year_home_talent', 'team_home_talent', 'talent', 'talent_away_talent', 'year_away_talent', 'team_away_talent'], axis=1, inplace=True, errors='ignore')

# Merge opponent-adjusted metrics (WEPA, use previous year to avoid lookahead)
df_wepa['year'] += 1
df_games = df_games.merge(df_wepa[['year', 'team', 'epa_total', 'epaAllowed_total', 'successRate_total', 'successRateAllowed_total']], left_on=['season', 'homeTeam'], right_on=['year', 'team'], suffixes=('', '_home_wepa'), how='left')
df_games = df_games.merge(df_wepa[['year', 'team', 'epa_total', 'epaAllowed_total', 'successRate_total', 'successRateAllowed_total']], left_on=['season', 'awayTeam'], right_on=['year', 'team'], suffixes=('', '_away_wepa'), how='left')
df_games['home_epa'] = df_games['epa_total']
df_games['home_epa_allowed'] = df_games['epaAllowed_total']
df_games['home_successRate'] = df_games['successRate_total']
df_games['home_successRate_allowed'] = df_games['successRateAllowed_total']
df_games['away_epa'] = df_games['epa_total_away_wepa']
df_games['away_epa_allowed'] = df_games['epaAllowed_total_away_wepa']
df_games['away_successRate'] = df_games['successRate_total_away_wepa']
df_games['away_successRate_allowed'] = df_games['successRateAllowed_total_away_wepa']
df_games.drop(['year_home_wepa', 'team_home_wepa', 'epa_total', 'epaAllowed_total', 'successRate_total', 'successRateAllowed_total',
               'year_away_wepa', 'team_away_wepa', 'epa_total_away_wepa', 'epaAllowed_total_away_wepa', 'successRate_total_away_wepa', 'successRateAllowed_total_away_wepa'], axis=1, inplace=True, errors='ignore')

# Merge FPI ratings (use previous year to avoid lookahead)
df_fpi['year'] += 1
df_games = df_games.merge(df_fpi[['year', 'team', 'fpi']], left_on=['season', 'homeTeam'], right_on=['year', 'team'], suffixes=('', '_home_fpi'), how='left')
df_games = df_games.merge(df_fpi[['year', 'team', 'fpi']], left_on=['season', 'awayTeam'], right_on=['year', 'team'], suffixes=('', '_away_fpi'), how='left')
df_games['home_fpi'] = df_games['fpi']
df_games['away_fpi'] = df_games['fpi_away_fpi']
df_games.drop(['year_home_fpi', 'team_home_fpi', 'fpi', 'year_away_fpi', 'team_away_fpi', 'fpi_away_fpi'], axis=1, inplace=True, errors='ignore')

# Merge season-level Elo (but we'll use per-game from games for rolling)
df_elo['year'] += 1
df_games = df_games.merge(df_elo[['year', 'team', 'elo']], left_on=['season', 'homeTeam'], right_on=['year', 'team'], suffixes=('', '_home_elo'), how='left')
df_games = df_games.merge(df_elo[['year', 'team', 'elo']], left_on=['season', 'awayTeam'], right_on=['year', 'team'], suffixes=('', '_away_elo'), how='left')
df_games['home_elo_season'] = df_games['elo']
df_games['away_elo_season'] = df_games['elo_away_elo']
df_games.drop(['year_home_elo', 'team_home_elo', 'elo', 'year_away_elo', 'team_away_elo', 'elo_away_elo'], axis=1, inplace=True, errors='ignore')

# Expand advanced stats (added more: explosiveness, havocTotal, pointsPerOpportunity)
df_advanced['off_ppa'] = df_advanced['offense'].apply(lambda x: x.get('ppa', np.nan) if isinstance(x, dict) else np.nan)
df_advanced['off_successRate'] = df_advanced['offense'].apply(lambda x: x.get('successRate', np.nan) if isinstance(x, dict) else np.nan)
df_advanced['off_explosiveness'] = df_advanced['offense'].apply(lambda x: x.get('explosiveness', np.nan) if isinstance(x, dict) else np.nan)
df_advanced['off_havocTotal'] = df_advanced['offense'].apply(lambda x: x.get('havoc', {}).get('total', np.nan) if isinstance(x, dict) else np.nan)
df_advanced['off_pointsPerOpportunity'] = df_advanced['offense'].apply(lambda x: x.get('pointsPerOpportunity', np.nan) if isinstance(x, dict) else np.nan)
df_advanced['def_ppa'] = df_advanced['defense'].apply(lambda x: x.get('ppa', np.nan) if isinstance(x, dict) else np.nan)
df_advanced['def_successRate'] = df_advanced['defense'].apply(lambda x: x.get('successRate', np.nan) if isinstance(x, dict) else np.nan)
df_advanced['def_explosiveness'] = df_advanced['defense'].apply(lambda x: x.get('explosiveness', np.nan) if isinstance(x, dict) else np.nan)
df_advanced['def_havocTotal'] = df_advanced['defense'].apply(lambda x: x.get('havoc', {}).get('total', np.nan) if isinstance(x, dict) else np.nan)
df_advanced['def_pointsPerOpportunity'] = df_advanced['defense'].apply(lambda x: x.get('pointsPerOpportunity', np.nan) if isinstance(x, dict) else np.nan)

# Step 4: Engineer rolling features (last 5 games, added more stats)
teams = df_advanced['team'].unique()
rolling_dfs = []

for team in teams:
    team_df = df_advanced[df_advanced['team'] == team].sort_values(['season', 'week'])
    rolling_stats = team_df[['off_ppa', 'off_successRate', 'off_explosiveness', 'off_havocTotal', 'off_pointsPerOpportunity',
                             'def_ppa', 'def_successRate', 'def_explosiveness', 'def_havocTotal', 'def_pointsPerOpportunity']].rolling(window=5, min_periods=1).mean().shift(1)  # Previous games
    rolling_stats.columns = [f"{col}_rolling" for col in rolling_stats.columns]
    rolling_stats['gameId'] = team_df['gameId']
    rolling_stats['team'] = team
    rolling_dfs.append(rolling_stats)

df_rolling = pd.concat(rolling_dfs)

# Merge to games (home and away)
df_games = df_games.merge(df_rolling, left_on=['id', 'homeTeam'], right_on=['gameId', 'team'], suffixes=('', '_home'), how='left')
df_games = df_games.merge(df_rolling, left_on=['id', 'awayTeam'], right_on=['gameId', 'team'], suffixes=('', '_away'), how='left')
df_games.drop(['gameId_home', 'team_home', 'gameId_away', 'team_away'], axis=1, inplace=True, errors='ignore')

# Engineer rolling Elo from per-game pre-game Elo
# Create a df for each team's Elo history
team_elo_dfs = []
all_teams = pd.unique(df_games[['homeTeam', 'awayTeam']].values.ravel('K'))
for team in all_teams:
    home_games = df_games[df_games['homeTeam'] == team][['season', 'week', 'id', 'homePregameElo']].rename(columns={'homePregameElo': 'pregame_elo', 'id': 'gameId'})
    away_games = df_games[df_games['awayTeam'] == team][['season', 'week', 'id', 'awayPregameElo']].rename(columns={'awayPregameElo': 'pregame_elo', 'id': 'gameId'})
    team_elo = pd.concat([home_games, away_games]).sort_values(['season', 'week'])
    team_elo['team'] = team
    team_elo_dfs.append(team_elo)

df_team_elo = pd.concat(team_elo_dfs)

# Compute rolling Elo
rolling_elo_dfs = []
for team in all_teams:
    team_df = df_team_elo[df_team_elo['team'] == team].sort_values(['season', 'week'])
    team_df['elo_rolling'] = team_df['pregame_elo'].rolling(window=5, min_periods=1).mean().shift(1)
    rolling_elo_dfs.append(team_df[['gameId', 'team', 'elo_rolling']])

df_rolling_elo = pd.concat(rolling_elo_dfs)

# Merge to df_games
df_games = df_games.merge(df_rolling_elo, left_on=['id', 'homeTeam'], right_on=['gameId', 'team'], suffixes=('', '_home_elo_roll'), how='left')
df_games = df_games.merge(df_rolling_elo, left_on=['id', 'awayTeam'], right_on=['gameId', 'team'], suffixes=('', '_away_elo_roll'), how='left')
df_games['home_elo_rolling'] = df_games['elo_rolling']
df_games['away_elo_rolling'] = df_games['elo_rolling_away_elo_roll']
df_games.drop(['gameId_home_elo_roll', 'team_home_elo_roll', 'elo_rolling', 'gameId_away_elo_roll', 'team_away_elo_roll', 'elo_rolling_away_elo_roll'], axis=1, inplace=True, errors='ignore')

# Additional features (added diffs for new stats and opponent-adjusted)
df_games['home_adv'] = 1  # Home advantage binary
df_games['sp_diff'] = df_games['home_sp'] - df_games['away_sp']
df_games['talent_diff'] = df_games['home_talent'] - df_games['away_talent']
df_games['fpi_diff'] = df_games['home_fpi'] - df_games['away_fpi']
df_games['elo_season_diff'] = df_games['home_elo_season'] - df_games['away_elo_season']
df_games['elo_rolling_diff'] = df_games['home_elo_rolling'] - df_games['away_elo_rolling']
df_games['off_ppa_diff'] = df_games['off_ppa_rolling'] - df_games['def_ppa_rolling_away']
df_games['def_ppa_diff'] = df_games['def_ppa_rolling'] - df_games['off_ppa_rolling_away']
df_games['off_success_diff'] = df_games['off_successRate_rolling'] - df_games['def_successRate_rolling_away']
df_games['def_success_diff'] = df_games['def_successRate_rolling'] - df_games['off_successRate_rolling_away']
df_games['off_explo_diff'] = df_games['off_explosiveness_rolling'] - df_games['def_explosiveness_rolling_away']
df_games['def_explo_diff'] = df_games['def_explosiveness_rolling'] - df_games['off_explosiveness_rolling_away']
df_games['off_havoc_diff'] = df_games['off_havocTotal_rolling'] - df_games['def_havocTotal_rolling_away']
df_games['def_havoc_diff'] = df_games['def_havocTotal_rolling'] - df_games['off_havocTotal_rolling_away']
df_games['off_ppo_diff'] = df_games['off_pointsPerOpportunity_rolling'] - df_games['def_pointsPerOpportunity_rolling_away']
df_games['def_ppo_diff'] = df_games['def_pointsPerOpportunity_rolling'] - df_games['off_pointsPerOpportunity_rolling_away']

# Opponent-adjusted diffs
df_games['adj_off_epa_diff'] = df_games['home_epa'] - df_games['away_epa_allowed']
df_games['adj_def_epa_diff'] = df_games['home_epa_allowed'] - df_games['away_epa']
df_games['adj_off_success_diff'] = df_games['home_successRate'] - df_games['away_successRate_allowed']
df_games['adj_def_success_diff'] = df_games['home_successRate_allowed'] - df_games['away_successRate']

# Weather features
df_games['is_rainy'] = (df_games['precipitation'] > 0).astype(int)
df_games['wind_high'] = (df_games['windSpeed'] > 10).astype(int)

# Drop NaN rows for training
df_train = df_games.dropna(subset=['actual_margin', 'actual_total', 'spread', 'overUnder'])

# Features list (expanded with new diffs and adjusted metrics)
features = ['home_adv', 'sp_diff', 'talent_diff', 'fpi_diff', 'elo_season_diff', 'elo_rolling_diff', 'off_ppa_diff', 'def_ppa_diff', 
            'off_success_diff', 'def_success_diff', 'off_explo_diff', 'def_explo_diff',
            'off_havoc_diff', 'def_havoc_diff', 'off_ppo_diff', 'def_ppo_diff',
            'adj_off_epa_diff', 'adj_def_epa_diff', 'adj_off_success_diff', 'adj_def_success_diff',
            'temperature', 'wind_high', 'is_rainy']

# Step 5: Train Models
X = df_train[features]
y_margin = df_train['actual_margin']
y_total = df_train['actual_total']

X_train, X_test, y_margin_train, y_margin_test = train_test_split(X, y_margin, test_size=0.2, random_state=42)
margin_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
margin_model.fit(X_train, y_margin_train)
print(f"Margin MAE: {mean_absolute_error(y_margin_test, margin_model.predict(X_test))}")

X_train, X_test, y_total_train, y_total_test = train_test_split(X, y_total, test_size=0.2, random_state=42)
total_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
total_model.fit(X_train, y_total_train)
print(f"Total MAE: {mean_absolute_error(y_total_test, total_model.predict(X_test))}")

# Step 6: Fetch current week data for picks (2025, current week)
current_games = get_data("/games", {"year": 2025, "week": current_week, "seasonType": "regular"})
df_current = pd.DataFrame(current_games)

current_lines = get_data("/lines", {"year": 2025, "week": current_week})
df_current_lines = pd.DataFrame(current_lines)
df_current = df_current.merge(df_current_lines[['id', 'lines']], on='id', how='left')
df_current = df_current.join(df_current.apply(extract_line, axis=1))
df_current['spread'] = pd.to_numeric(df_current['spread'], errors='coerce')

# Fetch current advanced for rolling
current_advanced = get_data("/stats/game/advanced", {"year": 2025})
df_current_advanced = pd.DataFrame(current_advanced)
df_current_advanced['off_ppa'] = df_current_advanced['offense'].apply(lambda x: x.get('ppa', np.nan) if isinstance(x, dict) else np.nan)
df_current_advanced['off_successRate'] = df_current_advanced['offense'].apply(lambda x: x.get('successRate', np.nan) if isinstance(x, dict) else np.nan)
df_current_advanced['off_explosiveness'] = df_current_advanced['offense'].apply(lambda x: x.get('explosiveness', np.nan) if isinstance(x, dict) else np.nan)
df_current_advanced['off_havocTotal'] = df_current_advanced['offense'].apply(lambda x: x.get('havoc', {}).get('total', np.nan) if isinstance(x, dict) else np.nan)
df_current_advanced['off_pointsPerOpportunity'] = df_current_advanced['offense'].apply(lambda x: x.get('pointsPerOpportunity', np.nan) if isinstance(x, dict) else np.nan)
df_current_advanced['def_ppa'] = df_current_advanced['defense'].apply(lambda x: x.get('ppa', np.nan) if isinstance(x, dict) else np.nan)
df_current_advanced['def_successRate'] = df_current_advanced['defense'].apply(lambda x: x.get('successRate', np.nan) if isinstance(x, dict) else np.nan)
df_current_advanced['def_explosiveness'] = df_current_advanced['defense'].apply(lambda x: x.get('explosiveness', np.nan) if isinstance(x, dict) else np.nan)
df_current_advanced['def_havocTotal'] = df_current_advanced['defense'].apply(lambda x: x.get('havoc', {}).get('total', np.nan) if isinstance(x, dict) else np.nan)
df_current_advanced['def_pointsPerOpportunity'] = df_current_advanced['defense'].apply(lambda x: x.get('pointsPerOpportunity', np.nan) if isinstance(x, dict) else np.nan)

# Fetch current WEPA
wepa_2025 = get_data("/wepa/team/season", {"year": 2025})
if not wepa_2025:
    wepa_2025 = get_data("/wepa/team/season", {"year": 2024})
df_wepa_current = pd.DataFrame(wepa_2025)

# Expand current WEPA
df_wepa_current['epa_total'] = df_wepa_current['epa'].apply(lambda x: x.get('total', np.nan) if isinstance(x, dict) else np.nan)
df_wepa_current['epaAllowed_total'] = df_wepa_current['epaAllowed'].apply(lambda x: x.get('total', np.nan) if isinstance(x, dict) else np.nan)
df_wepa_current['successRate_total'] = df_wepa_current['successRate'].apply(lambda x: x.get('total', np.nan) if isinstance(x, dict) else np.nan)
df_wepa_current['successRateAllowed_total'] = df_wepa_current['successRateAllowed'].apply(lambda x: x.get('total', np.nan) if isinstance(x, dict) else np.nan)

# Fetch current FPI
fpi_2025 = get_data("/ratings/fpi", {"year": 2025})
if not fpi_2025:
    fpi_2025 = get_data("/ratings/fpi", {"year": 2024})
df_fpi_current = pd.DataFrame(fpi_2025)

# Fetch current Elo
elo_2025 = get_data("/ratings/elo", {"year": 2025})
if not elo_2025:
    elo_2025 = get_data("/ratings/elo", {"year": 2024})
df_elo_current = pd.DataFrame(elo_2025)

# Merge SP+ and talent (use 2025 if available, else 2024)
sp_2025 = get_data("/ratings/sp", {"year": 2025})
if not sp_2025:
    sp_2025 = get_data("/ratings/sp", {"year": 2024})
df_sp_current = pd.DataFrame(sp_2025)
talent_2025 = get_data("/talent", {"year": 2025})
if not talent_2025:
    talent_2025 = get_data("/talent", {"year": 2024})
df_talent_current = pd.DataFrame(talent_2025)

# Merge similarly
df_current = df_current.merge(df_sp_current[['team', 'rating']], left_on='homeTeam', right_on='team', how='left')
df_current['home_sp'] = df_current['rating']
df_current.drop(['team', 'rating'], axis=1, inplace=True)
df_current = df_current.merge(df_sp_current[['team', 'rating']], left_on='awayTeam', right_on='team', how='left')
df_current['away_sp'] = df_current['rating']
df_current.drop(['team', 'rating'], axis=1, inplace=True)

df_current = df_current.merge(df_talent_current[['team', 'talent']], left_on='homeTeam', right_on='team', how='left')
df_current['home_talent'] = df_current['talent']
df_current.drop(['team', 'talent'], axis=1, inplace=True)

df_current = df_current.merge(df_talent_current[['team', 'talent']], left_on='awayTeam', right_on='team', how='left')
df_current['away_talent'] = df_current['talent']
df_current.drop(['team', 'talent'], axis=1, inplace=True)

# Merge WEPA for current
df_current = df_current.merge(df_wepa_current[['team', 'epa_total', 'epaAllowed_total', 'successRate_total', 'successRateAllowed_total']], left_on='homeTeam', right_on='team', how='left')
df_current['home_epa'] = df_current['epa_total']
df_current['home_epa_allowed'] = df_current['epaAllowed_total']
df_current['home_successRate'] = df_current['successRate_total']
df_current['home_successRate_allowed'] = df_current['successRateAllowed_total']
df_current.drop(['team', 'epa_total', 'epaAllowed_total', 'successRate_total', 'successRateAllowed_total'], axis=1, inplace=True)

df_current = df_current.merge(df_wepa_current[['team', 'epa_total', 'epaAllowed_total', 'successRate_total', 'successRateAllowed_total']], left_on='awayTeam', right_on='team', how='left')
df_current['away_epa'] = df_current['epa_total']
df_current['away_epa_allowed'] = df_current['epaAllowed_total']
df_current['away_successRate'] = df_current['successRate_total']
df_current['away_successRate_allowed'] = df_current['successRateAllowed_total']
df_current.drop(['team', 'epa_total', 'epaAllowed_total', 'successRate_total', 'successRateAllowed_total'], axis=1, inplace=True)

# Merge FPI for current
df_current = df_current.merge(df_fpi_current[['team', 'fpi']], left_on='homeTeam', right_on='team', how='left')
df_current['home_fpi'] = df_current['fpi']
df_current.drop(['team', 'fpi'], axis=1, inplace=True)

df_current = df_current.merge(df_fpi_current[['team', 'fpi']], left_on='awayTeam', right_on='team', how='left')
df_current['away_fpi'] = df_current['fpi']
df_current.drop(['team', 'fpi'], axis=1, inplace=True)

# Merge season-level Elo for current
df_current = df_current.merge(df_elo_current[['team', 'elo']], left_on='homeTeam', right_on='team', how='left')
df_current['home_elo_season'] = df_current['elo']
df_current.drop(['team', 'elo'], axis=1, inplace=True)

df_current = df_current.merge(df_elo_current[['team', 'elo']], left_on='awayTeam', right_on='team', how='left')
df_current['away_elo_season'] = df_current['elo']
df_current.drop(['team', 'elo'], axis=1, inplace=True)

# Engineer rolling Elo for current from per-game pre-game Elo in current_games
# For current, use the pre-game Elo from current_games for the upcoming games, but for rolling, need historical for 2025
# Fetch all 2025 games up to current week for Elo history
current_all_games = get_data("/games", {"year": 2025, "seasonType": "regular"})
df_current_all_games = pd.DataFrame(current_all_games)

# Create team Elo history for 2025
team_elo_current_dfs = []
all_teams_current = pd.unique(df_current_all_games[['homeTeam', 'awayTeam']].values.ravel('K'))
for team in all_teams_current:
    home_games = df_current_all_games[df_current_all_games['homeTeam'] == team][['season', 'week', 'id', 'homePregameElo']].rename(columns={'homePregameElo': 'pregame_elo', 'id': 'gameId'})
    away_games = df_current_all_games[df_current_all_games['awayTeam'] == team][['season', 'week', 'id', 'awayPregameElo']].rename(columns={'awayPregameElo': 'pregame_elo', 'id': 'gameId'})
    team_elo = pd.concat([home_games, away_games]).sort_values(['season', 'week'])
    team_elo['team'] = team
    team_elo_current_dfs.append(team_elo)

df_team_elo_current = pd.concat(team_elo_current_dfs)

# Compute rolling Elo for current
rolling_elo_current_dfs = []
for team in all_teams_current:
    team_df = df_team_elo_current[df_team_elo_current['team'] == team].sort_values(['season', 'week'])
    team_df['elo_rolling'] = team_df['pregame_elo'].rolling(window=5, min_periods=1).mean().shift(1)
    rolling_elo_current_dfs.append(team_df[['gameId', 'team', 'elo_rolling']])

df_rolling_elo_current = pd.concat(rolling_elo_current_dfs)

# Merge to df_current
df_current = df_current.merge(df_rolling_elo_current, left_on=['id', 'homeTeam'], right_on=['gameId', 'team'], suffixes=('', '_home_elo_roll'), how='left')
df_current = df_current.merge(df_rolling_elo_current, left_on=['id', 'awayTeam'], right_on=['gameId', 'team'], suffixes=('', '_away_elo_roll'), how='left')
df_current['home_elo_rolling'] = df_current['elo_rolling']
df_current['away_elo_rolling'] = df_current['elo_rolling_away_elo_roll']
df_current.drop(['gameId_home_elo_roll', 'team_home_elo_roll', 'elo_rolling', 'gameId_away_elo_roll', 'team_away_elo_roll', 'elo_rolling_away_elo_roll'], axis=1, inplace=True, errors='ignore')

# Engineer rolling for current (added more stats)
teams_current = df_current_advanced['team'].unique()
rolling_dfs_current = []

for team in teams_current:
    team_df = df_current_advanced[df_current_advanced['team'] == team].sort_values(['season', 'week'])
    rolling_stats = team_df[['off_ppa', 'off_successRate', 'off_explosiveness', 'off_havocTotal', 'off_pointsPerOpportunity',
                             'def_ppa', 'def_successRate', 'def_explosiveness', 'def_havocTotal', 'def_pointsPerOpportunity']].rolling(window=5, min_periods=1).mean().shift(1)
    rolling_stats.columns = [f"{col}_rolling" for col in rolling_stats.columns]
    rolling_stats['gameId'] = team_df['gameId']
    rolling_stats['team'] = team
    rolling_dfs_current.append(rolling_stats)

df_rolling_current = pd.concat(rolling_dfs_current)

# Merge to current games
df_current = df_current.merge(df_rolling_current, left_on=['id', 'homeTeam'], right_on=['gameId', 'team'], suffixes=('', '_home'), how='left')
df_current = df_current.merge(df_rolling_current, left_on=['id', 'awayTeam'], right_on=['gameId', 'team'], suffixes=('', '_away'), how='left')
df_current.drop(['gameId_home', 'team_home', 'gameId_away', 'team_away'], axis=1, inplace=True, errors='ignore')

# Additional features for current (added diffs for new stats and opponent-adjusted)
df_current['home_adv'] = 1
df_current['sp_diff'] = df_current['home_sp'] - df_current['away_sp']
df_current['talent_diff'] = df_current['home_talent'] - df_current['away_talent']
df_current['fpi_diff'] = df_current['home_fpi'] - df_current['away_fpi']
df_current['elo_season_diff'] = df_current['home_elo_season'] - df_current['away_elo_season']
df_current['elo_rolling_diff'] = df_current['home_elo_rolling'] - df_current['away_elo_rolling']
df_current['off_ppa_diff'] = df_current['off_ppa_rolling'] - df_current['def_ppa_rolling_away']
df_current['def_ppa_diff'] = df_current['def_ppa_rolling'] - df_current['off_ppa_rolling_away']
df_current['off_success_diff'] = df_current['off_successRate_rolling'] - df_current['def_successRate_rolling_away']
df_current['def_success_diff'] = df_current['def_successRate_rolling'] - df_current['off_successRate_rolling_away']
df_current['off_explo_diff'] = df_current['off_explosiveness_rolling'] - df_current['def_explosiveness_rolling_away']
df_current['def_explo_diff'] = df_current['def_explosiveness_rolling'] - df_current['off_explosiveness_rolling_away']
df_current['off_havoc_diff'] = df_current['off_havocTotal_rolling'] - df_current['def_havocTotal_rolling_away']
df_current['def_havoc_diff'] = df_current['def_havocTotal_rolling'] - df_current['off_havocTotal_rolling_away']
df_current['off_ppo_diff'] = df_current['off_pointsPerOpportunity_rolling'] - df_current['def_pointsPerOpportunity_rolling_away']
df_current['def_ppo_diff'] = df_current['def_pointsPerOpportunity_rolling'] - df_current['off_pointsPerOpportunity_rolling_away']

# Opponent-adjusted diffs for current
df_current['adj_off_epa_diff'] = df_current['home_epa'] - df_current['away_epa_allowed']
df_current['adj_def_epa_diff'] = df_current['home_epa_allowed'] - df_current['away_epa']
df_current['adj_off_success_diff'] = df_current['home_successRate'] - df_current['away_successRate_allowed']
df_current['adj_def_success_diff'] = df_current['home_successRate_allowed'] - df_current['away_successRate']

# For weather, fetch current weather if available, else skip or use averages
current_weather = get_data("/games/weather", {"year": 2025, "week": current_week})
df_current_weather = pd.DataFrame(current_weather)
df_current = df_current.merge(df_current_weather[['id', 'temperature', 'windSpeed', 'precipitation']], left_on='id', right_on='id', how='left')
df_current['is_rainy'] = (df_current['precipitation'] > 0).astype(int)
df_current['wind_high'] = (df_current['windSpeed'] > 10).astype(int)

# Fill NaNs for features (e.g., mean)
df_current[features] = df_current[features].fillna(df_train[features].mean())

X_current = df_current[features]

# Predict
df_current['predicted_margin'] = margin_model.predict(X_current)
df_current['predicted_total'] = total_model.predict(X_current)

# Step 7: Generate Picks
# Spread pick: Home covers if predicted_margin > spread (since spread negative for home fav)
def get_spread_pick(row):
    predicted = row['predicted_margin']
    spread = row['spread']
    if math.isnan(spread):
        return 'No line'
    # Spread is home spread, e.g., -7 for home fav
    if predicted > -spread:  # If home margin > expected (more positive)
        fav = row['homeTeam'] if spread < 0 else row['awayTeam']
        points = abs(spread)
        return f"{fav} -{points}"
    else:
        dog = row['awayTeam'] if spread < 0 else row['homeTeam']
        points = abs(spread)
        return f"{dog} +{points}"

df_current['spread_pick'] = df_current.apply(get_spread_pick, axis=1)

# Moneyline pick: Pick the predicted winner if value (simple threshold for ROI)
def get_ml_pick(row):
    predicted = row['predicted_margin']
    home_ml = row['homeMoneyline']
    away_ml = row['awayMoneyline']
    if math.isnan(home_ml) or math.isnan(away_ml):
        return 'No line'
    if predicted > 0:
        winner = row['homeTeam']
        ml = home_ml
    else:
        winner = row['awayTeam']
        ml = away_ml
    # For +ROI, add value check: estimate prob from margin (simple logistic)
    est_prob = 1 / (1 + np.exp(-predicted / 10))  # Rough, scale by std dev ~14 for CFB
    implied_prob = 100 / (100 + ml) if ml > 0 else -ml / (-ml + 100)
    if est_prob > implied_prob + 0.02:  # Edge threshold
        return f"{winner} ML ({ml})"
    return 'No value'

df_current['ml_pick'] = df_current.apply(get_ml_pick, axis=1)

# O/U pick with value
def get_ou_pick(row):
    predicted = row['predicted_total']
    ou = row['overUnder']
    if math.isnan(ou):
        return 'No line'
    diff = predicted - ou
    if abs(diff) > 2:  # Threshold for value
        return 'Over' if diff > 0 else 'Under'
    return 'No value'

df_current['ou_pick'] = df_current.apply(get_ou_pick, axis=1)

# Confidence
df_current['spread_conf'] = abs(df_current['predicted_margin'] + df_current['spread'])  # Distance from line (home view)
df_current['ou_conf'] = abs(df_current['predicted_total'] - df_current['overUnder'])
df_current['ml_conf'] = abs(df_current['predicted_margin']) / 10  # Scaled

# Step 8: Output to CSV
output_columns = ['id', 'homeTeam', 'awayTeam', 'spread_pick', 'ml_pick', 'ou_pick', 'spread_conf', 'ou_conf', 'ml_conf', 'predicted_margin', 'predicted_total']
df_current[output_columns].to_csv('college_football_picks.csv', index=False)
print("Picks saved to college_football_picks.csv")
