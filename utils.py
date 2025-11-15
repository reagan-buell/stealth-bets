# utils.py

from config import requests, BASE_URL, HEADERS, np, math, pd, features

# Function to fetch data from CFBD API
def get_data(endpoint, params=None):
    if params is None:
        params = {}
    response = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=HEADERS)
    response.raise_for_status()
    return response.json()

# Extract betting lines
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

# Compute rolling Elo
def compute_rolling_elo(df_games, all_teams):
    team_elo_dfs = []
    for team in all_teams:
        home_games = df_games[df_games['homeTeam'] == team][['season', 'week', 'id', 'homePregameElo']].rename(columns={'homePregameElo': 'pregame_elo', 'id': 'gameId'})
        away_games = df_games[df_games['awayTeam'] == team][['season', 'week', 'id', 'awayPregameElo']].rename(columns={'awayPregameElo': 'pregame_elo', 'id': 'gameId'})
        team_elo = pd.concat([home_games, away_games]).sort_values(['season', 'week'])
        team_elo['team'] = team
        team_elo_dfs.append(team_elo)

    df_team_elo = pd.concat(team_elo_dfs)

    rolling_elo_dfs = []
    for team in all_teams:
        team_df = df_team_elo[df_team_elo['team'] == team].sort_values(['season', 'week'])
        team_df['elo_rolling'] = team_df['pregame_elo'].rolling(window=5, min_periods=1).mean().shift(1)
        rolling_elo_dfs.append(team_df[['gameId', 'team', 'elo_rolling']])

    df_rolling_elo = pd.concat(rolling_elo_dfs)
    return df_rolling_elo

# Compute rolling advanced stats
def compute_rolling_advanced(df_advanced, teams):
    rolling_dfs = []
    for team in teams:
        team_df = df_advanced[df_advanced['team'] == team].sort_values(['season', 'week'])
        rolling_stats = team_df[['off_ppa', 'off_successRate', 'off_explosiveness', 'off_havocTotal', 'off_pointsPerOpportunity',
                                 'def_ppa', 'def_successRate', 'def_explosiveness', 'def_havocTotal', 'def_pointsPerOpportunity']].rolling(window=5, min_periods=1).mean().shift(1)
        rolling_stats.columns = [f"{col}_rolling" for col in rolling_stats.columns]
        rolling_stats['gameId'] = team_df['gameId']
        rolling_stats['team'] = team
        rolling_dfs.append(rolling_stats)

    df_rolling = pd.concat(rolling_dfs)
    return df_rolling

# Engineer features
def engineer_features(df, df_train=None):
    df['home_adv'] = 1
    df['sp_diff'] = df['home_sp'] - df['away_sp']
    df['talent_diff'] = df['home_talent'] - df['away_talent']
    df['fpi_diff'] = df['home_fpi'] - df['away_fpi']
    df['elo_season_diff'] = df['home_elo_season'] - df['away_elo_season']
    df['elo_rolling_diff'] = df['home_elo_rolling'] - df['away_elo_rolling']
    df['off_ppa_diff'] = df['off_ppa_rolling'] - df['def_ppa_rolling_away']
    df['def_ppa_diff'] = df['def_ppa_rolling'] - df['off_ppa_rolling_away']
    df['off_success_diff'] = df['off_successRate_rolling'] - df['def_successRate_rolling_away']
    df['def_success_diff'] = df['def_successRate_rolling'] - df['off_successRate_rolling_away']
    df['off_explo_diff'] = df['off_explosiveness_rolling'] - df['def_explosiveness_rolling_away']
    df['def_explo_diff'] = df['def_explosiveness_rolling'] - df['off_explosiveness_rolling_away']
    df['off_havoc_diff'] = df['off_havocTotal_rolling'] - df['def_havocTotal_rolling_away']
    df['def_havoc_diff'] = df['def_havocTotal_rolling'] - df['off_havocTotal_rolling_away']
    df['off_ppo_diff'] = df['off_pointsPerOpportunity_rolling'] - df['def_pointsPerOpportunity_rolling_away']
    df['def_ppo_diff'] = df['def_pointsPerOpportunity_rolling'] - df['off_pointsPerOpportunity_rolling_away']

    df['adj_off_epa_diff'] = df['home_epa'] - df['away_epa_allowed']
    df['adj_def_epa_diff'] = df['home_epa_allowed'] - df['away_epa']
    df['adj_off_success_diff'] = df['home_successRate'] - df['away_successRate_allowed']
    df['adj_def_success_diff'] = df['home_successRate_allowed'] - df['away_successRate']

    df['is_rainy'] = (df['precipitation'] > 0).astype(int)
    df['wind_high'] = (df['windSpeed'] > 10).astype(int)

    # Fill NaNs
    if df_train is not None:
        df[features] = df[features].fillna(df_train[features].mean())
    else:
        df[features] = df[features].fillna(df[features].mean())

    return df
