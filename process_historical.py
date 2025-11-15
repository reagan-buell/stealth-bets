# process_historical.py

from config import np, pd
from utils import extract_line, compute_rolling_elo, compute_rolling_advanced, engineer_features

def process_historical_data(df_games, df_lines, df_advanced, df_weather, df_sp, df_talent, df_wepa, df_fpi, df_elo):
    # Expand WEPA dicts
    df_wepa['epa_total'] = df_wepa['epa'].apply(lambda x: x.get('total', np.nan) if isinstance(x, dict) else np.nan)
    df_wepa['epaAllowed_total'] = df_wepa['epaAllowed'].apply(lambda x: x.get('total', np.nan) if isinstance(x, dict) else np.nan)
    df_wepa['successRate_total'] = df_wepa['successRate'].apply(lambda x: x.get('total', np.nan) if isinstance(x, dict) else np.nan)
    df_wepa['successRateAllowed_total'] = df_wepa['successRateAllowed'].apply(lambda x: x.get('total', np.nan) if isinstance(x, dict) else np.nan)

    # Merge games and lines on 'id'
    df_games = df_games.merge(df_lines[['id', 'lines']], on='id', how='left')

    # Extract betting lines
    df_games = df_games.join(df_games.apply(extract_line, axis=1))

    # Parse spread to numeric
    df_games['spread'] = pd.to_numeric(df_games['spread'], errors='coerce')

    # Add targets
    df_games['actual_margin'] = df_games['homePoints'] - df_games['awayPoints']
    df_games['actual_total'] = df_games['homePoints'] + df_games['awayPoints']

    # Merge weather
    df_games = df_games.merge(df_weather[['id', 'temperature', 'windSpeed', 'precipitation']], left_on='id', right_on='id', how='left')

    # Merge SP+ and talent (shift year +1 to avoid lookahead)
    df_sp['year'] += 1
    df_talent['year'] += 1
    df_games = df_games.merge(df_sp[['year', 'team', 'rating']], left_on=['season', 'homeTeam'], right_on=['year', 'team'], how='left').rename(columns={'rating': 'home_sp'})
    df_games = df_games.merge(df_sp[['year', 'team', 'rating']], left_on=['season', 'awayTeam'], right_on=['year', 'team'], how='left').rename(columns={'rating': 'away_sp'})
    df_games.drop(['year_x', 'team_x', 'year_y', 'team_y'], axis=1, inplace=True, errors='ignore')

    df_games = df_games.merge(df_talent[['year', 'team', 'talent']], left_on=['season', 'homeTeam'], right_on=['year', 'team'], how='left').rename(columns={'talent': 'home_talent'})
    df_games = df_games.merge(df_talent[['year', 'team', 'talent']], left_on=['season', 'awayTeam'], right_on=['year', 'team'], how='left').rename(columns={'talent': 'away_talent'})
    df_games.drop(['year_x', 'team_x', 'year_y', 'team_y'], axis=1, inplace=True, errors='ignore')

    # Merge WEPA (shift year +1)
    df_wepa['year'] += 1
    df_games = df_games.merge(df_wepa[['year', 'team', 'epa_total', 'epaAllowed_total', 'successRate_total', 'successRateAllowed_total']], left_on=['season', 'homeTeam'], right_on=['year', 'team'], how='left')
    df_games = df_games.rename(columns={'epa_total': 'home_epa', 'epaAllowed_total': 'home_epa_allowed', 'successRate_total': 'home_successRate', 'successRateAllowed_total': 'home_successRate_allowed'})
    df_games = df_games.merge(df_wepa[['year', 'team', 'epa_total', 'epaAllowed_total', 'successRate_total', 'successRateAllowed_total']], left_on=['season', 'awayTeam'], right_on=['year', 'team'], how='left')
    df_games = df_games.rename(columns={'epa_total': 'away_epa', 'epaAllowed_total': 'away_epa_allowed', 'successRate_total': 'away_successRate', 'successRateAllowed_total': 'away_successRate_allowed'})
    df_games.drop(['year_x', 'team_x', 'year_y', 'team_y'], axis=1, inplace=True, errors='ignore')

    # Merge FPI (shift year +1)
    df_fpi['year'] += 1
    df_games = df_games.merge(df_fpi[['year', 'team', 'fpi']], left_on=['season', 'homeTeam'], right_on=['year', 'team'], how='left').rename(columns={'fpi': 'home_fpi'})
    df_games = df_games.merge(df_fpi[['year', 'team', 'fpi']], left_on=['season', 'awayTeam'], right_on=['year', 'team'], how='left').rename(columns={'fpi': 'away_fpi'})
    df_games.drop(['year_x', 'team_x', 'year_y', 'team_y'], axis=1, inplace=True, errors='ignore')

    # Merge Elo (shift year +1)
    df_elo['year'] += 1
    df_games = df_games.merge(df_elo[['year', 'team', 'elo']], left_on=['season', 'homeTeam'], right_on=['year', 'team'], how='left').rename(columns={'elo': 'home_elo_season'})
    df_games = df_games.merge(df_elo[['year', 'team', 'elo']], left_on=['season', 'awayTeam'], right_on=['year', 'team'], how='left').rename(columns={'elo': 'away_elo_season'})
    df_games.drop(['year_x', 'team_x', 'year_y', 'team_y'], axis=1, inplace=True, errors='ignore')

    # Rolling Elo
    all_teams = pd.unique(df_games[['homeTeam', 'awayTeam']].values.ravel('K'))
    df_rolling_elo = compute_rolling_elo(df_games, all_teams)
    df_games = df_games.merge(df_rolling_elo, left_on=['id', 'homeTeam'], right_on=['gameId', 'team'], how='left').rename(columns={'elo_rolling': 'home_elo_rolling'})
    df_games = df_games.merge(df_rolling_elo, left_on=['id', 'awayTeam'], right_on=['gameId', 'team'], how='left').rename(columns={'elo_rolling': 'away_elo_rolling'})
    df_games.drop(['gameId_x', 'team_x', 'gameId_y', 'team_y'], axis=1, inplace=True, errors='ignore')

    # Expand advanced
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

    # Rolling advanced
    teams = df_advanced['team'].unique()
    df_rolling = compute_rolling_advanced(df_advanced, teams)
    df_games = df_games.merge(df_rolling, left_on=['id', 'homeTeam'], right_on=['gameId', 'team'], how='left')
    df_games = df_games.merge(df_rolling, left_on=['id', 'awayTeam'], right_on=['gameId', 'team'], how='left', suffixes=('', '_away'))
    df_games.drop(['gameId', 'team', 'gameId_away', 'team_away'], axis=1, inplace=True, errors='ignore')

    # Engineer features
    df_games = engineer_features(df_games)

    return df_games
