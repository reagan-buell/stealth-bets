# process_current.py

from config import np, pd, features
from utils import extract_line, compute_rolling_elo, compute_rolling_advanced, engineer_features

def process_current_data(df_current, df_current_lines, df_current_advanced, df_wepa_current, df_fpi_current, df_elo_current, df_sp_current, df_talent_current, df_current_weather, df_current_all_games, df_train):
    df_current = df_current.merge(df_current_lines[['id', 'lines']], on='id', how='left')
    df_current = df_current.join(df_current.apply(extract_line, axis=1))
    df_current['spread'] = pd.to_numeric(df_current['spread'], errors='coerce')

    # Expand advanced
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

    # Expand WEPA
    df_wepa_current['epa_total'] = df_wepa_current['epa'].apply(lambda x: x.get('total', np.nan) if isinstance(x, dict) else np.nan)
    df_wepa_current['epaAllowed_total'] = df_wepa_current['epaAllowed'].apply(lambda x: x.get('total', np.nan) if isinstance(x, dict) else np.nan)
    df_wepa_current['successRate_total'] = df_wepa_current['successRate'].apply(lambda x: x.get('total', np.nan) if isinstance(x, dict) else np.nan)
    df_wepa_current['successRateAllowed_total'] = df_wepa_current['successRateAllowed'].apply(lambda x: x.get('total', np.nan) if isinstance(x, dict) else np.nan)

    # Merge SP+
    df_current = df_current.merge(df_sp_current[['team', 'rating']], left_on='homeTeam', right_on='team', how='left').rename(columns={'rating': 'home_sp'}).drop('team', axis=1)
    df_current = df_current.merge(df_sp_current[['team', 'rating']], left_on='awayTeam', right_on='team', how='left').rename(columns={'rating': 'away_sp'}).drop('team', axis=1)

    # Merge talent
    df_current = df_current.merge(df_talent_current[['team', 'talent']], left_on='homeTeam', right_on='team', how='left').rename(columns={'talent': 'home_talent'}).drop('team', axis=1)
    df_current = df_current.merge(df_talent_current[['team', 'talent']], left_on='awayTeam', right_on='team', how='left').rename(columns={'talent': 'away_talent'}).drop('team', axis=1)

    # Merge WEPA
    df_current = df_current.merge(df_wepa_current[['team', 'epa_total', 'epaAllowed_total', 'successRate_total', 'successRateAllowed_total']], left_on='homeTeam', right_on='team', how='left')
    df_current = df_current.rename(columns={'epa_total': 'home_epa', 'epaAllowed_total': 'home_epa_allowed', 'successRate_total': 'home_successRate', 'successRateAllowed_total': 'home_successRate_allowed'}).drop('team', axis=1)
    df_current = df_current.merge(df_wepa_current[['team', 'epa_total', 'epaAllowed_total', 'successRate_total', 'successRateAllowed_total']], left_on='awayTeam', right_on='team', how='left')
    df_current = df_current.rename(columns={'epa_total': 'away_epa', 'epaAllowed_total': 'away_epa_allowed', 'successRate_total': 'away_successRate', 'successRateAllowed_total': 'away_successRate_allowed'}).drop('team', axis=1)

    # Merge FPI
    df_current = df_current.merge(df_fpi_current[['team', 'fpi']], left_on='homeTeam', right_on='team', how='left').rename(columns={'fpi': 'home_fpi'}).drop('team', axis=1)
    df_current = df_current.merge(df_fpi_current[['team', 'fpi']], left_on='awayTeam', right_on='team', how='left').rename(columns={'fpi': 'away_fpi'}).drop('team', axis=1)

    # Merge Elo
    df_current = df_current.merge(df_elo_current[['team', 'elo']], left_on='homeTeam', right_on='team', how='left').rename(columns={'elo': 'home_elo_season'}).drop('team', axis=1)
    df_current = df_current.merge(df_elo_current[['team', 'elo']], left_on='awayTeam', right_on='team', how='left').rename(columns={'elo': 'away_elo_season'}).drop('team', axis=1)

    # Rolling Elo
    all_teams_current = pd.unique(df_current_all_games[['homeTeam', 'awayTeam']].values.ravel('K'))
    df_rolling_elo_current = compute_rolling_elo(df_current_all_games, all_teams_current)
    df_current = df_current.merge(df_rolling_elo_current, left_on=['id', 'homeTeam'], right_on=['gameId', 'team'], how='left').rename(columns={'elo_rolling': 'home_elo_rolling'}).drop(['gameId', 'team'], axis=1)
    df_current = df_current.merge(df_rolling_elo_current, left_on=['id', 'awayTeam'], right_on=['gameId', 'team'], how='left').rename(columns={'elo_rolling': 'away_elo_rolling'}).drop(['gameId', 'team'], axis=1)

    # Rolling advanced
    teams_current = df_current_advanced['team'].unique()
    df_rolling_current = compute_rolling_advanced(df_current_advanced, teams_current)
    df_current = df_current.merge(df_rolling_current, left_on=['id', 'homeTeam'], right_on=['gameId', 'team'], how='left').drop(['gameId', 'team'], axis=1, errors='ignore')
    df_current = df_current.merge(df_rolling_current, left_on=['id', 'awayTeam'], right_on=['gameId', 'team'], how='left', suffixes=('', '_away')).drop(['gameId', 'team'], axis=1, errors='ignore')

    # Merge weather
    df_current = df_current.merge(df_current_weather[['id', 'temperature', 'windSpeed', 'precipitation']], left_on='id', right_on='id', how='left')

    # Engineer features
    df_current = engineer_features(df_current, df_train)

    return df_current
