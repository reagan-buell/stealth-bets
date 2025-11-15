# fetch_historical.py

from utils import get_data
from config import pd

def fetch_historical_data():
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
        
        # Elo ratings
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

    return df_games, df_lines, df_advanced, df_weather, df_sp, df_talent, df_wepa, df_fpi, df_elo
