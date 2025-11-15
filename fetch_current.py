# fetch_current.py

from utils import get_data
from config import pd

def fetch_current_data(current_week):
    current_games = get_data("/games", {"year": 2025, "week": current_week, "seasonType": "regular"})
    df_current = pd.DataFrame(current_games)

    current_lines = get_data("/lines", {"year": 2025, "week": current_week})
    df_current_lines = pd.DataFrame(current_lines)

    current_advanced = get_data("/stats/game/advanced", {"year": 2025})
    df_current_advanced = pd.DataFrame(current_advanced)

    wepa_2025 = get_data("/wepa/team/season", {"year": 2025})
    if not wepa_2025:
        wepa_2025 = get_data("/wepa/team/season", {"year": 2024})
    df_wepa_current = pd.DataFrame(wepa_2025)

    fpi_2025 = get_data("/ratings/fpi", {"year": 2025})
    if not fpi_2025:
        fpi_2025 = get_data("/ratings/fpi", {"year": 2024})
    df_fpi_current = pd.DataFrame(fpi_2025)

    elo_2025 = get_data("/ratings/elo", {"year": 2025})
    if not elo_2025:
        elo_2025 = get_data("/ratings/elo", {"year": 2024})
    df_elo_current = pd.DataFrame(elo_2025)

    sp_2025 = get_data("/ratings/sp", {"year": 2025})
    if not sp_2025:
        sp_2025 = get_data("/ratings/sp", {"year": 2024})
    df_sp_current = pd.DataFrame(sp_2025)

    talent_2025 = get_data("/talent", {"year": 2025})
    if not talent_2025:
        talent_2025 = get_data("/talent", {"year": 2024})
    df_talent_current = pd.DataFrame(talent_2025)

    current_weather = get_data("/games/weather", {"year": 2025, "week": current_week})
    df_current_weather = pd.DataFrame(current_weather)

    current_all_games = get_data("/games", {"year": 2025, "seasonType": "regular"})
    df_current_all_games = pd.DataFrame(current_all_games)

    return df_current, df_current_lines, df_current_advanced, df_wepa_current, df_fpi_current, df_elo_current, df_sp_current, df_talent_current, df_current_weather, df_current_all_games
