# stealth_picks.py (master file)

from config import date, datetime, timedelta, os, historical_file, pd
from utils import get_data
from fetch_historical import fetch_historical_data
from process_historical import process_historical_data
from train_models import train_models
from fetch_current import fetch_current_data
from process_current import process_current_data
from generate_picks import generate_picks

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

# Check if historical data is recent (last 7 days)
use_cached = False
if os.path.exists(historical_file):
    mtime = datetime.fromtimestamp(os.path.getmtime(historical_file))
    if datetime.now() - mtime < timedelta(days=7):
        print("Using cached historical data.")
        df_games = pd.read_csv(historical_file)
        use_cached = True

if not use_cached:
    print("Fetching new historical data.")
    df_games, df_lines, df_advanced, df_weather, df_sp, df_talent, df_wepa, df_fpi, df_elo = fetch_historical_data()
    df_games = process_historical_data(df_games, df_lines, df_advanced, df_weather, df_sp, df_talent, df_wepa, df_fpi, df_elo)
    df_games.to_csv(historical_file, index=False)

# Train models (always train or load if exists, but here we train if not cached or always)
train_models(df_games)

# Fetch and process current
df_current, df_current_lines, df_current_advanced, df_wepa_current, df_fpi_current, df_elo_current, df_sp_current, df_talent_current, df_current_weather, df_current_all_games = fetch_current_data(current_week)
df_current = process_current_data(df_current, df_current_lines, df_current_advanced, df_wepa_current, df_fpi_current, df_elo_current, df_sp_current, df_talent_current, df_current_weather, df_current_all_games, df_games)

# Generate picks
generate_picks(df_current)
