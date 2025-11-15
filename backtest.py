# backtest.py

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from config import features, historical_file
# Assuming utils and other imports if needed, but for backtest, we retrain anyway

# Pick functions (from generate_picks, but defined here for completeness)
def get_spread_pick(row):
    predicted = row['predicted_margin']
    spread = row['spread']
    if math.isnan(spread):
        return 'No line'
    if predicted > -spread:
        fav = row['homeTeam'] if spread < 0 else row['awayTeam']
        points = abs(spread)
        return f"{fav} -{points}"
    else:
        dog = row['awayTeam'] if spread < 0 else row['homeTeam']
        points = abs(spread)
        return f"{dog} +{points}"

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
    est_prob = 1 / (1 + np.exp(-predicted / 10))
    implied_prob = 100 / (100 + ml) if ml > 0 else -ml / (-ml + 100)
    if est_prob > implied_prob + 0.02:
        return f"{winner} ML ({ml})"
    return 'No value'

def get_ou_pick(row):
    predicted = row['predicted_total']
    ou = row['overUnder']
    if math.isnan(ou):
        return 'No line'
    diff = predicted - ou
    if abs(diff) > 2:
        return 'Over' if diff > 0 else 'Under'
    return 'No value'

# Simulate win functions
def simulate_spread_win(row):
    spread = row['spread']
    if math.isnan(spread):
        return None
    predicted = row['predicted_margin']
    actual = row['actual_margin']
    picked_home_to_cover = predicted > -spread
    if actual == -spread:
        return 'push'
    actual_home_covers = actual > -spread
    if picked_home_to_cover:
        return actual_home_covers
    else:
        return not actual_home_covers

def simulate_ou_win(row):
    ou = row['overUnder']
    if math.isnan(ou):
        return None
    predicted = row['predicted_total']
    actual = row['actual_total']
    picked_over = row['ou_pick'] == 'Over'
    if actual == ou:
        return 'push'
    actual_over = actual > ou
    return picked_over == actual_over

def simulate_ml_win(row):
    if 'No' in row['ml_pick']:
        return None
    predicted = row['predicted_margin']
    actual = row['actual_margin']
    picked_home = predicted > 0
    actual_home_win = actual > 0  # Assuming no ties
    return picked_home == actual_home_win

def calculate_ml_payout(ml, win):
    if not win:
        return -1
    if ml > 0:
        return ml / 100
    else:
        return 100 / -ml

def backtest(threshold=0.0, bet_types=['spread', 'ou', 'ml'], vig=-110):
    # Calculate payout for spread and ou (assuming -110, payout ~0.909)
    if vig < 0:
        payout = 100 / abs(vig)
    else:
        payout = vig / 100

    # Load historical data
    df_games = pd.read_csv(historical_file)

    # Get unique years
    years = sorted(df_games['season'].unique())

    # Overall stats
    overall_bets = 0
    overall_wins = 0
    overall_pushes = 0
    overall_profit = 0.0
    yearly_results = []

    for test_year in years[1:]:  # Start from second year to have training data
        # Train on previous years
        train_df = df_games[df_games['season'] < test_year].dropna(subset=['actual_margin', 'actual_total', 'spread', 'overUnder'])
        test_df = df_games[df_games['season'] == test_year].copy()

        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[features]
        y_margin_train = train_df['actual_margin']
        y_total_train = train_df['actual_total']

        margin_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        margin_model.fit(X_train, y_margin_train)

        total_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        total_model.fit(X_train, y_total_train)

        # Predict on test
        X_test = test_df[features]
        test_df['predicted_margin'] = margin_model.predict(X_test)
        test_df['predicted_total'] = total_model.predict(X_test)

        # Get picks
        test_df['spread_pick'] = test_df.apply(get_spread_pick, axis=1)
        test_df['ml_pick'] = test_df.apply(get_ml_pick, axis=1)
        test_df['ou_pick'] = test_df.apply(get_ou_pick, axis=1)

        # Get confidences
        test_df['spread_conf'] = abs(test_df['predicted_margin'] + test_df['spread'])  # Edge magnitude
        test_df['ou_conf'] = abs(test_df['predicted_total'] - test_df['overUnder'])
        test_df['ml_conf'] = abs(test_df['predicted_margin']) / 10

        # Year stats
        year_bets = 0
        year_wins = 0
        year_pushes = 0
        year_profit = 0.0

        for _, row in test_df.iterrows():
            # Spread bet
            if 'spread' in bet_types and row['spread_pick'] != 'No line' and row['spread_conf'] > threshold:
                outcome = simulate_spread_win(row)
                if outcome is not None:
                    year_bets += 1
                    if outcome == 'push':
                        year_pushes += 1
                        profit = 0
                    else:
                        profit = payout if outcome else -1
                        if outcome:
                            year_wins += 1
                    year_profit += profit

            # O/U bet
            if 'ou' in bet_types and row['ou_pick'] in ['Over', 'Under'] and row['ou_conf'] > threshold:
                outcome = simulate_ou_win(row)
                if outcome is not None:
                    year_bets += 1
                    if outcome == 'push':
                        year_pushes += 1
                        profit = 0
                    else:
                        profit = payout if outcome else -1
                        if outcome:
                            year_wins += 1
                    year_profit += profit

            # ML bet
            if 'ml' in bet_types and 'No' not in row['ml_pick'] and row['ml_conf'] > threshold:
                outcome = simulate_ml_win(row)
                if outcome is not None:
                    year_bets += 1
                    # No push for ML
                    ml = row['homeMoneyline'] if row['predicted_margin'] > 0 else row['awayMoneyline']
                    profit = calculate_ml_payout(ml, outcome)
                    year_profit += profit
                    if outcome:
                        year_wins += 1

        # Calculate year metrics
        if year_bets > 0:
            graded_bets = year_bets - year_pushes
            year_win_rate = (year_wins / graded_bets * 100) if graded_bets > 0 else 0.0
            year_roi = (year_profit / year_bets) * 100
            yearly_results.append({
                'year': test_year,
                'bets': year_bets,
                'wins': year_wins,
                'pushes': year_pushes,
                'win_rate': year_win_rate,
                'roi': year_roi
            })
            overall_bets += year_bets
            overall_wins += year_wins
            overall_pushes += year_pushes
            overall_profit += year_profit

    # Overall metrics
    if overall_bets > 0:
        overall_graded = overall_bets - overall_pushes
        overall_win_rate = (overall_wins / overall_graded * 100) if overall_graded > 0 else 0.0
        overall_roi = (overall_profit / overall_bets) * 100
        print(f"Overall Bets: {overall_bets}, Wins: {overall_wins}, Pushes: {overall_pushes}")
        print(f"Overall Win Rate: {overall_win_rate:.2f}% (over graded bets)")
        print(f"Overall ROI: {overall_roi:.2f}% (profit per unit risked)")
        print("Yearly Results:")
        for res in yearly_results:
            print(f"Year {res['year']}: Bets={res['bets']}, Wins={res['wins']}, Pushes={res['pushes']}, Win Rate={res['win_rate']:.2f}%, ROI={res['roi']:.2f}%")
    else:
        print("No bets placed in backtest.")

    return overall_win_rate, overall_roi

if __name__ == "__main__":
    # To find a threshold that achieves ~53-60% win rate, loop over possible thresholds
    # Adjust the range as needed; here using 0 to 20 in steps of 0.5
    target_win_min = 53
    target_win_max = 60
    found_threshold = None
    for thresh in np.arange(0, 20.5, 0.5):
        print(f"\nTesting threshold: {thresh}")
        win_rate, roi = backtest(threshold=thresh, bet_types=['spread', 'ou', 'ml'], vig=-110)
        if target_win_min <= win_rate <= target_win_max:
            print(f"Found suitable threshold {thresh} with win rate {win_rate:.2f}% and ROI {roi:.2f}%")
            found_threshold = thresh
            break  # Or continue to find all, but stop at first for example

    if found_threshold is None:
        print("No threshold found in range that achieves 53-60% win rate. Adjust range or parameters.")
