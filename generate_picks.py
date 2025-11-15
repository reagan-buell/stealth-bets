# generate_picks.py

from config import load, margin_model_file, total_model_file, picks_file, features, np, math, pd

def generate_picks(df_current):
    margin_model = load(margin_model_file)
    total_model = load(total_model_file)

    X_current = df_current[features]
    df_current['predicted_margin'] = margin_model.predict(X_current)
    df_current['predicted_total'] = total_model.predict(X_current)

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

    df_current['spread_pick'] = df_current.apply(get_spread_pick, axis=1)

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

    df_current['ml_pick'] = df_current.apply(get_ml_pick, axis=1)

    def get_ou_pick(row):
        predicted = row['predicted_total']
        ou = row['overUnder']
        if math.isnan(ou):
            return 'No line'
        diff = predicted - ou
        if abs(diff) > 2:
            return 'Over' if diff > 0 else 'Under'
        return 'No value'

    df_current['ou_pick'] = df_current.apply(get_ou_pick, axis=1)

    df_current['spread_conf'] = abs(df_current['predicted_margin'] + df_current['spread'])
    df_current['ou_conf'] = abs(df_current['predicted_total'] - df_current['overUnder'])
    df_current['ml_conf'] = abs(df_current['predicted_margin']) / 10

    output_columns = ['id', 'homeTeam', 'awayTeam', 'spread_pick', 'ml_pick', 'ou_pick', 'spread_conf', 'ou_conf', 'ml_conf', 'predicted_margin', 'predicted_total']
    df_current[output_columns].to_csv(picks_file, index=False)
    print("Picks saved to college_football_picks.csv")
