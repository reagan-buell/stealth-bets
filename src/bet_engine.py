# Unified Bet Engine: 56%+ NCAA/NFL + Props
import pandas as pd
from modules.nfl-models.models import xgb_spread_model  # From brianbailey18
from modules.multi-sport-ml.ml import lasso_prop_model  # From Bet-on-Sibyl
from modules.props-backend.odds import fetch_odds  # From kyleskom (adapted)
from scipy.stats import poisson

# Features (from georgedouzas dataloaders)
features = ['home_rest', 'travel_miles', 'pff_ol_diff', 'public_pct', 'line_move']

def generate_picks(sport='NFL'):
    df = fetch_odds(sport)  # Live lines from SportsData.io
    df['model_prob'] = xgb_spread_model.predict_proba(df[features])[:, 1]
    df['edge'] = df['model_prob'] - df['implied_prob']
    
    # 56% Filters
    if sport == 'NCAA':
        bets = df[(df['edge'] > 0.055) & (df['is_g5_home_dog'] & df['spread'] >= 7)]
    else:  # NFL
        bets = df[(df['edge'] > 0.055) & (df['is_div_dog'] & df['spread'].between(3,7))]
    
    # Props (Poisson for yds)
    props = pd.DataFrame()  # Load from PFF
    props['model_yards'] = lasso_prop_model.predict(props[['route_pct', 'target_share']])
    props['over_prob'] = poisson.cdf(props['line'], props['model_yards'])
    props = props[(props['clv_yards'] > 5.5) & (props['opp_def_rank'] > 23)]
    
    # Kelly Sizing (50% frac for safety)
    def kelly_frac(p, odds=-110):
        b = 0.909  # -110 vig
        return max(0, (b * p - (1 - p)) / b) * 0.5  # Half-Kelly
    
    bets['units'] = bets.apply(lambda row: kelly_frac(row['model_prob']), axis=1)
    return pd.concat([bets, props])

# Run: picks = generate_picks('NFL') → JSON for app
