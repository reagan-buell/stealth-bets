# config.py

import requests
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from joblib import dump, load
import os
import math

# API Configuration
API_KEY = "krRqEYZn0h4HQRNxo1PU/cPid/NMi9Ki1FJb6ajILhCdt9/ScLi1FNaCssp6GH+A"
BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Features list (shared across modules)
features = ['home_adv', 'sp_diff', 'talent_diff', 'fpi_diff', 'elo_season_diff', 'elo_rolling_diff', 'off_ppa_diff', 'def_ppa_diff', 
            'off_success_diff', 'def_success_diff', 'off_explo_diff', 'def_explo_diff',
            'off_havoc_diff', 'def_havoc_diff', 'off_ppo_diff', 'def_ppo_diff',
            'adj_off_epa_diff', 'adj_def_epa_diff', 'adj_off_success_diff', 'adj_def_success_diff',
            'temperature', 'wind_high', 'is_rainy']

# File paths
historical_file = 'historical_data.csv'
margin_model_file = 'margin_model.joblib'
total_model_file = 'total_model.joblib'
picks_file = 'college_football_picks.csv'
