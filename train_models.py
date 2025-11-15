# train_models.py

from config import features, train_test_split, mean_absolute_error, XGBRegressor, dump, margin_model_file, total_model_file, pd, np
import matplotlib.pyplot as plt  # For plotting feature importance

def train_models(df_games, plot_importance=True):  # Added param for optional plotting
    df_train = df_games.dropna(subset=['actual_margin', 'actual_total', 'spread', 'overUnder'])

    X = df_train[features]
    y_margin = df_train['actual_margin']
    y_total = df_train['actual_total']

    X_train, X_test, y_margin_train, y_margin_test = train_test_split(X, y_margin, test_size=0.2, random_state=42)
    margin_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    margin_model.fit(X_train, y_margin_train)
    print(f"Margin MAE: {mean_absolute_error(y_margin_test, margin_model.predict(X_test))}")

    X_train, X_test, y_total_train, y_total_test = train_test_split(X, y_total, test_size=0.2, random_state=42)
    total_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    total_model.fit(X_train, y_total_train)
    print(f"Total MAE: {mean_absolute_error(y_total_test, total_model.predict(X_test))}")

    # Save models
    dump(margin_model, margin_model_file)
    dump(total_model, total_model_file)

    # NEW: Feature Importance Reporting
    # Use 'gain' for importance (average gain from splits; more meaningful for predictions)
    def report_feature_importance(model, model_name, feature_names):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Print top 10 for console
        print(f"\nTop Feature Importances for {model_name} Model (Gain):")
        print(importance_df.head(10))
        
        # Save to CSV
        importance_df.to_csv(f'feature_importance_{model_name.lower()}.csv', index=False)
        print(f"Feature importance saved to feature_importance_{model_name.lower()}.csv")
        
        # Optional: Plot and save as PNG (for app dashboard)
        if plot_importance:
            plt.figure(figsize=(10, 8))
            plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
            plt.xlabel('Importance (Gain)')
            plt.title(f'Top 10 Features for {model_name} Model')
            plt.gca().invert_yaxis()  # Highest on top
            plt.savefig(f'feature_importance_{model_name.lower()}.png')
            plt.close()
            print(f"Plot saved to feature_importance_{model_name.lower()}.png")

    # Generate reports
    report_feature_importance(margin_model, 'Margin', features)
    report_feature_importance(total_model, 'Total', features)

    return margin_model, total_model
