import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape_val = np.mean(numerator / denominator) * 100
    return smape_val


def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv('sales_data.csv')
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_zero_values(df, check_targets=False):
    zero_count = (df['Units Sold'] == 0).sum()
    total_count = len(df)
    zero_percentage = (zero_count / total_count) * 100
    
    print(f"Total rows: {total_count}")
    print(f"Rows with Units Sold = 0: {zero_count} ({zero_percentage:.2f}%)")
    print(f"Rows with Units Sold > 0: {total_count - zero_count} ({100-zero_percentage:.2f}%)")
    
    # Check zero values per product
    print("\nZero values per Product ID (Units Sold):")
    zero_by_product = df.groupby('Product ID')['Units Sold'].apply(lambda x: (x == 0).sum())
    print(zero_by_product)
    
    # Check target columns if requested
    if check_targets:
        print("\nZero values in target columns:")
        for i in range(1, 8):
            target_col = f'demand_t+{i}'
            if target_col in df.columns:
                target_zero = (df[target_col] == 0).sum()
                target_total = df[target_col].notna().sum()
                if target_total > 0:
                    target_zero_pct = (target_zero / target_total) * 100
                    print(f"  {target_col}: {target_zero}/{target_total} ({target_zero_pct:.2f}%)")
    
def prepare_features(df):
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by Product ID, then Date
    print("Sorting by Product ID and Date...")
    df = df.sort_values(['Product ID', 'Date']).reset_index(drop=True)
    
    # Create lag features for Units Sold: 1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 30
    print("Creating lag features...")
    lag_periods = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 30]
    for lag in lag_periods:
        df[f'lag_{lag}'] = df.groupby('Product ID')['Units Sold'].shift(lag)
    
    # Create rolling mean features for Units Sold: windows 3, 7, 14, 21, 30
    print("Creating rolling mean features...")
    rolling_windows = [3, 7, 14, 21, 30]
    for window in rolling_windows:
        df[f'rolling_mean_{window}'] = df.groupby('Product ID')['Units Sold'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
    
    # Create rolling standard deviation features: windows 7, 14, 30
    print("Creating rolling standard deviation features...")
    rolling_std_windows = [7, 14, 30]
    for window in rolling_std_windows:
        df[f'rolling_std_{window}'] = df.groupby('Product ID')['Units Sold'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
    
    # Create day-based features
    print("Creating day-based features...")
    df['day_of_week'] = df['Date'].dt.dayofweek  # 0-6 (Monday=0, Sunday=6)
    df['week_of_year'] = df['Date'].dt.isocalendar().week  # 1-52/53
    df['month'] = df['Date'].dt.month  # 1-12
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # 0 or 1
    
    # Create target columns (demand_t+1 to demand_t+7)
    print("Creating target columns...")
    for i in range(1, 8):
        df[f'demand_t+{i}'] = df.groupby('Product ID')['Units Sold'].shift(-i)
    
    # Drop rows without available future targets (last 7 rows per product)
    print("Dropping rows without future targets...")
    initial_rows = len(df)
    df = df.dropna(subset=[f'demand_t+{i}' for i in range(1, 8)])
    dropped_rows = initial_rows - len(df)
    print(f"Dropped {dropped_rows} rows without future targets")
    
    print(f"Final dataset shape: {df.shape}")
    return df


def split_data(df, test_size=0.2):
    print(f"\nSplitting data (time-based, {test_size*100}% validation)...")
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    print(f"Training set: {len(train_df)} rows")
    print(f"Validation set: {len(val_df)} rows")
    
    return train_df, val_df


def get_feature_columns(df):
    exclude_cols = ['Date', 'Product ID', 'demand_t+1', 'demand_t+2', 'demand_t+3', 
                    'demand_t+4', 'demand_t+5', 'demand_t+6', 'demand_t+7']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    numeric_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
    
    return numeric_cols


def tune_hyperparameters(X_train, y_train, X_val, y_val):
    # Ensure numpy arrays to avoid memory issues with pandas DataFrames in multiprocessing
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)
    
    print("Starting hyperparameter tuning...")
    
    # Reduced parameter grid for faster tuning and less memory usage
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=1  # Sequential to avoid memory issues
    )
    
    # Grid search with RMSE as scoring
    # Using n_jobs=1 to avoid memory issues with multiprocessing
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=3,
        n_jobs=1,  # Sequential processing to avoid memory issues
        verbose=1
    )
    
    try:
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score (RMSE): {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    except MemoryError:
        print("Memory error during tuning, using default parameters...")
        # Fallback to default model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=1
        )
        model.fit(X_train, y_train)
        return model


def train_model(X_train, y_train, X_val, y_val, target_name, tune=True):

    print(f"\nTraining model for {target_name}...")
    
    if tune:
        model = tune_hyperparameters(X_train, y_train, X_val, y_val)
    else:
        # Initialize XGBoost regressor with basic hyperparameters
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=1  # Reduced to avoid memory issues
        )
        
        # Train the model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    
    return model


def evaluate_model(model, X_train, y_train, X_val, y_val, target_name):

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_smape = smape(y_train, y_train_pred)
    
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_smape = smape(y_val, y_val_pred)
    
    rmse_gap = train_rmse - val_rmse
    mae_gap = train_mae - val_mae
    
    print(f"\n{target_name} - Training Metrics:")
    print(f"  RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, SMAPE: {train_smape:.4f}%")
    print(f"{target_name} - Validation Metrics:")
    print(f"  RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, SMAPE: {val_smape:.4f}%")
    
    # Underfitting check
    if val_rmse > train_rmse * 1.2:
        print(f"  WARNING: Possible underfitting detected!")
        print(f"  Validation RMSE is {((val_rmse/train_rmse - 1) * 100):.2f}% higher than training RMSE")
    elif val_rmse < train_rmse:
        print(f"  NOTE: Validation RMSE is lower than training RMSE (unusual, may indicate data leakage)")
    else:
        print(f"  Model performance looks reasonable (gap: {rmse_gap:.4f})")
    
    return {
        'target': target_name,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_smape': train_smape,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_smape': val_smape,
        'y_val': y_val,
        'y_val_pred': y_val_pred
    }


def plot_predictions(results_dict, save_plots=True, max_points=5000):
    
    n_models = len(results_dict)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (target_name, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        y_val = results['y_val']
        y_val_pred = results['y_val_pred']
        
        # Sample points if dataset is too large
        if len(y_val) > max_points:
            indices = np.random.choice(len(y_val), max_points, replace=False)
            y_val_plot = y_val[indices]
            y_val_pred_plot = y_val_pred[indices]
        else:
            y_val_plot = y_val
            y_val_pred_plot = y_val_pred
        
        # Scatter plot
        ax.scatter(y_val_plot, y_val_pred_plot, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(min(y_val), min(y_val_pred))
        max_val = max(max(y_val), max(y_val_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Demand')
        ax.set_ylabel('Predicted Demand')
        ax.set_title(f'{target_name}\nRMSE: {results["val_rmse"]:.2f}, SMAPE: {results["val_smape"]:.2f}%')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove the last subplot if odd number of models
    if n_models < 8:
        fig.delaxes(axes[7])
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        print("Plots saved to 'predictions_vs_actual.png'")
    
    plt.show()


def save_model(model, filename):
    model.save_model(filename)
    print(f"Model saved to {filename}")


def load_models():
    """Load all 7 trained models."""
    models = {}
    for i in range(1, 8):
        model_filename = f'xgb_model_t+{i}.json'
        try:
            model = xgb.XGBRegressor()
            model.load_model(model_filename)
            models[f'demand_t+{i}'] = model
        except Exception as e:
            print(f"Warning: Could not load {model_filename}: {e}")
    return models


def construct_features_for_inference(latest_data_row, historical_data, product_id):
    """
    Construct features for inference from the latest data row and historical data.
    
    Args:
        latest_data_row: Series or dict with the latest row data (must include 'Date' and 'Units Sold')
        historical_data: DataFrame with historical data for the product (sorted by Date)
        product_id: Product ID string
        
    Returns:
        Dictionary of feature values ready for model prediction
    """
    # Ensure historical_data is sorted and filtered for this product
    product_data = historical_data[historical_data['Product ID'] == product_id].copy()
    product_data = product_data.sort_values('Date').reset_index(drop=True)
    
    # Get the latest date and units sold
    if isinstance(latest_data_row, pd.Series):
        latest_date = pd.to_datetime(latest_data_row['Date'])
        latest_units_sold = float(latest_data_row['Units Sold'])
    else:
        latest_date = pd.to_datetime(latest_data_row['Date'])
        latest_units_sold = float(latest_data_row['Units Sold'])
    
    # Create a temporary dataframe with historical + latest data for feature calculation
    temp_df = product_data[['Date', 'Units Sold']].copy()
    
    # Add latest row
    new_row = pd.DataFrame({
        'Date': [latest_date],
        'Units Sold': [latest_units_sold]
    })
    temp_df = pd.concat([temp_df, new_row], ignore_index=True)
    temp_df = temp_df.sort_values('Date').reset_index(drop=True)
    
    # Initialize features dictionary
    features = {}
    
    # Calculate lag features
    lag_periods = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 30]
    for lag in lag_periods:
        if len(temp_df) > lag:
            features[f'lag_{lag}'] = float(temp_df['Units Sold'].iloc[-lag-1])
        else:
            # Use the most recent available value or 0
            if len(temp_df) > 1:
                features[f'lag_{lag}'] = float(temp_df['Units Sold'].iloc[0])
            else:
                features[f'lag_{lag}'] = latest_units_sold
    
    # Calculate rolling mean features
    rolling_windows = [3, 7, 14, 21, 30]
    for window in rolling_windows:
        # Use data up to (but not including) the latest row
        if len(temp_df) > 1:
            window_data = temp_df['Units Sold'].iloc[-min(window+1, len(temp_df)-1):-1]
            if len(window_data) > 0:
                features[f'rolling_mean_{window}'] = float(window_data.mean())
            else:
                features[f'rolling_mean_{window}'] = latest_units_sold
        else:
            features[f'rolling_mean_{window}'] = latest_units_sold
    
    # Calculate rolling standard deviation features
    rolling_std_windows = [7, 14, 30]
    for window in rolling_std_windows:
        if len(temp_df) > 1:
            window_data = temp_df['Units Sold'].iloc[-min(window+1, len(temp_df)-1):-1]
            if len(window_data) > 1:
                std_val = window_data.std()
                features[f'rolling_std_{window}'] = float(std_val) if not pd.isna(std_val) else 0.0
            else:
                features[f'rolling_std_{window}'] = 0.0
        else:
            features[f'rolling_std_{window}'] = 0.0
    
    # Day-based features
    features['day_of_week'] = int(latest_date.dayofweek)
    features['week_of_year'] = int(latest_date.isocalendar().week)
    features['month'] = int(latest_date.month)
    features['is_weekend'] = 1 if latest_date.dayofweek >= 5 else 0
    
    # Add other numeric features from latest_data_row or historical data
    numeric_cols = ['Inventory Level', 'Units Ordered', 'Price', 'Discount', 
                    'Promotion', 'Competitor Pricing', 'Epidemic']
    for col in numeric_cols:
        if isinstance(latest_data_row, pd.Series) and col in latest_data_row.index:
            val = latest_data_row[col]
            features[col] = float(val) if pd.notna(val) else 0.0
        elif isinstance(latest_data_row, dict) and col in latest_data_row:
            val = latest_data_row[col]
            features[col] = float(val) if val is not None else 0.0
        else:
            # Use last known value from historical data
            if len(product_data) > 0 and col in product_data.columns:
                val = product_data[col].iloc[-1]
                features[col] = float(val) if pd.notna(val) else 0.0
            else:
                features[col] = 0.0
    
    return features


def forecast_next_7_days(latest_data_row, historical_data=None, models=None, feature_columns=None):
    if models is None:
        models = load_models()
        if len(models) == 0:
            raise ValueError("No models found. Please train models first.")
    
    # Get product ID
    if isinstance(latest_data_row, pd.Series):
        product_id = str(latest_data_row['Product ID'])
    else:
        product_id = str(latest_data_row['Product ID'])
    
    # Load historical data if not provided
    if historical_data is None:
        try:
            historical_data = pd.read_csv('sales_data.csv')
            historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        except Exception as e:
            raise ValueError(f"Could not load historical data: {e}")
    
    # Load feature columns if not provided
    if feature_columns is None:
        try:
            with open('feature_columns.json', 'r') as f:
                feature_columns = json.load(f)
        except:
            # Fallback: construct features first to get the keys
            features = construct_features_for_inference(latest_data_row, historical_data, product_id)
            feature_columns = [col for col in features.keys() if col not in ['Date', 'Product ID']]
    
    # Construct features
    features = construct_features_for_inference(latest_data_row, historical_data, product_id)
    
    # Prepare feature vector in the correct order
    feature_vector = []
    for col in feature_columns:
        if col in features:
            feature_vector.append(features[col])
        else:
            feature_vector.append(0.0)  # Default value for missing features
    
    feature_vector = np.array([feature_vector], dtype=np.float64)
    
    # Generate forecasts
    forecasts = {}
    for i in range(1, 8):
        model_key = f'demand_t+{i}'
        if model_key in models:
            try:
                pred = models[model_key].predict(feature_vector)[0]
                forecasts[f't+{i}'] = max(0.0, float(pred))  # Ensure non-negative
            except Exception as e:
                print(f"Warning: Error predicting for {model_key}: {e}")
                forecasts[f't+{i}'] = 0.0
        else:
            forecasts[f't+{i}'] = 0.0
    
    return forecasts


def main():
    data_file = 'sales_data.csv'

    df = load_data(data_file)
    
    inspect_zero_values(df)
    
    df = prepare_features(df)
    
    inspect_zero_values(df, check_targets=True)
    
    train_df, val_df = split_data(df, test_size=0.2)
    
    feature_cols = get_feature_columns(df)
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")
    
    # Save feature columns for inference
    with open('feature_columns.json', 'w') as f:
        json.dump(feature_cols, f)
    print("Feature columns saved to 'feature_columns.json'")
    
    # Prepare training and validation sets - convert to numpy arrays
    X_train = train_df[feature_cols].fillna(0).values
    X_val = val_df[feature_cols].fillna(0).values
    
    # Train and evaluate 7 models
    results = {}
    models = {}  # Store models for potential use
    
    for i in range(1, 8):
        target_name = f'demand_t+{i}'
        
        # Prepare targets - ensure numpy arrays
        y_train = np.array(train_df[target_name].values)
        y_val = np.array(val_df[target_name].values)
        
        # Train model (with hyperparameter tuning)
        model = train_model(X_train, y_train, X_val, y_val, target_name, tune=True)
        models[target_name] = model
        
        # Evaluate model
        result = evaluate_model(model, X_train, y_train, X_val, y_val, target_name)
        results[target_name] = result
        
        # Save model
        model_filename = f'xgb_model_t+{i}.json'
        save_model(model, model_filename)
    
    # Plot predictions vs actual
    plot_predictions(results, save_plots=True)
 
    print(f"\n{'Target':<15} {'Train RMSE':<12} {'Val RMSE':<12} {'Val MAE':<12} {'Val SMAPE':<12}")
    for target_name, result in results.items():
        print(f"{target_name:<15} {result['train_rmse']:<12.4f} {result['val_rmse']:<12.4f} "
              f"{result['val_mae']:<12.4f} {result['val_smape']:<12.4f}%")
    

    if len(val_df) > 0:
        # Use the last row from validation set as example
        example_row = val_df.iloc[-1]
        try:
            forecasts = forecast_next_7_days(
                example_row, 
                historical_data=df,
                models=models,
                feature_columns=feature_cols
            )
            print(f"Example forecast for Product {example_row['Product ID']} on {example_row['Date']}:")
            for horizon, forecast in forecasts.items():
                print(f"  {horizon}: {forecast:.2f}")
        except Exception as e:
            print(f"Could not run inference example: {e}")

if __name__ == "__main__":
    main()

