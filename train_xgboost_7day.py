"""
Demand Forecasting Module - XGBoost 7-Day Forecast
Trains 7 separate XGBoost models to forecast demand for the next 7 days.
Includes hyperparameter tuning, visualization, and underfitting detection.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    SMAPE handles zero values better than MAPE.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        SMAPE value as percentage
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape_val = np.mean(numerator / denominator) * 100
    return smape_val


def load_data(filepath):
    """
    Load the sales dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_zero_values(df, check_targets=False):
    """
    Inspect how many zero values exist in the target columns (Units Sold).
    
    Args:
        df: Input DataFrame
        check_targets: Whether to also check target columns (demand_t+1 to demand_t+7)
    """
    print("\n" + "="*60)
    print("ZERO VALUE INSPECTION")
    print("="*60)
    
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
    
    print("="*60 + "\n")


def prepare_features(df):
    """
    Prepare features including sorting, lag features, rolling features, and targets.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features and targets
    """
    print("\nPreparing features...")
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by Product ID, then Date
    print("Sorting by Product ID and Date...")
    df = df.sort_values(['Product ID', 'Date']).reset_index(drop=True)
    
    # Create lag features for Units Sold
    print("Creating lag features...")
    df['lag_1'] = df.groupby('Product ID')['Units Sold'].shift(1)
    df['lag_7'] = df.groupby('Product ID')['Units Sold'].shift(7)
    df['lag_30'] = df.groupby('Product ID')['Units Sold'].shift(30)
    
    # Create rolling mean features for Units Sold
    print("Creating rolling mean features...")
    df['rolling_7'] = df.groupby('Product ID')['Units Sold'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    df['rolling_30'] = df.groupby('Product ID')['Units Sold'].transform(
        lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()
    )
    
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
    """
    Split data into train and validation sets using time-based split.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data to use for validation (default 0.2)
        
    Returns:
        train_df, val_df: Training and validation DataFrames
    """
    print(f"\nSplitting data (time-based, {test_size*100}% validation)...")
    
    # Sort by Date to ensure time-based split
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate split index
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    print(f"Training set: {len(train_df)} rows")
    print(f"Validation set: {len(val_df)} rows")
    
    return train_df, val_df


def get_feature_columns(df):
    """
    Get list of feature columns to use for training.
    Excludes target columns, Date, and Product ID.
    Returns only numeric columns to avoid encoding issues.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of numeric feature column names
    """
    exclude_cols = ['Date', 'Product ID', 'demand_t+1', 'demand_t+2', 'demand_t+3', 
                    'demand_t+4', 'demand_t+5', 'demand_t+6', 'demand_t+7']
    
    # Get all columns except excluded ones
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Filter to only numeric columns
    numeric_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
    
    return numeric_cols


def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Tune XGBoost hyperparameters using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        Best XGBoost model with tuned hyperparameters
    """
    print("Tuning hyperparameters...")
    
    # Define parameter grid (optimized for faster search)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Base model
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    # Grid search with RMSE as scoring
    # Using 3-fold CV for faster tuning
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=3,
        n_jobs=-1,
        verbose=0
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score (RMSE): {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def train_model(X_train, y_train, X_val, y_val, target_name, tune=True):
    """
    Train an XGBoost regression model with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        target_name: Name of the target (for logging)
        tune: Whether to tune hyperparameters (default True)
        
    Returns:
        Trained XGBoost model
    """
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
            n_jobs=-1
        )
        
        # Train the model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    
    return model


def evaluate_model(model, X_train, y_train, X_val, y_val, target_name):
    """
    Evaluate model performance using RMSE, MAE, and SMAPE.
    Also checks for underfitting by comparing train vs validation performance.
    
    Args:
        model: Trained XGBoost model
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        target_name: Name of the target (for logging)
        
    Returns:
        Dictionary with metrics and predictions
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics for training set
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_smape = smape(y_train, y_train_pred)
    
    # Calculate metrics for validation set
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_smape = smape(y_val, y_val_pred)
    
    # Check for underfitting
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
    """
    Plot predictions vs actual values for each horizon.
    
    Args:
        results_dict: Dictionary containing results for all 7 models
        save_plots: Whether to save plots to files (default True)
        max_points: Maximum number of points to plot (for performance with large datasets)
    """
    print("\nGenerating prediction plots...")
    
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
    """
    Save XGBoost model to JSON file.
    
    Args:
        model: Trained XGBoost model
        filename: Output filename
    """
    model.save_model(filename)
    print(f"Model saved to {filename}")


def main():
    """
    Main function to orchestrate the training pipeline.
    """
    # File path
    data_file = 'sales_data.csv'
    
    # Load data
    df = load_data(data_file)
    
    # Inspect zero values
    inspect_zero_values(df)
    
    # Prepare features
    df = prepare_features(df)
    
    # Inspect zero values in target columns after feature engineering
    inspect_zero_values(df, check_targets=True)
    
    # Split data
    train_df, val_df = split_data(df, test_size=0.2)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")
    
    # Prepare training and validation sets
    X_train = train_df[feature_cols].fillna(0)
    X_val = val_df[feature_cols].fillna(0)
    
    # Train and evaluate 7 models
    results = {}
    
    for i in range(1, 8):
        target_name = f'demand_t+{i}'
        
        # Prepare targets
        y_train = train_df[target_name].values
        y_val = val_df[target_name].values
        
        # Train model (with hyperparameter tuning)
        model = train_model(X_train, y_train, X_val, y_val, target_name, tune=True)
        
        # Evaluate model
        result = evaluate_model(model, X_train, y_train, X_val, y_val, target_name)
        results[target_name] = result
        
        # Save model
        model_filename = f'xgb_model_t+{i}.json'
        save_model(model, model_filename)
    
    # Plot predictions vs actual
    plot_predictions(results, save_plots=True)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Target':<15} {'Train RMSE':<12} {'Val RMSE':<12} {'Val MAE':<12} {'Val SMAPE':<12}")
    print("-"*60)
    for target_name, result in results.items():
        print(f"{target_name:<15} {result['train_rmse']:<12.4f} {result['val_rmse']:<12.4f} "
              f"{result['val_mae']:<12.4f} {result['val_smape']:<12.4f}%")
    print("="*60)


if __name__ == "__main__":
    main()
