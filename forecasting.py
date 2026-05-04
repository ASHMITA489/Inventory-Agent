import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import json


def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape_val = np.mean(numerator / denominator) * 100
    return smape_val


def get_feature_columns(df):
    exclude_cols = ['Date', 'Product ID', 'demand_t+1', 'demand_t+2', 'demand_t+3', 
                    'demand_t+4', 'demand_t+5', 'demand_t+6', 'demand_t+7']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
    return numeric_cols


def split_data(df, test_size=0.2):
    df = df.sort_values('Date').reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    return train_df, val_df


def tune_hyperparameters(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=1
    )
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=3,
        n_jobs=1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def train_and_eval(features):
    feature_cols = get_feature_columns(features)
    
    train_df, val_df = split_data(features, test_size=0.2)
    
    X_train = train_df[feature_cols].fillna(0).values
    X_val = val_df[feature_cols].fillna(0).values
    
    # Train 7 models (one for each horizon)
    models = {}
    results = {}
    
    for i in range(1, 8):
        target_name = f'demand_t+{i}'
        
        y_train = np.array(train_df[target_name].values)
        y_val = np.array(val_df[target_name].values)
        
        model = tune_hyperparameters(X_train, y_train)
        models[target_name] = model
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_smape = smape(y_val, y_val_pred)
        
        results[target_name] = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_smape': val_smape
        }
        
        model_filename = f'xgb_model_t+{i}.json'
        model.save_model(model_filename)
    
    with open('feature_columns.json', 'w') as f:
        json.dump(feature_cols, f)
    
    if len(val_df) > 0:
        latest_row = val_df.iloc[-1]
        X_latest = latest_row[feature_cols].fillna(0).astype(float).values.reshape(1, -1)
        
        sample_forecasts = {}
        for i in range(1, 8):
            model_key = f'demand_t+{i}'
            pred = models[model_key].predict(X_latest)[0]
            sample_forecasts[f't+{i}'] = max(0.0, float(pred))
    else:
        sample_forecasts = {}
    
    return {
        'models': models,
        'metrics': results,
        'forecasts': sample_forecasts,
        'feature_columns': feature_cols
    }

