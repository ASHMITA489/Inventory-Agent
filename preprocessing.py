"""
Data Preprocessing Module
Loads and preprocesses the sales data for the demand forecasting pipeline.
"""

import pandas as pd


def run_preprocessing():
    """
    Load and preprocess raw sales data.
    
    Returns:
        pandas.DataFrame: Preprocessed dataset
    """
    # Load data
    df = pd.read_csv('sales_data.csv')
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by Product ID and Date
    df = df.sort_values(['Product ID', 'Date']).reset_index(drop=True)
    
    # Basic data validation
    # Remove any rows with missing critical columns
    critical_cols = ['Date', 'Product ID', 'Units Sold']
    df = df.dropna(subset=critical_cols)
    
    # Ensure Units Sold is numeric and non-negative
    df['Units Sold'] = pd.to_numeric(df['Units Sold'], errors='coerce')
    df = df[df['Units Sold'] >= 0]
    
    # Reset index after filtering
    df = df.reset_index(drop=True)
    
    return df


