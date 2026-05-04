import pandas as pd
from xgboost_7 import forecast_next_7_days, load_models
from inventory import InventoryDecisionAgent

def main():
    try:
        historical_data = pd.read_csv('sales_data.csv')
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        print(f"Loaded {len(historical_data)} rows of historical data")
    except FileNotFoundError:
        print("Error: sales_data.csv not found!")
        print("Please ensure the sales data file is in the current directory.")
        return
    
    # Step 2: Check if models exist, if not, inform user
    models = load_models()
    if len(models) == 0:
        print("No trained models found!")
        print("\nPlease run the following command first to train models:")
        print("  python xgboost_7.py")
        print("\nThis will:")
        print("  - Load and prepare the data")
        print("  - Create features (lags, rolling stats, day features)")
        print("  - Train 7 XGBoost models (one for each day t+1 to t+7)")
        print("  - Save models as xgb_model_t+1.json through xgb_model_t+7.json")
        return
    else:
        print(f"Found {len(models)} trained models")
    
    # Step 3: Select a product for demonstration
    historical_data_sorted = historical_data.sort_values('Date')
    product_ids = historical_data['Product ID'].unique()
    
    if len(product_ids) == 0:
        print("No products found in data!")
        return
    
    # Use the first product and get its latest row
    selected_product = product_ids[0]
    product_data = historical_data_sorted[historical_data_sorted['Product ID'] == selected_product]
    
    if len(product_data) == 0:
        print(f"✗ No data found for product {selected_product}!")
        return
    
    latest_row = product_data.iloc[-1]
    print(f"Selected Product: {selected_product}")
    print(f"  Latest Date: {latest_row['Date'].strftime('%Y-%m-%d')}")
    print(f"  Current Inventory: {latest_row.get('Inventory Level', 'N/A')}")
    print(f"  Last Units Sold: {latest_row['Units Sold']}")
    
    # Step 4: Generate demand forecast
    print("\n[Step 4] Generating 7-day demand forecast...")
    try:
        forecasts = forecast_next_7_days(
            latest_row,
            historical_data=historical_data,
            models=models
        )
        print("Forecast generated successfully:")
        for horizon, forecast_value in forecasts.items():
            print(f"  {horizon}: {forecast_value:.2f} units")
    except Exception as e:
        print(f"Error generating forecast: {e}")
        return
    
    # Step 5: Generate inventory recommendation
    print("\n[Step 5] Generating inventory recommendation...")
    try:
        agent = InventoryDecisionAgent()
        
        # Get current inventory (use Inventory Level from data or a default)
        current_inventory = float(latest_row.get('Inventory Level', 100))
        lead_time_days = 3  # Example: 3 days lead time
        
        recommendation = agent.generate_recommendation(
            forecast=forecasts,
            current_inventory=current_inventory,
            lead_time_days=lead_time_days
        )
 
        print(f"Current Inventory:     {recommendation['current_inventory']:.2f} units")
        print(f"Lead Time Demand:      {recommendation['lead_time_demand']:.2f} units")
        print(f"Safety Stock:          {recommendation['safety_stock']:.2f} units")
        print(f"Reorder Point:          {recommendation['reorder_point']:.2f} units")
        print(f"Reorder Status:         {'YES - Reorder Needed' if recommendation['reorder_status'] else 'NO - Sufficient Stock'}")
        print(f"Recommended Quantity:   {recommendation['reorder_quantity']:.2f} units")
        print(f"Days to Stockout:       {recommendation['days_to_stockout'] if recommendation['days_to_stockout'] > 0 else 'Beyond forecast horizon'}")
        print(f"Risk Level:             {recommendation['risk_level'].upper()}")
        print(recommendation['recommendation'])

        
    except Exception as e:
        print(f"Error generating recommendation: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()

