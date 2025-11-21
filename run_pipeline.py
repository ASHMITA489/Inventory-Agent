"""
Complete Pipeline Runner for Smart Demand & Inventory Agent

This script demonstrates the complete pipeline:
1. Load trained models (or train if needed)
2. Get demand forecasts for a product
3. Generate inventory recommendations

Usage:
    python run_pipeline.py
"""

import pandas as pd
from xgboost_7 import forecast_next_7_days, load_models
from inventory import InventoryDecisionAgent
from alerts import generate_alerts
from agent import ConversationalAgent


def main():
    """
    Main pipeline execution.
    """
    print("="*70)
    print("SMART DEMAND & INVENTORY AGENT - COMPLETE PIPELINE")
    print("="*70)
    
    # Step 1: Load historical data
    print("\n[Step 1] Loading historical data...")
    try:
        historical_data = pd.read_csv('sales_data.csv')
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        print(f"✓ Loaded {len(historical_data)} rows of historical data")
    except FileNotFoundError:
        print("✗ Error: sales_data.csv not found!")
        print("Please ensure the sales data file is in the current directory.")
        return
    
    # Step 2: Check if models exist, if not, inform user
    print("\n[Step 2] Checking for trained models...")
    models = load_models()
    if len(models) == 0:
        print("✗ No trained models found!")
        print("\nPlease run the following command first to train models:")
        print("  python xgboost_7.py")
        print("\nThis will:")
        print("  - Load and prepare the data")
        print("  - Create features (lags, rolling stats, day features)")
        print("  - Train 7 XGBoost models (one for each day t+1 to t+7)")
        print("  - Save models as xgb_model_t+1.json through xgb_model_t+7.json")
        return
    else:
        print(f"✓ Found {len(models)} trained models")
    
    # Step 3: Select a product for demonstration
    print("\n[Step 3] Selecting product for demonstration...")
    # Get the most recent data for a product
    historical_data_sorted = historical_data.sort_values('Date')
    product_ids = historical_data['Product ID'].unique()
    
    if len(product_ids) == 0:
        print("✗ No products found in data!")
        return
    
    # Use the first product and get its latest row
    selected_product = product_ids[0]
    product_data = historical_data_sorted[historical_data_sorted['Product ID'] == selected_product]
    
    if len(product_data) == 0:
        print(f"✗ No data found for product {selected_product}!")
        return
    
    latest_row = product_data.iloc[-1]
    print(f"✓ Selected Product: {selected_product}")
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
        print("✓ Forecast generated successfully:")
        for horizon, forecast_value in forecasts.items():
            print(f"  {horizon}: {forecast_value:.2f} units")
    except Exception as e:
        print(f"✗ Error generating forecast: {e}")
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
        
        print("✓ Recommendation generated successfully:")
        print("\n" + "-"*70)
        print("INVENTORY RECOMMENDATION SUMMARY")
        print("-"*70)
        print(f"Current Inventory:     {recommendation['current_inventory']:.2f} units")
        print(f"Lead Time Demand:      {recommendation['lead_time_demand']:.2f} units")
        print(f"Safety Stock:          {recommendation['safety_stock']:.2f} units")
        print(f"Reorder Point:          {recommendation['reorder_point']:.2f} units")
        print(f"Reorder Status:         {'YES - Reorder Needed' if recommendation['reorder_status'] else 'NO - Sufficient Stock'}")
        print(f"Recommended Quantity:   {recommendation['reorder_quantity']:.2f} units")
        print(f"Days to Stockout:       {recommendation['days_to_stockout'] if recommendation['days_to_stockout'] > 0 else 'Beyond forecast horizon'}")
        print(f"Risk Level:             {recommendation['risk_level'].upper()}")
        print("\n" + "-"*70)
        print("RECOMMENDATION:")
        print(recommendation['recommendation'])
        print("-"*70)
        
    except Exception as e:
        print(f"✗ Error generating recommendation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Generate alerts
    print("\n[Step 6] Generating alerts...")
    try:
        # Calculate forecast_7day sum for overstock detection
        forecast_7day_sum = sum(forecasts.values())
        
        alerts = generate_alerts(
            metrics_dict=recommendation,
            forecast_7day=forecast_7day_sum
        )
        
        print("✓ Alerts generated successfully:")
        print("\n" + "-"*70)
        print("ALERT SUMMARY")
        print("-"*70)
        print(f"Risk Level:             {alerts['risk_level']}")
        print(f"Overstock Detected:     {'YES' if alerts['overstock_flag'] else 'NO'}")
        print(f"Days to Stockout:       {alerts['days_to_stockout'] if alerts['days_to_stockout'] > 0 else 'Beyond forecast horizon'}")
        print(f"Inventory Coverage:     {alerts['inventory_coverage_days']:.1f} days" if alerts['inventory_coverage_days'] > 0 else "Inventory Coverage:     Beyond forecast horizon")
        print(f"7-Day Forecast Total:   {alerts['forecast_summary']['forecast_7day_total']:.2f} units")
        print(f"Estimated Daily Demand:  {alerts['forecast_summary']['estimated_daily_demand']:.2f} units")
        print("\n" + "-"*70)
        print("ALERT MESSAGE:")
        print(alerts['alert_message'])
        print("-"*70)
        
    except Exception as e:
        print(f"✗ Error generating alerts: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    
    # Optional: Start conversational interface
    print("\n" + "="*70)
    print("CONVERSATIONAL AGENT")
    print("="*70)
    print("\nWould you like to use the conversational agent? (yes/no)")
    try:
        user_input = input("> ").strip().lower()
        if user_input in ['yes', 'y', '1']:
            print("\nStarting conversational interface...\n")
            
            # Prepare data inputs for conversational agent
            data_inputs = {
                'latest_data_row': latest_row,
                'historical_data': historical_data,
                'models': models,
                'current_inventory': current_inventory,
                'lead_time_days': 3,
                'forecast': forecasts  # Already computed
            }
            
            # Initialize and run conversational agent
            conv_agent = ConversationalAgent()
            
            print("Conversational Agent Ready!")
            print("Ask questions like:")
            print("  - 'What is the demand forecast?'")
            print("  - 'Do I need to reorder?'")
            print("  - 'What is the risk level?'")
            print("  - 'Give me a summary'")
            print("\nType 'quit' or 'exit' to stop.\n")
            
            while True:
                try:
                    query = input("You: ").strip()
                    
                    if not query:
                        continue
                    
                    if query.lower() in ['quit', 'exit', 'q', 'bye']:
                        print("\nGoodbye!")
                        break
                    
                    print("\nAgent: ", end="")
                    response = conv_agent.respond(query, data_inputs)
                    print(response['response_text'])
                    
                    # Show additional details
                    if response.get('forecast'):
                        print(f"\n[Forecast: {response['forecast']}]")
                    if response.get('inventory'):
                        inv = response['inventory']
                        if inv:
                            print(f"\n[Inventory: Reorder needed={inv.get('reorder_status')}, "
                                  f"Quantity={inv.get('reorder_quantity')}]")
                    if response.get('alerts'):
                        alert = response['alerts']
                        if alert:
                            print(f"\n[Alert: Risk={alert.get('risk_level')}]")
                    
                    print()
                    
                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    break
                except Exception as e:
                    print(f"\nError: {e}")
        else:
            print("Skipping conversational interface.")
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()

