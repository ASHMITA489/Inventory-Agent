"""
Example usage of the Conversational Agent (Module 4)

This script demonstrates how to use the conversational agent
to interact with the Smart Demand & Inventory Agent system.
"""

import pandas as pd
from agent import ConversationalAgent
from xgboost_7 import load_models


def main():
    """Demonstrate conversational agent usage."""
    
    print("="*70)
    print("CONVERSATIONAL AGENT DEMONSTRATION")
    print("="*70)
    
    # Initialize agent
    agent = ConversationalAgent()
    
    # Load data
    print("\n[Loading data...]")
    try:
        historical_data = pd.read_csv('sales_data.csv')
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        
        # Get latest row for a product
        product_data = historical_data[historical_data['Product ID'] == 'P0001']
        if len(product_data) == 0:
            print("No data for P0001, using first available product...")
            product_data = historical_data[historical_data['Product ID'] == historical_data['Product ID'].iloc[0]]
        
        latest_row = product_data.iloc[-1]
        current_inventory = float(latest_row.get('Inventory Level', 100))
        
        print(f"✓ Loaded data for Product: {latest_row['Product ID']}")
        print(f"  Current Inventory: {current_inventory}")
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Load models
    print("\n[Loading models...]")
    models = load_models()
    if len(models) == 0:
        print("✗ No models found. Please train models first with: python xgboost_7.py")
        return
    print(f"✓ Loaded {len(models)} models")
    
    # Prepare data inputs
    data_inputs = {
        'latest_data_row': latest_row,
        'historical_data': historical_data,
        'models': models,
        'current_inventory': current_inventory,
        'lead_time_days': 3
    }
    
    # Example queries
    queries = [
        "What is the demand forecast for the next 7 days?",
        "Do I need to reorder?",
        "What is the risk level?",
        "Give me a summary of the inventory status"
    ]
    
    print("\n" + "="*70)
    print("EXAMPLE QUERIES AND RESPONSES")
    print("="*70)
    
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}]")
        print(f"User: {query}")
        print(f"Intent: {agent.detect_intent(query)}")
        print("\nResponse:")
        
        try:
            response = agent.respond(query, data_inputs)
            print(response['response_text'])
            
            # Show structured data if available
            if response.get('forecast'):
                print(f"\nForecast data: {response['forecast']}")
            if response.get('inventory'):
                inv = response['inventory']
                print(f"\nInventory metrics: Reorder point={inv.get('reorder_point')}, "
                      f"Reorder needed={inv.get('reorder_status')}")
            if response.get('alerts'):
                alert = response['alerts']
                print(f"\nAlert: Risk={alert.get('risk_level')}, "
                      f"Overstock={alert.get('overstock_flag')}")
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

