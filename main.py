"""
Terminal-based Inventory Intelligence Agent
Main CLI interface for interacting with the AI agent.
Supports dynamic SKU selection and real-time forecast generation.
"""

import json
import os
import pandas as pd
from dotenv import load_dotenv
from ai_agent import run_agent
from xgboost_7 import forecast_next_7_days, load_models
from inventory import InventoryDecisionAgent
from alerts import generate_alerts


def load_sales_data(filepath: str = "sales_data.csv") -> pd.DataFrame:
    """
    Load and preprocess sales data.
    
    Args:
        filepath: Path to sales data CSV file
        
    Returns:
        pd.DataFrame: Preprocessed sales data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Sales data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Product ID', 'Date']).reset_index(drop=True)
    
    return df


def get_available_skus(df: pd.DataFrame) -> list:
    """
    Get list of available SKUs from the dataset.
    
    Args:
        df: Sales data DataFrame
        
    Returns:
        list: List of unique Product IDs
    """
    return sorted(df['Product ID'].unique().tolist())


def get_product_info(df: pd.DataFrame, sku: str) -> dict:
    """
    Extract product information for a given SKU.
    
    Args:
        df: Sales data DataFrame
        sku: Product ID
        
    Returns:
        dict: Product information (name, category, region, etc.)
    """
    sku_data = df[df['Product ID'] == sku]
    if len(sku_data) == 0:
        return {}
    
    # Get product details from first occurrence (should be consistent)
    first_row = sku_data.iloc[0]
    
    product_info = {
        'sku': sku,
        'category': first_row.get('Category', 'Unknown'),
        'region': first_row.get('Region', 'Unknown'),
    }
    
    # Create a product name from category and SKU if no explicit name exists
    category = product_info['category']
    product_info['product_name'] = f"{category} - {sku}"
    
    # Add additional product details if available
    if 'Store ID' in first_row:
        product_info['store_id'] = first_row['Store ID']
    
    # Get average price for this product
    if 'Price' in sku_data.columns:
        avg_price = sku_data['Price'].mean()
        product_info['average_price'] = float(avg_price)
    
    return product_info


def select_sku(df: pd.DataFrame, sku: str = None) -> tuple:
    """
    Select a SKU and get its latest data row and product information.
    
    Args:
        df: Sales data DataFrame
        sku: Optional SKU to select. If None, prompts user.
        
    Returns:
        tuple: (selected_sku, latest_row, product_info)
    """
    available_skus = get_available_skus(df)
    
    if not available_skus:
        raise ValueError("No SKUs found in the dataset!")
    
    # Get product info for all SKUs to display
    sku_info_list = []
    for s in available_skus[:50]:  # Limit to first 50 for display
        info = get_product_info(df, s)
        sku_info_list.append((s, info))
    
    if sku:
        if sku not in available_skus:
            print(f"SKU '{sku}' not found. Available SKUs:")
            for s, info in sku_info_list[:10]:
                print(f"  {s} - {info.get('product_name', s)}")
            sku = None
    
    if not sku:
        print("\n" + "="*70)
        print("  AVAILABLE PRODUCTS")
        print("="*70)
        print(f"Found {len(available_skus)} products in dataset\n")
        
        # Show first 20 products with their info
        display_count = min(20, len(sku_info_list))
        for i, (s, info) in enumerate(sku_info_list[:display_count], 1):
            category = info.get('category', 'Unknown')
            product_name = info.get('product_name', s)
            print(f"  {i}. {s} - {product_name} ({category})")
        
        if len(available_skus) > display_count:
            print(f"  ... and {len(available_skus) - display_count} more products")
        
        print("\nEnter SKU to analyze (or press Enter for first product):")
        user_input = input("> ").strip()
        
        if user_input:
            if user_input in available_skus:
                sku = user_input
            else:
                print(f"SKU '{user_input}' not found. Using first SKU: {available_skus[0]}")
                sku = available_skus[0]
        else:
            sku = available_skus[0]
            print(f"Using first product: {sku}")
    
    # Get latest row for this SKU
    sku_data = df[df['Product ID'] == sku].sort_values('Date')
    if len(sku_data) == 0:
        raise ValueError(f"No data found for SKU: {sku}")
    
    latest_row = sku_data.iloc[-1]
    product_info = get_product_info(df, sku)
    
    return sku, latest_row, product_info


def generate_context_for_sku(
    sku: str,
    latest_row: pd.Series,
    df: pd.DataFrame,
    models: dict,
    product_info: dict,
    lead_time_days: int = 3
) -> dict:
    """
    Generate complete context dictionary for a SKU.
    
    Args:
        sku: Product ID
        latest_row: Latest data row for this SKU
        df: Full sales data DataFrame
        models: Trained XGBoost models
        product_info: Product information dictionary
        lead_time_days: Lead time in days
        
    Returns:
        dict: Complete context dictionary for AI agent
    """
    # Generate forecast
    forecasts = forecast_next_7_days(
        latest_row,
        historical_data=df,
        models=models
    )
    
    # Get current inventory (from data or use a default)
    current_inventory = float(latest_row.get('Inventory Level', 100))
    
    # Generate inventory recommendations
    inventory_agent = InventoryDecisionAgent()
    recommendation = inventory_agent.generate_recommendation(
        forecast=forecasts,
        current_inventory=current_inventory,
        lead_time_days=lead_time_days
    )
    
    # Generate alerts
    forecast_7day_sum = sum(forecasts.values())
    alerts = generate_alerts(
        metrics_dict=recommendation,
        forecast_7day=forecast_7day_sum
    )
    
    # Get recent sales history (last 14 days)
    sku_data = df[df['Product ID'] == sku].sort_values('Date')
    recent_sales = sku_data['Units Sold'].tail(14).tolist()
    
    # Get current price if available
    current_price = float(latest_row.get('Price', product_info.get('average_price', 0)))
    
    # Build context dictionary with product information
    context = {
        "sku": sku,
        "product_name": product_info.get('product_name', sku),
        "category": product_info.get('category', 'Unknown'),
        "region": product_info.get('region', 'Unknown'),
        "current_inventory": current_inventory,
        "current_price": current_price,
        "forecast": {f"t+{i+1}": float(forecasts.get(f"t+{i+1}", 0)) for i in range(7)},
        "days_to_stockout": recommendation.get('days_to_stockout', -1),
        "risk_level": alerts.get('risk_level', 'LOW'),
        "reorder_needed": recommendation.get('reorder_status', False),
        "recommended_reorder_quantity": recommendation.get('reorder_quantity', 0),
        "daily_sales_history": recent_sales,
        "metadata": {
            "lead_time_days": lead_time_days,
            "safety_stock": recommendation.get('safety_stock', 0),
            "reorder_point": recommendation.get('reorder_point', 0),
            "lead_time_demand": recommendation.get('lead_time_demand', 0),
            "inventory_coverage_days": alerts.get('inventory_coverage_days', 0),
            "average_price": product_info.get('average_price', current_price)
        }
    }
    
    return context


def display_context_summary(context: dict):
    """
    Display a summary of the current context.
    
    Args:
        context: Context dictionary
    """
    print("\n" + "="*70)
    print("  CURRENT CONTEXT")
    print("="*70)
    print(f"Product: {context.get('product_name', context.get('sku', 'N/A'))}")
    print(f"SKU: {context.get('sku', 'N/A')}")
    print(f"Category: {context.get('category', 'Unknown')}")
    print(f"Region: {context.get('region', 'Unknown')}")
    if context.get('current_price'):
        print(f"Current Price: ${context.get('current_price', 0):.2f}")
    print(f"\nCurrent Inventory: {context.get('current_inventory', 0):.2f} units")
    print(f"Risk Level: {context.get('risk_level', 'N/A')}")
    print(f"Days to Stockout: {context.get('days_to_stockout', 'N/A')}")
    print(f"Reorder Needed: {'Yes' if context.get('reorder_needed', False) else 'No'}")
    if context.get('recommended_reorder_quantity'):
        print(f"Recommended Quantity: {context.get('recommended_reorder_quantity', 0):.2f} units")
    
    # Show forecast summary
    forecast = context.get('forecast', {})
    if forecast:
        total = sum(forecast.values())
        avg = total / len(forecast) if forecast else 0
        print(f"\n7-Day Forecast:")
        print(f"  Total: {total:.2f} units")
        print(f"  Average daily: {avg:.2f} units/day")
    
    print("="*70)
    print()


def main():
    """
    Main entry point for the AI Agent CLI.
    """
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("="*70)
        print("  ERROR: GROQ_API_KEY not found")
        print("="*70)
        print("\nPlease create a .env file in the project root with:")
        print("  GROQ_API_KEY=your_api_key_here")
        print("\nGet your API key from: https://console.groq.com/")
        print()
        return
    
    print("="*70)
    print("  INVENTORY INTELLIGENCE AGENT")
    print("="*70)
    
    # Load sales data
    print("\n[1/4] Loading sales data...")
    try:
        df = load_sales_data()
        print(f"✓ Loaded {len(df)} rows of sales data")
    except Exception as e:
        print(f"✗ Error loading sales data: {e}")
        return
    
    # Load models
    print("\n[2/4] Loading trained models...")
    models = load_models()
    if len(models) == 0:
        print("✗ No trained models found!")
        print("\nPlease train models first by running:")
        print("  python xgboost_7.py")
        print("\nThis will train 7 XGBoost models for forecasting.")
        return
    print(f"✓ Loaded {len(models)} trained models")
    
    # Select SKU
    print("\n[3/4] Selecting product...")
    try:
        sku, latest_row, product_info = select_sku(df)
        print(f"✓ Selected Product: {product_info.get('product_name', sku)}")
        print(f"  SKU: {sku}")
        print(f"  Category: {product_info.get('category', 'Unknown')}")
        print(f"  Latest date: {latest_row['Date'].strftime('%Y-%m-%d')}")
        print(f"  Last units sold: {latest_row.get('Units Sold', 'N/A')}")
    except Exception as e:
        print(f"✗ Error selecting product: {e}")
        return
    
    # Generate context
    print("\n[4/4] Generating forecasts and inventory analysis...")
    try:
        context = generate_context_for_sku(
            sku=sku,
            latest_row=latest_row,
            df=df,
            models=models,
            product_info=product_info,
            lead_time_days=3  # Default lead time, can be made configurable
        )
        print("✓ Context generated successfully")
    except Exception as e:
        print(f"✗ Error generating context: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display context summary
    display_context_summary(context)
    
    print("Agent ready! Ask questions about inventory, forecasts, or risk.")
    print("Commands: 'change_sku' to switch SKU, 'context' to see summary, 'quit' to exit.\n")
    
    # Main interaction loop
    while True:
        try:
            # Get user query
            user_query = input("You: ").strip()
            
            if not user_query:
                continue
            
            # Check for exit commands
            if user_query.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\nGoodbye!")
                break
            
            # Special commands
            if user_query.lower() == 'context':
                display_context_summary(context)
                continue
            
            if user_query.lower() in ['change_sku', 'change sku', 'switch_sku', 'switch sku', 'change_product', 'change product']:
                try:
                    new_sku, new_latest_row, new_product_info = select_sku(df)
                    print(f"\nSwitching to Product: {new_product_info.get('product_name', new_sku)}")
                    context = generate_context_for_sku(
                        sku=new_sku,
                        latest_row=new_latest_row,
                        df=df,
                        models=models,
                        product_info=new_product_info,
                        lead_time_days=3
                    )
                    display_context_summary(context)
                    print("Context updated! Ask questions about the new product.\n")
                except Exception as e:
                    print(f"Error switching product: {e}\n")
                continue
            
            if user_query.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Ask any question about inventory, forecasts, or risk")
                print("  - 'context' - Show current context summary")
                print("  - 'change_sku' - Switch to a different SKU")
                print("  - 'help' - Show this help message")
                print("  - 'quit' - Exit the agent\n")
                continue
            
            # Get AI response
            print("\nAgent: ", end="", flush=True)
            response = run_agent(user_query, context)
            print(response)
            print()  # Empty line for readability
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()
