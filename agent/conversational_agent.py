import sys
import os

# Add parent directory to path to allow importing xgboost_7
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost_7 import forecast_next_7_days, load_models
from inventory import InventoryDecisionAgent
from alerts import generate_alerts


class ConversationalAgent:
    def __init__(self):
        self.inventory_agent = InventoryDecisionAgent()
        self.models = None  # Will be loaded on demand
    
    def detect_intent(self, query: str) -> str:
        query_lower = query.lower()
        
        # Check for forecast intent
        forecast_keywords = ["forecast", "demand", "prediction", "predict", "future", "next week"]
        if any(keyword in query_lower for keyword in forecast_keywords):
            return "forecast"
        
        # Check for risk intent
        risk_keywords = ["risk", "alert", "danger", "critical", "urgent", "warning"]
        if any(keyword in query_lower for keyword in risk_keywords):
            return "risk"
        
        # Check for reorder intent
        reorder_keywords = ["reorder", "order", "purchase", "buy", "restock", "stock"]
        if any(keyword in query_lower for keyword in reorder_keywords):
            return "reorder"
        
        # Check for summary intent
        summary_keywords = ["summary", "report", "overview", "status", "dashboard"]
        if any(keyword in query_lower for keyword in summary_keywords):
            return "summary"
        
        # Default to summary
        return "summary"
    
    def handle_forecast(self, data_inputs):
        try:
            # Extract inputs
            latest_data_row = data_inputs.get('latest_data_row')
            if latest_data_row is None:
                return {
                    'error': 'Missing latest_data_row in data_inputs',
                    'forecast': None,
                    'explanation': 'Cannot generate forecast without latest data.'
                }
            
            historical_data = data_inputs.get('historical_data')
            models = data_inputs.get('models')
            
            # Load models if not provided
            if models is None:
                if self.models is None:
                    self.models = load_models()
                models = self.models
            
            # Generate forecast
            forecasts = forecast_next_7_days(
                latest_data_row=latest_data_row,
                historical_data=historical_data,
                models=models
            )
            
            # Create explanation
            forecast_sum = sum(forecasts.values())
            avg_daily = forecast_sum / 7.0
            explanation = (
                f"7-day demand forecast generated. Total expected demand: {forecast_sum:.1f} units "
                f"({avg_daily:.1f} units/day average). "
                f"Day-by-day: {', '.join([f'{k}={v:.1f}' for k, v in forecasts.items()])}."
            )
            
            return {
                'forecast': forecasts,
                'explanation': explanation,
                'forecast_summary': {
                    'total_7day': round(forecast_sum, 2),
                    'average_daily': round(avg_daily, 2)
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'forecast': None,
                'explanation': f'Error generating forecast: {str(e)}'
            }
    
    def handle_inventory(self, data_inputs):
        try:
            # Extract inputs
            forecast = data_inputs.get('forecast')
            current_inventory = data_inputs.get('current_inventory')
            lead_time_days = data_inputs.get('lead_time_days', 3)  # Default 3 days
            
            if forecast is None or current_inventory is None:
                return {
                    'error': 'Missing forecast or current_inventory in data_inputs',
                    'inventory': None,
                    'explanation': 'Cannot generate inventory recommendation without forecast and inventory data.'
                }
            
            # Generate inventory recommendation
            recommendation = self.inventory_agent.generate_recommendation(
                forecast=forecast,
                current_inventory=float(current_inventory),
                lead_time_days=int(lead_time_days)
            )
            
            # Create explanation
            explanation = (
                f"Inventory status: {recommendation['current_inventory']:.0f} units on hand. "
                f"Reorder point: {recommendation['reorder_point']:.0f} units. "
            )
            
            if recommendation['reorder_status']:
                explanation += (
                    f"REORDER NEEDED. Recommended quantity: {recommendation['reorder_quantity']:.0f} units. "
                )
            else:
                explanation += "No reorder needed at this time. "
            
            if recommendation['days_to_stockout'] > 0:
                explanation += f"Estimated stockout in {recommendation['days_to_stockout']} days."
            else:
                explanation += "Stockout beyond forecast horizon."
            
            return {
                'inventory': recommendation,
                'explanation': explanation
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'inventory': None,
                'explanation': f'Error generating inventory recommendation: {str(e)}'
            }
    
    def handle_risk(self, data_inputs):
        try:
            # First get inventory metrics (Module 2)
            inventory_result = self.handle_inventory(data_inputs)
            
            if 'error' in inventory_result:
                return inventory_result
            
            inventory_metrics = inventory_result['inventory']
            
            # Then get alerts (Module 3)
            forecast = data_inputs.get('forecast')
            forecast_7day_sum = sum(forecast.values()) if forecast else None
            
            alerts = generate_alerts(
                metrics_dict=inventory_metrics,
                forecast_7day=forecast_7day_sum
            )
            
            # Create explanation
            explanation = (
                f"Risk Level: {alerts['risk_level']}. "
                f"{alerts['alert_message']} "
            )
            
            if alerts['overstock_flag']:
                explanation += "OVERSTOCK DETECTED: Inventory exceeds 2x forecasted demand."
            
            return {
                'risk': alerts,
                'inventory': inventory_metrics,
                'explanation': explanation
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'risk': None,
                'explanation': f'Error assessing risk: {str(e)}'
            }
    
    def handle_summary(self, data_inputs):
        try:
            # Get forecast (Module 1)
            forecast_result = self.handle_forecast(data_inputs)
            forecast = forecast_result.get('forecast')
            
            if forecast is None:
                return {
                    'error': 'Could not generate forecast',
                    'summary': None,
                    'explanation': 'Unable to generate summary without forecast.'
                }
            
            # Update data_inputs with forecast for inventory/risk calculations
            data_inputs['forecast'] = forecast
            
            # Get inventory metrics (Module 2)
            inventory_result = self.handle_inventory(data_inputs)
            inventory_metrics = inventory_result.get('inventory')
            
            # Get alerts (Module 3)
            forecast_7day_sum = sum(forecast.values())
            alerts = generate_alerts(
                metrics_dict=inventory_metrics,
                forecast_7day=forecast_7day_sum
            )
            
            # Create comprehensive explanation
            explanation = (
                f"OPERATIONAL SUMMARY\n"
                f"Forecast: {forecast_result['forecast_summary']['total_7day']:.0f} units expected over next 7 days "
                f"({forecast_result['forecast_summary']['average_daily']:.1f} units/day).\n"
                f"Inventory: {inventory_metrics['current_inventory']:.0f} units on hand. "
                f"Reorder point: {inventory_metrics['reorder_point']:.0f} units. "
            )
            
            if inventory_metrics['reorder_status']:
                explanation += f"REORDER NEEDED: {inventory_metrics['reorder_quantity']:.0f} units.\n"
            else:
                explanation += "No reorder needed.\n"
            
            explanation += (
                f"Risk: {alerts['risk_level']}. "
                f"Stockout in {alerts['days_to_stockout']} days. "
            )
            
            if alerts['overstock_flag']:
                explanation += "OVERSTOCK detected."
            
            return {
                'forecast': forecast,
                'inventory': inventory_metrics,
                'alerts': alerts,
                'explanation': explanation
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'summary': None,
                'explanation': f'Error generating summary: {str(e)}'
            }
    
    def respond(self, query: str, data_inputs: dict):
        intent = self.detect_intent(query)
        
        # Route to appropriate handler
        if intent == "forecast":
            result = self.handle_forecast(data_inputs)
            return {
                'intent': intent,
                'response_text': result.get('explanation', 'Forecast generated.'),
                'forecast': result.get('forecast'),
                'inventory': None,
                'alerts': None
            }
        
        elif intent == "reorder":
            result = self.handle_inventory(data_inputs)
            return {
                'intent': intent,
                'response_text': result.get('explanation', 'Inventory analysis completed.'),
                'forecast': None,
                'inventory': result.get('inventory'),
                'alerts': None
            }
        
        elif intent == "risk":
            result = self.handle_risk(data_inputs)
            return {
                'intent': intent,
                'response_text': result.get('explanation', 'Risk assessment completed.'),
                'forecast': None,
                'inventory': result.get('inventory'),
                'alerts': result.get('risk')
            }
        
        else:  # summary
            result = self.handle_summary(data_inputs)
            return {
                'intent': intent,
                'response_text': result.get('explanation', 'Summary generated.'),
                'forecast': result.get('forecast'),
                'inventory': result.get('inventory'),
                'alerts': result.get('alerts')
             }


def main():
    """
    Interactive main function for the conversational agent.
    Allows users to ask questions and get responses.
    """
    import pandas as pd
    
    print("="*70)
    print("SMART DEMAND & INVENTORY AGENT - CONVERSATIONAL INTERFACE")
    print("="*70)
    print("\nInitializing agent and loading data...")
    
    # Initialize agent
    agent = ConversationalAgent()
    
    # Load data
    try:
        historical_data = pd.read_csv('sales_data.csv')
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        print("✓ Data loaded successfully")
    except FileNotFoundError:
        print("✗ Error: sales_data.csv not found!")
        print("Please ensure the sales data file is in the current directory.")
        return
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Load models
    try:
        models = load_models()
        if len(models) == 0:
            print("✗ No trained models found!")
            print("\nPlease run the following command first to train models:")
            print("  python xgboost_7.py")
            return
        print(f"✓ Loaded {len(models)} trained models")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return
    
    # Get product selection
    print("\n" + "-"*70)
    product_ids = historical_data['Product ID'].unique()
    if len(product_ids) == 0:
        print("✗ No products found in data!")
        return
    
    print(f"Available products: {', '.join(product_ids[:10])}{'...' if len(product_ids) > 10 else ''}")
    
    # Use first product or let user choose
    selected_product = product_ids[0]
    print(f"\nUsing product: {selected_product}")
    print("(You can modify the code to select a different product)")
    
    # Get latest row for selected product
    product_data = historical_data[historical_data['Product ID'] == selected_product]
    product_data = product_data.sort_values('Date')
    
    if len(product_data) == 0:
        print(f"✗ No data found for product {selected_product}!")
        return
    
    latest_row = product_data.iloc[-1]
    current_inventory = float(latest_row.get('Inventory Level', 100))
    
    print(f"Latest date: {latest_row['Date'].strftime('%Y-%m-%d')}")
    print(f"Current inventory: {current_inventory:.0f} units")
    
    # Prepare data inputs
    data_inputs = {
        'latest_data_row': latest_row,
        'historical_data': historical_data,
        'models': models,
        'current_inventory': current_inventory,
        'lead_time_days': 3
    }
    
    # Interactive loop
    print("\n" + "="*70)
    print("CONVERSATIONAL AGENT READY")
    print("="*70)
    print("\nYou can ask questions like:")
    print("  - 'What is the demand forecast?'")
    print("  - 'Do I need to reorder?'")
    print("  - 'What is the risk level?'")
    print("  - 'Give me a summary'")
    print("\nType 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            # Get user query
            query = input("You: ").strip()
            
            if not query:
                continue
            
            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\nGoodbye!")
                break
            
            # Get response
            print("\nAgent: ", end="")
            response = agent.respond(query, data_inputs)
            
            # Display response
            print(response['response_text'])
            
            # Show additional details if available
            if response.get('forecast'):
                print(f"\n[Forecast Details: {response['forecast']}]")
            
            if response.get('inventory'):
                inv = response['inventory']
                if inv:
                    print(f"\n[Inventory: Reorder point={inv.get('reorder_point', 'N/A')}, "
                          f"Reorder needed={inv.get('reorder_status', 'N/A')}, "
                          f"Days to stockout={inv.get('days_to_stockout', 'N/A')}]")
            
            if response.get('alerts'):
                alert = response['alerts']
                if alert:
                    print(f"\n[Alert: Risk={alert.get('risk_level', 'N/A')}, "
                          f"Overstock={alert.get('overstock_flag', 'N/A')}]")
            
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
 
