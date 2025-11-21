"""
Decision Agent Module
Generates inventory recommendations from forecasts.
"""

from inventory import InventoryDecisionAgent
from alerts import generate_alerts


def generate_recommendations(forecasts):
    """
    Generate inventory recommendations from forecasts.
    
    Args:
        forecasts: dict, forecast results from forecasting module
                 Expected format: {'forecasts': {'t+1': value, ...}, ...}
    
    Returns:
        dict: Complete recommendations with inventory metrics and alerts
    """
    # Extract forecast dictionary
    if isinstance(forecasts, dict):
        forecast_dict = forecasts.get('forecasts', forecasts)
    else:
        forecast_dict = forecasts
    
    # Initialize inventory decision agent
    agent = InventoryDecisionAgent()
    
    # Example: Use a default current inventory (in real scenario, this would come from data)
    current_inventory = 250.0  # Default value - should be passed as parameter in production
    lead_time_days = 3
    
    # Generate inventory recommendation
    inventory_metrics = agent.generate_recommendation(
        forecast=forecast_dict,
        current_inventory=current_inventory,
        lead_time_days=lead_time_days
    )
    
    # Generate alerts
    forecast_7day_sum = sum(forecast_dict.values()) if isinstance(forecast_dict, dict) else 0
    alerts = generate_alerts(
        metrics_dict=inventory_metrics,
        forecast_7day=forecast_7day_sum
    )
    
    # Combine results
    recommendations = {
        'forecasts': forecast_dict,
        'inventory_metrics': inventory_metrics,
        'alerts': alerts,
        'summary': {
            'current_inventory': inventory_metrics['current_inventory'],
            'reorder_needed': inventory_metrics['reorder_status'],
            'reorder_quantity': inventory_metrics['reorder_quantity'],
            'risk_level': alerts['risk_level'],
            'days_to_stockout': alerts['days_to_stockout']
        }
    }
    
    return recommendations


