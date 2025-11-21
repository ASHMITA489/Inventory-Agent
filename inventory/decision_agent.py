from .inventory_engine import InventoryEngine

class InventoryDecisionAgent:    
    def __init__(self):
        self.engine = InventoryEngine()
    
    def generate_recommendation(self, forecast, current_inventory, lead_time_days):
        current_inventory = float(current_inventory)
        lead_time_days = int(lead_time_days)
        
        # Calculate core inventory metrics
        lead_time_demand = self.engine.calculate_lead_time_demand(forecast, lead_time_days)
        safety_stock = self.engine.calculate_safety_stock(lead_time_demand)
        reorder_point = self.engine.calculate_reorder_point(lead_time_demand, safety_stock)
        reorder_status = self.engine.should_reorder(current_inventory, reorder_point)
        reorder_quantity = self.engine.calculate_reorder_quantity(current_inventory, reorder_point)
        
        # Calculate days to stockout
        days_to_stockout = self._calculate_days_to_stockout(forecast, current_inventory)
        
        # Assess risk level
        risk_level = self._assess_risk_level(current_inventory, reorder_point, days_to_stockout, lead_time_days)
        
        # Generate recommendation message
        recommendation = self._generate_recommendation_message(
            reorder_status, reorder_quantity, risk_level, days_to_stockout, 
            current_inventory, reorder_point
        )
        
        # Return structured recommendation
        return {
            'lead_time_demand': round(lead_time_demand, 2),
            'safety_stock': round(safety_stock, 2),
            'reorder_point': round(reorder_point, 2),
            'reorder_status': reorder_status,
            'reorder_quantity': round(reorder_quantity, 2) if reorder_status else 0.0,
            'days_to_stockout': days_to_stockout,
            'risk_level': risk_level,
            'current_inventory': round(current_inventory, 2),
            'recommendation': recommendation
        }
    
    def _calculate_days_to_stockout(self, forecast, current_inventory):
        if current_inventory <= 0:
            return 0
        
        # Extract forecast values
        if isinstance(forecast, dict):
            forecast_values = [float(forecast.get(f't+{i}', 0)) for i in range(1, 8)]
        elif isinstance(forecast, (list, tuple)):
            forecast_values = [float(x) for x in forecast[:7]]
        else:
            # Single value - assume constant daily demand
            forecast_values = [float(forecast)] * 7
        
        # Calculate cumulative demand
        cumulative_demand = 0.0
        for day, daily_demand in enumerate(forecast_values, start=1):
            cumulative_demand += daily_demand
            if cumulative_demand >= current_inventory:
                return day
        
        # If inventory lasts beyond forecast period
        return -1  # Indicates stockout beyond forecast horizon
    
    def _assess_risk_level(self, current_inventory, reorder_point, days_to_stockout, lead_time_days):

        # High risk: Inventory below reorder point and stockout within lead time
        if current_inventory <= reorder_point:
            if days_to_stockout > 0 and days_to_stockout <= lead_time_days:
                return 'high'
            elif days_to_stockout > 0 and days_to_stockout <= lead_time_days * 1.5:
                return 'medium'
            else:
                return 'medium'
        
        # Medium risk: Inventory above reorder point but close to it
        elif current_inventory <= reorder_point * 1.2:
            return 'medium'
        
        # Low risk: Inventory well above reorder point
        else:
            return 'low'
    
    def _generate_recommendation_message(self, reorder_status, reorder_quantity, 
                                        risk_level, days_to_stockout, 
                                        current_inventory, reorder_point):

        if reorder_status:
            if risk_level == 'high':
                message = f"URGENT: Reorder immediately! Current inventory ({current_inventory:.1f}) "
                message += f"is below reorder point ({reorder_point:.1f}). "
                if days_to_stockout > 0:
                    message += f"Stockout expected in {days_to_stockout} days. "
                message += f"Recommended reorder quantity: {reorder_quantity:.1f} units."
            elif risk_level == 'medium':
                message = f"Reorder recommended. Current inventory ({current_inventory:.1f}) "
                message += f"is at or below reorder point ({reorder_point:.1f}). "
                if days_to_stockout > 0:
                    message += f"Stockout expected in {days_to_stockout} days. "
                message += f"Recommended reorder quantity: {reorder_quantity:.1f} units."
            else:
                message = f"Consider reordering. Current inventory ({current_inventory:.1f}) "
                message += f"is at reorder point ({reorder_point:.1f}). "
                message += f"Recommended reorder quantity: {reorder_quantity:.1f} units."
        else:
            if risk_level == 'low':
                message = f"Inventory level is healthy. Current inventory ({current_inventory:.1f}) "
                message += f"is above reorder point ({reorder_point:.1f}). No action needed."
            elif risk_level == 'medium':
                message = f"Monitor inventory closely. Current inventory ({current_inventory:.1f}) "
                message += f"is above reorder point ({reorder_point:.1f}) but should be watched. "
                if days_to_stockout > 0:
                    message += f"Estimated stockout in {days_to_stockout} days."
            else:
                message = f"Inventory status: Current inventory ({current_inventory:.1f}) "
                message += f"is above reorder point ({reorder_point:.1f})."
        
        return message

