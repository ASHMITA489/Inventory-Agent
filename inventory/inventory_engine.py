class InventoryEngine:
    @staticmethod
    def calculate_lead_time_demand(forecast, lead_time_days):

        if isinstance(forecast, dict):
            # Extract values from dictionary and sum up to lead_time_days
            total_demand = 0.0
            for i in range(1, min(lead_time_days + 1, 8)):  # Max 7 days forecast
                key = f't+{i}'
                if key in forecast:
                    total_demand += float(forecast[key])
        elif isinstance(forecast, (list, tuple)):
            # Sum first lead_time_days values
            total_demand = sum(float(x) for x in forecast[:lead_time_days])
        else:
            # Assume it's a single value or array-like
            try:
                import numpy as np
                forecast_array = np.asarray(forecast)
                total_demand = float(np.sum(forecast_array[:lead_time_days]))
            except:
                # Fallback: if lead_time_days is 1, use the forecast value directly
                if lead_time_days == 1:
                    total_demand = float(forecast) if not isinstance(forecast, dict) else float(list(forecast.values())[0])
                else:
                    total_demand = float(forecast) * lead_time_days
        
        return max(0.0, total_demand)
    
    @staticmethod
    def calculate_safety_stock(lead_time_demand):
        safety_stock = 0.2 * lead_time_demand
        return max(0.0, safety_stock)
    
    @staticmethod
    def calculate_reorder_point(lead_time_demand, safety_stock):
        reorder_point = lead_time_demand + safety_stock
        return max(0.0, reorder_point)
    
    @staticmethod
    def should_reorder(current_inventory, reorder_point):
        return float(current_inventory) <= float(reorder_point)
    
    @staticmethod
    def calculate_reorder_quantity(current_inventory, reorder_point):
        if current_inventory <= reorder_point:
            # Calculate quantity needed to reach reorder point plus a buffer
            buffer = reorder_point * 0.5  # 50% buffer above reorder point
            reorder_quantity = reorder_point - current_inventory + buffer
            return max(0.0, reorder_quantity)
        else:
            return 0.0

