class AlertEngine:
    @staticmethod
    def classify_risk(days_to_stockout):
        if days_to_stockout == -1:
            # Stockout beyond forecast horizon - low risk
            return "LOW"
        elif days_to_stockout <= 2:
            return "HIGH"
        elif days_to_stockout <= 5:
            return "MEDIUM"
        else:
            return "LOW"
    
    @staticmethod
    def generate_alert_message(risk_level, days_to_stockout, reorder_needed, reorder_quantity):

        risk_level = risk_level.upper()
        
        if risk_level == "HIGH":
            if reorder_needed:
                if days_to_stockout > 0:
                    message = f"URGENT: Critical inventory level. Stockout expected in {days_to_stockout} day(s). "
                else:
                    message = "URGENT: Critical inventory level. Immediate action required. "
                message += f"Recommended reorder: {reorder_quantity:.0f} units."
            else:
                message = f"WARNING: High risk detected. Stockout expected in {days_to_stockout} day(s). Monitor closely."
        
        elif risk_level == "MEDIUM":
            if reorder_needed:
                if days_to_stockout > 0:
                    message = f"ALERT: Reorder recommended. Stockout expected in {days_to_stockout} day(s). "
                else:
                    message = "ALERT: Reorder recommended. "
                message += f"Recommended quantity: {reorder_quantity:.0f} units."
            else:
                if days_to_stockout > 0:
                    message = f"NOTICE: Medium risk. Stockout expected in {days_to_stockout} day(s). Continue monitoring."
                else:
                    message = "NOTICE: Medium risk level. Continue monitoring inventory."
        
        else:  # LOW
            if reorder_needed:
                message = f"INFO: Consider reordering. Recommended quantity: {reorder_quantity:.0f} units."
            else:
                if days_to_stockout == -1:
                    message = "INFO: Inventory levels are healthy. No immediate action required."
                elif days_to_stockout > 0:
                    message = f"INFO: Low risk. Stockout expected in {days_to_stockout} day(s)."
                else:
                    message = "INFO: Inventory levels are adequate."
        
        return message
    
    @staticmethod
    def detect_overstock(current_inventory, forecast_7day):
        if forecast_7day <= 0:
            return False  # Cannot determine overstock without forecast
        
        threshold = 2.0 * forecast_7day
        return float(current_inventory) > threshold
    
    def build_alert_payload(self, metrics_dict, forecast_7day=None):

        # Extract metrics from dictionary
        current_inventory = float(metrics_dict.get('current_inventory', 0))
        days_to_stockout = metrics_dict.get('days_to_stockout', -1)
        reorder_status = bool(metrics_dict.get('reorder_status', False))
        reorder_quantity = float(metrics_dict.get('reorder_quantity', 0))
        
        # Classify risk using days_to_stockout
        risk_level = self.classify_risk(days_to_stockout)
        
        # Generate alert message
        alert_message = self.generate_alert_message(
            risk_level=risk_level,
            days_to_stockout=days_to_stockout,
            reorder_needed=reorder_status,
            reorder_quantity=reorder_quantity
        )
        
        # Calculate forecast_7day if not provided
        if forecast_7day is None:
            # Estimate from days_to_stockout and current_inventory
            if days_to_stockout > 0 and days_to_stockout <= 7:
                # Estimate daily demand rate
                estimated_daily_demand = current_inventory / days_to_stockout if days_to_stockout > 0 else 0
                forecast_7day = estimated_daily_demand * 7
            else:
                # Use lead_time_demand as proxy if available
                lead_time_demand = float(metrics_dict.get('lead_time_demand', 0))
                if lead_time_demand > 0:
                    # Estimate 7-day from lead time demand (assuming 3-day lead time)
                    forecast_7day = (lead_time_demand / 3) * 7 if lead_time_demand > 0 else 0
                else:
                    forecast_7day = 0
        
        # Detect overstock
        overstock_flag = self.detect_overstock(current_inventory, forecast_7day)
        
        # Calculate inventory coverage days
        if forecast_7day > 0:
            daily_demand = forecast_7day / 7.0
            if daily_demand > 0:
                inventory_coverage_days = current_inventory / daily_demand
            else:
                inventory_coverage_days = -1  # Infinite coverage
        else:
            inventory_coverage_days = -1  # Cannot calculate without forecast
        
        # Build forecast summary
        forecast_summary = {
            'forecast_7day_total': round(forecast_7day, 2),
            'estimated_daily_demand': round(forecast_7day / 7.0, 2) if forecast_7day > 0 else 0.0
        }
        
        # Build and return payload
        payload = {
            'risk_level': risk_level,
            'alert_message': alert_message,
            'overstock_flag': overstock_flag,
            'days_to_stockout': days_to_stockout,
            'reorder_needed': reorder_status,
            'reorder_quantity': round(reorder_quantity, 2),
            'inventory_coverage_days': round(inventory_coverage_days, 2) if inventory_coverage_days > 0 else -1,
            'forecast_summary': forecast_summary,
            'current_inventory': round(current_inventory, 2)
        }
        return payload

def generate_alerts(metrics_dict, forecast_7day=None):
    engine = AlertEngine()
    return engine.build_alert_payload(metrics_dict, forecast_7day=forecast_7day)

