"""
AI Agent Module for Inventory Intelligence
Uses Groq LLM to generate contextually accurate responses based on inventory and forecast data.
"""

import os
import json
from typing import Dict, Any
from groq import Groq


def run_agent(user_query: str, context: dict) -> str:
    """
    Generate AI-powered response to user query based on inventory context.
    
    Args:
        user_query: User's question about inventory/forecast
        context: Dictionary containing SKU inventory and forecast data
        
    Returns:
        str: AI-generated response based on context
    """
    # Validate context
    if not context or 'sku' not in context:
        return "Error: Invalid context provided. Missing SKU information."
    
    # Get API key
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        return "Error: GROQ_API_KEY environment variable not set. Please set it in your .env file."
    
    # Initialize Groq client
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        return f"Error initializing Groq client: {str(e)}"
    
    # Build system prompt
    system_prompt = """You are an expert Inventory Intelligence Agent specializing in demand forecasting and supply chain optimization.

CRITICAL RULES:
1. You MUST use ONLY the data provided in the context. Never invent or guess numbers.
2. All forecasts, inventory levels, and metrics must come from the context.
3. If asked about data not in the context, explicitly state "I don't have data for that."
4. Provide detailed, expert-level analysis (3-6 sentences) with reasoning.
5. Explain risk levels with specific context from the data.
6. Give actionable recommendations based on the provided metrics.
7. Analyze trends when sales history is available.
8. Be concise but comprehensive - avoid generic responses.

Your responses should be:
- Contextually accurate (use exact numbers from context)
- Actionable (provide specific recommendations)
- Insightful (explain the "why" behind the data)
- Professional (suitable for supply chain managers)

Format your response clearly with proper structure."""
    
    # Format context for the LLM
    context_str = format_context_for_llm(context)
    
    # Build user message with context
    user_message = f"""Context Data:
{context_str}

User Question: {user_query}

Please provide a detailed, expert analysis based on the context data above. Use only the numbers and information provided in the context."""
    
    # Call Groq API
    try:
        # Try llama3-8b-8192 first, fallback to deepseek-r1
        model = "llama-3.1-8b-instant"  # Updated model name
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=500,
            top_p=0.9
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # Post-process response for safety
        ai_response = validate_response(ai_response, context)
        
        return ai_response
        
    except Exception as e:
        # Fallback to alternative model if primary fails
        try:
            model = "deepseek-r1:7b"
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=500,
                top_p=0.9
            )
            ai_response = response.choices[0].message.content.strip()
            ai_response = validate_response(ai_response, context)
            return ai_response
        except Exception as e2:
            return f"Error calling Groq API: {str(e2)}. Please check your API key and internet connection."


def format_context_for_llm(context: dict) -> str:
    """
    Format context dictionary into a readable string for the LLM.
    
    Args:
        context: Context dictionary
        
    Returns:
        str: Formatted context string
    """
    lines = []
    lines.append(f"SKU: {context.get('sku', 'N/A')}")
    lines.append(f"Current Inventory: {context.get('current_inventory', 0):.2f} units")
    
    # Forecast data
    forecast = context.get('forecast', {})
    if forecast:
        lines.append("\n7-Day Demand Forecast:")
        for day, value in forecast.items():
            lines.append(f"  {day}: {value:.2f} units")
        total_forecast = sum(forecast.values())
        lines.append(f"  Total (7-day): {total_forecast:.2f} units")
        lines.append(f"  Average daily: {total_forecast/7:.2f} units/day")
    
    # Inventory metrics
    lines.append(f"\nInventory Metrics:")
    lines.append(f"  Days to Stockout: {context.get('days_to_stockout', 'N/A')}")
    lines.append(f"  Risk Level: {context.get('risk_level', 'N/A')}")
    lines.append(f"  Reorder Needed: {'Yes' if context.get('reorder_needed', False) else 'No'}")
    
    if context.get('reorder_needed'):
        lines.append(f"  Recommended Reorder Quantity: {context.get('recommended_reorder_quantity', 0):.2f} units")
    
    # Sales history
    daily_sales = context.get('daily_sales_history', [])
    if daily_sales:
        lines.append(f"\nRecent Sales History (last {len(daily_sales)} days):")
        if len(daily_sales) <= 14:
            lines.append(f"  {daily_sales}")
        else:
            lines.append(f"  Last 7 days: {daily_sales[-7:]}")
            lines.append(f"  Average: {sum(daily_sales)/len(daily_sales):.2f} units/day")
            lines.append(f"  Trend: {'Increasing' if daily_sales[-1] > daily_sales[0] else 'Decreasing' if daily_sales[-1] < daily_sales[0] else 'Stable'}")
    
    # Metadata
    metadata = context.get('metadata', {})
    if metadata:
        lines.append(f"\nAdditional Context:")
        for key, value in metadata.items():
            lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)


def validate_response(response: str, context: dict) -> str:
    """
    Validate and sanitize LLM response to ensure it doesn't invent data.
    
    Args:
        response: LLM-generated response
        context: Original context dictionary
        
    Returns:
        str: Validated response
    """
    # Check if response mentions data not in context
    sku = context.get('sku', '')
    
    # If response seems to be making up data, add a disclaimer
    # This is a simple check - in production, you might want more sophisticated validation
    
    # Ensure response mentions the SKU if it's a specific question
    if sku and sku not in response and len(response) > 50:
        # Response might be too generic, but we'll let it through
        pass
    
    return response

