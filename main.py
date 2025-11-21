"""
Terminal-based Inventory Intelligence Agent
Main CLI interface for interacting with the AI agent.
"""

import json
import os
from dotenv import load_dotenv
from ai_agent import run_agent


def load_context(filepath: str = "context_example.json") -> dict:
    """
    Load context from JSON file or return mock context.
    
    Args:
        filepath: Path to context JSON file
        
    Returns:
        dict: Context dictionary
    """
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            print("Using mock context instead.\n")
    
    # Return mock context if file doesn't exist
    return {
        "sku": "P0001",
        "current_inventory": 195.0,
        "forecast": {
            "t+1": 95.23,
            "t+2": 98.45,
            "t+3": 102.10,
            "t+4": 105.32,
            "t+5": 98.76,
            "t+6": 101.89,
            "t+7": 107.45
        },
        "days_to_stockout": 2,
        "risk_level": "HIGH",
        "reorder_needed": True,
        "recommended_reorder_quantity": 159.94,
        "daily_sales_history": [102, 95, 110, 98, 105, 103, 108, 97, 112, 99, 106, 104, 101, 109],
        "metadata": {
            "lead_time_days": 3,
            "safety_stock": 59.16,
            "reorder_point": 354.94
        }
    }


def display_context_summary(context: dict):
    """Display a summary of the current context."""
    print("\n" + "="*70)
    print("  CURRENT CONTEXT")
    print("="*70)
    print(f"SKU: {context.get('sku', 'N/A')}")
    print(f"Current Inventory: {context.get('current_inventory', 0):.2f} units")
    print(f"Risk Level: {context.get('risk_level', 'N/A')}")
    print(f"Days to Stockout: {context.get('days_to_stockout', 'N/A')}")
    print(f"Reorder Needed: {'Yes' if context.get('reorder_needed', False) else 'No'}")
    if context.get('reorder_needed'):
        print(f"Recommended Quantity: {context.get('recommended_reorder_quantity', 0):.2f} units")
    print("="*70 + "\n")


def main():
    """Main CLI loop for the Inventory Intelligence Agent."""
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    if not os.getenv('GROQ_API_KEY'):
        print("="*70)
        print("  ERROR: GROQ_API_KEY not found")
        print("="*70)
        print("\nPlease set your Groq API key:")
        print("1. Create a .env file in the project root")
        print("2. Add: GROQ_API_KEY=your_api_key_here")
        print("3. Get your API key from: https://console.groq.com/\n")
        return
    
    # Load context
    print("="*70)
    print("  INVENTORY INTELLIGENCE AGENT")
    print("="*70)
    print("\nLoading context...")
    
    context = load_context()
    display_context_summary(context)
    
    print("Agent ready! Ask questions about inventory, forecasts, or risk.")
    print("Type 'quit', 'exit', or 'q' to stop.\n")
    
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
            
            if user_query.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Ask any question about inventory, forecasts, or risk")
                print("  - 'context' - Show current context summary")
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
