# Inventory Intelligence Agent

AI-powered terminal-based agent that provides expert analysis of inventory and demand forecasting data using Groq LLM.

## Features

- **Context-Aware Responses**: Uses only provided inventory and forecast data
- **Expert Analysis**: Generates detailed, actionable insights (3-6 sentences)
- **Risk Assessment**: Explains risk levels with reasoning
- **Reorder Recommendations**: Provides specific reorder guidance
- **Trend Analysis**: Analyzes sales history patterns
- **Safety Checks**: Prevents hallucination by strictly using context data

## Setup

### 1. Install Dependencies

```bash
pip install groq python-dotenv
```

### 2. Get Groq API Key

1. Sign up at [https://console.groq.com/](https://console.groq.com/)
2. Create an API key
3. Copy your API key

### 3. Configure Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
```

**Important**: Never commit your `.env` file to version control. Add it to `.gitignore`.

## Usage

### Basic Usage

```bash
python main.py
```

The agent will:
1. Load context (from `context_example.json` or use mock data)
2. Display context summary
3. Enter interactive loop
4. Respond to your questions

### Example Queries

- "What is the current risk level and why?"
- "Do I need to reorder? How much?"
- "Analyze the sales trend"
- "What's the forecast for the next week?"
- "Explain the inventory situation"
- "What's the stockout risk?"

### Commands

- `context` - Show current context summary
- `help` - Show help message
- `quit` / `exit` / `q` - Exit the agent

## Integrating with Real Data

### Option 1: Update context_example.json

Replace the mock data in `context_example.json` with real data from your forecasting pipeline:

```python
# In your main pipeline (main.py or run_pipeline.py)
import json

# After generating forecasts and recommendations
context = {
    "sku": "P0001",
    "current_inventory": float(latest_row['Inventory Level']),
    "forecast": forecasts['forecasts'],  # From forecasting module
    "days_to_stockout": recommendations['inventory_metrics']['days_to_stockout'],
    "risk_level": recommendations['alerts']['risk_level'],
    "reorder_needed": recommendations['inventory_metrics']['reorder_status'],
    "recommended_reorder_quantity": recommendations['inventory_metrics']['reorder_quantity'],
    "daily_sales_history": [...],  # Last 14 days of sales
    "metadata": {
        "lead_time_days": 3,
        "safety_stock": recommendations['inventory_metrics']['safety_stock'],
        "reorder_point": recommendations['inventory_metrics']['reorder_point']
    }
}

# Save to file
with open('context_example.json', 'w') as f:
    json.dump(context, f, indent=2)
```

### Option 2: Modify main.py

Update the `load_context()` function to pull data directly from your pipeline:

```python
def load_context(filepath: str = "context_example.json") -> dict:
    # Load from your forecasting pipeline
    from forecasting import train_and_eval
    from decision_agent import generate_recommendations
    # ... build context from real data
    return context
```

## File Structure

```
.
├── ai_agent.py              # LLM logic and Groq API integration
├── main.py                   # CLI terminal interface
├── context_example.json      # Example context schema
├── .env                      # Your API key (create this)
└── README_AI_AGENT.md       # This file
```

## Context Schema

The context dictionary must follow this structure:

```python
{
    "sku": str,                          # Product SKU identifier
    "current_inventory": float,          # Current stock level
    "forecast": {                        # 7-day demand forecast
        "t+1": float,
        "t+2": float,
        ...
        "t+7": float
    },
    "days_to_stockout": float,          # Estimated days until stockout
    "risk_level": "LOW" | "MEDIUM" | "HIGH",
    "reorder_needed": bool,             # Whether reorder is needed
    "recommended_reorder_quantity": float,
    "daily_sales_history": list,         # List of daily sales (last 14 days)
    "metadata": {                        # Optional additional data
        "lead_time_days": int,
        "safety_stock": float,
        "reorder_point": float,
        ...
    }
}
```

## API Models

The agent tries these models in order:
1. `llama-3.1-8b-instant` (primary)
2. `deepseek-r1:7b` (fallback)

You can modify the model in `ai_agent.py` if needed.

## Safety Features

- **Strict Context Usage**: System prompt enforces using only provided data
- **Validation**: Response validation prevents hallucination
- **Error Handling**: Graceful fallbacks if API calls fail
- **Out-of-Context Detection**: Agent will state if data is unavailable

## Troubleshooting

### "GROQ_API_KEY not found"
- Ensure `.env` file exists in project root
- Check that `GROQ_API_KEY=your_key` is in `.env`
- Restart terminal after creating `.env`

### "Error calling Groq API"
- Check internet connection
- Verify API key is correct
- Check Groq service status
- Ensure you have API credits/quota

### Generic or inaccurate responses
- Verify context data is complete and accurate
- Check that all required fields are present
- Ensure forecast values are realistic

## Production Considerations

1. **Rate Limiting**: Implement rate limiting for API calls
2. **Caching**: Cache responses for identical queries
3. **Logging**: Log all queries and responses
4. **Error Recovery**: Implement retry logic for API failures
5. **Context Validation**: Add stricter validation of context data
6. **Multi-SKU Support**: Extend to handle multiple SKUs

## Example Session

```
======================================================================
  INVENTORY INTELLIGENCE AGENT
======================================================================

Loading context...

======================================================================
  CURRENT CONTEXT
======================================================================
SKU: P0001
Current Inventory: 195.00 units
Risk Level: HIGH
Days to Stockout: 2
Reorder Needed: Yes
Recommended Quantity: 159.94 units
======================================================================

Agent ready! Ask questions about inventory, forecasts, or risk.
Type 'quit', 'exit', or 'q' to stop.

You: What is the current risk level and why?

Agent: The current risk level is HIGH, which is critically concerning given 
that you only have 2 days until stockout. With current inventory at 195 units 
and a 7-day forecast totaling 709.2 units (average 101.3 units/day), your 
inventory will be depleted well before new stock arrives (assuming a 3-day 
lead time). The recommended reorder quantity of 159.94 units should be placed 
immediately to prevent stockout and maintain service levels.

You: quit

Goodbye!
```

