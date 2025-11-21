# Inventory Intelligence Agent

An AI-powered inventory management system that combines demand forecasting (XGBoost) with natural language AI analysis (Groq LLM) to help retail companies optimize inventory, reduce stockouts, and improve cash flow.

> **Perfect for clothing/footwear startups** - Prevents stockouts, reduces overstock, and optimizes cash flow through intelligent demand forecasting and AI-powered recommendations.

## Features

- **7-Day Demand Forecasting**: XGBoost models predict demand for next 7 days
- **Inventory Optimization**: Calculates optimal reorder points and quantities
- **Risk Assessment**: Classifies risk levels (HIGH/MEDIUM/LOW) with detailed analysis
- **AI-Powered Analysis**: Groq LLM provides natural language insights and recommendations
- **Terminal-Based Interface**: Clean CLI for easy interaction
- **Production-Ready**: Modular architecture, error handling, validation

## Architecture

```
┌─────────────────┐
│   main.py       │  ← AI Agent CLI (Terminal Interface)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ai_agent.py    │  ← Groq LLM Integration
└────────┬────────┘
         │
         ├──► xgboost_7.py          (Forecasting Engine)
         ├──► inventory/            (Inventory Calculations)
         └──► alerts/                (Risk & Alert System)
```

## Project Structure

```
Inventory Agent/
├── main.py                    # AI Agent CLI (run this for demo)
├── ai_agent.py                # AI Agent core (Groq LLM integration)
├── xgboost_7.py               # Forecasting engine (XGBoost models)
├── run_pipeline.py            # Full pipeline runner (alternative)
├── context_example.json       # Example inventory context
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
│
├── inventory/                 # Inventory management module
│   ├── __init__.py
│   ├── inventory_engine.py    # Core calculations (reorder points, safety stock)
│   └── decision_agent.py     # Recommendation engine
│
├── alerts/                    # Alert system module
│   ├── __init__.py
│   └── alert_engine.py       # Risk classification & alert generation
│
└── agent/                     # Rule-based conversational interface
    ├── __init__.py
    └── conversational_agent.py
```

## Which File to Run?

- **`python main.py`** → **AI Agent** (Groq LLM) - **Main demo for internship**
- **`python run_pipeline.py`** → Full pipeline (forecasting + inventory + alerts)
- **`python xgboost_7.py`** → Train forecasting models (one-time setup)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Groq API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
```

Get your API key from [https://console.groq.com/](https://console.groq.com/)

### 3. Train Models (First Time - Optional)

If you want to use real forecasting, train models first:

```bash
python xgboost_7.py
```

This will:
- Load and preprocess data from `sales_data.csv`
- Create features (lags, rolling stats, day features)
- Train 7 XGBoost models (one for each day t+1 to t+7)
- Save models as `xgb_model_t+1.json` through `xgb_model_t+7.json`
- Generate evaluation metrics

**Note**: This takes ~10-30 minutes. Models are saved and reused afterward.

**For quick demo**: You can skip this step and use the example context in `context_example.json`.

### 4. Run the AI Agent

```bash
python main.py
```

Then ask questions like:
- "What is the current risk level and why?"
- "Do I need to reorder? How much?"
- "Analyze the sales trend"
- "What's the forecast for next week?"
- "Explain the inventory situation"

## Usage Examples

### Basic Usage

```bash
python main.py
```


### Using with Your Own Data
Update `context_example.json` with your inventory data:
```json
{
  "sku": "YOUR_SKU",
  "current_inventory": 250.0,
  "forecast": {
    "t+1": 100.0,
    "t+2": 95.0,
    ...
  },
  "days_to_stockout": 3,
  "risk_level": "MEDIUM",
  "reorder_needed": true,
  "recommended_reorder_quantity": 150.0,
  "daily_sales_history": [102, 95, 110, ...],
  "metadata": {...}
}
```

## Modules

### Forecasting Module (`xgboost_7.py`)

- **Features**: Lag features (1-30 days), rolling means/stds, day-based features
- **Models**: 7 XGBoost regressors (one per forecast horizon)
- **Output**: 7-day demand forecasts

### Inventory Module (`inventory/`)

- **InventoryEngine**: Core calculations (lead time demand, safety stock, reorder points)
- **InventoryDecisionAgent**: Generates recommendations based on forecasts

### Alert Module (`alerts/`)

- **AlertEngine**: Risk classification, alert generation, overstock detection
- **Risk Levels**: HIGH (≤2 days), MEDIUM (≤5 days), LOW (>5 days)

### AI Agent Module (`ai_agent.py`)

- **Groq LLM Integration**: Uses Llama 3.1 for natural language analysis
- **Context-Aware**: Strictly uses provided data (no hallucination)
- **Expert Analysis**: 3-6 sentence detailed responses

## Data Format

### Input Data (`sales_data.csv`)

Required columns:
- `Date`: Date of sale
- `Product ID`: Product identifier
- `Units Sold`: Units sold (target variable)
- `Inventory Level`: Current inventory
- Other features: Price, Discount, Promotion, etc.

### Context Format (`context_example.json`)

```json
{
  "sku": "string",
  "current_inventory": float,
  "forecast": {"t+1": float, ..., "t+7": float},
  "days_to_stockout": int,
  "risk_level": "LOW|MEDIUM|HIGH",
  "reorder_needed": bool,
  "recommended_reorder_quantity": float,
  "daily_sales_history": [float, ...],
  "metadata": {...}
}
```

## How It Helps Retail Startups

### Problems Solved
1. **Stockouts**: Prevents running out of popular items during peak seasons
2. **Overstock**: Reduces dead inventory (unsold items)
3. **Cash Flow**: Optimizes inventory investment to free up capital
4. **Seasonality**: Handles seasonal demand spikes (holidays, back-to-school)
5. **Multi-SKU Management**: Scales to manage hundreds of products

### Real-World Benefits
- **Demand Forecasting**: Predicts what customers will buy
- **Automated Reordering**: Tells you when and how much to order
- **Risk Management**: Alerts before problems occur
- **AI Insights**: Natural language explanations of complex data
- **Cost Savings**: Reduces inventory waste and stockout losses

## Development

### Training New Models

```bash
python xgboost_7.py
```

### Testing Individual Modules

```python
from inventory import InventoryDecisionAgent
from alerts import generate_alerts
from ai_agent import run_agent

# Test inventory agent
agent = InventoryDecisionAgent()
recommendation = agent.generate_recommendation(
    forecast={'t+1': 100, 't+2': 95, ...},
    current_inventory=250,
    lead_time_days=3
)

# Test AI agent
response = run_agent(
    "What is the risk level?",
    context=recommendation
)
```

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Environment Variables

Create `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## Demo Video Guide

For internship/portfolio demonstration:

1. **Run**: `python main.py`
2. **Show**: Context summary (inventory, risk, forecasts)
3. **Ask**: Questions about risk, reordering, trends
4. **Highlight**: AI's detailed, contextually accurate responses
5. **Emphasize**: How it helps clothing/footwear startups

**Key Talking Points:**
- AI-powered analysis using Groq LLM
- Demand forecasting with XGBoost
- Automated inventory recommendations
- Risk assessment and alerts
- Natural language interface

## Security & Git Notes

- **Never commit `.env` file** (already in `.gitignore`)
- **Keep API keys secure** - never share or commit them
- **Model files**: The `xgb_model_t+*.json` files are large (~1-2MB each). 
  - Option 1: Include them (repo will be larger but complete)
  - Option 2: Exclude them (users need to train models first)
  - To exclude: Uncomment model file rules in `.gitignore`


