"""
Competitive Pricing CLI Pricing App (single-file, no Streamlit)

This version removes Streamlit and provides a command-line interface so you won't hit Streamlit-specific quota/UI issues.

Features:
- Same sample data for three retailers: THB, SuperBuy, MartD
- CLI: pass product name and your retailer as arguments
- Computes competitive prices, lowest/second-lowest, gaps
- Optional GPT-4 explanation (disabled by default to avoid API quota usage). Enable with --explain
- Loads .env automatically if present

Run examples:
1) Install requirements:
   pip install pandas python-dotenv openai

2) Put your keys in a .env file (optional):
   OPENAI_API_KEY=sk-xxxx
   OPENAI_API_BASE=https://api.openai.com/v1

3) Run without GPT explanation (recommended to avoid quotas):
   python competitive_pricing_app.py --product "Wireless Mouse X100" --retailer THB

4) Run with GPT explanation (will call OpenAI API):
   python competitive_pricing_app.py --product "Wireless Mouse X100" --retailer THB --explain

Notes:
- If you don't provide OPENAI_API_KEY or you don't pass --explain, the app will not call the OpenAI API.
- This is a demo. Tweak pricing logic before using in production.
"""

import os
import math
import argparse
import pandas as pd
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# OpenAI client (optional)
try:
    from openai import OpenAI
    openai_available = True
except Exception:
    openai_available = False

# -------------------- OpenAI Config --------------------
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------- Sample data --------------------
SAMPLE_DATA = [
    {"product": "Wireless Mouse X100", "retailer": "THB", "price": 499.0},
    {"product": "Wireless Mouse X100", "retailer": "SuperBuy", "price": 479.0},
    {"product": "Wireless Mouse X100", "retailer": "MartD", "price": 489.0},

    {"product": "USB-C Charger 30W", "retailer": "THB", "price": 799.0},
    {"product": "USB-C Charger 30W", "retailer": "SuperBuy", "price": 749.0},
    {"product": "USB-C Charger 30W", "retailer": "MartD", "price": 769.0},

    {"product": "Noise Cancelling Headphones Z", "retailer": "THB", "price": 3499.0},
    {"product": "Noise Cancelling Headphones Z", "retailer": "SuperBuy", "price": 3299.0},
    {"product": "Noise Cancelling Headphones Z", "retailer": "MartD", "price": 3399.0},

    {"product": "Portable SSD 1TB", "retailer": "THB", "price": 6499.0},
    {"product": "Portable SSD 1TB", "retailer": "SuperBuy", "price": 6299.0},
    {"product": "Portable SSD 1TB", "retailer": "MartD", "price": 6399.0},
]

DF = pd.DataFrame(SAMPLE_DATA)
RETAILERS = sorted(DF['retailer'].unique())

# -------------------- Pricing logic --------------------

def compute_competitive_recommendation(product_rows, your_retailer, min_margin_pct=0.02):
    table = product_rows.copy().sort_values('price')
    lowest_row = table.iloc[0]
    lowest_price = float(lowest_row['price'])
    lowest_retailer = lowest_row['retailer']

    second_lowest_price = None
    if len(table) > 1:
        second_lowest_price = float(table.iloc[1]['price'])

    your_row = table[table['retailer'] == your_retailer]
    your_price = float(your_row['price'].iloc[0]) if not your_row.empty else None

    epsilon = 1.0
    recommended_price = None
    strategy = ""

    if your_price is None:
        recommended_price = max(max(0.0, lowest_price - epsilon), 0.0)
        strategy = f"Your retailer ({your_retailer}) has no current listing. Recommend market-entry price = lowest - {epsilon}"
    else:
        if your_price > lowest_price:
            candidate = round(max(lowest_price - epsilon, 0.0), 2)
            floor = round(your_price * (1 - 0.5 * min_margin_pct), 2)
            recommended_price = max(candidate, floor)
            strategy = f"Undercut current lowest ({lowest_retailer}) by {epsilon}, but not below safety floor."
        elif math.isclose(your_price, lowest_price, rel_tol=1e-9) or your_price == lowest_price:
            if second_lowest_price is not None:
                gap = second_lowest_price - lowest_price
                bump = round(min(gap * 0.25, 5.0), 2)
                recommended_price = round(min(your_price + bump, second_lowest_price - 0.01), 2)
                strategy = "You're the current lowest — consider a small price increase toward the second-lowest to improve margin."
            else:
                recommended_price = your_price
                strategy = "Only seller — maintain price and monitor."
        else:
            recommended_price = your_price
            strategy = "You're already priced below competitors. Review margin."

    return {
        'table': table.reset_index(drop=True),
        'lowest_price': lowest_price,
        'lowest_retailer': lowest_retailer,
        'second_lowest_price': second_lowest_price,
        'your_price': your_price,
        'recommended_price': float(recommended_price) if recommended_price is not None else None,
        'strategy_text': strategy,
    }

# -------------------- GPT-4 explanation helper --------------------


def generate_gpt4_explanation(openai_client, product_name, price_table, rec):
    rows_text = "\n".join([f"- {r['retailer']}: {r['price']}" for _, r in price_table.iterrows()])
    prompt = (
        f"You are a pricing analyst. Provide a short (3-5 sentence) explanation and two bullet-point action steps for product '{product_name}':\n"
        f"Current prices:\n{rows_text}\n"
        f"Recommended price: {rec['recommended_price']}.\n"
        f"Explain rationale, risks, and two quick actions. Keep concise."
    )
    response = openai_client.responses.create(model="gpt-4", input=prompt)
    text = getattr(response, 'output_text', None)
    if text is None:
        choices = response.output if hasattr(response, 'output') else None
        if choices and isinstance(choices, list) and len(choices) > 0:
            text = choices[0].get('content', {}).get('text', '')
        else:
            text = str(response)
    return text


# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(description="Competitive pricing CLI (no Streamlit)")
    parser.add_argument('--product', '-p', required=True, help='Product name (partial match allowed)')
    parser.add_argument('--retailer', '-r', required=True, choices=RETAILERS, help='Your retailer')
    parser.add_argument('--explain', action='store_true', help='Call GPT-4 to generate explanation (requires OPENAI_API_KEY)')
    args = parser.parse_args()

    product_input = args.product.strip()
    user_retailer = args.retailer

    mask = DF['product'].str.lower().str.contains(product_input.lower())
    if not mask.any():
        print("No matching product found in sample data. Available products:")
        for p in sorted(DF['product'].unique()):
            print(" -", p)
        return

    product_rows = DF[mask][['retailer','price']]
    rec = compute_competitive_recommendation(product_rows, user_retailer)

    print("Price table:")
    print(rec['table'].to_string(index=False))
    print(f"Lowest price: {rec['lowest_price']} (by {rec['lowest_retailer']})")
    if rec['second_lowest_price'] is not None:
        print(f"Second lowest: {rec['second_lowest_price']}")
    print(f"Your current price: {rec['your_price']}")
    print(f"Recommended price: {rec['recommended_price']}")
    print("Rationale:", rec['strategy_text'])

    if args.explain:
        if not openai_available or not OPENAI_API_KEY:
            print('GPT-4 explanation skipped — OpenAI client not installed or OPENAI_API_KEY not set in environment.')
            return
        try:
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE or None)
            explanation = generate_gpt4_explanation(client, product_input, rec['table'], rec)
            print('GPT-4 explanation:')
            print(explanation)
        except Exception as e:
            print(f"Could not call OpenAI API: {e}")

if __name__ == '__main__':
    main()
