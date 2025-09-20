"""
Competitive Pricing CLI — load sample input from file `input_data_comp_pricing` (CSV/JSON/XLSX)

Behavior:
- The script now looks for an input file named one of:
  - `input_data_comp_pricing.csv`
  - `input_data_comp_pricing.json`
  - `input_data_comp_pricing.xlsx`

- If one of these files is present in the script working directory, the script will load pricing sample data from it.
- The input file must contain columns (case-insensitive): `product`, `retailer`, `price`.
- If no input file is found, the script will automatically create a **sample CSV** named `input_data_comp_pricing.csv` in the project folder with example rows, and will use that sample during the run. This helps you get started quickly.

Usage examples:
- Single product:
  python competitive_pricing_app.py --product "Wireless Mouse X100"
- Batch from CSV you prepared (file name must match one of the above):
  python competitive_pricing_app.py --input-csv input_products.csv --output-csv results.csv
- If you don't provide any args the script will prompt for interactive input.

Notes:
- The file `input_data_comp_pricing.csv` is created only if no input file exists. It contains the same sample rows used previously.
- Keep the input file in the same directory as this script when running.
"""

import os
import argparse
import pandas as pd
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Optional OpenAI client import is kept for the --explain flag
try:
    from openai import OpenAI
    openai_available = True
except Exception:
    openai_available = False

# -------------------- Input file handling --------------------
POSSIBLE_FILES = [
    'input_data_comp_pricing.csv',
    'input_data_comp_pricing.json',
    'input_data_comp_pricing.xlsx'
]

SAMPLE_ROWS = [
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


def find_and_load_input_file():
    """Search for a supported input file and load it into a dataframe.
       If none found, create a sample CSV and load it.
    """
    for fname in POSSIBLE_FILES:
        if os.path.isfile(fname):
            try:
                if fname.lower().endswith('.csv'):
                    df = pd.read_csv(fname)
                elif fname.lower().endswith('.json'):
                    df = pd.read_json(fname)
                elif fname.lower().endswith('.xlsx'):
                    df = pd.read_excel(fname)
                else:
                    continue
                print(f"Loaded input data from {fname}")
                return df
            except Exception as e:
                print(f"Found {fname} but failed to load it: {e}")
                break

    # No file found or failed to load — create a sample CSV for user convenience
    sample_df = pd.DataFrame(SAMPLE_ROWS)
    sample_fname = 'input_data_comp_pricing.csv'
    try:
        sample_df.to_csv(sample_fname, index=False)
        print(f"No input file found. A sample file has been created at '{sample_fname}'. Edit it or replace it with your own file named 'input_data_comp_pricing.csv' (or .json/.xlsx) and re-run.")
    except Exception as e:
        print(f"No input file found and failed to write sample file: {e}")
    return sample_df

# Load DF from file or sample
DF = find_and_load_input_file()

# Normalize column names for robustness
DF.columns = [c.strip().lower() for c in DF.columns]
required_cols = {'product', 'retailer', 'price'}
if not required_cols.issubset(set(DF.columns)):
    print('\nERROR: Input data must contain columns: product, retailer, price (case-insensitive).')
    print('Found columns:', list(DF.columns))
    print('\nPlease format your input file accordingly. Sample CSV format:')
    sample_preview = pd.DataFrame(SAMPLE_ROWS).head()
    print(sample_preview.to_csv(index=False))
    raise SystemExit(1)

# Re-map to expected column names
DF = DF.rename(columns={
    'product': 'product',
    'retailer': 'retailer',
    'price': 'price'
})

ALL_RETAILERS = sorted(DF['retailer'].dropna().unique())

# -------------------- Helpers --------------------

def get_price_table_for_product(product_input):
    mask = DF['product'].str.lower().str.contains(product_input.lower())
    if not mask.any():
        return None
    table = DF[mask][['retailer', 'price']].copy()
    # Ensure all retailers are present (if a retailer is missing, show as NaN)
    table = table.set_index('retailer').reindex(ALL_RETAILERS).reset_index()
    return table


def lowest_and_second_lowest(table):
    valid = table.dropna(subset=['price']).copy()
    if valid.empty:
        return None, None
    valid = valid.sort_values('price').reset_index(drop=True)
    lowest = {'retailer': valid.loc[0, 'retailer'], 'price': float(valid.loc[0, 'price'])}
    second = None
    if len(valid) > 1:
        second = {'retailer': valid.loc[1, 'retailer'], 'price': float(valid.loc[1, 'price'])}
    return lowest, second


def process_single_product(product_input):
    table = get_price_table_for_product(product_input)
    if table is None:
        return {'product_query': product_input, 'error': 'No matching product in input data'}

    lowest, second = lowest_and_second_lowest(table)

    # Format table to a dict for easier CSV output later
    retailers_info = {row['retailer']: (row['price'] if not pd.isna(row['price']) else None) for _, row in table.to_dict('records')}

    result = {
        'product_query': product_input,
        'prices': retailers_info,
        'lowest_retailer': lowest['retailer'] if lowest else None,
        'lowest_price': lowest['price'] if lowest else None,
        'second_lowest_retailer': second['retailer'] if second else None,
        'second_lowest_price': second['price'] if second else None,
    }
    return result

# -------------------- GPT-4 helper (optional) --------------------

def generate_gpt4_explanation_for_product(client, product_query, prices_dict, lowest, second):
    prices_lines = []
    for r in ALL_RETAILERS:
        price = prices_dict.get(r)
        prices_lines.append(f"{r}: {'N/A' if price is None else price}")
    prices_text = "\n".join(prices_lines)

    prompt = (
        f"You are a concise pricing analyst. For the product query '{product_query}', current prices are:\n{prices_text}\n\n"
        f"Lowest: {lowest['price']} by {lowest['retailer'] if lowest else 'N/A'}.\n"
        f"Second lowest: {second['price']} by {second['retailer'] if second else 'N/A'}.\n\n"
        "Provide a very short (2-4 sentences) plain-language summary of the pricing situation and two quick action steps the retailer could consider."
    )

    try:
        resp = client.responses.create(model="gpt-4", input=prompt)
        text = getattr(resp, 'output_text', None)
        if text is None:
            out = getattr(resp, 'output', None)
            if isinstance(out, list) and out:
                text = out[0].get('content', {}).get('text', '')
            else:
                text = str(resp)
        return text
    except Exception as e:
        return f"(GPT call failed: {e})"

# -------------------- CLI / Interactive --------------------

def re_split_products(raw):
    if raw is None:
        return []
    return [s.strip() for s in raw.replace(';', ',').split(',')]


def prompt_for_products_interactive():
    import sys
    print("No input detected. How would you like to provide products?")
    print("  1) Enter products manually (comma- or semicolon-separated)")
    print("  2) Provide an input CSV path (CSV must contain a 'product' column)")
    print("  3) Cancel")
    choice = input("Choose 1, 2 or 3: ").strip()
    if choice == '1':
        raw = input("Enter product names separated by comma or semicolon:\n")
        parts = [p.strip() for p in re_split_products(raw) if p.strip()]
        return parts
    elif choice == '2':
        path = input("Enter path to input CSV: ").strip()
        if not os.path.isfile(path):
            print("File not found:", path)
            return []
        try:
            df_in = pd.read_csv(path)
        except Exception as e:
            print("Failed to read CSV:", e)
            return []
        prod_col = None
        for c in df_in.columns:
            if c.strip().lower() == 'product':
                prod_col = c
                break
        if prod_col is None:
            print("Input CSV must contain a 'product' column (case-insensitive). Columns found:", list(df_in.columns))
            return []
        return [str(x).strip() for x in df_in[prod_col].dropna().astype(str).tolist()]
    else:
        print("Cancelled by user.")
        return []


def main():
    import sys
    parser = argparse.ArgumentParser(description="Batch competitive pricing CLI with input file support and optional GPT-4 explanations")
    parser.add_argument('--product', '-p', help='Single product name (partial match allowed)')
    parser.add_argument('--products', help='Multiple products, separated by comma or semicolon')
    parser.add_argument('--input-csv', help='Path to input CSV file with a column named "product"')
    parser.add_argument('--output-csv', help='Optional path to save output CSV')
    parser.add_argument('--explain', action='store_true', help='Call GPT-4 to generate a short explanation for each product (requires OPENAI_API_KEY)')
    args = parser.parse_args()

    products = []

    if args.product:
        products = [args.product.strip()]
    elif args.products:
        products = [p for p in re_split_products(args.products) if p]
    elif args.input_csv:
        if not os.path.isfile(args.input_csv):
            print("Input CSV not found:", args.input_csv)
            sys.exit(1)
        try:
            df_in = pd.read_csv(args.input_csv)
        except Exception as e:
            print("Failed to read input CSV:", e)
            sys.exit(1)
        prod_col = None
        for c in df_in.columns:
            if c.strip().lower() == 'product':
                prod_col = c
                break
        if prod_col is None:
            print("Input CSV must contain a 'product' column (case-insensitive). Columns found:", list(df_in.columns))
            sys.exit(1)
        products = [str(x).strip() for x in df_in[prod_col].dropna().astype(str).tolist()]
    else:
        # Interactive prompt
        try:
            products = prompt_for_products_interactive()
        except Exception as e:
            print("Interactive prompt failed:", e)
            sys.exit(1)

    if not products:
        print("No products to process. Exiting.")
        return

    # Prepare OpenAI client if needed
    openai_client = None
    if args.explain:
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
        if not openai_available:
            print("OpenAI Python client not installed. Install with: pip install openai")
            print("Proceeding without explanations.")
            args.explain = False
        elif not OPENAI_API_KEY:
            print("OPENAI_API_KEY not set in environment or .env. Set it to enable --explain. Proceeding without explanations.")
            args.explain = False
        else:
            try:
                openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE or None)
            except Exception as e:
                print("Failed to initialize OpenAI client:", e)
                args.explain = False

    results = []
    for prod in products:
        res = process_single_product(prod)
        # Print summary for the user
        if 'error' in res:
            print(f"\nProduct query: '{prod}' -> ERROR: {res['error']}")
        else:
            print(f"\nProduct query: '{prod}'")
            for r in ALL_RETAILERS:
                price = res['prices'].get(r)
                price_str = str(price) if price is not None else 'N/A'
                print(f" - {r}: {price_str}")
            print(f"Lowest: {res['lowest_price']} (by {res['lowest_retailer']})")
            second = res['second_lowest_price'] if res['second_lowest_price'] is not None else 'N/A'
            second_r = res['second_lowest_retailer'] if res['second_lowest_retailer'] is not None else 'N/A'
            print(f"Second lowest: {second} (by {second_r})")

            # Optional GPT explanation
            if args.explain and openai_client is not None:
                print('\nGenerating short GPT-4 explanation (this may consume tokens)...')
                try:
                    explanation = generate_gpt4_explanation_for_product(openai_client, prod, res['prices'],
                                                                          {'retailer': res['lowest_retailer'], 'price': res['lowest_price']} if res['lowest_price'] is not None else None,
                                                                          {'retailer': res['second_lowest_retailer'], 'price': res['second_lowest_price']} if res['second_lowest_price'] is not None else None)
                    print('\nGPT-4 explanation:')
                    print(explanation)
                except Exception as e:
                    print('GPT explanation failed:', e)
        results.append(res)

    # If output-csv requested, flatten results and save
    if args.output_csv:
        rows = []
        for r in results:
            if 'error' in r:
                rows.append({'product_query': r['product_query'], 'retailer': None, 'price': None, 'lowest_retailer': None, 'lowest_price': None, 'second_lowest_retailer': None, 'second_lowest_price': None})
            else:
                for retailer, price in r['prices'].items():
                    rows.append({
                        'product_query': r['product_query'],
                        'retailer': retailer,
                        'price': price,
                        'lowest_retailer': r['lowest_retailer'],
                        'lowest_price': r['lowest_price'],
                        'second_lowest_retailer': r['second_lowest_retailer'],
                        'second_lowest_price': r['second_lowest_price'],
                    })
        df_out = pd.DataFrame(rows)
        try:
            df_out.to_csv(args.output_csv, index=False)
            print(f"\nSaved results to {args.output_csv}")
        except Exception as e:
            print("Failed to write output CSV:", e)

if __name__ == '__main__':
    main()
