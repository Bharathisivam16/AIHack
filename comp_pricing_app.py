"""
Competitive Pricing App — CLI + Streamlit front end + LangChain ChatOpenAI LLM support (GPT-4 default)

Instructions:
- Put your OPENAI_API_KEY and optional OPENAI_API_BASE in a .env file next to this script (or export them):
    OPENAI_API_KEY=sk-...
    OPENAI_API_BASE=https://genailab.tcs.in

- Install dependencies:
    pip install pandas python-dotenv langchain-openai httpx streamlit

- Run CLI:
    python comp_pricing_v2.py --product "Wireless Mouse X100" --explain --model gpt-4

- Run Streamlit UI:
    streamlit run comp_pricing_v2.py
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# dotenv
from dotenv import load_dotenv, find_dotenv

# LangChain + httpx
from langchain_openai import ChatOpenAI
import httpx

# Streamlit import detection + page config (must run before any other Streamlit commands)
try:
    import streamlit as st
    streamlit_available = True
    # set_page_config must be called as the first Streamlit command
    try:
        st.set_page_config(page_title='Competitive Pricing App', layout='centered')
    except Exception as e:
        # If it fails because Streamlit already configured the page, ignore.
        print("DEBUG: st.set_page_config() skipped:", e)
except Exception:
    streamlit_available = False

# -------------------- Robust .env loading --------------------
env_path = find_dotenv(usecwd=True)
if not env_path:
    env_path = str(Path(__file__).parent / ".env")
# load and override so behavior is predictable
load_dotenv(env_path, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

# Debug masked prints (remove or silence in production)
if OPENAI_API_KEY:
    masked = OPENAI_API_KEY[:6] + "..." + OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 12 else "(too-short)"
    print(f"DEBUG: OPENAI_API_KEY present: True; masked: {masked}")
else:
    print("DEBUG: OPENAI_API_KEY not present; LLM explanations will be disabled unless manual fallback is used.")
print("DEBUG: OPENAI_API_BASE:", OPENAI_API_BASE)

# Try to import the connectivity helper (Option A)
try:
    import mass_connectivity_check as maas_connect
except Exception:
    maas_connect = None

# -------------------- LangChain ChatOpenAI client (for GPT explanations) --------------------
# Local fallback initializer (kept for environments where mass_connectivity_check isn't available)
import traceback, time

def init_langchain_llm(api_key=None, base_url=None, model_name="gpt-4",
                       verify_tls=False, timeout=10.0, max_retries=1, do_diagnostics=True):
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = (base_url or os.getenv("OPENAI_API_BASE") or "https://genailab.tcs.in").rstrip('/')
    if not api_key:
        print("LLM init: no API key; skipping LLM init.")
        return None
    masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 12 else "(too-short)"
    print(f"LLM init: key={masked}, base={base_url}, model={model_name}, verify_tls={verify_tls}")

    last_err = None
    for attempt in range(1, max_retries + 2):
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
                "User-Agent": "comp_pricing_v2/diag"
            }
            client = httpx.Client(verify=verify_tls, timeout=timeout, headers=headers)
            # quick GET root
            try:
                r = client.get(base_url + '/', follow_redirects=True)
                print(f"[GET] {base_url}/ -> {r.status_code}")
            except Exception as e:
                print("[GET] root failed:", repr(e))
            # try POST to /v1/responses
            if do_diagnostics:
                for path in ['/v1/responses', '/v1', '/']:
                    try:
                        url = base_url + path
                        print("DIAG POST:", url)
                        resp = client.post(url, json={"input":"ping"}, timeout=5.0)
                        print(" ->", resp.status_code)
                        print("   headers:", dict(resp.headers))
                        print("   body:", (resp.text or "")[:300])
                    except Exception as e:
                        print("   POST failed to", path, ":", repr(e))
            # create LLM
            llm = ChatOpenAI(base_url=base_url, model=model_name, api_key=api_key, http_client=client)
            print("LLM init succeeded.")
            return llm
        except Exception as e:
            last_err = e
            print("LLM init attempt", attempt, "failed:", repr(e))
            traceback.print_exc()
            time.sleep(0.6 * attempt)
    raise RuntimeError("LLM init failed: " + str(last_err))
    

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
        print(f"No input file found. A sample file has been created at '{sample_fname}'. Edit or replace it and re-run.")
    except Exception as e:
        print(f"No input file found and failed to write sample file: {e}")
    return sample_df

# Load DF from file or sample
DF = find_and_load_input_file()

# Normalize and validate column names
DF.columns = [c.strip().lower() for c in DF.columns]
required_cols = {'product', 'retailer', 'price'}
if not required_cols.issubset(set(DF.columns)):
    print('ERROR: Input data must contain columns: product, retailer, price (case-insensitive).')
    print('Found columns:', list(DF.columns))
    print('Please format your input file accordingly. Sample CSV format:')
    sample_preview = pd.DataFrame(SAMPLE_ROWS).head()
    print(sample_preview.to_csv(index=False))
    raise SystemExit(1)

# Re-map to expected column names (lowercase keys preserved)
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
    retailers_info = {row['retailer']: (row['price'] if not pd.isna(row['price']) else None) for row in table.to_dict('records')}

    result = {
        'product_query': product_input,
        'prices': retailers_info,
        'lowest_retailer': lowest['retailer'] if lowest else None,
        'lowest_price': lowest['price'] if lowest else None,
        'second_lowest_retailer': second['retailer'] if second else None,
        'second_lowest_price': second['price'] if second else None,
    }
    return result

# -------------------- LangChain-based explanation helper --------------------
def generate_gpt4_explanation_for_product(llm, product_query, prices_dict, lowest, second):
    if llm is None:
        return "(LLM disabled: no client available)"

    prices_lines = []
    for r in ALL_RETAILERS:
        price = prices_dict.get(r)
        prices_lines.append(f"{r}: {'N/A' if price is None else price}")
    prices_text = "; ".join(prices_lines)

    prompt = (
        f"You are a concise pricing analyst. For the product query '{product_query}', current prices are: {prices_text}. "
        f"Lowest: {lowest['price']} by {lowest['retailer'] if lowest else 'N/A'}. "
        f"Second lowest: {second['price']} by {second['retailer'] if second else 'N/A'}. "
        "Provide a very short (2-4 sentences) plain-language summary of the pricing situation and two quick action steps the retailer could consider."
    )

    try:
        # Try a simple invoke/call - depending on langchain_openai version you may need to adjust
        # to llm.call() or llm.generate([...]) if invoke isn't available.
        try:
            resp = llm.invoke(prompt)
            if hasattr(resp, "content"):
                return resp.content
            return str(resp)
        except AttributeError:
            # fallbacks for different LangChain versions
            try:
                resp = llm.call(prompt)
                return getattr(resp, "text", str(resp))
            except Exception:
                resp = llm.generate([prompt])
                # try to extract text
                if hasattr(resp, "generations"):
                    gens = resp.generations
                    if isinstance(gens, list) and gens and isinstance(gens[0], list) and gens[0]:
                        return getattr(gens[0][0], "text", str(gens[0][0]))
                return str(resp)
    except Exception as e:
        return f"(LangChain LLM call failed: {e})"

# -------------------- CLI / Interactive helpers --------------------
def re_split_products(raw):
    if raw is None:
        return []
    return [s.strip() for s in raw.replace(';', ',').split(',')]

def prompt_for_products_interactive():
    print("No input detected. How would you like to provide products?")
    print("  1) Enter products manually (comma- or semicolon-separated)")
    print("  2) Provide an input CSV path (CSV must contain a 'product' column)")
    print("  3) Cancel")
    choice = input("Choose 1, 2 or 3: ").strip()
    if choice == '1':
        raw = input("Enter product names separated by comma or semicolon:")
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

def cli_main():
    parser = argparse.ArgumentParser(description="Batch competitive pricing CLI with input file support and optional LLM explanations")
    parser.add_argument('--product', '-p', help='Single product name (partial match allowed)')
    parser.add_argument('--products', help='Multiple products, separated by comma or semicolon')
    parser.add_argument('--input-csv', help='Path to input CSV file with a column named "product"')
    parser.add_argument('--output-csv', help='Optional path to save output CSV')
    parser.add_argument('--explain', action='store_true', help='Call LLM to generate a short explanation for each product (requires OPENAI_API_KEY)')
    parser.add_argument('--model', default='gpt-4', help='Model name to use for LLM explanations (default: gpt-4)')
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

    # Prepare LangChain LLM if needed
    llm = None
    if args.explain:
        # Prefer the helper from mass_connectivity_check if available (Option A)
        if maas_connect is not None and hasattr(maas_connect, 'init_langchain_llm'):
            try:
                llm = maas_connect.init_langchain_llm(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE,
                                                     model_name=args.model, verify_tls=False, do_diagnostics=False)
            except Exception as e:
                print('Failed to init LLM via mass_connectivity_check:', e)
                llm = None
        else:
            llm = init_langchain_llm(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE, model_name=args.model, verify_tls=False, do_diagnostics=False)

        if llm is None:
            print("Proceeding without explanations (LLM not initialized).")
            args.explain = False

    results = []
    for prod in products:
        res = process_single_product(prod)
        # Print summary for the user
        if 'error' in res:
            print(f"Product query: '{prod}' -> ERROR: {res['error']}")
        else:
            print(f"Product query: '{prod}'")
            for r in ALL_RETAILERS:
                price = res['prices'].get(r)
                price_str = str(price) if price is not None else 'N/A'
                print(f" - {r}: {price_str}")
            print(f"Lowest: {res['lowest_price']} (by {res['lowest_retailer']})")
            second = res['second_lowest_price'] if res['second_lowest_price'] is not None else 'N/A'
            second_r = res['second_lowest_retailer'] if res['second_lowest_retailer'] is not None else 'N/A'
            print(f"Second lowest: {second} (by {second_r})")

            # Optional LLM explanation
            if args.explain and llm is not None:
                print('Generating short LLM explanation (this may consume tokens)...')
                try:
                    explanation = generate_gpt4_explanation_for_product(
                        llm, prod, res['prices'],
                        {'retailer': res['lowest_retailer'], 'price': res['lowest_price']} if res['lowest_price'] is not None else None,
                        {'retailer': res['second_lowest_retailer'], 'price': res['second_lowest_price']} if res['second_lowest_price'] is not None else None
                    )
                    print('LLM explanation:')
                    print(explanation)
                except Exception as e:
                    print('LLM explanation failed:', e)
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
            print(f"Saved results to {args.output_csv}")
        except Exception as e:
            print("Failed to write output CSV:", e)

# -------------------- Streamlit UI --------------------
def run_streamlit_app():
    if not streamlit_available:
        raise RuntimeError('streamlit is not installed. Install with: pip install streamlit')

    # NOTE: st.set_page_config() is called at module import time above.
    st.title('Retail Dynamic Competitive Pricing Generator')
    st.write('Enter a product name, or upload a CSV of products to find prices of THB / SuperBuy / MartD. Toggle LLM explanation to get a short LLM summary .')

    col1, col2 = st.columns([2,1])
    with col1:
        product_input = st.text_input('Product name (partial match allowed)', '')
    with col2:
        explain_toggle = st.checkbox('Enable LLM explanation', value=False)
        model_choice = st.selectbox('Model', options=['gpt-4', 'gpt-4o', 'gpt-4o-mini', 'gpt'], index=0,
                                    help='Choose the model name your proxy supports. Default: gpt-4')

    uploaded = st.file_uploader('Optional: upload CSV of products (column name: product)', type=['csv'])

    # handle CSV upload
    products = []
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            prod_col = None
            for c in df_up.columns:
                if c.strip().lower() == 'product':
                    prod_col = c
                    break
            if prod_col is None:
                st.error("Uploaded CSV must contain a 'product' column (case-insensitive).")
            else:
                products = [str(x).strip() for x in df_up[prod_col].dropna().astype(str).tolist()]
        except Exception as e:
            st.error(f'Failed to read uploaded CSV: {e}')

    # If CSV not uploaded, use single product input if provided
    if not products and product_input.strip():
        products = [product_input.strip()]

    if st.button('Get prices'):
        if not products:
            st.warning('Please enter a product or upload a CSV.')
        else:
            # prepare LangChain LLM if requested
            llm = None
            if explain_toggle:
                # Prefer connectivity helper if available
                if maas_connect is not None and hasattr(maas_connect, 'init_langchain_llm'):
                    try:
                        llm = maas_connect.init_langchain_llm(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"),
                                                             model_name=model_choice, verify_tls=False, do_diagnostics=False)
                    except Exception as e:
                        st.error(f'Failed to init LLM via mass_connectivity_check: {e}')
                        llm = None
                else:
                    llm = init_langchain_llm(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"), model_name=model_choice, verify_tls=False, do_diagnostics=False)

                if llm is None:
                    st.error('LLM client not available. Check OPENAI_API_KEY/OPENAI_API_BASE or manual key.')
                    explain_toggle = False

            # Process each product and display
            all_results = []
            for prod in products:
                res = process_single_product(prod)
                if 'error' in res:
                    st.write(f"Product query: '{prod}' -> ERROR: {res['error']}")
                    all_results.append(res)
                    continue

                st.subheader(f"Product: {prod}")
                table = get_price_table_for_product(prod)
                display_df = table.rename(columns={'retailer': 'Retailer', 'price': 'Price'})
                st.table(display_df)

                st.write(f"Lowest: {res['lowest_price']} (by {res['lowest_retailer']})")
                second_price = res['second_lowest_price'] if res['second_lowest_price'] is not None else 'N/A'
                second_retailer = res['second_lowest_retailer'] if res['second_lowest_retailer'] is not None else 'N/A'
                st.write(f"Second lowest: {second_price} (by {second_retailer})")

                if explain_toggle and llm is not None:
                    with st.spinner('Generating LLM explanation...'):
                        try:
                            explanation = generate_gpt4_explanation_for_product(
                                llm, prod, res['prices'],
                                {'retailer': res['lowest_retailer'], 'price': res['lowest_price']} if res['lowest_price'] is not None else None,
                                {'retailer': res['second_lowest_retailer'], 'price': res['second_lowest_price']} if res['second_lowest_price'] is not None else None
                            )
                            st.markdown('**LLM explanation:**')
                            st.write(explanation)
                        except Exception as e:
                            st.error(f'LLM explanation failed: {e}')

                all_results.append(res)

            # Offer CSV download of results
            try:
                rows = []
                for r in all_results:
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
                csv_bytes = df_out.to_csv(index=False).encode('utf-8')
                st.download_button('Download results CSV', data=csv_bytes, file_name='comp_pricing_results.csv')
            except Exception as e:
                st.error(f'Failed to prepare download: {e}')

# -------------------- Entrypoint --------------------
if 'streamlit' in sys.modules:
    # Running via `streamlit run comp_pricing_v2.py`
    try:
        run_streamlit_app()
    except Exception as e:
        # Show a helpful message in Streamlit if the app fails to start
        if streamlit_available:
            st.error(f'Failed to start Streamlit app: {e}')
        else:
            print('Failed to start Streamlit app:', e)
elif __name__ == '__main__':
    cli_main()
