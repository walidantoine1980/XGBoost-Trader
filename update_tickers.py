import pandas as pd
import json
import requests
from io import StringIO

def get_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

def get_table_with_columns(dfs, req_cols):
    for df in dfs:
        cols = set(df.columns)
        if all(c in cols for c in req_cols):
            return df
    raise ValueError(f"Table with columns {req_cols} not found.")

def fetch_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = get_html(url)
    dfs = pd.read_html(StringIO(html))
    df = get_table_with_columns(dfs, ['Symbol', 'Security'])
    tickers = {}
    for _, row in df.iterrows():
        symbol = str(row['Symbol'])
        name = str(row['Security'])
        symbol = symbol.replace('.', '-')
        tickers[f"{name} (US)"] = f"NASDAQ:{symbol}" # Using NASDAQ here, will be stripped correctly by the app
    return tickers

def fetch_cac40():
    url = "https://en.wikipedia.org/wiki/CAC_40"
    html = get_html(url)
    dfs = pd.read_html(StringIO(html))
    df = get_table_with_columns(dfs, ['Ticker', 'Company'])
    tickers = {}
    for _, row in df.iterrows():
        symbol = str(row['Ticker'])
        name = str(row['Company'])
        symbol = symbol.replace('.PA', '')
        tickers[f"{name} (FR)"] = f"EPA:{symbol}"
    return tickers

def fetch_dax():
    url = "https://en.wikipedia.org/wiki/DAX"
    html = get_html(url)
    dfs = pd.read_html(StringIO(html))
    df = get_table_with_columns(dfs, ['Ticker', 'Company'])
    tickers = {}
    for _, row in df.iterrows():
        symbol = str(row['Ticker'])
        name = str(row['Company'])
        symbol = symbol.replace('.DE', '')
        tickers[f"{name} (DE)"] = f"FRA:{symbol}"
    return tickers

def main():
    try:
        print("Fetching S&P 500...")
        sp500 = fetch_sp500()
        print("Fetching CAC 40...")
        cac40 = fetch_cac40()
        print("Fetching DAX...")
        dax = fetch_dax()

        all_tickers = {"--- Saisir manuellement ---": "CUSTOM"}
        all_tickers.update(sp500)
        all_tickers.update(cac40)
        all_tickers.update(dax)

        with open("tickers_db.py", "w", encoding="utf-8") as f:
            f.write("# Base de données locale des grandes actions (EU & US)\n")
            f.write("# Format attendu par notre convertisseur : Google Finance (Bourse:Symbole)\n\n")
            f.write("MAJOR_STOCKS = {\n")
            for i, (name, symbol) in enumerate(all_tickers.items()):
                safe_name = name.replace('"', '\\"')
                comma = "," if i < len(all_tickers) - 1 else ""
                f.write(f'    "{safe_name}": "{symbol}"{comma}\n')
            f.write("}\n")
        print("Successfully generated tickers_db.py")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
