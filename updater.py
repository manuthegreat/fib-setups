"""
updater.py (FAST VERSION)

✓ Builds SP500, HSI, STI universes
✓ Downloads last 600 days of OHLC from Yahoo
✓ Parallel download using ThreadPoolExecutor (20 workers)
✓ Returns ONE CLEAN MERGED DATAFRAME
✓ No parquet saving, no disk IO
"""

import pandas as pd
import requests
from io import StringIO
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st




# ==========================================================
# 1. UNIVERSE BUILDERS
# ==========================================================

def get_sp500_universe():
    """Scrapes SP500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(r.text))

    for t in tables:
        if "Symbol" in t.columns:
            df = t.copy()
            break

    df["Ticker"] = df["Symbol"].str.replace(".", "-")
    df["Name"] = df["Security"]
    df["Sector"] = df["GICS Sector"]

    return df[["Ticker", "Name", "Sector"]]



def get_hsi_universe():
    """Scrapes Hang Seng Index constituents."""
    url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(r.text))

    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any(x in cols for x in ["ticker", "constituent", "sub-index"]):
            df = t.copy()
            break

    df.columns = [str(c).lower() for c in df.columns]

    ticker_col = None
    for c in df.columns:
        if "sehk" in c or "ticker" in c or "code" in c:
            ticker_col = c
            break

    df["Ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(str)
        .str.zfill(4)
        + ".HK"
    )

    name_col = "name" if "name" in df.columns else [c for c in df.columns if c != ticker_col][0]

    df["Name"] = df[name_col]
    df["Sector"] = df.get("sub-index", df.get("industry", None))

    return df[["Ticker", "Name", "Sector"]]



def get_sti_universe():
    """Hardcoded STI constituents (accurate as of 2024)."""
    data = [
        ("D05.SI", "DBS Group Holdings", "Financials"),
        ("U11.SI", "United Overseas Bank", "Financials"),
        ("O39.SI", "Oversea-Chinese Banking Corporation", "Financials"),
        ("C07.SI", "Jardine Matheson", "Conglomerate"),
        ("C09.SI", "City Developments", "Real Estate"),
        ("C38U.SI", "CapitaLand Integrated Commercial Trust", "Real Estate"),
        ("C52.SI", "ComfortDelGro", "Transportation"),
        ("F34.SI", "Frasers Logistics & Commercial Trust", "Real Estate"),
        ("G13.SI", "Genting Singapore", "Entertainment"),
        ("H78.SI", "Hongkong Land", "Real Estate"),
        ("J36.SI", "Jardine Cycle & Carriage", "Industrial"),
        ("M44U.SI", "Mapletree Logistics Trust", "Real Estate"),
        ("ME8U.SI", "Mapletree Industrial Trust", "Real Estate"),
        ("N2IU.SI", "NetLink NBN Trust", "Utilities"),
        ("S63.SI", "Singapore Airlines", "Transportation"),
        ("S68.SI", "Singapore Exchange", "Financials"),
        ("S58.SI", "Sembcorp Industries", "Utilities"),
        ("U96.SI", "SATS Ltd", "Services"),
        ("S07.SI", "Singapore Technologies Engineering", "Industrial"),
        ("Z74.SI", "Singtel", "Telecom"),
        ("BN4.SI", "Keppel Corporation", "Industrial"),
        ("M01.SI", "Micro-Mechanics", "Industrial"),
        ("A17U.SI", "CapitaLand Ascendas REIT", "Real Estate"),
        ("BS6.SI", "Yangzijiang Shipbuilding", "Industrial"),
        ("C31.SI", "CapitaLand Investment", "Real Estate"),
        ("E5H.SI", "Emperador Inc", "Consumer Staples"),
        ("5DP.SI", "Delfi Limited", "Consumer Goods"),
        ("D01.SI", "Dairy Farm International", "Consumer Staples"),
        ("K71U.SI", "Keppel DC REIT", "Real Estate"),
        ("H78.SI", "Hongkong Land Holdings", "Real Estate"),
    ]

    return pd.DataFrame(data, columns=["Ticker", "Name", "Sector"])



# ==========================================================
# 2. FAST YAHOO DOWNLOADER (Parallel Threads)
# ==========================================================

def fast_fetch(ticker, period="600d", label="UNKNOWN"):
    """
    Fetch OHLC data for ONE ticker.
    Called inside ThreadPoolExecutor.
    """
    try:
        df = yf.download(ticker, period=period, auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return None
        
        df = df.reset_index()
        df["Ticker"] = ticker
        df["Index"] = label
        return df

    except Exception:
        return None



def download_yahoo_prices(tickers, label, period="600d", max_workers=20):
    """
    FAST VERSION:
    - Downloads each ticker in parallel (20 threads default)
    - Much faster than batch download
    - Handles failures gracefully
    """
    print(f"\nDownloading {label}: {len(tickers)} tickers in parallel…")

    frames = []
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fast_fetch, t, period, label): t for t in tickers}

        for future in as_completed(futures):
            t = futures[future]
            df = future.result()

            if df is None:
                failed.append(t)
            else:
                frames.append(df)

    print(f"{label} completed — {len(frames)} OK, {len(failed)} failed")
    return frames



# ==========================================================
# 3. MASTER FUNCTION CALLED BY ENGINE / DASHBOARD
# ==========================================================
@st.cache_data(show_spinner=True, ttl=3600)
def load_all_market_data():
    """
    Returns a merged OHLC dataframe for:
    - SP500
    - HSI
    - STI
    Downloads ~600 tickers quickly using multithreading.
    """
    print("Building universes…")

    sp500 = get_sp500_universe()
    hsi   = get_hsi_universe()
    sti   = get_sti_universe()

    print("SP500:", len(sp500))
    print("HSI:  ", len(hsi))
    print("STI:  ", len(sti))

    # Parallel fetch for each universe
    sp = download_yahoo_prices(sp500["Ticker"].tolist(), "SP500")
    hs = download_yahoo_prices(hsi["Ticker"].tolist(),   "HSI")
    st = download_yahoo_prices(sti["Ticker"].tolist(),   "STI")

    # Merge ALL
    combined = pd.concat(sp + hs + st, ignore_index=True)

    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    print("\nFINAL dataframe shape:", combined.shape)
    return combined


