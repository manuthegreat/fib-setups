"""
data_loader.py

✓ Builds SP500, HSI, STI universes
✓ Downloads last 600 days of OHLC from Yahoo
✓ Batches requests (fast + avoids throttles)
✓ Returns ONE CLEAN MERGED DATAFRAME
✓ No parquet saving, no disk IO
"""

import pandas as pd
import requests
from io import StringIO
import yfinance as yf
from datetime import datetime, timedelta



# ==========================================================
# 1. UNIVERSE BUILDERS
# ==========================================================

def get_sp500_universe():
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
# 2. YAHOO DOWNLOADER (600 days)
# ==========================================================

def download_yahoo_prices(tickers, label, period="600d"):
    """
    Downloads last 600 days of OHLC for a list of tickers.
    Batches of 40 tickers to avoid throttling.
    Returns a list of clean DataFrames.
    """
    print(f"\nDownloading {label}: {len(tickers)} tickers")

    batch_size = 40
    frames = []
    failed = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"  Batch {i//batch_size + 1}: {len(batch)} tickers")

        try:
            data = yf.download(batch, period=period, group_by="ticker", auto_adjust=False, threads=True)
        except Exception:
            failed.extend(batch)
            continue

        for t in batch:
            try:
                df = data[t].dropna().copy()
                df["Ticker"] = t
                df["Index"] = label
                frames.append(df.reset_index())
            except Exception:
                failed.append(t)

    print(f"Completed {label}: {len(frames)} OK, {len(failed)} failed")
    return frames



# ==========================================================
# 3. MASTER FUNCTION (dashboard/engine will call this)
# ==========================================================

def load_all_market_data():
    """
    Returns a merged OHLC dataframe for SP500 + HSI + STI
    without saving anything to disk.
    """
    print("Building universes...")

    sp500 = get_sp500_universe()
    hsi = get_hsi_universe()
    sti = get_sti_universe()

    print("SP500:", len(sp500))
    print("HSI:  ", len(hsi))
    print("STI:  ", len(sti))

    sp = download_yahoo_prices(sp500["Ticker"].tolist(), "SP500")
    hs = download_yahoo_prices(hsi["Ticker"].tolist(), "HSI")
    st = download_yahoo_prices(sti["Ticker"].tolist(), "STI")

    combined = pd.concat(sp + hs + st, ignore_index=True)

    # Standardize column order
    combined.rename(columns={"Date": "Date"}, inplace=True)
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    print("\nFinal merged dataframe shape:", combined.shape)
    return combined
