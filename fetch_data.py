"""
fetch_data.py: download the inputs into data/. The repo ships code, not data.

Run (needs network access to zenodo.org and Yahoo Finance):
    python fetch_data.py

Writes:
    data/news.csv             Zenodo record 16991843, the "Indian Stock Market
                              News Dataset" (Moneycontrol articles), CC BY 4.0
    data/zenodo_record.json   the record's metadata as served by the Zenodo
                              API (title, creators, DOI, licence, files)
    data/nifty50_prices.csv   daily bars for the 50 tickers in TICKERS, yfinance

TICKER SYMBOLS: each ticker is downloaded as <TICKER>.NS except where
YAHOO_SYMBOL says otherwise. Tata Motors is downloaded as TMPV.NS: NSE renamed
TATAMOTORS to TMPV on 2025-10-24 after the commercial-vehicle demerger
(effective 2025-10-01, record date 2025-10-14), and Yahoo serves the full
history, including the pre-demerger period, under TMPV.NS. That history is NOT
adjusted for the demerger (Close falls 40% on 2025-10-14), which lies outside
the article window (2025-02-07 .. 2025-08-21). The file stores it under the
ticker TATAMOTORS.NS so it matches pipeline.NAMES.
KOTAKBANK and TATAMOTORS were added on 2026-11-22 after both passed a probe
with criteria fixed before any download succeeded (full trading-day coverage
of the article window, no missing Close, no daily move over 20%). 

CHECKS against the files the reported results used:
    news.csv   SHA-256 of the whole file (must match).
    PRIMARY price checks (must pass):
      dates    SHA-256 of the sorted (ticker, Date) pairs: same tickers, same
               trading days, nothing missing or extra.
      returns  per-ticker summary statistics of daily close-to-close returns,
               compared with reference/price_return_stats.csv to within
               RETURN_TOL. Statistics: number of returns, sum of returns, sum
               of absolute returns, and a time-weighted sum (each return times
               its position / n, so a shifted or reordered day changes it).
               A change of more than about RETURN_TOL in any single return
               moves the sum of absolute returns by that much and fails.
    SECONDARY (informational only): SHA-256 of the (ticker, Date, Close)
      triples. Expected to change whenever Yahoo re-adjusts a ticker's
      history for a split, as it did for Kotak's 1:5 split.
    Why returns are compared with a tolerance and not hashed: returns are
    invariant to split adjustment in exact arithmetic, but Yahoo stores
    prices at float32 precision, so re-adjusting a series moves returns by up
    to ~1.5e-7. A simulated 1:5 re-adjustment of all 50 series changed a hash
    of returns even after rounding to 4 decimal places (51 of 35,650 returns
    rounded differently), while every per-ticker statistic moved by < 1e-5.
    Why no whole-file hash: yfinance re-downloads differ in ways that do not
    matter (Volume written as 2898619 vs 2898619.0, ticker block order,
    float32-level noise in Adj Close; see compare_prices.py).

Regenerate the reference statistics (only when the reference data change,
and commit the result):
    python fetch_data.py --write-return-reference
"""
import hashlib
import json
import os
import sys

import pandas as pd
import requests

ZENODO_RECORD = "16991843"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

START = "2024-01-01"
END = "2026-11-21"          # yfinance `end` is exclusive: last bar 2026-11-20

TICKERS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BEL", "BHARTIARTL", "CIPLA",
    "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH",
    "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR",
    "ICICIBANK", "INDUSINDBK", "INFY", "ITC", "JIOFIN", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA",
    "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN",
    "TRENT", "ULTRACEMCO", "WIPRO",
]
YAHOO_SYMBOL = {"TATAMOTORS": "TMPV.NS"}

REFERENCE_NEWS_SHA256 = "fa38c44557e7e7964ffc0c724dc5e2447c5f6ecca9ae2a23426f280da5e30535"
REFERENCE_DATES_SHA256 = "5432c436a813362d2e0b283f2d742c8acf3b1e691b0f3eea631efb4430db00b1"
REFERENCE_RETURN_STATS = "reference/price_return_stats.csv"
RETURN_TOL = 1e-5
REFERENCE_CLOSE_SHA256 = "ec4d925643aeb1bf810b9bf77ded2fe5f5d5949d83b6ddeec2d3869358403d12"
STAT_COLS = ["n_returns", "sum_ret", "sum_abs_ret", "sum_time_weighted_ret"]

PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

# Expected citation fields; compared with the Zenodo record.
EXPECTED_CITATION = {
    "doi": "10.5281/zenodo.16991843",
    "creators": ["Chakraborty, Sayan", "Barman, Debaditya", "Sufian, Abu"],
    "title": "Indian Stock Market News Dataset",
    "license": "cc-by-4.0",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def close_fingerprint(path):
    """SHA-256 of sorted 'ticker,Date,Close' lines, Close written with
    repr() so every bit of the float counts."""
    p = pd.read_csv(path)
    p["Date"] = pd.to_datetime(p["Date"]).dt.strftime("%Y-%m-%d")
    lines = sorted(f"{t},{d},{c!r}" for t, d, c in zip(p["ticker"], p["Date"], p["Close"]))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def load_prices(path):
    p = pd.read_csv(path)
    p["Date"] = pd.to_datetime(p["Date"]).dt.strftime("%Y-%m-%d")
    return p.sort_values(["ticker", "Date"]).reset_index(drop=True)


def dates_fingerprint(path):
    p = load_prices(path)
    lines = sorted(f"{t},{d}" for t, d in zip(p["ticker"], p["Date"]))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def return_stats(path):
    """Per-ticker summary statistics of daily close-to-close returns."""
    p = load_prices(path)
    p["ret"] = p.groupby("ticker")["Close"].pct_change()
    rows = []
    for t, g in p.groupby("ticker"):
        r = g["ret"].dropna().to_numpy()
        w = (pd.RangeIndex(len(r)).to_numpy() + 1) / len(r)
        rows.append(dict(ticker=t, n_returns=len(r), sum_ret=r.sum(),
                         sum_abs_ret=abs(r).sum(), sum_time_weighted_ret=(w * r).sum()))
    return pd.DataFrame(rows).set_index("ticker")


def check_returns(path):
    ref = pd.read_csv(REFERENCE_RETURN_STATS).set_index("ticker")
    got = return_stats(path)
    if set(ref.index) != set(got.index):
        print(f"  returns: ticker sets differ: only reference "
              f"{sorted(set(ref.index) - set(got.index))}, only new "
              f"{sorted(set(got.index) - set(ref.index))}")
        return False
    got = got.loc[ref.index]
    diff = (got[STAT_COLS] - ref[STAT_COLS]).abs()
    bad = diff[(diff > RETURN_TOL).any(axis=1)]
    print(f"  {'returns (per-ticker statistics)':<34} "
          f"{'match' if bad.empty else 'DIFFER from'} the reference "
          f"(largest difference {diff.to_numpy().max():.1e}, tolerance {RETURN_TOL:.0e})")
    if not bad.empty:
        print(bad.to_string(float_format=lambda x: f"{x:.2e}"))
    return bad.empty


def write_return_reference(path="data/nifty50_prices.csv"):
    os.makedirs(os.path.dirname(REFERENCE_RETURN_STATS), exist_ok=True)
    return_stats(path).to_csv(REFERENCE_RETURN_STATS, float_format="%.12g")
    print(f"wrote {REFERENCE_RETURN_STATS}")
    print(f"dates fingerprint: {dates_fingerprint(path)}")


def check(label, got, want):
    ok = got == want
    print(f"  {label:<34} {'matches' if ok else 'DIFFERS from'} the reference")
    if not ok:
        print(f"    reference {want}\n    got       {got}")
    return ok


def fetch_news():
    meta = requests.get(ZENODO_API, timeout=60).json()
    with open("data/zenodo_record.json", "w") as f:
        json.dump(meta, f, indent=2)
    csvs = [f for f in meta["files"] if f["key"].lower().endswith(".csv")]
    if len(csvs) != 1:
        sys.exit(f"Expected one CSV in the Zenodo record, found {[f['key'] for f in csvs]}")
    f = csvs[0]
    r = requests.get(f["links"]["self"], timeout=300)
    r.raise_for_status()
    algo, want = f["checksum"].split(":", 1)
    if hashlib.new(algo, r.content).hexdigest() != want:
        sys.exit(f"{f['key']}: checksum does not match the Zenodo record")
    with open("data/news.csv", "wb") as out:
        out.write(r.content)
    md = meta.get("metadata", {})
    print(f"  news.csv <- {f['key']}")
    got = {"doi": meta.get("doi") or md.get("doi"),
           "creators": [c.get("name") for c in md.get("creators", [])],
           "title": md.get("title"),
           "license": (md.get("license") or {}).get("id")}
    for k, want in EXPECTED_CITATION.items():
        same = got[k] == want
        print(f"  citation {k:<9} {'matches expected' if same else 'DIFFERS from expected'}"
              + ("" if same else f": record has {got[k]!r}, expected {want!r}"))


def bars(symbol):
    import yfinance as yf
    df = yf.download(symbol, start=START, end=END, auto_adjust=False,
                     progress=False, multi_level_index=False)
    if df.empty:
        raise RuntimeError(f"no data returned for {symbol}")
    df = df[PRICE_COLS].reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df


def fetch_prices():
    frames, failed = [], []
    for t in TICKERS:
        sym = YAHOO_SYMBOL.get(t, f"{t}.NS")
        try:
            frames.append(bars(sym).assign(ticker=f"{t}.NS"))
        except Exception as e:  # report every failure, do not skip silently
            failed.append(f"{t} ({sym}): {e}")
    if failed:
        sys.exit("Price download failed for:\n  " + "\n  ".join(failed))
    pd.concat(frames, ignore_index=True)[["Date"] + PRICE_COLS + ["ticker"]] \
        .to_csv("data/nifty50_prices.csv", index=False)
    print(f"  prices for {len(TICKERS)} tickers, {START} to {END} (exclusive)")


def main():
    os.makedirs("data", exist_ok=True)
    print("Fetching")
    fetch_news()
    fetch_prices()
    print("Checking against the files the reported results used")
    ok = check("news.csv (whole file)", sha256_file("data/news.csv"), REFERENCE_NEWS_SHA256)
    px = "data/nifty50_prices.csv"
    ok &= check("prices: (ticker, Date) pairs", dates_fingerprint(px), REFERENCE_DATES_SHA256)
    ok &= check_returns(px)
    print("Secondary (informational; changes when Yahoo re-adjusts for a split)")
    check("prices: (ticker, Date, Close)", close_fingerprint(px), REFERENCE_CLOSE_SHA256)
    print("\nPRIMARY CHECKS " + ("PASSED" if ok else "FAILED: results may not reproduce"))


if __name__ == "__main__":
    if sys.argv[1:] == ["--write-return-reference"]:
        write_return_reference()
    else:
        main()
