"""
pipeline.py — contamination measurement for Indian financial news.

Run:
    python pipeline.py data/news.csv data/nifty50_prices.csv

Produces the extractor comparison table, the v3 accuracy detail, the worst
remaining extraction errors, and results_v3_matched.csv.

No tone or sentiment regressions live here. An earlier version regressed
returns on sign(extracted move); that was circular and has been retracted and
removed. Title-tone regressions are in title_tone.py.
"""
import re
import sys
import numpy as np
import pandas as pd

NAMES = {
    "RELIANCE": ["reliance industries"], "TCS": ["tata consultancy", r"\btcs\b"],
    "HDFCBANK": ["hdfc bank"], "ICICIBANK": ["icici bank"], "INFY": ["infosys"],
    "BHARTIARTL": ["bharti airtel"], "ITC": [r"\bitc\b"], "SBIN": ["state bank of india"],
    "LT": ["larsen", r"l&t\b"], "HINDUNILVR": ["hindustan unilever", r"\bhul\b"],
    "AXISBANK": ["axis bank"], "KOTAKBANK": ["kotak mahindra bank", "kotak bank"],
    "MARUTI": ["maruti"], "SUNPHARMA": ["sun pharma"], "TITAN": ["titan"],
    "ASIANPAINT": ["asian paints"], "BAJFINANCE": ["bajaj finance"],
    "NESTLEIND": ["nestle"], "WIPRO": ["wipro"], "HCLTECH": ["hcl tech"],
    "ULTRACEMCO": ["ultratech"], "TATAMOTORS": ["tata motors"], "TATASTEEL": ["tata steel"],
    "NTPC": [r"\bntpc\b"], "POWERGRID": ["power grid", "powergrid"], "ONGC": [r"\bongc\b"],
    "COALINDIA": ["coal india"], "ADANIENT": ["adani enterprises"],
    "ADANIPORTS": ["adani ports"], "JSWSTEEL": ["jsw steel"], "TECHM": ["tech mahindra"],
    "CIPLA": ["cipla"], "DRREDDY": [r"dr\.? reddy"], "GRASIM": ["grasim"],
    "HINDALCO": ["hindalco"], "EICHERMOT": ["eicher"], "HEROMOTOCO": ["hero motocorp"],
    "BAJAJ-AUTO": ["bajaj auto"], "M&M": ["mahindra & mahindra"],
    "INDUSINDBK": ["indusind"], "APOLLOHOSP": ["apollo hospital"],
    "TATACONSUM": ["tata consumer"], "BEL": ["bharat electronics"],
    "TRENT": ["trent"], "SHRIRAMFIN": ["shriram finance"], "SBILIFE": ["sbi life"],
    "HDFCLIFE": ["hdfc life"], "BAJAJFINSV": ["bajaj finserv"], "JIOFIN": ["jio financial"],
    "ETERNAL": ["eternal", "zomato"],
}

# Non-NIFTY names that commonly appear alongside, used only to detect
# multi-stock sentences.
OTHERS = [
    "ioc", "bpcl", "hpcl", "nalco", "vedanta", "swiggy", "yes bank", "rites",
    "cochin shipyard", "south indian bank", "paras defence", "hal",
    "hindustan aeronautics", "bdl", "mazagon", "cummins", "avanti feeds",
    "berger paints", "dabur", "colgate", "britannia", "godrej", "jubilant",
    "dmart", "avenue supermarts", "sbi cards", "rbl bank", "hdb financial",
    "tata communications", "raymond", "dlf", "prestige", "glenmark",
    "aurobindo", "lupin", "zydus", "granules", "torrent", "ajanta",
    "laurus", "biocon", "alkem", "divi", "mankind", "natco", "ipca",
    "gland pharma", "abbot", "tvs motor", "voltas", "havells", "mrf",
    "chambal", "ideaforge", "motilal", "jefferies", "macquarie", "nomura",
    "goldman sachs", "clsa", "ubs", "hsbc", "kotak securities",
]

UP = r"(?:surge[sd]?|jump(?:s|ed)?|rall(?:y|ies|ied)|rise[sn]?|rose|gain(?:s|ed)?|climb(?:s|ed)?|soar(?:s|ed)?|advance[sd]?|higher|zoom(?:s|ed)?|spike[sd]?|up)"
DOWN = r"(?:fall[s]?|fell|drop(?:s|ped)?|decline[sd]?|tumble[sd]?|slip(?:s|ped)?|slid[es]?|sank|sink[s]?|plunge[sd]?|crash(?:es|ed)?|lower|dip(?:s|ped)?|shed[s]?|lose[s]?|lost|down)"
PCT = r"(\d+(?:\.\d+)?)\s*(?:%|per\s?cent)"

EQUITY = re.compile(r"\b(?:share|shares|stock|stocks|scrip|share price|stock price)\b", re.I)
FUNDAMENTAL = re.compile(
    r"net sales|revenue|net profit|profit|margin|ebitda|\beps\b|dividend|"
    r"y-?o-?y|q-?o-?q|yoy|qoq|order book|aum|nii|asset quality|"
    r"growth in|grew|crore|market share|stake|\bipo\b|gdp|inflation|repo",
    re.I)
FORECAST = re.compile(
    r"target|upside|downside|can rally|can rise|could rally|could rise|sees?\s+\d|"
    r"initiat\w+ coverage|implies|potential|expect|forecast|estimate|valuation|"
    r"recommend|'?buy'?|'?sell'?|'?hold'?|outperform|underperform|overweight|neutral",
    re.I)
# NEW in v3
RANGE = re.compile(
    r"up to\s+\d|as much as\s+\d|between\s+\d|\d+\s*[-–]\s*\d+\s*(?:%|per\s?cent)", re.I)
CORP_ACTION = re.compile(
    r"bonus|stock split|share split|split|adjust\w*|ex-date|record date|"
    r"rights issue|buyback|demerger|face value", re.I)
RE_UP = re.compile(rf"{UP}[^.;]{{0,25}}?{PCT}|{PCT}[^.;]{{0,25}}?{UP}", re.I)
RE_DN = re.compile(rf"{DOWN}[^.;]{{0,25}}?{PCT}|{PCT}[^.;]{{0,25}}?{DOWN}", re.I)

ALL_NAME_PATS = [("|".join(f), t) for t, f in NAMES.items()]

# Tickers in NAMES deliberately left without price data, with the reason.
# Articles matched to these are excluded and counted, never dropped silently.
# check_price_coverage() fails if any other matched ticker lacks prices, or if
# a ticker listed here turns out to have prices.
EXCLUDED_TICKERS = {}
# History: KOTAKBANK and TATAMOTORS were listed here until 2026-11-22 (no price
# data in the original price file). Both were then downloaded and passed a
# pre-set probe; Tata Motors is sourced from TMPV.NS (see fetch_data.py).


def check_price_coverage(article_tickers, price_tickers):
    """Fail loudly unless every ticker matched by an article has prices or is
    listed in EXCLUDED_TICKERS. Returns a boolean mask of excluded articles."""
    article_tickers = pd.Series(article_tickers)
    have = set(price_tickers)
    stale = sorted(t for t in EXCLUDED_TICKERS if t in have)
    if stale:
        raise RuntimeError(f"EXCLUDED_TICKERS lists tickers that now have "
                           f"prices: {stale}. Remove them from the list.")
    used = set(article_tickers.dropna())
    missing = sorted(used - have - set(EXCLUDED_TICKERS))
    if missing:
        n = int(article_tickers.isin(missing).sum())
        raise RuntimeError(f"{n} articles match tickers with no price data and "
                           f"no recorded exclusion: {missing}")
    excl = article_tickers.isin(list(EXCLUDED_TICKERS))
    if excl.any():
        counts = article_tickers[excl].value_counts().to_dict()
        print(f"  excluded (no price data, see EXCLUDED_TICKERS): {counts}")
    return excl.to_numpy()


def names_in(sentence):
    """How many distinct companies does this sentence name?"""
    n = sum(1 for pat, _ in ALL_NAME_PATS if re.search(pat, sentence, re.I))
    n += sum(1 for o in OTHERS if re.search(re.escape(o), sentence, re.I))
    return n


def first_pct(m):
    return float(next(g for g in m.groups() if g is not None))


def extract(text, level):
    """level: 'loose' | 'strict' | 'v3'. Returns (signed_pct, kind)."""
    for s in re.split(r"(?<=[.;])\s+", str(text))[:6]:
        u, d = RE_UP.search(s), RE_DN.search(s)
        if not (u or d):
            continue
        if u and d:
            m, sign = (u, 1.0) if u.start() < d.start() else (d, -1.0)
        elif u:
            m, sign = u, 1.0
        else:
            m, sign = d, -1.0
        if FORECAST.search(s):
            return sign * first_pct(m), "target"
        if level in ("strict", "v3"):
            if not EQUITY.search(s) or FUNDAMENTAL.search(s) or RANGE.search(s):
                continue
        if level == "v3":
            if CORP_ACTION.search(s) or names_in(s) > 1:
                continue
        return sign * first_pct(m), "realized"
    return np.nan, None


def match_single(hay):
    hits = [t for pat, t in ALL_NAME_PATS if re.search(pat, hay, re.I)]
    return hits[0] if len(hits) == 1 else None


def main(news_path, price_path):
    news = pd.read_csv(news_path)
    news["date"] = pd.to_datetime(news["Date"])
    news["hay"] = (news["Title"].fillna("").astype(str) + " | "
                   + news["Keywords"].fillna("").astype(str))
    news["ticker"] = news["hay"].map(match_single)

    px = pd.read_csv(price_path)
    px["date"] = pd.to_datetime(px["Date"])
    px["tic"] = px["ticker"].str.replace(".NS", "", regex=False)
    px = px.sort_values(["tic", "date"])
    px["ret"] = px.groupby("tic")["Close"].pct_change()
    px["ret_next"] = px.groupby("tic")["ret"].shift(-1)
    px = px[["tic", "date", "ret", "ret_next"]]
    excl = check_price_coverage(news["ticker"], px["tic"].unique())
    news.loc[excl, "ticker"] = None

    results = {}
    for level in ["loose", "strict", "v3"]:
        t = news["Title"].map(lambda s, L=level: extract(s, L))
        b = news["Content"].map(lambda s, L=level: extract(s, L))
        mv = [x[0] if not pd.isna(x[0]) else y[0] for x, y in zip(t, b)]
        kd = [x[1] if x[1] else y[1] for x, y in zip(t, b)]
        n2 = news.assign(move=mv, kind=kd)
        real = n2["kind"] == "realized"
        m = n2[real & n2["ticker"].notna()].merge(
            px, left_on=["ticker", "date"], right_on=["tic", "date"], how="inner")
        m = m[m["ret"].notna()]
        results[level] = (real, m)

    print("=" * 78)
    print("EXTRACTOR COMPARISON — does the stated move equal the actual return?")
    print("=" * 78)
    hdr = f"{'extractor':<10}{'% of corpus':>13}{'matched':>9}{'corr':>8}{'sign agr':>10}{'med err':>10}"
    print(hdr)
    print("-" * 78)
    rows = []
    for level, (real, m) in results.items():
        st, ac = m["move"] / 100, m["ret"]
        print(f"{level:<10}{real.mean():>12.1%}{len(m):>9}"
              f"{st.corr(ac):>+8.3f}{(np.sign(st) == np.sign(ac)).mean():>9.1%}"
              f"{abs(st-ac).median()*100:>9.2f}pp")
        rows.append(dict(extractor=level, n_flagged=int(real.sum()),
                         share_flagged=real.mean(), matched=len(m),
                         corr=st.corr(ac),
                         sign_agreement=(np.sign(st) == np.sign(ac)).mean(),
                         median_abs_error_pp=abs(st - ac).median() * 100,
                         within_0_5pp=(abs(st - ac) < 0.005).mean(),
                         within_1pp=(abs(st - ac) < 0.01).mean()))
    print()
    pd.DataFrame(rows).to_csv("results/extractor_table.csv", index=False)

    real, m = results["v3"]
    print("=" * 78)
    print("v3 DETAIL")
    print("=" * 78)
    st, ac = m["move"] / 100, m["ret"]
    print(f"  articles flagged as restating a realized price move: "
          f"{real.sum():,} of {len(news):,} ({real.mean():.1%})")
    print(f"  validation sample (one ticker, trading day):         {len(m):,}")
    print(f"  within 0.5pp of actual: {(abs(st-ac) < 0.005).mean():.1%}"
          f"    within 1.0pp: {(abs(st-ac) < 0.01).mean():.1%}")
    print()

    print("  remaining worst errors:")
    d2 = m.assign(err=abs(m["move"]/100 - m["ret"])).nlargest(6, "err")
    for _, r in d2.iterrows():
        print(f"    stated {r['move']:+6.2f}%  actual {r['ret']*100:+6.2f}%  "
              f"{r['ticker']:<11} {str(r['Title'])[:58]}")

    m.to_csv("results_v3_matched.csv", index=False)
    print("\n  wrote results_v3_matched.csv")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
