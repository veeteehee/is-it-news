"""
title_tone.py: same-day and next-day return on title tone, by what the title
says about the price move.

Run:
    python title_tone.py data/news.csv data/nifty50_prices.csv

Writes:
    results/title_tone_table.csv              main table
    results/title_tone_table_no_gainers.csv   robustness: without gainer(s)/loser(s)
    results/up_to_fix_comparison.csv          main table before / after the
                                              "up to" fix
    results/up_to_fix_changes.csv             titles whose tone the fix changed
    results/direction_words.csv               word list for the appendix

TONE, from the Title text only (rule 7: nothing from extraction enters it).
    Count whole words that denote price direction. Rise words: the UP pattern
    in pipeline.py (surge, jump, rally, rise, rose, gain, climb, soar, advance,
    higher, zoom, spike, up, with inflections) plus gainer/gainers. Fall
    words: the DOWN pattern (fall, fell, drop, decline, tumble, slip, slid,
    sank, sink, plunge, crash, lower, dip, shed, lose, lost, down, with
    inflections) plus loser/losers. Matches are whole words only, so
    "Enterprises" or "Lupin" do not count.
    tone = +1 if rise words > fall words, -1 if fewer, 0 if equal.
    Rows with tone == 0 are dropped.
    RANGE MARKER: "up to" (e.g. "shares fall up to 3%", "raise up to Rs 500
    crore") is a range marker, not a direction, and is removed before
    counting. Fixed 2026-11-22. Before the fix, "fall up to 3%" counted one
    rise and one fall word, netted to 0 and was dropped. Other range markers
    ("as much as", "as high as", "down to", ...) contain no direction word or
    do not occur in the matched titles, so need no rule.
    results/up_to_fix_comparison.csv reports the table before and after.

SAMPLE: Title + Keywords name exactly one NIFTY 50 company
    (pipeline.match_single), merged to that company's close-to-close return
    on the article date; rows without a valid same-day return are dropped.

ROWS of the table. Each article falls in exactly one of the last three:
    v3_move      v3 extractor accepts a realized move in the Title
    pct_rejected Title contains a percentage, but v3 does not accept it
                 (ranges, forecasts, multi-stock, results figures, etc.)
    no_pct       Title contains no percentage at all
    The extractor defines these rows only, never the tone.
    Only no_pct speaks to "does tone track the return without a number?".

Standard errors clustered by date.
"""
import re
import sys

import numpy as np
import pandas as pd

import pipeline as P

UP_EXTRA = r"gainers?"
DOWN_EXTRA = r"losers?"

RE_PCT = re.compile(P.PCT, re.I)
RE_RANGE_MARKER = re.compile(r"\bup\s+to\b", re.I)

ROW_ORDER = ["all", "v3_move", "pct_rejected", "no_pct"]
ROW_LABEL = {
    "all": "all matched articles",
    "v3_move": "title states a v3-accepted move",
    "pct_rejected": "title has a percentage v3 rejected",
    "no_pct": "title has no percentage",
}


def direction_patterns(include_gainers=True):
    up, dn = P.UP, P.DOWN
    if include_gainers:
        up = f"(?:{up}|{UP_EXTRA})"
        dn = f"(?:{dn}|{DOWN_EXTRA})"
    return re.compile(rf"\b{up}\b", re.I), re.compile(rf"\b{dn}\b", re.I)


def title_tone(title, up, dn, strip_ranges=True):
    s = str(title)
    if strip_ranges:
        s = RE_RANGE_MARKER.sub(" ", s)
    return int(np.sign(len(up.findall(s)) - len(dn.findall(s))))


def title_row(title):
    s = str(title)
    if P.extract(s, "v3")[1] == "realized":
        return "v3_move"
    if RE_PCT.search(s):
        return "pct_rejected"
    return "no_pct"


def matched_articles(news_path, price_path, drop_tickers=()):
    """One row per article naming exactly one NIFTY 50 company on a day with a
    valid same-day return. article_id = row number in news.csv."""
    news = pd.read_csv(news_path)
    news["article_id"] = news.index
    news["date"] = pd.to_datetime(news["Date"])
    news["hay"] = (news["Title"].fillna("").astype(str) + " | "
                   + news["Keywords"].fillna("").astype(str))
    news["ticker"] = news["hay"].map(P.match_single)

    px = pd.read_csv(price_path)
    px["date"] = pd.to_datetime(px["Date"])
    px["tic"] = px["ticker"].str.replace(".NS", "", regex=False)
    px = px.sort_values(["tic", "date"])
    px["ret"] = px.groupby("tic")["Close"].pct_change()
    px["ret_next"] = px.groupby("tic")["ret"].shift(-1)

    # drop_tickers: only for re-running earlier specifications (e.g. the
    # 48-ticker sample in next_day_sensitivity.py). Default: none dropped.
    news.loc[news["ticker"].isin(list(drop_tickers)), "ticker"] = None
    excl = P.check_price_coverage(news["ticker"], px["tic"].unique())
    news.loc[excl, "ticker"] = None
    m = news[news["ticker"].notna()].merge(
        px[["tic", "date", "ret", "ret_next"]],
        left_on=["ticker", "date"], right_on=["tic", "date"], how="inner")
    m = m[m["ret"].notna()].reset_index(drop=True)
    m["row"] = m["Title"].map(title_row)
    return m


def tone_series(m, include_gainers=True, strip_ranges=True):
    up, dn = direction_patterns(include_gainers)
    return m["Title"].map(lambda s: title_tone(s, up, dn, strip_ranges))


def with_tone(m, include_gainers=True, strip_ranges=True):
    t = tone_series(m, include_gainers, strip_ranges)
    return m.assign(tone=t)[t != 0]


def ols_clustered(x, y, cluster):
    """y = a + b*x, SEs clustered by `cluster`. Returns b, t, R2, n clusters."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(x, float)])
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    r = y - X @ b
    g = np.asarray(cluster)
    meat = np.zeros((k, k))
    for c in np.unique(g):
        s = X[g == c].T @ r[g == c]
        meat += np.outer(s, s)
    G = len(np.unique(g))
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    r2 = 1 - (r @ r) / ((y - y.mean()) ** 2).sum()
    return b[1], b[1] / np.sqrt(V[1, 1]), r2, G


def tone_return_row(s, name):
    """Same-day and next-day return (in %) on tone for one subsample."""
    b, t, r2, G = ols_clustered(s["tone"], s["ret"] * 100, s["date"].dt.date)
    nx = s[s["ret_next"].notna()]
    bn, tn, r2n, _ = ols_clustered(nx["tone"], nx["ret_next"] * 100, nx["date"].dt.date)
    return dict(sample=name, n=len(s), clusters=G, same_day_r2=r2,
                tone_coef=b, tone_t=t,
                sign_agreement=(np.sign(s["ret"]) == s["tone"]).mean(),
                next_day_n=len(nx), next_day_coef=bn, next_day_t=tn,
                next_day_r2=r2n)


def table(d):
    rows = []
    for key in ROW_ORDER:
        s = d if key == "all" else d[d["row"] == key]
        rows.append(tone_return_row(s, ROW_LABEL[key]))
    return pd.DataFrame(rows)


def show(tab, title):
    print(title)
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(tab.to_string(index=False))
    print()


def main(news_path, price_path):
    m = matched_articles(news_path, price_path)
    print(f"single-ticker, trading-day articles: {len(m)}")
    print("  by title row:", m["row"].value_counts().to_dict(), "\n")

    d = with_tone(m, include_gainers=True)
    tab = table(d)
    show(tab, f"MAIN: tone incl. gainer(s)/loser(s), tone != 0: {len(d)}")
    tab.to_csv("results/title_tone_table.csv", index=False)

    d2 = with_tone(m, include_gainers=False)
    tab2 = table(d2)
    show(tab2, f"ROBUSTNESS: tone without gainer(s)/loser(s), tone != 0: {len(d2)}")
    tab2.to_csv("results/title_tone_table_no_gainers.csv", index=False)

    # ---- "up to" fix: before / after, main specification ----
    before = table(with_tone(m, include_gainers=True, strip_ranges=False))
    cmp_ = pd.concat([before.assign(version="before fix"),
                      tab.assign(version="after fix")], ignore_index=True)
    cmp_ = cmp_[["version"] + [c for c in cmp_.columns if c != "version"]]
    show(cmp_, "\"UP TO\" FIX: main table before and after")
    cmp_.to_csv("results/up_to_fix_comparison.csv", index=False)

    t0 = tone_series(m, strip_ranges=False)
    t1 = tone_series(m, strip_ranges=True)
    ch = m.assign(tone_before=t0, tone_after=t1)[t0 != t1]
    ch = ch.assign(same_day_ret_pct=ch["ret"] * 100,
                   sign_agrees_after=np.where(ch["tone_after"] != 0,
                                              np.sign(ch["ret"]) == ch["tone_after"],
                                              np.nan))
    ch[["article_id", "Title", "row", "tone_before", "tone_after",
        "same_day_ret_pct", "sign_agrees_after"]].to_csv(
        "results/up_to_fix_changes.csv", index=False)
    print(f"titles whose tone the fix changed: {len(ch)}")
    for (a, b), g in ch.groupby(["tone_before", "tone_after"]):
        rows = g["row"].value_counts().to_dict()
        agr = (f", sign agreement after fix {g['sign_agrees_after'].mean():.1%}"
               if b != 0 else "")
        print(f"  tone {a:+d} -> {b:+d}: {len(g):>3}  rows {rows}{agr}")
    print()

    words = direction_word_list(d)
    words.to_csv("results/direction_words.csv", index=False)
    print("wrote results/title_tone_table.csv, results/title_tone_table_no_gainers.csv,")
    print("      results/up_to_fix_comparison.csv, results/up_to_fix_changes.csv,")
    print(f"      results/direction_words.csv ({len(words)} word forms)")


# ---- word list for the appendix -------------------------------------------
try:
    import re._parser as _sre      # Python 3.11+
except ImportError:                # pragma: no cover
    import sre_parse as _sre


def _expand(items):
    """All strings matched by a parsed regex made of literals, character
    classes, alternation, groups and optional parts (the only constructs the
    UP / DOWN / extra patterns use). Word boundaries are ignored."""
    C = _sre._constants if hasattr(_sre, "_constants") else _sre
    out = [""]
    for op, av in items:
        if op == C.LITERAL:
            opts = [chr(av)]
        elif op == C.IN:
            opts = []
            for o, a in av:
                if o == C.LITERAL:
                    opts.append(chr(a))
                elif o == C.RANGE:
                    opts += [chr(c) for c in range(a[0], a[1] + 1)]
                else:
                    raise ValueError(f"unsupported class item {o}")
        elif op == C.BRANCH:
            opts = [w for alt in av[1] for w in _expand(alt)]
        elif op == C.SUBPATTERN:
            opts = _expand(av[-1])
        elif op in (C.MAX_REPEAT, C.MIN_REPEAT):
            lo, hi, sub = av
            if hi > 3:
                raise ValueError("unbounded repeat")
            base = _expand(sub)
            opts = []
            for k in range(lo, hi + 1):
                acc = [""]
                for _ in range(k):
                    acc = [a + b for a in acc for b in base]
                opts += acc
        elif op == C.AT:
            continue
        else:
            raise ValueError(f"unsupported regex op {op}")
        out = [a + b for a in out for b in opts]
    return out


def expand_pattern(p):
    return sorted(set(w.lower() for w in _expand(_sre.parse(p))))


def direction_word_list(d):
    """Every word form the tone measure counts, its source, and how many
    titles in the tone sample `d` contain it (as a whole word)."""
    rows = []
    for direction, pat, src in [("rise", P.UP, "pipeline.py UP"),
                                ("rise", UP_EXTRA, "added: gainer(s)"),
                                ("fall", P.DOWN, "pipeline.py DOWN"),
                                ("fall", DOWN_EXTRA, "added: loser(s)")]:
        for w in expand_pattern(pat):
            rx = re.compile(rf"\b{re.escape(w)}\b", re.I)
            rows.append(dict(direction=direction, word=w, source=src,
                             titles_in_tone_sample=int(d["Title"].astype(str)
                                                       .str.contains(rx).sum())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
