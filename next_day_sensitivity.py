"""
next_day_sensitivity.py: how the next-day estimate moves across the
specification changes made on 2026-11-22.

Run:
    python next_day_sensitivity.py data/news.csv data/nifty50_prices.csv

Writes results/next_day_spec_sensitivity.csv.

Specifications, in the order they were adopted (main tone measure, incl.
gainer(s)/loser(s), whole-word matching throughout):
    A  "up to" counted as a rise word; 48 tickers (KOTAKBANK and TATAMOTORS
       dropped, as when they had no price data)
    B  "up to" removed before counting; 48 tickers
    C  "up to" removed; 50 tickers (current specification)
Each change was a bug fix or a coverage fix decided on its own grounds, not
chosen by its effect on this estimate. The earlier substring-matching table
(n = 688) is not included: it was a bug, not a specification.

For each specification and each table row: next-day return (%) on tone,
SEs clustered by date, the t-statistic, and a two-sided p-value from a
t distribution with (clusters - 1) degrees of freedom.

Why this exists: the all-matched next-day t-statistic moved with every change
(A -> B -> C). None is significant at 5%, but the paper should report the
range across specifications and call the next-day estimate
specification-sensitive, not quote the last value as if it were stable.
"""
import sys

import pandas as pd
from scipy import stats

import title_tone as T

SPECS = [
    ("A", "up-to counted, 48 tickers", dict(strip_ranges=False), ("KOTAKBANK", "TATAMOTORS")),
    ("B", "up-to removed, 48 tickers", dict(strip_ranges=True), ("KOTAKBANK", "TATAMOTORS")),
    ("C", "up-to removed, 50 tickers (current)", dict(strip_ranges=True), ()),
]


def main(news_path, price_path):
    rows = []
    for code, desc, tone_kw, drop in SPECS:
        m = T.matched_articles(news_path, price_path, drop_tickers=drop)
        d = T.with_tone(m, include_gainers=True, **tone_kw)
        for key in T.ROW_ORDER:
            s = d if key == "all" else d[d["row"] == key]
            nx = s[s["ret_next"].notna()]
            b, t, r2, G = T.ols_clustered(nx["tone"], nx["ret_next"] * 100,
                                          nx["date"].dt.date)
            rows.append(dict(spec=code, spec_description=desc,
                             sample=T.ROW_LABEL[key], n=len(nx), clusters=G,
                             next_day_coef=b, next_day_t=t,
                             next_day_p=2 * stats.t.sf(abs(t), G - 1),
                             next_day_r2=r2))
    tab = pd.DataFrame(rows)
    tab.to_csv("results/next_day_spec_sensitivity.csv", index=False)

    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(tab.drop(columns="spec_description").to_string(index=False))
    print()
    for sample, g in tab.groupby("sample", sort=False):
        print(f"{sample:<36} t range {g['next_day_t'].min():+.2f} .. "
              f"{g['next_day_t'].max():+.2f}   p range {g['next_day_p'].min():.3f} .. "
              f"{g['next_day_p'].max():.3f}")
    print("\nwrote results/next_day_spec_sensitivity.csv")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
