"""
finbert_regressions.py: Tables 2 and 3 re-run with the FinBERT tone measure,
beside the word-count tone, on matching and on common samples.

Run (after title_tone.py, analyse_labels.py and score_finbert.py):
    python finbert_regressions.py data/news.csv data/nifty50_prices.csv

Writes results/finbert_regressions.csv (long format, one row per
measure x sample x group).

Measures (mapping fixed in score_finbert.py before any result was seen):
    wordcount   title_tone word-count tone (main specification)
    finbert     sign of FinBERT argmax class (primary)
    finbert_cont  P(positive) - P(negative) (secondary; check only)
Samples:
    own           each measure on its own non-zero rows (as Table 2 does)
    intersection  rows where both word-count and FinBERT tone are non-zero
    finbert_only  FinBERT non-zero, word-count zero
    all_rows      every matched row (continuous measure only; no zero to drop)
Groups: the four Table 2 rows, and the two hand-label groups of Table 3.
Hand labels exist only for the 153 no-percentage headlines with non-zero
word-count tone, so label groups are restricted to those.
Specification as title_tone.tone_return_row: same-day and next-day return
(%) on tone, SEs clustered by date; p from t with (clusters - 1) df.
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

import title_tone as T
from score_finbert import CACHE, finbert_tone


def reg(s, measure, sample, group):
    if len(s) < 3 or s["date"].dt.date.nunique() < 2:
        return dict(measure=measure, sample=sample, group=group, n=len(s))
    r = T.tone_return_row(s, group)
    r["sign_agreement"] = (np.sign(s["ret"]) == np.sign(s["tone"])).mean()
    r["same_day_p"] = 2 * stats.t.sf(abs(r["tone_t"]), r["clusters"] - 1)
    r["next_day_p"] = 2 * stats.t.sf(abs(r["next_day_t"]), r["clusters"] - 1)
    r.update(measure=measure, sample=sample, group=group)
    return r


def groups(d):
    yield "all", d
    for key in ["v3_move", "pct_rejected", "no_pct"]:
        yield key, d[d["row"] == key]
    lab = d.dropna(subset=["label"])
    for key in ["describes_price_move", "describes_event"]:
        yield f"label:{key}", lab[lab["label"] == key]


def main(news_path, price_path):
    m = T.matched_articles(news_path, price_path)
    m["wc"] = T.tone_series(m)
    c = pd.read_parquet(CACHE)
    m = m.merge(c, on="article_id", validate="one_to_one")
    m["fb"] = finbert_tone(m)
    m["fb_cont"] = m["p_positive"] - m["p_negative"]
    lab = pd.read_csv("results/labelling_sample.csv")[["article_id", "label"]]
    m = m.merge(lab, on="article_id", how="left")

    rows = []
    specs = [
        ("wordcount", "own", m[m["wc"] != 0], "wc"),
        ("finbert", "own", m[m["fb"] != 0], "fb"),
        ("wordcount", "intersection", m[(m["wc"] != 0) & (m["fb"] != 0)], "wc"),
        ("finbert", "intersection", m[(m["wc"] != 0) & (m["fb"] != 0)], "fb"),
        ("finbert", "finbert_only", m[(m["fb"] != 0) & (m["wc"] == 0)], "fb"),
        ("finbert_cont", "all_rows", m, "fb_cont"),
        ("finbert_cont", "own_wordcount_rows", m[m["wc"] != 0], "fb_cont"),
    ]
    for measure, sample, d, col in specs:
        d = d.assign(tone=d[col])
        for g, s in groups(d):
            rows.append(reg(s, measure, sample, g))
    tab = pd.DataFrame(rows)
    cols = ["measure", "sample", "group", "n", "clusters", "same_day_r2", "tone_coef",
            "tone_t", "same_day_p", "sign_agreement", "next_day_coef", "next_day_t",
            "next_day_p"]
    tab = tab.reindex(columns=cols)
    tab.to_csv("results/finbert_regressions.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200,
                           "display.float_format", "{:.3f}".format):
        for (meas, samp), g in tab.groupby(["measure", "sample"], sort=False):
            print(f"\n== {meas} | {samp}")
            print(g.drop(columns=["measure", "sample"]).to_string(index=False))
    fo = m[(m["fb"] != 0) & (m["wc"] == 0)]
    print(f"\nFinBERT-only headlines: {len(fo)}; by row {fo['row'].value_counts().to_dict()}; "
          f"labelled: {int(fo['label'].notna().sum())}")
    print("wrote results/finbert_regressions.csv")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
