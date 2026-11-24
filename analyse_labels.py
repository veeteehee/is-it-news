"""
analyse_labels.py: tone-return relationship by hand label. Pre-specified on
2026-11-22, before any label was assigned. Run only once every row of
results/labelling_sample.csv has a label. The file covers the whole no_pct
population, and the script checks that it does.

Run:
    python analyse_labels.py data/news.csv data/nifty50_prices.csv

Reports:
    - ambiguous rate: share of the labelled sample marked ambiguous
    - for describes_price_move and describes_event separately: same-day and
      next-day return (%) on title tone, SEs clustered by date, plus sign
      agreement. Same specification as title_tone.py.
Ambiguous rows are counted but not regressed.
Writes results/label_analysis.csv.
"""
import sys

import pandas as pd

import title_tone as T
from make_labelling_sample import LABELS

SAMPLE = "results/labelling_sample.csv"
GROUPS = ["describes_price_move", "describes_event"]


def main(news_path, price_path):
    lab = pd.read_csv(SAMPLE, dtype={"label": str})
    lab["label"] = lab["label"].fillna("").str.strip()
    missing = lab["label"] == ""
    bad = ~lab["label"].isin(LABELS) & ~missing
    if missing.any() or bad.any():
        sys.exit(f"Not run: {int(missing.sum())} unlabelled rows, "
                 f"{int(bad.sum())} rows with a label not in {LABELS}.")

    d = T.with_tone(T.matched_articles(news_path, price_path), include_gainers=True)
    m = lab[["article_id", "label"]].merge(d, on="article_id", how="left",
                                            validate="one_to_one")
    if m["tone"].isna().any():
        sys.exit("Not run: some labelled articles are no longer in the population.")
    pop = d[d["row"] == "no_pct"]
    if set(pop["article_id"]) != set(m["article_id"]):
        sys.exit("Not run: labelled titles are not exactly the no_pct population.")

    n = len(m)
    counts = m["label"].value_counts().reindex(LABELS, fill_value=0)
    print(f"labelled: {n}")
    for k, v in counts.items():
        print(f"  {k:<22} {v:>4}  ({v / n:.1%})")
    print(f"ambiguous rate: {counts['ambiguous'] / n:.1%}\n")

    rows = [T.tone_return_row(m[m["label"] == g], g) for g in GROUPS]
    tab = pd.DataFrame(rows)
    tab.insert(1, "share_of_sample", [counts[g] / n for g in GROUPS])
    T.show(tab, "Tone-return relationship by label")
    tab.to_csv("results/label_analysis.csv", index=False)
    pd.DataFrame({"label": LABELS, "count": counts.values,
                  "share": counts.values / n}).to_csv(
        "results/label_counts.csv", index=False)
    print("wrote results/label_analysis.csv, results/label_counts.csv")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
