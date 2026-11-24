"""
make_labelling_sample.py: write the hand-labelling file for the second
finding.

Run:
    python make_labelling_sample.py data/news.csv data/nifty50_prices.csv

Writes results/labelling_sample.csv with columns: article_id, title, label.
article_id is the row number in data/news.csv. label is left empty.

POPULATION, labelled in full (no sampling): every title in the "title has no
percentage" row of the main title-tone table (title_tone.py, tone incl.
gainer(s)/loser(s), "up to" removed, tone != 0). 153 titles as of
2026-11-22 (155 before the "up to" fix; the two dropped were SBI fundraising
headlines whose only direction word was the "up" in "raise up to").

ORDER: rows are shuffled with numpy default_rng(SEED) so titles are not
labelled in article_id order. SEED was fixed on 2026-11-22, before any label
was assigned. (An earlier version used the same seed to draw 100 of the 155;
superseded by labelling the whole population.)

BLINDING: return columns and tone are used only to decide membership and are
never printed or written. The output holds no returns, tone, date, ticker or
extractor output.

LABELS (one per title). Operational test, as finally applied: could a reader
infer the direction of the subject company's share price move from the
headline alone? (Refined during labelling from "what does the direction word
refer to?".)
    describes_price_move  yes
    describes_event       no
    ambiguous             cannot tell

The labels were filled in by the author on 2026-11-22. This script refuses
to overwrite a labelling file that already contains labels.
"""
import sys

import numpy as np

import title_tone as T

SEED = 20261122
LABELS = ["describes_price_move", "describes_event", "ambiguous"]
OUT = "results/labelling_sample.csv"


def population(news_path, price_path):
    """article_id and Title of the no_pct, tone != 0 rows. Nothing else."""
    d = T.with_tone(T.matched_articles(news_path, price_path), include_gainers=True)
    d = d[d["row"] == "no_pct"]
    return d[["article_id", "Title"]].sort_values("article_id").reset_index(drop=True)


def main(news_path, price_path):
    import os
    import pandas as pd
    if os.path.exists(OUT) and pd.read_csv(OUT)["label"].notna().any():
        sys.exit(f"Not run: {OUT} already contains labels; it would be overwritten.")
    pop = population(news_path, price_path)
    order = np.random.default_rng(SEED).permutation(len(pop))
    out = pop.iloc[order].rename(columns={"Title": "title"})
    out["label"] = ""
    out.to_csv(OUT, index=False)
    print(f"population: {len(pop)} titles, all written, order shuffled with seed {SEED}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
