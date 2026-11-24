"""
session_qualifiers.py: do partial-session headlines explain the imperfect sign
agreement among hand-labelled price-move headlines?

Run (after analyse_labels.py):
    python session_qualifiers.py data/news.csv data/nifty50_prices.csv

Writes results/session_qualifiers.csv.

A headline written during the session ("ITC shares fall in early trade")
describes a partial-session move, which need not match the close-to-close
return. Among the headlines labelled describes_price_move, this counts those
that carry a session qualifier and reports sign agreement (tone vs same-day
return) separately for headlines with and without one.

QUALIFIERS, fixed on 2026-11-22 before any count was seen; matched
case-insensitively, with straight and curly apostrophes treated alike:
    early trade, morning, intraday, afternoon, in today's session,
    day's low, day's high
Also reported, as a count only: headlines without a listed qualifier that
contain "session" or "trade", to show what the fixed list does not cover.
The split is descriptive; no test is run.
"""
import re
import sys

import numpy as np
import pandas as pd

import title_tone as T

QUALIFIERS = ["early trade", "morning", "intraday", "afternoon",
              "in today's session", "day's low", "day's high"]
RE_Q = re.compile("|".join(re.escape(q) for q in QUALIFIERS), re.I)
RE_OTHER = re.compile(r"\bsession\b|\btrade\b", re.I)


def norm(s):
    return str(s).replace("’", "'")


def main(news_path, price_path):
    lab = pd.read_csv("results/labelling_sample.csv")
    d = T.with_tone(T.matched_articles(news_path, price_path))
    m = lab[["article_id", "label"]].merge(d, on="article_id", validate="one_to_one")
    pm = m[m["label"] == "describes_price_move"].copy()
    title = pm["Title"].map(norm)
    pm["qualifier"] = title.str.contains(RE_Q)
    pm["other_session_word"] = ~pm["qualifier"] & title.str.contains(RE_OTHER)
    pm["sign_agrees"] = np.sign(pm["ret"]) == pm["tone"]

    rows = [dict(group="all price-move headlines", n=len(pm),
                 share=1.0, sign_agreement=pm["sign_agrees"].mean())]
    for flag, name in [(True, "with a listed session qualifier"),
                       (False, "without a listed session qualifier")]:
        g = pm[pm["qualifier"] == flag]
        rows.append(dict(group=name, n=len(g), share=len(g) / len(pm),
                         sign_agreement=g["sign_agrees"].mean()))
    g = pm[pm["other_session_word"]]
    # Count only, as declared above: no sign agreement for this post-hoc group.
    rows.append(dict(group="without a listed qualifier but containing 'session' or 'trade'",
                     n=len(g), share=len(g) / len(pm), sign_agreement=np.nan))
    tab = pd.DataFrame(rows)
    tab.to_csv("results/session_qualifiers.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(tab.to_string(index=False))
    print("\nper qualifier (headline counts):")
    for q in QUALIFIERS:
        print(f"  {q:<20} {int(title.str.contains(re.escape(q), case=False).sum())}")
    print("\nwrote results/session_qualifiers.csv")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
