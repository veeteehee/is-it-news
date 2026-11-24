"""
score_finbert.py: score every matched-sample headline with FinBERT and cache
the class probabilities. Scoring runs once; later runs read the cache.

Run:
    python score_finbert.py data/news.csv data/nifty50_prices.csv

Writes:
    data/sentiment_cache.parquet   one row per article_id: p_positive,
                                   p_negative, p_neutral, model_id, revision
    results/finbert_descriptives.csv

MODEL: tahp0604/finbert-sentfin (FinBERT fine-tuned on SEntFiN 1.0), pinned
to REVISION below. Labels from the model config, checked against the model
card: 0 = positive, 1 = negative, 2 = neutral.

INPUT: the Title field only, for every article in the matched sample
(title_tone.matched_articles: exactly one NIFTY 50 company named, valid
same-day return). No returns are read or used here.

MAPPING, fixed 2026-11-24 before any result was looked at:
    primary    finbert_tone = sign of the argmax class:
               positive -> +1, negative -> -1, neutral -> 0
               (like-for-like with the word-count tone)
    secondary  finbert_score = P(positive) - P(negative), continuous

DESCRIPTIVES ONLY (no returns): headlines scored, class distribution, how
many get tone 0 under each measure, the intersection where neither is 0,
and sign agreement between the two measures on that intersection.
"""
import os
import sys

# Download weights over plain HTTPS (huggingface.co -> us.aws.cdn.hf.co)
# rather than the Xet client, whose storage host is not on this
# environment's allowlist. Does not change the files downloaded.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pandas as pd

import title_tone as T

MODEL_ID = "tahp0604/finbert-sentfin"
REVISION = "2e1df5cfb7ad5022cfcd2d9c0f674e7c9f5f805f"
CACHE = "data/sentiment_cache.parquet"
LABELS = {0: "positive", 1: "negative", 2: "neutral"}
BATCH = 32
MAX_LEN = 128


def score(titles):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=REVISION)
    model.eval()
    got = {int(k): v for k, v in model.config.id2label.items()}
    if got != LABELS:
        sys.exit(f"Label mapping changed: model says {got}, expected {LABELS}")
    out = []
    with torch.no_grad():
        for i in range(0, len(titles), BATCH):
            enc = tok(titles[i:i + BATCH], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
            out.append(torch.softmax(model(**enc).logits, dim=-1).numpy())
    return np.vstack(out)


def load_or_score(m):
    if os.path.exists(CACHE):
        c = pd.read_parquet(CACHE)
        if (set(c["article_id"]) >= set(m["article_id"])
                and (c["revision"] == REVISION).all()):
            print(f"using cache {CACHE} ({len(c)} rows)")
            return c
        sys.exit(f"{CACHE} exists but does not cover the matched sample at this "
                 f"revision; move it aside deliberately before re-scoring.")
    titles = m["Title"].fillna("").astype(str).tolist()
    p = score(titles)
    c = pd.DataFrame({"article_id": m["article_id"].to_numpy(),
                      "p_positive": p[:, 0], "p_negative": p[:, 1],
                      "p_neutral": p[:, 2], "model_id": MODEL_ID,
                      "revision": REVISION})
    c.to_parquet(CACHE, index=False)
    print(f"scored {len(c)} headlines, wrote {CACHE}")
    return c


def finbert_tone(c):
    cls = c[["p_positive", "p_negative", "p_neutral"]].to_numpy().argmax(axis=1)
    return pd.Series(np.select([cls == 0, cls == 1], [1, -1], 0), index=c.index)


def main(news_path, price_path):
    m = T.matched_articles(news_path, price_path)
    c = load_or_score(m)
    d = m[["article_id", "Title"]].merge(c, on="article_id", validate="one_to_one")
    d["finbert_tone"] = finbert_tone(d)
    d["finbert_score"] = d["p_positive"] - d["p_negative"]
    d["wordcount_tone"] = T.tone_series(d.rename(columns={}))  # main word-count spec

    n = len(d)
    cls = d["finbert_tone"].map({1: "positive", -1: "negative", 0: "neutral"})
    both = d[(d["finbert_tone"] != 0) & (d["wordcount_tone"] != 0)]
    rows = [("headlines scored", n, 1.0)]
    for k in ["positive", "negative", "neutral"]:
        rows.append((f"FinBERT class {k}", int((cls == k).sum()), (cls == k).mean()))
    rows += [
        ("FinBERT tone 0 (neutral)", int((d["finbert_tone"] == 0).sum()), (d["finbert_tone"] == 0).mean()),
        ("word-count tone 0", int((d["wordcount_tone"] == 0).sum()), (d["wordcount_tone"] == 0).mean()),
        ("both non-zero (intersection)", len(both), len(both) / n),
        ("sign agreement on intersection", int((both["finbert_tone"] == both["wordcount_tone"]).sum()),
         (both["finbert_tone"] == both["wordcount_tone"]).mean() if len(both) else np.nan),
    ]
    tab = pd.DataFrame(rows, columns=["quantity", "count", "share"])
    tab.to_csv("results/finbert_descriptives.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(tab.to_string(index=False))
    print("\n(share of 'sign agreement' is relative to the intersection; all other "
          "shares are relative to headlines scored)")
    print("wrote results/finbert_descriptives.csv")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
