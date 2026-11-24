# Financial News That Restates the Return

Code, hand-labelled data, and LaTeX source for **"Financial News That Restates the Return: Measuring Price-Move Contamination in an Indian News Corpus"** by Sree Vidya Ghandikota (Independent researcher, Hyderabad, India).

## Summary

Studies relating financial news tone to stock returns typically treat each article as information arriving at the market. In practice, much financial news reports a price move that has already occurred. Using a public corpus of 3,348 Indian financial news articles from Moneycontrol (7 February 2025 to 21 August 2025, 173 distinct dates) matched against daily NIFTY 50 close-to-close returns, this note measures:

- **Extraction precision & accuracy:** A loose extractor flags **64.1%** of articles, but its stated moves are uncorrelated with actual returns (**-0.016**). A filtered extractor (`v3`) flags **27.5%** of articles (921) as stating a realised price move; on single-company NIFTY 50 matches (n = 376), stated moves correlate **+0.847** with actual close-to-close returns (89.6% sign agreement; 0.45 pp median absolute error).
- **Same-day tone regressions:** A headline direction-word tone score has a same-day R^2 of **0.492** among headlines quoting a `v3` move, versus **0.154** among headlines that contain a price-direction word but no percentage.
- **Number-free headlines:** Blinded hand-labelling of all **153** tone-bearing headlines without a percentage shows that **79.7%** (122) still describe the company's own price move in words.
- **FinBERT robustness check:** Scoring all 1,218 matched headlines with `tahp0604/finbert-sentfin` yields **99.2%** sign agreement on the 651-headline intersection where both measures are non-zero. Among the 289 additional headlines that only FinBERT scores, tone is unrelated to the same-day return (R^2 = 0.004; 51.2% sign agreement), confirming that the contamination result applies to headlines carrying price-direction vocabulary rather than number-free headlines in general.
- **Next-day returns:** The sample cannot detect a next-day relationship between headline tone and returns under either measure.

## Repository Structure

- `fetch_data.py` — Downloads the news corpus from Zenodo and NIFTY 50 daily prices from Yahoo Finance into `data/` (git-ignored) and verifies them against SHA-256 hashes and per-ticker return tolerances (`reference/price_return_stats.csv`).
- `pipeline.py` — Runs the three extractor levels (`loose`, `strict`, `v3`) and validates stated moves against actual returns.
- `title_tone.py` — Computes headline direction-word tone and runs date-clustered same-day and next-day regressions.
- `next_day_sensitivity.py` — Checks next-day estimate sensitivity across specification updates.
- `make_labelling_sample.py` & `analyse_labels.py` — Generates the blinded labelling file and runs the pre-specified by-label regressions.
- `session_qualifiers.py` — Checks whether partial-session wording explains the 76.2% sign agreement in the price-move group.
- `score_finbert.py` & `finbert_regressions.py` — Scores all 1,218 matched headlines with `tahp0604/finbert-sentfin` (cached in `data/sentiment_cache.parquet`) and runs the side-by-side FinBERT robustness regressions.
- `make_paper_inputs.py` — Auto-generates all in-text numbers, tables, and the appendix word list in `paper/generated/`.
- `compare_prices.py` — Diagnostic utility comparing two Yahoo Finance price downloads.
- `results/` — Output CSVs, including the 153 hand-labelled headlines (`labelling_sample.csv`) and the 14-article misattribution audit (`misattribution_review.csv`).
- `paper/` — LaTeX source (`paper.tex`) and generated inputs (`paper/generated/`).

## Reproducing the Results

Raw news text is not redistributed in this repository. To download the inputs from their original sources and regenerate every number and table in the paper:

```bash
pip install -r requirements.txt
python fetch_data.py
python pipeline.py data/news.csv data/nifty50_prices.csv
python title_tone.py data/news.csv data/nifty50_prices.csv
python next_day_sensitivity.py data/news.csv data/nifty50_prices.csv
python analyse_labels.py data/news.csv data/nifty50_prices.csv
python session_qualifiers.py data/news.csv data/nifty50_prices.csv
python score_finbert.py data/news.csv data/nifty50_prices.csv
python finbert_regressions.py data/news.csv data/nifty50_prices.csv
python make_paper_inputs.py data/news.csv data/nifty50_prices.csv
```

To compile the paper PDF:

```bash
cd paper
pdflatex paper.tex
pdflatex paper.tex
```

## Data Attribution

News corpus: Chakraborty, S., Barman, D., & Sufian, A. (2025). *Indian Stock Market News Dataset (v1)* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.16991843 (CC BY 4.0).
