"""
make_paper_inputs.py: write every number and table the paper uses into
paper/generated/, so nothing in the paper is typed by hand.

Run after the analysis scripts:
    python pipeline.py data/news.csv data/nifty50_prices.csv
    python title_tone.py data/news.csv data/nifty50_prices.csv
    python next_day_sensitivity.py data/news.csv data/nifty50_prices.csv
    python analyse_labels.py data/news.csv data/nifty50_prices.csv
    python session_qualifiers.py data/news.csv data/nifty50_prices.csv
    python score_finbert.py data/news.csv data/nifty50_prices.csv
    python finbert_regressions.py data/news.csv data/nifty50_prices.csv
    python make_paper_inputs.py data/news.csv data/nifty50_prices.csv

Writes:
    paper/generated/numbers.tex         \\newcommand macros for in-text numbers
    paper/generated/tab_extractor.tex   Table 1
    paper/generated/tab_tone.tex        Table 2
    paper/generated/tab_labels.tex      Table 3
    paper/generated/tab_finbert.tex     Table 4 (FinBERT robustness)
    paper/generated/appendix_words.tex  direction-word list

Reads results/*.csv, plus corpus counts computed here from news.csv and the
matched sample (title_tone.matched_articles).
"""
import os
import sys

import pandas as pd
from scipy import stats

import pipeline as P
import title_tone as T

OUT = "paper/generated"
R = "results"


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def num(x, d=3, sign=False):
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def p_from_t(t, clusters):
    return 2 * stats.t.sf(abs(t), clusters - 1)


def pfmt(p, d=2):
    return "$<$0.001" if p < 0.001 else num(p, d)


class Macros(dict):
    """A dict that refuses to define the same macro twice."""
    def __setitem__(self, k, v):
        if k in self:
            raise KeyError(f"macro \\{k} defined twice")
        super().__setitem__(k, v)


def main(news_path, price_path):
    os.makedirs(OUT, exist_ok=True)
    M = Macros()

    # ---- corpus -----------------------------------------------------------
    news = pd.read_csv(news_path)
    d = pd.to_datetime(news["Date"])
    M["nArticles"] = f"{len(news):,}"
    M["nDates"] = str(d.dt.date.nunique())
    M["dateFirst"] = d.min().strftime("%-d %B %Y")
    M["dateLast"] = d.max().strftime("%-d %B %Y")
    M["shareMoneycontrol"] = pct(news["URL"].astype(str).str.contains("moneycontrol.com").mean())
    auth = news["Author"].fillna("").astype(str).str.strip()
    for key, name in [("AlphaDesk", "Alpha Desk"), ("BrokerResearch", "Broker Research")]:
        M[f"n{key}"] = f"{int((auth == name).sum()):,}"
        M[f"share{key}"] = pct((auth == name).mean())
    other = ~auth.isin(["Alpha Desk", "Broker Research"])
    M["nOtherAuthors"] = f"{int(other.sum()):,}"
    M["shareOtherAuthors"] = pct(other.mean())

    # ---- matching -----------------------------------------------------------
    hay = news["Title"].fillna("").astype(str) + " | " + news["Keywords"].fillna("").astype(str)
    single = hay.map(P.match_single).notna()
    M["nSingleName"] = f"{int(single.sum()):,}"
    M["shareSingleName"] = pct(single.mean())
    m = T.matched_articles(news_path, price_path)
    M["nMatched"] = f"{len(m):,}"
    rows = m["row"].value_counts()
    M["nRowVthree"] = str(rows["v3_move"])
    M["nRowPctRejected"] = str(rows["pct_rejected"])
    M["nRowNoPct"] = str(rows["no_pct"])
    M["nTickers"] = str(len(P.NAMES))

    # ---- extractor table ----------------------------------------------------
    ex = pd.read_csv(f"{R}/extractor_table.csv").set_index("extractor")
    for lv, tag in [("loose", "Loose"), ("strict", "Strict"), ("v3", "Vthree")]:
        r = ex.loc[lv]
        M[f"flag{tag}"] = pct(r["share_flagged"])
        M[f"nFlag{tag}"] = f"{int(r['n_flagged']):,}"
        M[f"matched{tag}"] = str(int(r["matched"]))
        M[f"corr{tag}"] = num(r["corr"], sign=True)
        M[f"sign{tag}"] = pct(r["sign_agreement"])
        M[f"err{tag}"] = num(r["median_abs_error_pp"], 2)
    M["withinHalfVthree"] = pct(ex.loc["v3", "within_0_5pp"])
    M["withinOneVthree"] = pct(ex.loc["v3", "within_1pp"])
    lines = []
    for lv, name in [("loose", "Loose"), ("strict", "Strict"), ("v3", "v3")]:
        r = ex.loc[lv]
        lines.append(f"{name} & {pct(r['share_flagged'])} & {int(r['matched'])} & "
                     f"{num(r['corr'], sign=True)} & {pct(r['sign_agreement'])} & "
                     f"{num(r['median_abs_error_pp'], 2)} & {pct(r['within_1pp'])} \\\\")
    with open(f"{OUT}/tab_extractor.tex", "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- tone table ---------------------------------------------------------
    tt = pd.read_csv(f"{R}/title_tone_table.csv")
    ng = pd.read_csv(f"{R}/title_tone_table_no_gainers.csv")
    tags = ["All", "Vthree", "PctRej", "NoPct"]
    names = ["All matched", "States a v3 move", "Percentage, v3-rejected",
             "No percentage"]
    lines = []
    for tag, name, (_, r), (_, g) in zip(tags, names, tt.iterrows(), ng.iterrows()):
        pn = p_from_t(r["next_day_t"], r["clusters"])
        M[f"toneN{tag}"] = str(int(r["n"]))
        M[f"toneCl{tag}"] = str(int(r["clusters"]))
        M[f"toneRsq{tag}"] = num(r["same_day_r2"])
        M[f"toneCoef{tag}"] = num(r["tone_coef"], sign=True)
        M[f"toneT{tag}"] = num(r["tone_t"], 2)
        M[f"toneSign{tag}"] = pct(r["sign_agreement"])
        M[f"toneNextT{tag}"] = num(r["next_day_t"], 2, sign=True)
        M[f"toneNextP{tag}"] = pfmt(pn, 3)
        M[f"toneRsqNoGain{tag}"] = num(g["same_day_r2"])
        lines.append(f"{name} & {int(r['n'])} & {int(r['clusters'])} & {num(r['same_day_r2'])} & "
                     f"{num(r['tone_coef'], sign=True)} ({num(r['tone_t'], 2)}) & "
                     f"{pct(r['sign_agreement'])} & {num(r['next_day_coef'], sign=True)} "
                     f"({num(r['next_day_t'], 2)}) & {pfmt(pn)} \\\\")
    with open(f"{OUT}/tab_tone.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    M["nToneSample"] = str(int(tt.iloc[0]["n"]))

    # ---- "up to" fix --------------------------------------------------------
    ch = pd.read_csv(f"{R}/up_to_fix_changes.csv")
    rec = ch[(ch["tone_before"] == 0) & (ch["tone_after"] != 0)]
    M["nUpToChanged"] = str(len(ch))
    M["nUpToRecovered"] = str(len(rec))
    M["signUpToRecovered"] = pct(rec["sign_agrees_after"].mean(), 0)

    # ---- next-day sensitivity ----------------------------------------------
    ns = pd.read_csv(f"{R}/next_day_spec_sensitivity.csv")
    a = ns[ns["sample"] == "all matched articles"]
    M["nextTAllMin"] = num(a["next_day_t"].min(), 2)
    M["nextTAllMax"] = num(a["next_day_t"].max(), 2)
    M["nextPAllMin"] = num(a["next_day_p"].min(), 2)
    M["nextPAllMax"] = num(a["next_day_p"].max(), 2)

    # ---- labels -------------------------------------------------------------
    lc = pd.read_csv(f"{R}/label_counts.csv").set_index("label")
    M["nLabelled"] = str(int(lc["count"].sum()))
    for lab, tag in [("describes_price_move", "Move"), ("describes_event", "Event"),
                     ("ambiguous", "Ambig")]:
        M[f"nLab{tag}"] = str(int(lc.loc[lab, "count"]))
        M[f"shareLab{tag}"] = pct(lc.loc[lab, "share"])
    la = pd.read_csv(f"{R}/label_analysis.csv").set_index("sample")
    lines = []
    for lab, tag, name in [("describes_price_move", "Move", "Price move"),
                           ("describes_event", "Event", "Event")]:
        r = la.loc[lab]
        pn = p_from_t(r["next_day_t"], r["clusters"])
        M[f"labN{tag}"] = str(int(r["n"]))
        M[f"labCl{tag}"] = str(int(r["clusters"]))
        M[f"labRsq{tag}"] = num(r["same_day_r2"])
        M[f"labCoef{tag}"] = num(r["tone_coef"], sign=True)
        M[f"labT{tag}"] = num(r["tone_t"], 2)
        M[f"labP{tag}"] = pfmt(p_from_t(r["tone_t"], r["clusters"]))
        M[f"labSign{tag}"] = pct(r["sign_agreement"])
        M[f"labNextT{tag}"] = num(r["next_day_t"], 2, sign=True)
        M[f"labNextP{tag}"] = pfmt(pn)
        lines.append(f"{name} & {int(r['n'])} & {pct(r['share_of_sample'])} & "
                     f"{int(r['clusters'])} & {num(r['same_day_r2'])} & "
                     f"{num(r['tone_coef'], sign=True)} ({num(r['tone_t'], 2)}) & "
                     f"{pct(r['sign_agreement'])} & {num(r['next_day_coef'], sign=True)} "
                     f"({num(r['next_day_t'], 2)}) \\\\")
    with open(f"{OUT}/tab_labels.tex", "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- session qualifiers (price-move headlines) ---------------------------
    sq = pd.read_csv(f"{R}/session_qualifiers.csv").set_index("group")
    q, nq = sq.loc["with a listed session qualifier"], sq.loc["without a listed session qualifier"]
    oth = sq.loc["without a listed qualifier but containing 'session' or 'trade'"]
    M["nSessQual"] = str(int(q["n"]))
    M["shareSessQual"] = pct(q["share"])
    M["signSessQual"] = pct(q["sign_agreement"])
    M["nSessNoQual"] = str(int(nq["n"]))
    M["signSessNoQual"] = pct(nq["sign_agreement"])
    M["nSessOther"] = str(int(oth["n"]))

    # ---- misattribution review (author's judgement, 14 keyword-only matches)
    mr = pd.read_csv(f"{R}/misattribution_review.csv")
    n_lab = int(lc["count"].sum())
    clear = int((mr["classification"] == "clear").sum())
    border = int((mr["classification"] == "borderline").sum())
    M["nKeywordOnly"] = str(len(mr))
    M["nMisClear"] = str(clear)
    M["shareMisClear"] = pct(clear / n_lab)
    M["nMisUpper"] = str(clear + border)
    M["shareMisUpper"] = pct((clear + border) / n_lab)

    # ---- FinBERT robustness ---------------------------------------------------
    import score_finbert as SF
    M["fbModel"] = SF.MODEL_ID
    M["fbRev"] = SF.REVISION[:8]
    fd = pd.read_csv(f"{R}/finbert_descriptives.csv").set_index("quantity")
    M["fbNScored"] = f"{int(fd.loc['headlines scored', 'count']):,}"
    M["fbNNeutral"] = str(int(fd.loc["FinBERT tone 0 (neutral)", "count"]))
    M["fbNInt"] = str(int(fd.loc["both non-zero (intersection)", "count"]))
    M["fbNAgree"] = str(int(fd.loc["sign agreement on intersection", "count"]))
    M["fbAgree"] = pct(fd.loc["sign agreement on intersection", "share"])
    fr = pd.read_csv(f"{R}/finbert_regressions.csv")
    def F(measure, sample, group):
        r = fr[(fr["measure"] == measure) & (fr["sample"] == sample) & (fr["group"] == group)]
        assert len(r) == 1, (measure, sample, group)
        return r.iloc[0]
    gtags = [("all", "All"), ("v3_move", "Vthree"), ("pct_rejected", "PctRej"), ("no_pct", "NoPct")]
    lines = []
    for (g, tag), name in zip(gtags, names):
        w, f, wi, fi = (F("wordcount", "own", g), F("finbert", "own", g),
                        F("wordcount", "intersection", g), F("finbert", "intersection", g))
        M[f"fbN{tag}"] = str(int(f["n"]))
        M[f"fbCl{tag}"] = str(int(f["clusters"]))
        M[f"fbRsq{tag}"] = num(f["same_day_r2"])
        M[f"intN{tag}"] = str(int(wi["n"]))
        M[f"intRsqWc{tag}"] = num(wi["same_day_r2"])
        M[f"intRsqFb{tag}"] = num(fi["same_day_r2"])
        lines.append(f"{name} & {int(w['n'])} & {int(w['clusters'])} & {num(w['same_day_r2'])} & "
                     f"{int(f['n'])} & {int(f['clusters'])} & {num(f['same_day_r2'])} & "
                     f"{int(wi['n'])} & {num(wi['same_day_r2'])} & {num(fi['same_day_r2'])} \\\\")
    with open(f"{OUT}/tab_finbert.tex", "w") as f_:
        f_.write("\n".join(lines) + "\n")
    fa = F("finbert", "own", "all")
    M["fbNextTAll"] = num(fa["next_day_t"], 2, sign=True)
    M["fbNextPAll"] = pfmt(fa["next_day_p"])
    fo, fon = F("finbert", "finbert_only", "all"), F("finbert", "finbert_only", "no_pct")
    M["fbOnlyN"] = str(int(fo["n"]))
    M["fbOnlyCl"] = str(int(fo["clusters"]))
    M["fbOnlyRsq"] = num(fo["same_day_r2"])
    M["fbOnlyT"] = num(fo["tone_t"], 2)
    M["fbOnlyP"] = pfmt(fo["same_day_p"])
    M["fbOnlySign"] = pct(fo["sign_agreement"])
    M["fbOnlyNoPctN"] = str(int(fon["n"]))
    M["fbOnlyNoPctRsq"] = num(fon["same_day_r2"])
    lm, le = F("finbert", "own", "label:describes_price_move"), F("finbert", "own", "label:describes_event")
    M["fbLabNMove"] = str(int(lm["n"]))
    M["fbLabRsqMove"] = num(lm["same_day_r2"])
    M["fbLabNEvent"] = str(int(le["n"]))
    M["fbLabNEventNeutral"] = str(int(la.loc["describes_event", "n"]) - int(le["n"]))
    M["fbLabTEvent"] = num(le["tone_t"], 2)
    M["fbLabPEvent"] = pfmt(le["same_day_p"])
    cv, cn = F("finbert_cont", "all_rows", "v3_move"), F("finbert_cont", "all_rows", "no_pct")
    M["fbContRsqVthree"] = num(cv["same_day_r2"])
    M["fbContRsqNoPct"] = num(cn["same_day_r2"])
    M["fbContNNoPct"] = str(int(cn["n"]))
    nd = fr[fr["next_day_p"].notna() & fr["measure"].str.startswith("finbert")]
    M["fbMinNextP"] = num(nd["next_day_p"].min(), 2)

    # ---- word list ----------------------------------------------------------
    w = pd.read_csv(f"{R}/direction_words.csv")
    M["nWordForms"] = str(len(w))
    parts = []
    for direction, label in [("rise", "Rise"), ("fall", "Fall")]:
        g = w[w["direction"] == direction]
        parts.append(f"\\textbf{{{label} ({len(g)} forms):}} " + ", ".join(g["word"]) + ".")
    with open(f"{OUT}/appendix_words.tex", "w") as f:
        f.write("\n\n".join(parts) + "\n")

    with open(f"{OUT}/numbers.tex", "w") as f:
        f.write("% Generated by make_paper_inputs.py. Do not edit.\n")
        for k, v in M.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(M)} macros and 5 table/appendix files to {OUT}/")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
