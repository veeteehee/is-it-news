"""
compare_prices.py: explain why two price files differ, beyond "the hash
differs".

Run:
    python compare_prices.py REFERENCE.csv NEW.csv

Reports, in this order:
  1. FORMAT: column names and order, row order, how numbers are written
     (e.g. Volume "2898619.0" vs "2898619"), line endings. Differences here
     change the hash without changing any value.
  2. COVERAGE: tickers and (ticker, Date) rows present in only one file,
     and each file's date range.
  3. VALUES, on rows present in both: for each numeric column, how many rows
     differ, and the distribution of absolute and relative differences.
  4. EFFECT ON RETURNS: the same for close-to-close returns, which is what the
     analysis uses.
Prints a list of the largest differences. Changes nothing.
"""
import sys

import numpy as np
import pandas as pd

KEY = ["ticker", "Date"]
NUM = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def raw_format(path):
    with open(path, "rb") as f:
        b = f.read()
    return dict(bytes=len(b), crlf=b.count(b"\r\n"), lines=b.count(b"\n"),
                trailing_newline=b.endswith(b"\n"))


def describe(x):
    x = np.abs(np.asarray(x, float))
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "n/a"
    q = np.quantile(x, [0.5, 0.9, 0.99, 1.0])
    return f"median {q[0]:.3g}  p90 {q[1]:.3g}  p99 {q[2]:.3g}  max {q[3]:.3g}"


def main(ref_path, new_path):
    rs, ns = pd.read_csv(ref_path, dtype=str), pd.read_csv(new_path, dtype=str)
    print("=" * 72, "\n1. FORMAT\n" + "=" * 72)
    print(f"  columns ref: {list(rs.columns)}")
    print(f"  columns new: {list(ns.columns)}")
    print(f"  raw ref: {raw_format(ref_path)}")
    print(f"  raw new: {raw_format(new_path)}")
    k0 = (rs["ticker"].iloc[0], rs["Date"].iloc[0])
    nk = ns[(ns["ticker"] == k0[0]) & (ns["Date"] == k0[1])]
    print(f"  example row {k0}: present in new: {len(nk) == 1}")
    for c in [c for c in NUM if c in rs and c in ns]:
        ex_r = rs[c].iloc[0]
        ex_n = nk[c].iloc[0] if len(nk) else None
        dot_r = rs[c].str.contains(r"\.", na=False).mean()
        dot_n = ns[c].str.contains(r"\.", na=False).mean()
        if ex_r != ex_n or abs(dot_r - dot_n) > 0:
            print(f"  {c:<10} first value ref {ex_r!r} new {ex_n!r}; "
                  f"share with a decimal point ref {dot_r:.0%} new {dot_n:.0%}")
    print(f"  Date format ref {rs['Date'].iloc[0]!r}  new {ns['Date'].iloc[0]!r}"
          f"  (lengths ref {sorted(rs['Date'].str.len().unique())}"
          f" new {sorted(ns['Date'].str.len().unique())})")
    order_r = list(zip(rs["ticker"], rs["Date"]))
    order_n = list(zip(ns["ticker"], ns["Date"]))
    same_set = set(order_r) == set(order_n)
    print(f"  identical row order: {order_r == order_n}  (same set of rows: {same_set})")

    r, n = rs.copy(), ns.copy()
    for d in (r, n):
        d["Date"] = pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d")
        for c in NUM:
            if c in d:
                d[c] = pd.to_numeric(d[c], errors="coerce")

    print("\n" + "=" * 72, "\n2. COVERAGE\n" + "=" * 72)
    tr, tn = set(r["ticker"]), set(n["ticker"])
    print(f"  tickers: ref {len(tr)}, new {len(tn)}; only ref {sorted(tr - tn)}; "
          f"only new {sorted(tn - tr)}")
    print(f"  dates: ref {r['Date'].min()}..{r['Date'].max()}  "
          f"new {n['Date'].min()}..{n['Date'].max()}")
    dup_r, dup_n = r.duplicated(KEY).sum(), n.duplicated(KEY).sum()
    print(f"  duplicate (ticker, Date) rows: ref {dup_r}, new {dup_n}")
    m = r.merge(n, on=KEY, how="outer", suffixes=("_ref", "_new"), indicator=True)
    for side in ["left_only", "right_only"]:
        s = m[m["_merge"] == side]
        name = "only in ref" if side == "left_only" else "only in new"
        print(f"  rows {name}: {len(s)}")
        if len(s):
            print("    by date:", s["Date"].value_counts().sort_index().head(10).to_dict())
            print("    by ticker:", s["ticker"].value_counts().head(10).to_dict())

    b = m[m["_merge"] == "both"].copy()
    print("\n" + "=" * 72, f"\n3. VALUES on {len(b)} rows in both files\n" + "=" * 72)
    print("  'exact' = any difference at all; 'real' = relative difference above\n"
          "  1e-9, i.e. beyond float rounding from re-writing the file")
    any_diff = np.zeros(len(b), bool)
    any_real = np.zeros(len(b), bool)
    for c in NUM:
        if f"{c}_ref" not in b:
            continue
        x, y = b[f"{c}_ref"], b[f"{c}_new"]
        diff = ~((x == y) | (x.isna() & y.isna()))
        rel_all = ((y - x) / x.replace(0, np.nan)).abs()
        real = diff & ~(rel_all <= 1e-9)
        any_diff |= diff.to_numpy()
        any_real |= real.to_numpy()
        ad = (y - x)[real]
        rd = rel_all[real]
        print(f"  {c:<10} rows differing: exact {int(diff.sum()):>6}   "
              f"real {int(real.sum()):>6} ({real.mean():.2%})")
        diff = real
        if diff.any():
            print(f"             |abs diff|  {describe(ad)}")
            print(f"             |rel diff|  {describe(rd)}")
            print(f"             by ticker   {b.loc[diff, 'ticker'].value_counts().head(5).to_dict()}")
            print(f"             by date     {b.loc[diff, 'Date'].value_counts().sort_index().tail(5).to_dict()}")
    print(f"  rows with any value differing: exact {int(any_diff.sum())}, "
          f"real {int(any_real.sum())}, of {len(b)}")

    print("\n" + "=" * 72, "\n4. EFFECT ON CLOSE-TO-CLOSE RETURNS\n" + "=" * 72)
    for d in (r, n):
        d.sort_values(KEY, inplace=True)
        d["ret"] = d.groupby("ticker")["Close"].pct_change()
    rr = r[KEY + ["ret"]].merge(n[KEY + ["ret"]], on=KEY, suffixes=("_ref", "_new"))
    rr = rr.dropna(subset=["ret_ref", "ret_new"], how="all")
    dret = (rr["ret_new"] - rr["ret_ref"]).abs()
    print(f"  returns compared: {len(rr)}; differing by > 1e-9: {int((dret > 1e-9).sum())}; "
          f"> 1bp (0.0001): {int((dret > 1e-4).sum())}")
    print(f"  |return diff|  {describe(dret)}")
    top = rr.assign(d=dret).nlargest(10, "d")
    top = top[top["d"] > 1e-12]
    if len(top):
        print("  largest return differences:")
        for _, t in top.iterrows():
            print(f"    {t['ticker']:<15} {t['Date']}  ref {t['ret_ref']:+.6f}  "
                  f"new {t['ret_new']:+.6f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
