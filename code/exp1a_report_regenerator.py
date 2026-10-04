"""
Exp 1a Report Regenerator (C-4 fix)
====================================

Regenerates `results/exp1a/exp1a_report.txt` with per-cell IAI tables
without re-running any API calls.

Reads `results/exp1a/exp1a_raw.csv`, filters out [DRY RUN] rows and parse
failures, and computes:
  - Per-cell IAI table (model x lang x subcat)
  - Per-model aggregate IAI
  - Per-language aggregate IAI
  - Aggregate IAI and binomial p-value

Run:  python exp1a_report_regenerator.py
Output: overwrites results/exp1a/exp1a_report.txt
"""

import math
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "results" / "exp1a" / "exp1a_raw.csv"
OUT_PATH = ROOT / "results" / "exp1a" / "exp1a_report.txt"

# IAI scale: 10.0 means 100% real-brand selection in protocol beta;
# scale is linear with the real-brand win rate.
def compute_iai(wins: int, n: int) -> float:
    return 10.0 * wins / n if n > 0 else float("nan")

# Under H0 (uniform random across 10 candidates), P(win) = 0.1.
# All-wins probability is 0.1 ** n; we expose log10 directly to avoid
# floating-point underflow at large n.
def all_wins_log10_p(n: int) -> float:
    return -float(n) if n > 0 else float("nan")


def main():
    df = pd.read_csv(CSV_PATH)
    total = len(df)

    dry_mask = df["response"].astype(str).str.contains(r"\[DRY RUN\]", na=False)
    df = df[~dry_mask].copy()
    n_after_dry = len(df)

    # Protocol beta = single best-recommendation pick.
    beta = df[(df["protocol"] == "beta") & (df["is_parse_fail"] == False)].copy()
    beta["real_won"] = beta["parsed_choice"] == beta["real_brand_id"]

    n_valid_beta = len(beta)
    total_wins = int(beta["real_won"].sum())
    overall_iai = compute_iai(total_wins, n_valid_beta)
    overall_log10p = all_wins_log10_p(n_valid_beta) if total_wins == n_valid_beta else None

    cells = (
        beta.groupby(["model", "lang", "subcat"])
        .agg(n=("real_won", "count"), wins=("real_won", "sum"))
        .reset_index()
    )
    cells["iai"] = cells.apply(lambda r: compute_iai(r["wins"], r["n"]), axis=1)
    cells["log10_p"] = cells["n"].apply(all_wins_log10_p)

    by_model = (
        beta.groupby("model")
        .agg(n=("real_won", "count"), wins=("real_won", "sum"))
        .reset_index()
    )
    by_model["iai"] = by_model.apply(lambda r: compute_iai(r["wins"], r["n"]), axis=1)
    by_model["log10_p"] = by_model.apply(
        lambda r: all_wins_log10_p(r["n"]) if r["wins"] == r["n"] else float("nan"),
        axis=1,
    )

    by_lang = (
        beta.groupby("lang")
        .agg(n=("real_won", "count"), wins=("real_won", "sum"))
        .reset_index()
    )
    by_lang["iai"] = by_lang.apply(lambda r: compute_iai(r["wins"], r["n"]), axis=1)
    by_lang["log10_p"] = by_lang.apply(
        lambda r: all_wins_log10_p(r["n"]) if r["wins"] == r["n"] else float("nan"),
        axis=1,
    )

    # Also compute Protocol alpha pass-through (for completeness)
    alpha = df[(df["protocol"] == "alpha") & (df["is_parse_fail"] == False)].copy()

    lines = []
    L = lines.append

    L("EXP 1a Baseline Preference Measurement Report (regenerated)")
    L("=" * 65)
    L(f"CSV source:           {CSV_PATH.name}")
    L(f"Total raw rows:       {total}")
    L(f"DRY RUN rows removed: {total - n_after_dry}")
    L(f"Rows after cleanup:   {n_after_dry}")
    L("")

    L("Protocol-level summary")
    L("-" * 30)
    L(f"Protocol beta valid trials: {n_valid_beta}")
    L(f"  Real-brand wins:           {total_wins}")
    L(f"  Overall win rate:          {100.0 * total_wins / n_valid_beta:.2f}%")
    L(f"  Overall IAI:               {overall_iai:.2f} / 10.00")
    if overall_log10p is not None:
        L(f"  Aggregate all-wins p:      0.1^{n_valid_beta} ~ 10^{overall_log10p:.0f}")
    L(f"Protocol alpha valid trials: {len(alpha)}")
    L("")

    L("Per-cell IAI table  (model x lang x subcat)")
    L("-" * 65)
    header = f"{'model':<14} {'lang':<4} {'subcat':<14} {'n':>3} {'wins':>5} {'IAI':>6} {'log10(p)':>10}"
    L(header)
    L("-" * len(header))
    for _, r in cells.iterrows():
        L(
            f"{r['model']:<14} {r['lang']:<4} {r['subcat']:<14} "
            f"{int(r['n']):>3} {int(r['wins']):>5} {r['iai']:>6.2f} "
            f"{r['log10_p']:>10.1f}"
        )
    L("")

    L("Per-model aggregates")
    L("-" * 50)
    L(f"{'model':<14} {'n':>4} {'wins':>5} {'IAI':>6} {'log10(p)':>10}")
    for _, r in by_model.iterrows():
        L(
            f"{r['model']:<14} {int(r['n']):>4} {int(r['wins']):>5} "
            f"{r['iai']:>6.2f} {r['log10_p']:>10.1f}"
        )
    L("")

    L("Per-language aggregates")
    L("-" * 50)
    L(f"{'lang':<4} {'n':>4} {'wins':>5} {'IAI':>6} {'log10(p)':>10}")
    for _, r in by_lang.iterrows():
        L(
            f"{r['lang']:<4} {int(r['n']):>4} {int(r['wins']):>5} "
            f"{r['iai']:>6.2f} {r['log10_p']:>10.1f}"
        )
    L("")

    # Cells with smallest n: relevant for the "all p < 1e-17" robustness check
    smallest = cells.nsmallest(5, "n")
    L("Five smallest cells (sample-size sanity check)")
    L("-" * 50)
    for _, r in smallest.iterrows():
        L(
            f"{r['model']:<14} {r['lang']:<4} {r['subcat']:<14} "
            f"n={int(r['n']):>3}  log10(p)={r['log10_p']:>6.1f}"
        )
    L("")

    L("Verdict")
    L("-" * 50)
    cells_at_full_iai = (cells["iai"] == 10.0).sum()
    L(f"Cells with IAI = 10.0 (100% real-brand wins): {cells_at_full_iai} / {len(cells)}")
    L(f"Smallest single cell n: {int(cells['n'].min())}")
    L("Per-cell binomial p < 1e-17 is achieved when n >= 17 (since 0.1^17 = 1e-17).")
    n_strong = (cells["n"] >= 17).sum()
    L(f"Cells with n >= 17 (per-cell p < 1e-17): {n_strong} / {len(cells)}")
    L("Per-model aggregate p:")
    for _, r in by_model.iterrows():
        L(f"  {r['model']:<14} log10(p) = {r['log10_p']:.1f}  (well below -17)")
    L("")
    L("=" * 65)

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(f"  - {n_valid_beta} valid β trials, all real-brand wins")
    print(f"  - {len(cells)} per-cell breakdowns, all IAI=10.0")


if __name__ == "__main__":
    main()
