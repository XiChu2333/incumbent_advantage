"""
Exp 1d Report Regenerator (M-6 fix)
====================================

Regenerates `results/exp1d/exp1d_report.txt` with:
  - One-way ANOVA effect sizes per main factor (matches paper §3.2)
  - Brand x Params interaction (eta^2 from 6-group decomposition)
  - Per-model effect-size breakdown
  - Parse-failure accounting (N = 9,545 of 9,600 designed calls)

Reads `results/exp1d/exp1d_raw.csv`. No new API calls.

Run:    python exp1d_report_regenerator.py
Output: overwrites results/exp1d/exp1d_report.txt
"""

import math
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "results" / "exp1d" / "exp1d_raw.csv"
OUT_PATH = ROOT / "results" / "exp1d" / "exp1d_report.txt"


def one_way_eta_sq(groups):
    """One-way ANOVA: returns dict with F, eta_sq, df_between, df_within."""
    k = len(groups)
    N = sum(len(g) for g in groups)
    if k < 2 or N <= k:
        return None
    flat = [v for g in groups for v in g]
    grand_mean = sum(flat) / N
    ssb = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups if g)
    ssw = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups if g)
    sst = ssb + ssw
    if sst == 0 or ssw == 0:
        return None
    df_b = k - 1
    df_w = N - k
    msb = ssb / df_b
    msw = ssw / df_w
    F = msb / msw
    eta_sq = ssb / sst
    return {"F": F, "eta_sq": eta_sq, "df_b": df_b, "df_w": df_w, "ssb": ssb, "ssw": ssw}


def f_pvalue_log10(F, df1, df2):
    """Log10 of one-tailed p-value for F-distribution; uses scipy if available."""
    try:
        from scipy.stats import f as fdist
        sf = fdist.sf(F, df1, df2)
        if sf > 0:
            return math.log10(sf)
        return float("-inf")
    except ImportError:
        return float("nan")


def main():
    df = pd.read_csv(CSV_PATH)
    n_total = len(df)
    v = df[df["is_parse_fail"] == False].copy()
    n_valid = len(v)
    n_pf = n_total - n_valid

    # Convert target_position to int once
    v["target_position"] = v["target_position"].astype(int)

    # === Main effects (one-way ANOVA per factor) ===
    factors = [
        ("brand", lambda r: "real" if r["target_is_real"] else "fictional"),
        ("params", lambda r: r["target_param_level"]),
        ("position", lambda r: int(r["target_position"])),
    ]

    main_results = {}
    for fname, key_fn in factors:
        gd = defaultdict(list)
        for _, r in v.iterrows():
            gd[key_fn(r)].append(r["target_rank"])
        groups = [gd[k] for k in sorted(gd.keys())]
        main_results[fname] = one_way_eta_sq(groups)

    # === Brand x Params interaction (6-group decomposition) ===
    gd = defaultdict(list)
    for _, r in v.iterrows():
        key = ("real" if r["target_is_real"] else "fictional", r["target_param_level"])
        gd[key].append(r["target_rank"])
    six_groups = [gd[k] for k in sorted(gd.keys())]
    six_result = one_way_eta_sq(six_groups)
    eta_brand_params = six_result["eta_sq"] - main_results["brand"]["eta_sq"] - main_results["params"]["eta_sq"]

    # === Per-model main effects ===
    per_model = {}
    for model in sorted(v["model"].unique()):
        m_data = v[v["model"] == model]
        per_model[model] = {}
        for fname, key_fn in factors:
            gd = defaultdict(list)
            for _, r in m_data.iterrows():
                gd[key_fn(r)].append(r["target_rank"])
            groups = [gd[k] for k in sorted(gd.keys())]
            per_model[model][fname] = one_way_eta_sq(groups)

    # === Build report ===
    L = []
    add = L.append

    add("EXP 1d F-Statistic Causal Decomposition Report (regenerated)")
    add("=" * 70)
    add(f"CSV source:           {CSV_PATH.name}")
    add(f"Total rows:           {n_total}")
    add(f"Parse failures:       {n_pf} ({100*n_pf/n_total:.2f}%)")
    add(f"Valid rows (N):       {n_valid}")
    add(f"Models tested:        {sorted(v['model'].unique())} (Gemini excluded — see Limitations)")
    add(f"Subcategories:        {sorted(v['subcat'].unique())}")
    add(f"Languages:            {sorted(v['lang'].unique())}")
    add(f"Param levels:         {sorted(v['target_param_level'].unique())}")
    add(f"Positions:            {sorted(v['target_position'].unique())}")
    add("")

    add("Main effects (one-way ANOVA per factor) — overall")
    add("-" * 70)
    add(f"{'factor':<10} {'F':>14} {'df_b':>6} {'df_w':>8} {'eta^2':>8} {'log10(p)':>10}")
    for fname in ["brand", "params", "position"]:
        r = main_results[fname]
        log10p = f_pvalue_log10(r["F"], r["df_b"], r["df_w"])
        log10p_str = f"{log10p:.1f}" if not math.isnan(log10p) else "n/a"
        add(f"{fname:<10} {r['F']:>14.1f} {r['df_b']:>6} {r['df_w']:>8} {r['eta_sq']:>8.4f} {log10p_str:>10}")
    add("")

    add(f"  --> Headline: eta^2_params = {main_results['params']['eta_sq']:.4f}, "
        f"eta^2_position = {main_results['position']['eta_sq']:.4f}, "
        f"eta^2_brand = {main_results['brand']['eta_sq']:.4f}")
    add("")

    add("Brand x Params interaction (6-group one-way ANOVA decomposition)")
    add("-" * 70)
    add(f"6-group eta^2:                 {six_result['eta_sq']:.4f}")
    add(f"  - main effect (brand):       {main_results['brand']['eta_sq']:.4f}")
    add(f"  - main effect (params):      {main_results['params']['eta_sq']:.4f}")
    add(f"  = Brand x Params interaction: {eta_brand_params:.4f}")
    add(f"6-group F-stat:                {six_result['F']:.1f} (df = {six_result['df_b']}, {six_result['df_w']})")
    add("")

    add("Cell means (target_rank by Brand x Params)")
    add("-" * 70)
    cell = v.groupby(["target_is_real", "target_param_level"])["target_rank"].agg(["mean", "count"]).reset_index()
    cell["brand"] = cell["target_is_real"].map({True: "real", False: "fictional"})
    add(f"{'brand':<10} {'params':<8} {'mean_rank':>10} {'n':>6}")
    for _, r in cell.iterrows():
        add(f"{r['brand']:<10} {r['target_param_level']:<8} {r['mean']:>10.2f} {int(r['count']):>6}")
    add("")

    add("Per-model main effects")
    add("-" * 70)
    add(f"  {'model':<14} {'eta^2_brand':>12} {'eta^2_params':>14} {'eta^2_position':>16}")
    for model, results in per_model.items():
        add(f"  {model:<14} "
            f"{results['brand']['eta_sq']:>12.4f} "
            f"{results['params']['eta_sq']:>14.4f} "
            f"{results['position']['eta_sq']:>16.4f}")
    add("")

    add("Verdict")
    add("-" * 70)
    add(f"Product parameters dominate ranking variance: eta^2 = "
        f"{main_results['params']['eta_sq']*100:.1f}%")
    add(f"Position bias contributes: eta^2 = "
        f"{main_results['position']['eta_sq']*100:.1f}%")
    add(f"Brand name alone contributes: eta^2 = "
        f"{main_results['brand']['eta_sq']*100:.1f}%")
    add(f"Brand x Params interaction confirms gating: eta^2 = "
        f"{eta_brand_params*100:.2f}%")
    add("")
    add("=" * 70)

    OUT_PATH.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(f"  eta^2_brand    = {main_results['brand']['eta_sq']:.4f}")
    print(f"  eta^2_params   = {main_results['params']['eta_sq']:.4f}")
    print(f"  eta^2_position = {main_results['position']['eta_sq']:.4f}")
    print(f"  eta^2_brand_x_params = {eta_brand_params:.4f}")


if __name__ == "__main__":
    main()
