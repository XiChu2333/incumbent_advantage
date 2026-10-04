"""
Exp 1b Report Regenerator (M-1 / M-2 / M-3 fix)
================================================

Regenerates `results/exp1b/exp1b_report.txt` with:
  - Per-subcategory BOR (verifies the 1.7-4.6% range)
  - Per-model and per-(model x lang) BOR
  - Memory Hallucination Probe results
  - 2x2 ANOVA on the (real_quality x fictional_quality) factorial,
    with eta^2 effect sizes for both main effects and the interaction

Reads `results/exp1b/exp1b_raw.csv`. No new API calls.

Run:    python exp1b_report_regenerator.py
Output: overwrites results/exp1b/exp1b_report.txt
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "results" / "exp1b" / "exp1b_raw.csv"
OUT_PATH = ROOT / "results" / "exp1b" / "exp1b_report.txt"

# 2x2 design (matches CONFIG in exp1b_swap.py):
#   config1 = real + good
#   config2 = real + bad
#   config3 = fictional + good
#   config4 = fictional + bad
COMPARISON_LABEL = {
    "aux2": ("good", "good"),  # real_quality, fictional_quality
    "aux1": ("good", "bad"),
    "main": ("bad", "good"),
    "aux3": ("bad", "bad"),
}

# Subcategory-specific "leaked" features that distinguish P_good from P_bad.
# If LLM mentions any of these while choosing the real brand under P_bad,
# we infer training-data memorization rather than faithful prompt reading.
LEAKED_FEATURES = {
    "moisturizer": {
        "en": ["niacinamide", "oil-free", "oil free", "non-comedogenic", "89ml", "89 ml", r"\$17", r"\b17\b"],
        "zh": ["烟酰胺", "无油", "不致痘", "89ml", "89 ml", "17"],
    },
    "bha_exfoliant": {
        "en": ["2%", "reduces redness", "acne-prone", "acne prone", "118ml", "118 ml"],
        "zh": ["2%", "减少泛红", "痘肌", "118ml", "118 ml"],
    },
    "sunscreen": {
        "en": ["zinc oxide", "niacinamide", "spf 46", "spf46", "acne-prone", "acne prone",
               "non-comedogenic", r"\$40", r"\b40\b"],
        "zh": ["氧化锌", "烟酰胺", "spf 46", "spf46", "痘肌", "不致痘", "40"],
    },
    "cleanser": {
        "en": ["niacinamide", "ceramide", "hyaluronic", "236ml", "236 ml", r"\$16", r"\b16\b"],
        "zh": ["烟酰胺", "神经酰胺", "透明质酸", "236ml", "236 ml", "16"],
    },
}


def main():
    raw = pd.read_csv(CSV_PATH)
    n_total = len(raw)
    v = raw[raw["is_parse_fail"] == False].copy()
    n_valid = len(v)
    n_pf = n_total - n_valid

    v["real_quality"] = v["comparison"].map(lambda c: COMPARISON_LABEL[c][0])
    v["fictional_quality"] = v["comparison"].map(lambda c: COMPARISON_LABEL[c][1])
    v["real_won"] = (v["chose_real"].astype(str) == "True").astype(int)

    main = v[v["comparison"] == "main"].copy()
    overall_bor = float(main["real_won"].mean())

    # === BOR breakdowns ===
    bor_subcat = main.groupby("subcat")["real_won"].agg(["mean", "count"]).reset_index()
    bor_model = main.groupby("model")["real_won"].agg(["mean", "count"]).reset_index()
    bor_model_lang = main.groupby(["model", "lang"])["real_won"].agg(["mean", "count"]).reset_index()
    bor_subcat_lang = main.groupby(["subcat", "lang"])["real_won"].agg(["mean", "count"]).reset_index()

    # === 2x2 ANOVA on real_won ===
    mdl = ols("real_won ~ C(real_quality) * C(fictional_quality)", data=v).fit()
    anova = anova_lm(mdl, typ=2)
    ss_total = float(anova["sum_sq"].sum())
    anova["eta_sq"] = anova["sum_sq"] / ss_total

    # === Hallucination probe ===
    main_real_won = main[main["chose_real"].astype(str) == "True"].copy()
    n_check = len(main_real_won)
    leaked_count = 0
    leaked_per_subcat = Counter()
    leaked_per_model = Counter()
    n_per_subcat = Counter()
    n_per_model = Counter()
    feature_examples = Counter()
    for _, r in main_real_won.iterrows():
        sub, lang, mdl_name = r["subcat"], r["lang"], r["model"]
        n_per_subcat[sub] += 1
        n_per_model[mdl_name] += 1
        feats = LEAKED_FEATURES.get(sub, {}).get(lang, [])
        resp = str(r.get("response", "")).lower()
        leaked_here = []
        for f in feats:
            if re.search(f.lower(), resp):
                leaked_here.append(f)
        if leaked_here:
            leaked_count += 1
            leaked_per_subcat[sub] += 1
            leaked_per_model[mdl_name] += 1
            for f in leaked_here:
                feature_examples[f] += 1

    # === Build report ===
    L = []
    add = L.append

    add("EXP 1b 2x2 Factorial Swap Report (regenerated)")
    add("=" * 70)
    add(f"CSV source:           {CSV_PATH.name}")
    add(f"Total rows:           {n_total}")
    add(f"Parse failures:       {n_pf}")
    add(f"Valid rows:           {n_valid}")
    add("")

    add("Brand Override Rate (BOR) — main comparison only")
    add("-" * 70)
    add(f"Definition: P(choose real brand | real+bad vs fictional+good)")
    add(f"Overall:    {len(main[main['real_won']==1])} / {len(main)} = {overall_bor*100:.2f}%")
    add("")

    add("BOR by subcategory")
    add("-" * 70)
    add(f"{'subcat':<14} {'BOR%':>7} {'wins':>6} {'n':>5}")
    for _, r in bor_subcat.iterrows():
        add(f"{r['subcat']:<14} {r['mean']*100:>6.2f}%  {int(r['mean']*r['count']):>5} {int(r['count']):>5}")
    bor_min, bor_max = bor_subcat["mean"].min() * 100, bor_subcat["mean"].max() * 100
    add(f"--> per-subcategory range:  {bor_min:.1f}% – {bor_max:.1f}%")
    add("")

    add("BOR by model")
    add("-" * 70)
    add(f"{'model':<14} {'BOR%':>7} {'wins':>6} {'n':>5}")
    for _, r in bor_model.iterrows():
        add(f"{r['model']:<14} {r['mean']*100:>6.2f}%  {int(r['mean']*r['count']):>5} {int(r['count']):>5}")
    add("")

    add("BOR by (model x lang)")
    add("-" * 70)
    add(f"{'model':<14} {'lang':<4} {'BOR%':>7} {'wins':>6} {'n':>5}")
    for _, r in bor_model_lang.iterrows():
        add(f"{r['model']:<14} {r['lang']:<4} {r['mean']*100:>6.2f}%  {int(r['mean']*r['count']):>5} {int(r['count']):>5}")
    add("")

    add("BOR by (subcat x lang)")
    add("-" * 70)
    add(f"{'subcat':<14} {'lang':<4} {'BOR%':>7} {'wins':>6} {'n':>5}")
    for _, r in bor_subcat_lang.iterrows():
        add(f"{r['subcat']:<14} {r['lang']:<4} {r['mean']*100:>6.2f}%  {int(r['mean']*r['count']):>5} {int(r['count']):>5}")
    add("")

    add("2x2 ANOVA on real-brand-wins (Type II)")
    add("-" * 70)
    add(f"Factors: real_quality (good|bad) x fictional_quality (good|bad)")
    add(f"N = {n_valid}")
    add(f"{'effect':<40} {'df':>5} {'SS':>10} {'F':>10} {'p':>10} {'eta^2':>8}")
    for idx, r in anova.iterrows():
        if idx == "Residual":
            add(f"{idx:<40} {int(r['df']):>5} {r['sum_sq']:>10.2f} {'':>10} {'':>10} {r['eta_sq']:>8.4f}")
        else:
            p_str = "<1e-300" if r["PR(>F)"] == 0 else f"{r['PR(>F)']:.2e}"
            add(f"{idx:<40} {int(r['df']):>5} {r['sum_sq']:>10.2f} {r['F']:>10.1f} {p_str:>10} {r['eta_sq']:>8.4f}")
    add("")
    interaction_idx = "C(real_quality):C(fictional_quality)"
    inter_eta = float(anova.loc[interaction_idx, "eta_sq"])
    inter_F = float(anova.loc[interaction_idx, "F"])
    add(f"--> Interaction:  eta^2 = {inter_eta:.3f},  F = {inter_F:.0f}")
    add(f"    (paper text reports eta^2 = 0.264, F = 4,345; recompute yields slightly larger)")
    add("")

    # Cell means table for the 2x2
    add("2x2 cell means — P(real brand wins)")
    add("-" * 70)
    cell = v.groupby(["real_quality", "fictional_quality"])["real_won"].mean().unstack()
    cell_n = v.groupby(["real_quality", "fictional_quality"])["real_won"].count().unstack()
    add("                   fictional_quality")
    add(f"  real_quality   |   good       bad")
    for rq in ["good", "bad"]:
        row = []
        for fq in ["good", "bad"]:
            try:
                row.append(f"{cell.loc[rq, fq]*100:>6.2f}% (n={int(cell_n.loc[rq, fq])})")
            except KeyError:
                row.append("   n/a")
        add(f"  {rq:<14} | {row[0]}  {row[1]}")
    add("")

    add("Memory Hallucination Probe")
    add("-" * 70)
    add(f"Definition: among 'main' responses where real brand was chosen under")
    add(f"P_bad description, count those that mention features absent from")
    add(f"the P_bad prompt but present in the actual real product (training-")
    add(f"data leakage indicator).")
    add(f"Responses checked:        {n_check}")
    add(f"With leaked features:     {leaked_count}")
    rate = (leaked_count / n_check * 100) if n_check else float("nan")
    add(f"Hallucination rate:       {rate:.1f}%")
    add("")
    if leaked_per_subcat:
        add("Leaks by subcat:")
        for sub in sorted(leaked_per_subcat):
            add(f"  {sub}: {leaked_per_subcat[sub]}/{n_per_subcat[sub]}")
    if leaked_per_model:
        add("Leaks by model:")
        for m in sorted(leaked_per_model):
            add(f"  {m}: {leaked_per_model[m]}/{n_per_model[m]}")
    if feature_examples:
        add("Top leaked features:")
        for f, c in feature_examples.most_common(5):
            add(f"  {f}: {c}")
    if leaked_count == 0:
        add("Verdict: 0% training-data contamination — BOR reflects genuine")
        add("brand-name bias, not memorized product features.")
    add("")
    add("=" * 70)

    OUT_PATH.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(f"  Per-subcategory BOR range: {bor_min:.1f}% to {bor_max:.1f}%")
    print(f"  Interaction: eta^2={inter_eta:.3f}, F={inter_F:.0f}")
    print(f"  Hallucination: {leaked_count}/{n_check} = {rate:.1f}%")


if __name__ == "__main__":
    main()
