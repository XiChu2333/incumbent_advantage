"""
Exp 3 Language Effects Analysis — EN vs ZH
============================================
Run this script in the same directory as results/exp3/exp3_raw.csv
(i.e., from the 'LLM preference and advertising' folder).

Output:
  1. ISR by Language × Scenario (overall + per model)
  2. χ² / Fisher tests for EN vs ZH at each scenario
  3. Language × Model interaction at S4
  4. Mean Rank (Protocol α) by Language × Scenario
  5. Summary table for paper inclusion

Usage:
  python exp3_lang_analysis.py
"""

import csv
import numpy as np
from scipy import stats
from collections import defaultdict

CSV_PATH = "results/exp3/exp3_raw.csv"

# ═══════════════════════════════════════════════════════════════════
# 1. Load and filter data
# ═══════════════════════════════════════════════════════════════════

rows = []
with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"Total rows: {len(rows)}")

# Filter valid only
valid_beta = [r for r in rows if r["protocol"] == "beta"
              and r["is_parse_fail"] != "True"
              and r["is_refusal"] != "True"]
valid_alpha = [r for r in rows if r["protocol"] == "alpha"
               and r["is_parse_fail"] != "True"
               and r["is_refusal"] != "True"]

print(f"Valid β trials: {len(valid_beta)}")
print(f"Valid α trials: {len(valid_alpha)}")

scenarios = ["S0", "S1", "S2", "S3", "S4"]
models = ["claude-sonnet", "gemini-flash", "gpt-4o-mini"]
model_labels = {"claude-sonnet": "Claude", "gemini-flash": "Gemini", "gpt-4o-mini": "GPT"}
langs = ["en", "zh"]

# ═══════════════════════════════════════════════════════════════════
# 2. ISR by Language × Scenario (Protocol β)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  Protocol β: ISR by Language × Scenario")
print("=" * 72)

# Aggregate counts
def compute_isr(trials):
    """Return (n_total, n_chose_real, isr%)."""
    n = len(trials)
    chose = sum(1 for r in trials if r["chose_real"] == "True")
    return n, chose, (chose / n * 100) if n > 0 else 0.0

# Overall: Lang × Scenario
print(f"\n{'Scenario':<10} {'EN n':>6} {'EN ISR':>8} {'ZH n':>6} {'ZH ISR':>8} {'Δ(EN-ZH)':>9} {'p-value':>10} {'sig':>5}")
print("-" * 72)

for scen in scenarios:
    en_trials = [r for r in valid_beta if r["scenario"] == scen and r["lang"] == "en"]
    zh_trials = [r for r in valid_beta if r["scenario"] == scen and r["lang"] == "zh"]
    en_n, en_cr, en_isr = compute_isr(en_trials)
    zh_n, zh_cr, zh_isr = compute_isr(zh_trials)
    delta = en_isr - zh_isr

    # Fisher's exact test (or χ² if cells large enough)
    table = np.array([[en_cr, en_n - en_cr],
                      [zh_cr, zh_n - zh_cr]])
    if table.min() < 5 or (table == 0).any():
        _, p = stats.fisher_exact(table)
        test = "Fisher"
    else:
        chi2, p, _, _ = stats.chi2_contingency(table, correction=True)
        test = "χ²"

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{scen:<10} {en_n:>6} {en_isr:>7.1f}% {zh_n:>6} {zh_isr:>7.1f}% {delta:>+8.1f}pp {p:>10.4f} {sig:>5}")

# ═══════════════════════════════════════════════════════════════════
# 3. ISR by Language × Scenario × Model
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  Protocol β: ISR by Language × Scenario × Model")
print("=" * 72)

for model in models:
    mlabel = model_labels[model]
    print(f"\n  --- {mlabel} ---")
    print(f"  {'Scenario':<10} {'EN n':>5} {'EN ISR':>8} {'ZH n':>5} {'ZH ISR':>8} {'Δ':>8} {'p':>10} {'sig':>4}")
    for scen in scenarios:
        en_t = [r for r in valid_beta if r["scenario"] == scen and r["lang"] == "en" and r["model"] == model]
        zh_t = [r for r in valid_beta if r["scenario"] == scen and r["lang"] == "zh" and r["model"] == model]
        en_n, en_cr, en_isr = compute_isr(en_t)
        zh_n, zh_cr, zh_isr = compute_isr(zh_t)
        delta = en_isr - zh_isr

        table = np.array([[en_cr, en_n - en_cr],
                          [zh_cr, zh_n - zh_cr]])
        if table.min() < 5 or (table == 0).any():
            _, p = stats.fisher_exact(table)
        else:
            _, p, _, _ = stats.chi2_contingency(table, correction=True)

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {scen:<10} {en_n:>5} {en_isr:>7.1f}% {zh_n:>5} {zh_isr:>7.1f}% {delta:>+7.1f}pp {p:>10.4f} {sig:>4}")

# ═══════════════════════════════════════════════════════════════════
# 4. Mean Rank by Language × Scenario (Protocol α)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  Protocol α: Mean Rank by Language × Scenario")
print("=" * 72)

print(f"\n{'Scenario':<10} {'EN n':>5} {'EN Rank':>9} {'ZH n':>5} {'ZH Rank':>9} {'Δ':>7} {'p (MW)':>10} {'sig':>4}")
print("-" * 72)

for scen in scenarios:
    en_t = [r for r in valid_alpha if r["scenario"] == scen and r["lang"] == "en" and r["real_rank"]]
    zh_t = [r for r in valid_alpha if r["scenario"] == scen and r["lang"] == "zh" and r["real_rank"]]

    en_ranks = [int(r["real_rank"]) for r in en_t if r["real_rank"]]
    zh_ranks = [int(r["real_rank"]) for r in zh_t if r["real_rank"]]

    if en_ranks and zh_ranks:
        en_mean = np.mean(en_ranks)
        zh_mean = np.mean(zh_ranks)
        delta = en_mean - zh_mean
        # Mann-Whitney U test
        u_stat, p = stats.mannwhitneyu(en_ranks, zh_ranks, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{scen:<10} {len(en_ranks):>5} {en_mean:>8.2f} {len(zh_ranks):>5} {zh_mean:>8.2f} {delta:>+6.2f} {p:>10.4f} {sig:>4}")
    else:
        print(f"{scen:<10}  insufficient data")

# ═══════════════════════════════════════════════════════════════════
# 5. Overall Language Effect — pooled across S1-S4
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  Pooled Language Effect (S1-S4, Protocol β)")
print("=" * 72)

en_s1s4 = [r for r in valid_beta if r["scenario"] != "S0" and r["lang"] == "en"]
zh_s1s4 = [r for r in valid_beta if r["scenario"] != "S0" and r["lang"] == "zh"]

en_n, en_cr, en_isr = compute_isr(en_s1s4)
zh_n, zh_cr, zh_isr = compute_isr(zh_s1s4)

table = np.array([[en_cr, en_n - en_cr],
                  [zh_cr, zh_n - zh_cr]])
chi2, p, dof, _ = stats.chi2_contingency(table, correction=True)
# Cohen's h
h1 = 2 * np.arcsin(np.sqrt(en_cr / en_n))
h2 = 2 * np.arcsin(np.sqrt(zh_cr / zh_n))
cohen_h = abs(h1 - h2)

print(f"\n  EN: n={en_n}, ISR={en_isr:.1f}% ({en_cr}/{en_n})")
print(f"  ZH: n={zh_n}, ISR={zh_isr:.1f}% ({zh_cr}/{zh_n})")
print(f"  Δ(EN-ZH) = {en_isr - zh_isr:+.1f} pp")
print(f"  χ²({dof}) = {chi2:.3f}, p = {p:.4f}")
print(f"  Cohen's h = {cohen_h:.4f} ({'negligible' if cohen_h < 0.2 else 'small' if cohen_h < 0.5 else 'medium'})")

# ═══════════════════════════════════════════════════════════════════
# 6. Cochran-Mantel-Haenszel-like: Language effect controlling for scenario
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  Stratified Analysis: Language Effect Controlling for Scenario")
print("=" * 72)

# Breslow-Day test for homogeneity of ORs across strata is complex;
# we'll compute per-stratum ORs and a Mantel-Haenszel summary OR

mh_num = 0.0
mh_den = 0.0
print(f"\n  {'Stratum':<10} {'EN ISR':>8} {'ZH ISR':>8} {'OR':>8} {'Weight':>8}")
for scen in ["S1", "S2", "S3", "S4"]:
    en_t = [r for r in valid_beta if r["scenario"] == scen and r["lang"] == "en"]
    zh_t = [r for r in valid_beta if r["scenario"] == scen and r["lang"] == "zh"]
    en_n_, en_cr_, en_isr_ = compute_isr(en_t)
    zh_n_, zh_cr_, zh_isr_ = compute_isr(zh_t)

    # 2×2 table: [chose_real, chose_other] × [en, zh]
    a, b = en_cr_, en_n_ - en_cr_
    c, d = zh_cr_, zh_n_ - zh_cr_
    n_total = a + b + c + d

    # Odds ratio (with 0.5 continuity correction if needed)
    a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ = (a_ * d_) / (b_ * c_) if (b_ * c_) > 0 else float('inf')

    # MH weights
    mh_num += a * d / n_total if n_total > 0 else 0
    mh_den += b * c / n_total if n_total > 0 else 0

    w = n_total
    print(f"  {scen:<10} {en_isr_:>7.1f}% {zh_isr_:>7.1f}% {or_:>7.2f} {w:>8}")

mh_or = mh_num / mh_den if mh_den > 0 else float('inf')
print(f"\n  Mantel-Haenszel OR = {mh_or:.3f}")
print(f"  (OR > 1 → EN more likely to choose real brand; OR < 1 → ZH more likely)")

# ═══════════════════════════════════════════════════════════════════
# 7. Summary for paper
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  SUMMARY: Language Effects")
print("=" * 72)
print(f"""
  Protocol β (ISR):
    EN overall (S1-S4): ISR = {en_isr:.1f}% (n = {en_n})
    ZH overall (S1-S4): ISR = {zh_isr:.1f}% (n = {zh_n})
    Δ = {en_isr - zh_isr:+.1f} pp
    χ² p = {p:.4f}, Cohen's h = {cohen_h:.4f}
    Mantel-Haenszel OR = {mh_or:.3f}

  Interpretation:
    [Fill in based on results above]
    - If p > 0.05 and Cohen's h < 0.2: "No significant language effect"
    - If p < 0.05 but Cohen's h < 0.2: "Statistically significant but negligible effect"
    - If p < 0.05 and Cohen's h >= 0.2: "Meaningful language effect — discuss direction"

  Per-scenario pattern:
    [Check if EN-ZH gap is consistent or varies by scenario]
    [Check if any specific model shows stronger language sensitivity]
""")

print("Script complete. Copy the relevant numbers into the paper section.")
