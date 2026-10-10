#!/usr/bin/env python3
"""
EXP 3 — Report Regenerator (no API calls)
=========================================
Recomputes every Experiment 3 number reported in the paper from
results/exp3/exp3_raw.csv and writes results/exp3/exp3_report.txt.

Covers: Table 3 (ISR by scenario x model), mean incumbent rank (Protocol α),
per-brand GEO payoff, HHI, H3a–H3e, KL divergence (E.3), language and
subcategory effects (E.4), non-participation penalty with exact CI.

Usage (from the repository root):
  python code/exp3_report_regenerator.py
"""

import os, sys, math, csv
from datetime import datetime
from collections import Counter
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "results", "exp3", "exp3_raw.csv")
OUT_PATH = os.path.join(ROOT, "results", "exp3", "exp3_report.txt")

SCEN = ["S0", "S1", "S2", "S3", "S4"]
K = {"S0": 0, "S1": 1, "S2": 3, "S3": 6, "S4": 9}
MODELS = ["claude-sonnet", "gpt-4o-mini", "gemini-flash"]
MLAB = {"claude-sonnet": "Claude", "gpt-4o-mini": "GPT", "gemini-flash": "Gemini"}
SUBCATS = ["moisturizer", "sunscreen", "cleanser", "bha_exfoliant"]
KL_EPS = 1e-6  # same as exp3_game.py analyze_exp3

out = []
def P(s=""):
    out.append(s); print(s)

def isr(trials):
    n = len(trials); c = sum(r["chose_real"] == "True" for r in trials)
    return c, n, (100 * c / n if n else float("nan"))

def chi2_2x2(a, b, c, d, correction=True):
    chi2, p, _, _ = stats.chi2_contingency([[a, b], [c, d]], correction=correction)
    return chi2, p

def cochran_armitage(counts, totals, scores):
    """Cochran–Armitage trend test (z and two-sided p)."""
    counts, totals, scores = map(np.asarray, (counts, totals, scores))
    N = totals.sum(); R = counts.sum(); pbar = R / N
    sbar = (scores * totals).sum() / N
    T = (counts * (scores - sbar)).sum()
    var = pbar * (1 - pbar) * (totals * (scores - sbar) ** 2).sum()
    z = T / math.sqrt(var)
    return z, 2 * stats.norm.sf(abs(z))

# ── Load ──
rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
valid = [r for r in rows if r["is_parse_fail"] != "True" and r["is_refusal"] != "True"]
beta = [r for r in valid if r["protocol"] == "beta"]
alpha = [r for r in valid if r["protocol"] == "alpha" and r["real_rank"] not in ("", "not_found")]

P("EXP 3 — Multi-Agent GEO Competition Report")
P(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
P(f"Total rows: {len(rows)}  (designed calls: 4,800; the CSV may contain a retried duplicate)")
P(f"Parse failures: {sum(r['is_parse_fail']=='True' for r in rows)}   Refusals: {sum(r['is_refusal']=='True' for r in rows)}")
P(f"Valid: {len(valid)} ({100*len(valid)/len(rows):.1f}% of rows)")
P(f"Valid Protocol β (single choice): {len(beta)}   Valid Protocol α (ranking): {len(alpha)}")

# ── Table 3: ISR ──
P("\n" + "=" * 72); P("  TABLE 3 — Incumbent Survival Rate (%) — Protocol β"); P("=" * 72)
P(f"{'Scen':<5}{'k':>3} | {'Claude':>14} {'GPT':>14} {'Gemini':>14} | {'All':>14}")
ISR = {}
for s in SCEN:
    sr = [r for r in beta if r["scenario"] == s]
    cells = []
    ISR[s] = {}
    for m in MODELS:
        c, n, v = isr([r for r in sr if r["model"] == m]); ISR[s][m] = (c, n, v)
        cells.append(f"{v:5.1f} ({c:>3}/{n:<3})")
    c, n, v = isr(sr); ISR[s]["all"] = (c, n, v)
    P(f"{s:<5}{K[s]:>3} | " + " ".join(cells) + f" | {v:5.1f} ({c}/{n})")

# ── Mean rank (Protocol α) ──
P("\n" + "=" * 72); P("  Mean incumbent rank (Protocol α; 1 = top)"); P("=" * 72)
for s in SCEN:
    sr = [r for r in alpha if r["scenario"] == s]
    P(f"{s}: " + "  ".join(f"{MLAB[m]}={np.mean([int(r['real_rank']) for r in sr if r['model']==m]):.2f}" for m in MODELS)
      + f"  All={np.mean([int(r['real_rank']) for r in sr]):.2f}  (n={len(sr)})")

# ── Payoff, HHI, visibility ──
P("\n" + "=" * 72); P("  GEO payoff per brand, HHI, challenger visibility (Protocol β)"); P("=" * 72)
s0 = [r for r in beta if r["scenario"] == "S0"]
base_per_brand = (sum(r["chose_real"] != "True" for r in s0) / len(s0)) / 9  # fictional share at S0, per brand
P(f"Per-brand fictional share at S0 (baseline for payoff): {base_per_brand:.4f}")
pay = {}; nongeo_total = 0; nongeo_denom_s1s3 = 0
P(f"{'Scen':<5}{'k':>3} {'ISR':>7} {'GEO share':>10} {'payoff/brand':>13} {'nonGEO chosen':>14} {'HHI':>7}")
for s in SCEN:
    sr = [r for r in beta if r["scenario"] == s]; k = K[s]; n = len(sr)
    real = sum(r["chose_real"] == "True" for r in sr) / n
    geo = sum(r["chose_geo"] == "True" for r in sr) / n
    nongeo_n = sum(r["chose_real"] != "True" and r["chose_geo"] != "True" for r in sr)
    nongeo = nongeo_n / n
    hhi = real ** 2 + (k * (geo / k) ** 2 if k else 0) + ((9 - k) * (nongeo / (9 - k)) ** 2 if 9 - k else 0)
    payoff = (geo / k - base_per_brand) if k else float("nan")
    if k: pay[k] = payoff
    if 1 <= k <= 6:
        nongeo_total += nongeo_n; nongeo_denom_s1s3 += n
    P(f"{s:<5}{k:>3} {100*real:6.1f}% {geo:10.4f} {payoff:13.4f} {nongeo_n:>8}/{n:<5} {hhi:7.3f}")
P(f"Total challenger visibility: k=1 {sum(r['chose_geo']=='True' for r in beta if r['scenario']=='S1')/len([r for r in beta if r['scenario']=='S1']):.3f}"
  f"  ->  k=9 {sum(r['chose_geo']=='True' for r in beta if r['scenario']=='S4')/len([r for r in beta if r['scenario']=='S4']):.3f}")

# ── Hypothesis tests ──
P("\n" + "=" * 72); P("  E.2 — Hypothesis tests"); P("=" * 72)
c0, n0, _ = ISR["S0"]["all"]; c1, n1, _ = ISR["S1"]["all"]
chi, p = chi2_2x2(c0, n0 - c0, c1, n1 - c1, correction=False)
chiY, pY = chi2_2x2(c0, n0 - c0, c1, n1 - c1, correction=True)
P(f"H3a  S0 vs S1 ISR: chi2(1) = {chi:.1f} (Yates {chiY:.1f}), p = {p:.2e}")
cs = [ISR[s]["all"][0] for s in SCEN[1:]]; ns = [ISR[s]["all"][1] for s in SCEN[1:]]
z, p = cochran_armitage(cs, ns, [K[s] for s in SCEN[1:]])
z2, p2 = cochran_armitage(cs[1:], ns[1:], [K[s] for s in SCEN[2:]])
P(f"H3b  Cochran–Armitage trend S1->S4: z = {z:.1f}, p = {p:.1e};  S2->S4: z = {z2:.1f}, p = {p2:.1e}")
ub95 = 1 - 0.025 ** (1 / nongeo_denom_s1s3)
n_s1s4 = sum(ISR[s]["all"][1] for s in SCEN[1:])
P(f"H3c  non-GEO fictional recommended: {nongeo_total}/{nongeo_denom_s1s3} valid β trials in S1–S3 "
  f"(0/{n_s1s4} over S1–S4); exact 95% upper bound (Clopper–Pearson) = {100*ub95:.2f}%; "
  f"binomial P(0 | n={nongeo_denom_s1s3}, p=0.01) = {0.99**nongeo_denom_s1s3:.1e}")
tab = [[ISR['S4'][m][0], ISR['S4'][m][1] - ISR['S4'][m][0]] for m in MODELS]
chi, p, _, _ = stats.chi2_contingency(tab)
P(f"H3d  S4 model differences: chi2(2) = {chi:.2f}, p = {p:.1e}  (" + ", ".join(f"{ISR['S4'][m][0]}/{ISR['S4'][m][1]}" for m in MODELS) + ")")
ks = np.array(sorted(pay)); ys = np.array([pay[k] for k in ks])
f = lambda k, a, b: a * np.exp(-b * (k - 1))
(a, b), _ = curve_fit(f, ks, ys, p0=[0.8, 0.5])
r2 = 1 - ((ys - f(ks, a, b)) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
sl, ic, rv, _, _ = stats.linregress(ks, np.log(ys))
P(f"H3e  payoff decay: NLS y = {a:.2f}·exp(-{b:.2f}(k-1)), R² = {r2:.3f}, half-life = {math.log(2)/b:.2f} brands; "
  f"log-linear slope {-sl:.2f} -> half-life {math.log(2)/-sl:.2f} brands")

# ── KL ──
P("\n" + "=" * 72); P(f"  E.3 — KL(S4 || S0), two-category {{incumbent, fictional}}, natural log, eps = {KL_EPS:g}"); P("=" * 72)
P("NOTE: P(fictional | S0) = 0 for every model, so magnitudes depend on eps; read as an ordering only.")
for m in MODELS:
    p4 = np.array([ISR["S4"][m][2] / 100, 1 - ISR["S4"][m][2] / 100])
    p0 = np.array([ISR["S0"][m][2] / 100, 1 - ISR["S0"][m][2] / 100])
    p4 = np.clip(p4, KL_EPS, None); p4 /= p4.sum(); p0 = np.clip(p0, KL_EPS, None); p0 /= p0.sum()
    P(f"{MLAB[m]:<7} KL = {float(np.sum(p4*np.log(p4/p0))):.3f}   (1 - ISR_S4 = {100-ISR['S4'][m][2]:.1f}%)")

# ── Language ──
P("\n" + "=" * 72); P("  E.4 — Language effect (pooled S1–S4, Protocol β)"); P("=" * 72)
pool = [r for r in beta if r["scenario"] != "S0"]
def lang_block(trials, label):
    en = [r for r in trials if r["lang"] == "en"]; zh = [r for r in trials if r["lang"] == "zh"]
    ce, ne, ve = isr(en); cz, nz, vz = isr(zh)
    chiY, pY = chi2_2x2(ce, ne - ce, cz, nz - cz, True); chi, p = chi2_2x2(ce, ne - ce, cz, nz - cz, False)
    h = abs(2 * math.asin(math.sqrt(ce / ne)) - 2 * math.asin(math.sqrt(cz / nz)))
    P(f"{label:<8} EN {ve:5.1f}% ({ce}/{ne})  ZH {vz:5.1f}% ({cz}/{nz})  chi2(1) = {chiY:.2f}, p = {pY:.4f} (Yates; {chi:.2f}, p = {p:.4f} uncorrected)  Cohen's h = {h:.2f}")
lang_block(pool, "All")
for m in MODELS:
    lang_block([r for r in pool if r["model"] == m], MLAB[m])

# ── Subcategory ──
P("\n" + "=" * 72); P("  E.4 — Subcategory effect (pooled S1–S4, Protocol β)"); P("=" * 72)
tab = [[isr([r for r in pool if r["subcat"] == sc])[0], isr([r for r in pool if r["subcat"] == sc])[1] - isr([r for r in pool if r["subcat"] == sc])[0]] for sc in SUBCATS]
chi, p, dof, _ = stats.chi2_contingency(tab)
P(f"chi2({dof}) = {chi:.2f}, p = {p:.1e}")
for sc in SUBCATS:
    c, n, v = isr([r for r in pool if r["subcat"] == sc])
    v1 = isr([r for r in beta if r["scenario"] == "S1" and r["subcat"] == sc])[2]
    v4 = isr([r for r in beta if r["scenario"] == "S4" and r["subcat"] == sc])[2]
    P(f"  {sc:<14} ISR pooled {v:5.1f}% ({c}/{n})   S1 {v1:5.1f}%   S4 {v4:5.1f}%")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
open(OUT_PATH, "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"\nReport written to {OUT_PATH}")
