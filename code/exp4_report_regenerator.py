#!/usr/bin/env python3
"""
EXP 4 — RAG Probe Report Regenerator (no API calls)
===================================================
Recomputes every Experiment 4 / Appendix F number reported in the paper from
results/exp4/exp4_raw.csv and writes results/exp4/exp4_report.txt.

Covers: F.1 validity, ISR by pipeline x GEO condition, F.2 retrieval-survival /
generation-selection decomposition (all four RAG cells), F.3 embedding
similarity (neutral vs GEO vs real brand), F.4 model differences, F.5 the
K5-S1 survival cases and their mechanism.

Usage (from the repository root):
  python code/exp4_report_regenerator.py
"""

import os, csv, math
from datetime import datetime
from collections import Counter
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "results", "exp4", "exp4_raw.csv")
OUT_PATH = os.path.join(ROOT, "results", "exp4", "exp4_report.txt")

PIPES = ["direct", "rag-k10", "rag-k5"]
CONDS = ["S0", "S1"]
MODELS = ["claude-sonnet", "gpt-4o-mini", "gemini-flash"]
MLAB = {"claude-sonnet": "Claude", "gpt-4o-mini": "GPT", "gemini-flash": "Gemini"}

out = []
def P(s=""):
    out.append(s); print(s)

def pct(c, n): return 100 * c / n if n else float("nan")
def parse_kv(s, cast=float): return {p.split(":")[0]: cast(p.split(":")[1]) for p in s.split(",")}

rows_all = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
rows = [r for r in rows_all if r["is_parse_fail"] != "True" and r["is_refusal"] != "True"]

P("EXP 4 — RAG Probe Report")
P(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
P(f"Total rows: {len(rows_all)}   Parse failures: {sum(r['is_parse_fail']=='True' for r in rows_all)}   "
  f"Refusals: {sum(r['is_refusal']=='True' for r in rows_all)}   Valid: {len(rows)} ({pct(len(rows),len(rows_all)):.1f}%)")
P("Design: " + " x ".join(f"{k}={len(set(r[k] for r in rows_all))}" for k in ["pipeline", "geo_condition", "subcat", "model", "lang", "repeat"])
  + f"   subcats = {sorted(set(r['subcat'] for r in rows_all))}")

# ── 1. ISR table ──
P("\n" + "=" * 72); P("  1. ISR (%) by pipeline x GEO condition (Protocol β)"); P("=" * 72)
P(f"{'pipeline':<9}{'cond':<5} " + "".join(f"{MLAB[m]:>16}" for m in MODELS) + f"{'All':>16}")
ISR = {}
for pl in PIPES:
    for gc in CONDS:
        sub = [r for r in rows if r["pipeline"] == pl and r["geo_condition"] == gc]
        line = f"{pl:<9}{gc:<5} "
        for m in MODELS:
            ms = [r for r in sub if r["model"] == m]; c = sum(r["chose_real"] == "True" for r in ms)
            ISR[(pl, gc, m)] = (c, len(ms)); line += f"{pct(c,len(ms)):>7.1f} ({c:>2}/{len(ms):<2})"
        c = sum(r["chose_real"] == "True" for r in sub); ISR[(pl, gc)] = (c, len(sub))
        P(line + f"{pct(c,len(sub)):>7.1f} ({c:>3}/{len(sub):<3})")

# ── 2. Retrieval survival / generation selection ──
P("\n" + "=" * 72); P("  2. F.2 — Retrieval survival vs generation selection"); P("=" * 72)
for pl in ["rag-k10", "rag-k5"]:
    for gc in CONDS:
        sub = [r for r in rows if r["pipeline"] == pl and r["geo_condition"] == gc]
        surv = [r for r in sub if r["real_retrieved"] == "True"]
        gen = sum(r["chose_real"] == "True" for r in surv)
        gsel = f"{pct(gen,len(surv)):.1f}% ({gen}/{len(surv)})" if surv else "undefined (never retrieved)"
        P(f"{pl} {gc}: retrieval survival = {pct(len(surv),len(sub)):.1f}% ({len(surv)}/{len(sub)}); generation selection = {gsel}")
        if gc == "S1":
            g_in = sum(r["geo_retrieved"] == "True" for r in sub); g_ch = sum(r["chose_geo"] == "True" for r in sub)
            P(f"           GEO brand: in top-K {pct(g_in,len(sub)):.1f}% ({g_in}/{len(sub)}); chosen {pct(g_ch,len(sub)):.1f}% ({g_ch}/{len(sub)})")
d1 = [r for r in rows if r["pipeline"] == "direct" and r["geo_condition"] == "S1"]
P(f"direct  S1: GEO brand chosen {pct(sum(r['chose_geo']=='True' for r in d1),len(d1)):.1f}%")

# ── 3. Similarity analysis ──
P("\n" + "=" * 72); P("  3. F.3 — Embedding similarity (cosine to query), reconstructed from retrieval_scores_all"); P("=" * 72)
rag = [r for r in rows if r["pipeline"].startswith("rag")]
neutral, geo, real, fict_all, paired, geo_rank, cells = [], [], [], [], [], [], {}
for r in rag:
    sc = parse_kv(r["retrieval_scores_all"]); rk = parse_kv(r["retrieval_ranks_all"], int)
    rr = int(float(r["real_retrieval_rank"])); gr = int(float(r["geo_retrieval_rank"])) if r["geo_retrieval_rank"] not in ("", "nan") else None
    real_pid = next(k for k, v in rk.items() if v == rr); geo_pid = next(k for k, v in rk.items() if v == gr) if gr else None
    neut = [v for k, v in sc.items() if k not in (real_pid, geo_pid)]
    neutral += neut; real.append(sc[real_pid]); fict_all += [v for k, v in sc.items() if k != real_pid]
    key = (r["subcat"], r["lang"]); cells.setdefault(key, {"n": [], "g": [], "r": []}); cells[key]["n"] += neut; cells[key]["r"].append(sc[real_pid])
    if geo_pid:
        geo.append(sc[geo_pid]); paired.append((np.mean(neut), sc[geo_pid])); geo_rank.append(gr); cells[key]["g"].append(sc[geo_pid])
P(f"neutral fictional descriptions: mean = {np.mean(neutral):.3f}, SD = {np.std(neutral,ddof=1):.3f}, n = {len(neutral)}")
P(f"GEO (Authority) descriptions:   mean = {np.mean(geo):.3f}, SD = {np.std(geo,ddof=1):.3f}, n = {len(geo)}")
P(f"real brand:                     mean = {np.mean(real):.3f}, SD = {np.std(real,ddof=1):.3f}, n = {len(real)};  fictional-brand median = {np.median(fict_all):.3f}")
d = (np.mean(geo) - np.mean(neutral)) / math.sqrt((np.var(geo, ddof=1) + np.var(neutral, ddof=1)) / 2)
t, p = stats.ttest_ind(geo, neutral)
P(f"ΔSim (GEO − neutral) = {np.mean(geo)-np.mean(neutral):+.4f};  Cohen's d = {d:.2f};  unpaired t = {t:.2f}, p = {p:.3f}")
a = np.array([x[1] for x in paired]); b = np.array([x[0] for x in paired]); t, p = stats.ttest_rel(a, b)
P(f"paired per-trial (GEO vs mean neutral in same trial, n = {len(a)}): t({len(a)-1}) = {t:.2f}, p = {p:.1e}, d_z = {(a-b).mean()/(a-b).std(ddof=1):.2f}")
for key in sorted(cells):
    c = cells[key]
    P(f"   {key[0]:<12}{key[1]}: neutral {np.mean(c['n']):.4f}  GEO {np.mean(c['g']):.4f}  real {np.mean(c['r']):.4f}  Δ = {np.mean(c['g'])-np.mean(c['n']):+.4f}")
# ranks
real_ranks = [int(float(r["real_retrieval_rank"])) for r in rag]
P(f"real brand mean retrieval rank: {np.mean(real_ranks):.2f}/10   GEO brand mean retrieval rank (S1): {np.mean(geo_rank):.2f}/10 "
  f"(expected for a neutral fictional brand ≈ {(55 - np.mean(real_ranks)) / 9:.2f})")
k5s1 = [r for r in rows if r["pipeline"] == "rag-k5" and r["geo_condition"] == "S1"]
g_in = sum(r["geo_retrieved"] == "True" for r in k5s1)
P(f"GEO brand enters top-5 in {pct(g_in,len(k5s1)):.1f}% of RAG-K5 S1 trials; chance rate for a neutral fictional brand ≈ {100*(5 - np.mean([1 for r in k5s1 if r['real_retrieved']=='True'] + [0]*(len(k5s1)-sum(r['real_retrieved']=='True' for r in k5s1)))) / 9:.1f}%")

# ── 4. Model differences ──
P("\n" + "=" * 72); P("  4. F.4 — Model differences at S1"); P("=" * 72)
for pl in ["rag-k5", "direct"]:
    tab = [[ISR[(pl, "S1", m)][0], ISR[(pl, "S1", m)][1] - ISR[(pl, "S1", m)][0]] for m in MODELS]
    chi, p, _, _ = stats.chi2_contingency(tab)
    P(f"{pl:<8} S1: " + ", ".join(f"{MLAB[m]} {pct(*ISR[(pl,'S1',m)]):.1f}%" for m in MODELS) + f";  chi2(2) = {chi:.2f}, p = {p:.2f}")

# ── 5. K5-S1 survival mechanism ──
P("\n" + "=" * 72); P("  5. F.5 — RAG-K5 S1 trials in which the real brand entered the top-5"); P("=" * 72)
surv = [r for r in k5s1 if r["real_retrieved"] == "True"]
P(f"count: {len(surv)}/{len(k5s1)} ({pct(len(surv),len(k5s1)):.1f}%); cells: {dict(Counter((r['subcat'], r['lang']) for r in surv))}")
P(f"real brand rank in these trials: {dict(Counter(int(float(r['real_retrieval_rank'])) for r in surv))};  GEO brand rank: {dict(Counter(int(float(r['geo_retrieval_rank'])) for r in surv))}")
P(f"real similarity: {sorted(set(round(float(r['real_similarity']),4) for r in surv))};  GEO similarity: {sorted(set(round(float(r['geo_similarity']),4) for r in surv))}")
P(f"chose real when retrieved: {sum(r['chose_real']=='True' for r in surv)}/{len(surv)}")
s0k5 = [r for r in rows if r["pipeline"] == "rag-k5" and r["geo_condition"] == "S0"]
P(f"RAG-K5 S0: real brand best rank = {min(int(float(r['real_retrieval_rank'])) for r in s0k5)} (never in top-5: {sum(r['real_retrieved']=='True' for r in s0k5)}/{len(s0k5)})")
k10s1 = [r for r in rows if r["pipeline"] == "rag-k10" and r["geo_condition"] == "S1"]
P(f"RAG-K10 S1 trials with the same rank shift (real rank 5, GEO rank 7): "
  f"{sum(int(float(r['real_retrieval_rank']))==5 and int(float(r['geo_retrieval_rank']))==7 for r in k10s1)}/{len(k10s1)}")
P("Interpretation: in the EN-moisturizer cell the Authority text LOWERS the GEO brand's similarity below the real brand's,")
P("so the GEO brand drops to rank 7 and the real brand rises from rank 6 to rank 5 (into the top-5).")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
open(OUT_PATH, "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"\nReport written to {OUT_PATH}")
