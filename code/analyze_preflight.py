#!/usr/bin/env python3
"""
Quick analysis of Phase 2 + Phase 3 pre-flight results.
Run from the project root:  python analyze_preflight.py
"""
import csv, json, os
from collections import Counter

RESULTS = os.path.join(os.path.dirname(__file__), "results", "preflight")

# ── Phase 2: Prompt Template ──
csv_path = os.path.join(RESULTS, "prompt_template_raw.csv")
rows = []
with open(csv_path, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r.get("subcat"):
            rows.append(r)

print("=" * 70)
print("PHASE 2 — CHOICE DISTRIBUTION ANALYSIS")
print("=" * 70)
print(f"Total valid rows: {len(rows)}")

success = [r for r in rows if r.get("is_parse_fail") == "False" and r.get("is_refusal") == "False"]
pf = [r for r in rows if r.get("is_parse_fail") == "True"]
print(f"Successful parses: {len(success)}  |  Parse failures: {len(pf)}  |  Refusals: {len(rows) - len(success) - len(pf)}")

# Overall choice
choices = Counter(r.get("parsed_choice", "?") for r in success)
print(f"\nOverall choice distribution:")
for k, v in choices.most_common():
    print(f"  {k}: {v} ({v/len(success)*100:.1f}%)")

# By subcategory
print(f"\n{'─'*70}")
print("By subcategory:")
for subcat in sorted(set(r["subcat"] for r in rows)):
    sub_s = [r for r in success if r["subcat"] == subcat]
    sub_pf = [r for r in pf if r["subcat"] == subcat]
    sc = Counter(r.get("parsed_choice", "?") for r in sub_s)
    total = len([r for r in rows if r["subcat"] == subcat])
    print(f"  {subcat} (total={total}, success={len(sub_s)}, parse_fail={len(sub_pf)}):")
    for k, v in sc.most_common():
        print(f"    Choice {k}: {v} ({v/len(sub_s)*100:.1f}%)" if sub_s else "    (no data)")

# By model
print(f"\n{'─'*70}")
print("By model:")
for model in sorted(set(r["model"] for r in rows)):
    ms = [r for r in success if r["model"] == model]
    mpf = [r for r in pf if r["model"] == model]
    mc = Counter(r.get("parsed_choice", "?") for r in ms)
    print(f"  {model} (success={len(ms)}, parse_fail={len(mpf)}):")
    for k, v in mc.most_common():
        print(f"    Choice {k}: {v} ({v/len(ms)*100:.1f}%)" if ms else "    (no data)")

# By template
print(f"\n{'─'*70}")
print("By template:")
for tpl in ["A", "B", "C"]:
    ts = [r for r in success if r["template"] == tpl]
    tpf = [r for r in pf if r["template"] == tpl]
    tc = Counter(r.get("parsed_choice", "?") for r in ts)
    print(f"  Template {tpl} (success={len(ts)}, parse_fail={len(tpf)}):")
    for k, v in tc.most_common():
        print(f"    Choice {k}: {v} ({v/len(ts)*100:.1f}%)" if ts else "    (no data)")

# By language
print(f"\n{'─'*70}")
print("By language:")
for lang in ["en", "zh"]:
    ls = [r for r in success if r["lang"] == lang]
    lpf = [r for r in pf if r["lang"] == lang]
    lc = Counter(r.get("parsed_choice", "?") for r in ls)
    print(f"  {lang} (success={len(ls)}, parse_fail={len(lpf)}):")
    for k, v in lc.most_common():
        print(f"    Choice {k}: {v} ({v/len(ls)*100:.1f}%)" if ls else "    (no data)")

# Gemini parse fail detail
print(f"\n{'─'*70}")
print("Parse failure detail (Gemini Flash):")
gem_pf = [r for r in pf if r["model"] == "gemini-flash"]
for lang in ["en", "zh"]:
    for tpl in ["A", "B", "C"]:
        cnt = len([r for r in gem_pf if r["lang"] == lang and r["template"] == tpl])
        if cnt > 0:
            print(f"  {lang} / Template {tpl}: {cnt} failures")

# ── Phase 3: Fictional Brands ──
print(f"\n{'='*70}")
print("PHASE 3 — FICTIONAL BRAND POOL")
print("=" * 70)
json_path = os.path.join(RESULTS, "fictional_brands.json")
try:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Status: {data.get('status', '?')}")
    cands = data.get("candidates", {})
    for subcat, info in cands.items():
        if isinstance(info, dict):
            validated = info.get("validated", [])
            all_cands = info.get("all_candidates", [])
            print(f"\n  {subcat}:")
            print(f"    Total candidates generated: {len(all_cands)}")
            print(f"    Validated (all models say UNKNOWN): {len(validated)}")
            if validated:
                print(f"    Names: {', '.join(validated)}")
except Exception as e:
    print(f"Error reading fictional_brands.json: {e}")

print(f"\n{'='*70}")
print("SUMMARY & RECOMMENDATION")
print("=" * 70)
real_pct = choices.get("A", 0) / len(success) * 100 if success else 0
print(f"1. Real brand (A) chosen in {real_pct:.1f}% of successful parses → STRONG incumbent advantage")
print(f"2. Parse failures concentrated in Gemini Flash Chinese ({len(gem_pf)} / {len(pf)} total)")
print(f"3. Recommended template: B (persona required for Exp 1, fix Gemini parser separately)")
