#!/usr/bin/env python3
"""
Exp 1d — Reparse existing CSV responses (no API calls needed).

Reads exp1d_raw.csv, re-parses each response with improved logic:
1. Apply the 7-strategy parser
2. If target is NOT in parsed ranking but ranking has 9 items → infer target = rank 10
3. If ranking has 8 items and target not found → infer target = rank 9 (next slot)
4. Update target_rank and is_parse_fail columns
5. Write back to CSV

Usage:
    python exp1d_reparse.py              # reparse and overwrite
    python exp1d_reparse.py --dry-run    # just report stats without writing
"""

import csv, re, sys, math
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

CSV_PATH = Path("results/exp1d/exp1d_raw.csv")

def parse_ranking(response_text):
    """Same parser as exp1d_decompose.py — 7 strategies."""
    lines = response_text.strip().split("\n")

    # Strategy 1: RANKING: [...] format
    for line in lines:
        m = re.search(r'RANKING:\s*\[?\s*([A-J](?:\s*[,，\s>→\-|/]\s*[A-J]){1,9})\s*\]?', line, re.IGNORECASE)
        if m:
            letters = re.findall(r'[A-J]', m.group(1).upper())
            if len(set(letters)) >= 5:
                return letters

    # Strategy 2: "Ranking:" variants
    for line in lines:
        m = re.search(r'(?:my\s+)?rank(?:ing)?[:\s]+\[?\s*([A-J](?:\s*[,，\s>→\-|/]\s*[A-J]){1,9})\s*\]?', line, re.IGNORECASE)
        if m:
            letters = re.findall(r'[A-J]', m.group(1).upper())
            if len(set(letters)) >= 5:
                return letters

    # Strategy 3: numbered list (1. A, 2. B, ...)
    all_matches = re.findall(r'(?:^|\n)\s*(\d+)[.）)\]:]\s*(?:产品\s*|Product\s*)?([A-J])\b', response_text, re.IGNORECASE)
    if len(all_matches) >= 5:
        return [m[1].upper() for m in all_matches]

    # Strategy 4: bullet/dash list
    bullet_matches = re.findall(r'(?:^|\n)\s*[-*•]\s*(?:产品\s*|Product\s*)?([A-J])\b', response_text, re.IGNORECASE)
    if len(set(bullet_matches)) >= 5:
        return [m.upper() for m in bullet_matches]

    # Strategy 5: ordinal
    ordinal_matches = re.findall(r'(?:^|\n)\s*(?:\d+(?:st|nd|rd|th)?|第\d+)[.:\s）)]*(?:名?[:\s]*)?(?:产品\s*|Product\s*)?([A-J])\b', response_text, re.IGNORECASE)
    if len(set(ordinal_matches)) >= 5:
        return [m.upper() for m in ordinal_matches]

    # Strategy 6: last few lines with letter sequence
    for line in reversed(lines[-5:]):
        letters_in_line = re.findall(r'\b([A-J])\b', line.upper())
        if len(set(letters_in_line)) >= 5 and len(letters_in_line) <= 12:
            seen = set()
            ordered = []
            for l in letters_in_line:
                if l not in seen:
                    seen.add(l)
                    ordered.append(l)
            if len(ordered) >= 5:
                return ordered

    # Strategy 7: "Product X" mentions in order
    product_mentions = re.findall(r'(?:产品|Product)\s*([A-J])\b', response_text, re.IGNORECASE)
    if len(product_mentions) >= 10:
        seen = set()
        ordered = []
        for l in product_mentions:
            lu = l.upper()
            if lu not in seen:
                seen.add(lu)
                ordered.append(lu)
        if len(ordered) >= 5:
            return ordered

    return "PARSE_FAIL"


def infer_target_rank(ranking, target_letter):
    """
    If target is in ranking, return its 1-based position.
    If target is NOT in ranking but ranking has 8+ items,
    infer target is at position len(ranking)+1 (capped at 10).
    """
    if target_letter in ranking:
        return ranking.index(target_letter) + 1

    # Target not found in parsed ranking
    n = len(set(ranking))
    if n >= 8:
        # Infer: if 9 products are ranked and target is missing, it's implicitly 10th
        # If 8 products are ranked and target is missing, it's implicitly 9th
        return min(n + 1, 10)

    return "not_found"


def main():
    dry_run = "--dry-run" in sys.argv

    if not CSV_PATH.exists():
        print(f"✗ File not found: {CSV_PATH}")
        return

    # Read all rows
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)

    print(f"Total rows: {len(rows)}")

    # Stats before
    before_valid = sum(1 for r in rows if r.get("target_rank", "").isdigit())
    before_pf = sum(1 for r in rows if r.get("is_parse_fail") == "True")
    before_nf = sum(1 for r in rows if r.get("target_rank") == "not_found")
    print(f"Before: valid={before_valid}, not_found={before_nf}, parse_fail={before_pf}")

    # Re-parse
    stats = Counter()
    for r in rows:
        response = r.get("response", "")
        target_letter = r.get("target_letter", "")

        if not response or not target_letter:
            stats["skip_empty"] += 1
            continue

        ranking = parse_ranking(response)

        if isinstance(ranking, str) and ranking == "PARSE_FAIL":
            r["is_parse_fail"] = "True"
            r["parsed_ranking"] = ""
            r["target_rank"] = ""
            stats["parse_fail"] += 1
            continue

        r["is_parse_fail"] = "False"
        r["parsed_ranking"] = ",".join(ranking)

        rank = infer_target_rank(ranking, target_letter)
        r["target_rank"] = str(rank)

        if isinstance(rank, int):
            if target_letter in ranking:
                stats["found_direct"] += 1
            else:
                stats["found_inferred"] += 1
        else:
            stats["not_found"] += 1

    # Stats after
    after_valid = sum(1 for r in rows if r.get("target_rank", "").isdigit())
    after_pf = sum(1 for r in rows if r.get("is_parse_fail") == "True")
    after_nf = sum(1 for r in rows if r.get("target_rank") == "not_found")

    print(f"\nAfter reparse:")
    print(f"  valid={after_valid}, not_found={after_nf}, parse_fail={after_pf}")
    print(f"  Improvement: {after_valid - before_valid} more valid rows")
    print(f"\nBreakdown:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    # Per-model stats
    print(f"\nPer-model valid rows:")
    model_stats = defaultdict(lambda: {"valid": 0, "nf": 0, "pf": 0, "total": 0})
    for r in rows:
        m = r.get("model", "unknown")
        model_stats[m]["total"] += 1
        if r.get("target_rank", "").isdigit():
            model_stats[m]["valid"] += 1
        elif r.get("is_parse_fail") == "True":
            model_stats[m]["pf"] += 1
        else:
            model_stats[m]["nf"] += 1
    for m in sorted(model_stats):
        s = model_stats[m]
        pct = s["valid"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {m:<16}: {s['valid']:>5}/{s['total']:>5} valid ({pct:.1f}%), "
              f"not_found={s['nf']}, parse_fail={s['pf']}")

    if dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Write back
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Updated {CSV_PATH}")
    print(f"  Now run: python exp1d_decompose.py --analyze-only")


if __name__ == "__main__":
    main()
