#!/usr/bin/env python3
"""
Exp 1a — Reparse existing CSV responses (no API calls needed).

Reads exp1a_raw.csv, re-parses each response with improved logic:

Protocol Alpha (ranking task):
  1. Try standard ranking parser (7 strategies from exp1d)
  2. For partial responses: infer real_rank from mentions of "Product X is #1"
  3. If 9 products are ranked and real brand missing → infer rank 10

Protocol Beta (single recommendation):
  1. Try RECOMMEND: [产品X] pattern
  2. Try "推荐/recommend 产品X" in natural language
  3. Try first product letter mentioned prominently

Usage:
    python exp1a_reparse.py              # reparse and overwrite
    python exp1a_reparse.py --dry-run    # just report stats without writing
"""

import csv, re, sys
from pathlib import Path
from collections import Counter, defaultdict

CSV_PATH = Path("results/exp1a/exp1a_raw.csv")


def parse_ranking(response_text):
    """7-strategy ranking parser (same as exp1d_reparse.py)."""
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
    all_matches = re.findall(r'(?:^|\n)\s*(\d+)[.）)\]:]\s*(?:\*\*)?(?:产品\s*|Product\s*)?([A-J])\b', response_text, re.IGNORECASE)
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

    # Strategy 7: "Product X" mentions in order (need at least 10)
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


def infer_real_rank_from_partial(response_text, real_brand_id):
    """
    For truncated alpha responses: try to infer the real brand's rank.

    Strategies:
    1. Numbered list with at least 1 entry mentioning the real brand
    2. Explicit "Product X is #1" / "首推产品X" patterns
    3. First product mentioned in a recommendation context
    """
    real_id = real_brand_id.upper()

    # Strategy A: numbered list (even partial, >=1 entry)
    nums = re.findall(r'(?:^|\n)\s*(\d+)[.）)\]:\s]+\s*(?:\*\*)?(?:产品|Product)?\s*([A-J])\b', response_text, re.IGNORECASE)
    if nums:
        # Build partial ranking
        ranking = [m[1].upper() for m in nums]
        if real_id in ranking:
            return ranking.index(real_id) + 1
        # If we have a partial ranking of 9 items without the target, infer last
        if len(set(ranking)) >= 9 and real_id not in ranking:
            return 10

    # Strategy B: Explicit top-1 mentions
    top1_patterns = [
        # "1. **Product X" or "1. Product X"
        r'(?:^|\n)\s*1[.）)\]:]\s*(?:\*\*)?(?:产品|Product)?\s*' + real_id + r'\b',
        r'(?:首推|首选|最佳|第一|top\s*(?:choice|pick|1)|best|ranked?\s*(?:first|#?1))[^.。\n]{0,30}(?:产品|Product)?\s*' + real_id + r'\b',
        r'(?:产品|Product)\s*' + real_id + r'[^.。\n]{0,50}(?:排名第一|first|#1|top|最(?:优|佳|好))',
        # "Product X (BrandName)" as the clear winner - bold or section header
        r'\*\*(?:产品|Product)\s*' + real_id + r'\b[^*]*\*\*.*?(?:首|first|top|最|推荐|recommend)',
    ]
    for pat in top1_patterns:
        if re.search(pat, response_text, re.IGNORECASE):
            return 1

    # Strategy C: "Product X is ranked first" reversed word order
    rev_patterns = [
        r'(?:推荐|recommend|选择|choose|pick)\s*(?:产品|Product)?\s*' + real_id + r'\b',
    ]
    # Only use this if we see clear recommendation context AND the real brand
    for pat in rev_patterns:
        if re.search(pat, response_text, re.IGNORECASE):
            # Make sure it's a top recommendation, not "don't recommend"
            m = re.search(pat, response_text, re.IGNORECASE)
            # Check no negation before
            start = max(0, m.start() - 10)
            prefix = response_text[start:m.start()]
            if not re.search(r'不|否|don\'t|not|除了', prefix):
                return 1

    return None


def parse_beta_choice(response_text):
    """
    Parse single product recommendation from beta protocol response.

    Strategies:
    1. RECOMMEND: [产品X] format
    2. "推荐产品X" / "recommend Product X"
    3. First prominently mentioned product letter
    """
    # Strategy 1: RECOMMEND: [...] format
    m = re.search(r'RECOMMEND[:\s]*\[?\s*(?:产品|Product)?\s*([A-J])\b', response_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r'(?:推荐|首推|首选|recommend|suggest|choose|pick)[:\s]*(?:产品|Product)?\s*([A-J])\b', response_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r'(?:最佳|最好|最适合|best|top|ideal)[^.。\n]{0,30}(?:产品|Product)\s*([A-J])\b', response_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r'\*\*(?:产品|Product)\s*([A-J])\b', response_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r'(?:选择?|pick|chose?n?)\s*(?:产品|Product)?\s*([A-J])\b', response_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


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
    alpha_rows = [r for r in rows if r.get("protocol") == "alpha"]
    beta_rows = [r for r in rows if r.get("protocol") == "beta"]

    before_alpha_valid = sum(1 for r in alpha_rows if r.get("real_rank", "").replace(".", "").isdigit())
    before_alpha_pf = sum(1 for r in alpha_rows if r.get("is_parse_fail") == "True")
    before_beta_valid = sum(1 for r in beta_rows if r.get("parsed_choice", "").strip())
    before_beta_pf = sum(1 for r in beta_rows if r.get("is_parse_fail") == "True")

    print(f"\nBefore reparse:")
    print(f"  Alpha: valid={before_alpha_valid}, parse_fail={before_alpha_pf} (total={len(alpha_rows)})")
    print(f"  Beta:  valid={before_beta_valid}, parse_fail={before_beta_pf} (total={len(beta_rows)})")

    # Re-parse
    stats = Counter()

    for r in rows:
        response = r.get("response", "") or ""
        protocol = r.get("protocol", "")
        model = r.get("model", "")

        # Skip DRY RUN placeholders
        if response.startswith("[DRY RUN]"):
            stats["skip_dryrun"] += 1
            continue

        if not response.strip():
            stats["skip_empty"] += 1
            continue

        if protocol == "alpha":
            # Try full ranking parse
            ranking = parse_ranking(response)

            if isinstance(ranking, list):
                # Full ranking recovered
                real_brand_id = r.get("real_brand_id", "").upper()
                r["parsed_ranking"] = ",".join(ranking)
                r["is_parse_fail"] = "False"

                if real_brand_id in ranking:
                    r["real_rank"] = str(ranking.index(real_brand_id) + 1)
                    stats["alpha_full_ranking"] += 1
                elif len(set(ranking)) >= 9:
                    r["real_rank"] = str(min(len(set(ranking)) + 1, 10))
                    stats["alpha_inferred_rank"] += 1
                else:
                    # Partial ranking without target
                    r["real_rank"] = ""
                    stats["alpha_partial_no_target"] += 1
            else:
                # Try partial inference
                real_brand_id = r.get("real_brand_id", "").upper()
                inferred = infer_real_rank_from_partial(response, real_brand_id)

                if inferred is not None:
                    r["real_rank"] = str(inferred)
                    r["is_parse_fail"] = "False"
                    r["parsed_ranking"] = ""  # No full ranking available
                    stats["alpha_partial_inferred"] += 1
                else:
                    # Truly unrecoverable
                    r["is_parse_fail"] = "True"
                    stats["alpha_unrecoverable"] += 1

        elif protocol == "beta":
            # Try to parse single choice
            choice = parse_beta_choice(response)

            if choice:
                r["parsed_choice"] = choice
                r["is_parse_fail"] = "False"

                # Determine if choice matches real brand
                real_brand_id = r.get("real_brand_id", "").upper()
                # real_rank for beta: 1 if chose real brand, else leave as-is
                # Actually for beta, the key metric is parsed_choice matching real_brand_id
                stats["beta_recovered"] += 1
            else:
                # Check if already valid
                if r.get("is_parse_fail") != "True" and r.get("parsed_choice", "").strip():
                    stats["beta_already_valid"] += 1
                else:
                    r["is_parse_fail"] = "True"
                    stats["beta_unrecoverable"] += 1

    # Stats after
    after_alpha_valid = sum(1 for r in rows if r.get("protocol") == "alpha" and
                           r.get("real_rank", "").replace(".", "").replace("-", "").isdigit() and
                           r.get("is_parse_fail") != "True")
    after_alpha_pf = sum(1 for r in rows if r.get("protocol") == "alpha" and r.get("is_parse_fail") == "True")
    after_beta_valid = sum(1 for r in rows if r.get("protocol") == "beta" and
                          r.get("parsed_choice", "").strip() and r.get("is_parse_fail") != "True")
    after_beta_pf = sum(1 for r in rows if r.get("protocol") == "beta" and r.get("is_parse_fail") == "True")

    print(f"\nAfter reparse:")
    print(f"  Alpha: valid={after_alpha_valid}, parse_fail={after_alpha_pf}")
    print(f"  Beta:  valid={after_beta_valid}, parse_fail={after_beta_pf}")
    print(f"  Alpha improvement: +{after_alpha_valid - before_alpha_valid}")
    print(f"  Beta improvement:  +{after_beta_valid - before_beta_valid}")

    print(f"\nBreakdown:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    # Per-model stats
    print(f"\nPer-model valid rows (Alpha):")
    model_stats = defaultdict(lambda: {"valid": 0, "pf": 0, "total": 0})
    for r in rows:
        if r.get("protocol") != "alpha":
            continue
        m = r.get("model", "unknown")
        model_stats[m]["total"] += 1
        if r.get("is_parse_fail") != "True" and r.get("real_rank", "").replace(".", "").replace("-", "").isdigit():
            model_stats[m]["valid"] += 1
        elif r.get("is_parse_fail") == "True":
            model_stats[m]["pf"] += 1
    for m in sorted(model_stats):
        s = model_stats[m]
        pct = s["valid"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {m:<16}: {s['valid']:>4}/{s['total']:>4} valid ({pct:.1f}%), parse_fail={s['pf']}")

    print(f"\nPer-model valid rows (Beta):")
    model_stats_b = defaultdict(lambda: {"valid": 0, "pf": 0, "total": 0})
    for r in rows:
        if r.get("protocol") != "beta":
            continue
        m = r.get("model", "unknown")
        model_stats_b[m]["total"] += 1
        if r.get("is_parse_fail") != "True" and r.get("parsed_choice", "").strip():
            model_stats_b[m]["valid"] += 1
        elif r.get("is_parse_fail") == "True":
            model_stats_b[m]["pf"] += 1
    for m in sorted(model_stats_b):
        s = model_stats_b[m]
        pct = s["valid"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {m:<16}: {s['valid']:>4}/{s['total']:>4} valid ({pct:.1f}%), parse_fail={s['pf']}")

    if dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Write back
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Updated {CSV_PATH}")


if __name__ == "__main__":
    main()
