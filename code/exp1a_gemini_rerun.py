#!/usr/bin/env python3
"""
Exp 1a — Rerun Gemini alpha protocol only (RANKING-first prompt).

Problem: Original exp1a gave Gemini max_tokens=1024 with "last line RANKING" instruction.
Gemini wrote long analyses and got truncated before outputting the ranking.
Fix: Use RANKING-first prompt (output ranking on first line) + max_tokens=4096.

This script:
1. Reads exp1a_raw.csv
2. Identifies Gemini alpha parse_fail rows
3. Reruns ONLY those rows with the new prompt
4. Merges results back into exp1a_raw.csv

Usage:
    python exp1a_gemini_rerun.py              # rerun and merge
    python exp1a_gemini_rerun.py --dry-run    # test without API calls
"""

import os, json, csv, time, re, sys, signal, threading, random, argparse
from datetime import datetime
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Graceful exit ──
_STOP = threading.Event()
def _handle_sigint(sig, frame):
    print("\n\n⚠ Interrupt received, graceful exit…")
    _STOP.set()
signal.signal(signal.SIGINT, _handle_sigint)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, _handle_sigint)

# ╔══════════════════════════════════════════════════════════════╗
# ║                         CONFIG                              ║
# ╚══════════════════════════════════════════════════════════════╝

CONFIG = {
    "google_api_key": os.environ.get("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY"),
    "model_id": "gemini-3-flash-preview",
    "temperature": 0.7,
    "max_tokens": 4096,  # Increased from 1024
    "max_parallel_workers": 10,
    "retry_attempts": 3,
    "retry_delay_seconds": 5,
    "output_dir": "results/exp1a",
    "fictional_brands_path": "results/preflight/fictional_brands.json",
    "dry_run": False,
}

# ╔══════════════════════════════════════════════════════════════╗
# ║                    SUBCATEGORIES & PRODUCTS                  ║
# ╚══════════════════════════════════════════════════════════════╝

SUBCATEGORIES = {
    "moisturizer": {
        "zh": "保湿乳液", "en": "facial moisturizer",
        "brand": "CeraVe", "product_en": "PM Facial Moisturizing Lotion", "product_zh": "PM 保湿乳液",
        "desc_en": "A lightweight, oil-free facial moisturizing lotion with ceramides, hyaluronic acid, and niacinamide. Helps restore and maintain the skin's natural barrier. Non-comedogenic, fragrance-free. Suitable for normal to oily skin. 89ml, $17.",
        "desc_zh": "轻盈无油保湿乳液，含神经酰胺、透明质酸和烟酰胺。帮助修复和维护肌肤天然屏障。不致痘、无香精。适合中性至油性肤质。89ml，$17。",
    },
    "bha_exfoliant": {
        "zh": "BHA精华液", "en": "BHA liquid exfoliant",
        "brand": "Paula's Choice", "product_en": "Skin Perfecting 2% BHA Liquid Exfoliant", "product_zh": "2% BHA 水杨酸精华液",
        "desc_en": "A leave-on exfoliant with 2% salicylic acid (BHA). Gently unclogs pores, smooths skin texture, and reduces redness. Suitable for all skin types including oily and acne-prone. Fragrance-free, paraben-free. 118ml, $35.",
        "desc_zh": "免洗型去角质精华，含2%水杨酸(BHA)。温和疏通毛孔，改善肤质纹理，减少泛红。适合所有肤质，包括油性和痘肌。无香精、无防腐剂。118ml，$35。",
    },
    "sunscreen": {
        "zh": "防晒霜", "en": "facial sunscreen",
        "brand": "EltaMD", "product_en": "UV Clear Broad-Spectrum SPF 46", "product_zh": "UV Clear 广谱防晒 SPF 46",
        "desc_en": "A lightweight, oil-free facial sunscreen with 9.0% zinc oxide and niacinamide (vitamin B3). Broad-spectrum SPF 46 protection. Calms and protects sensitive, acne-prone skin. Fragrance-free, paraben-free, non-comedogenic. 48g, $40.",
        "desc_zh": "轻盈无油面部防晒霜，含9.0%氧化锌和烟酰胺(维生素B3)。广谱SPF 46防护。舒缓并保护敏感、痘肌肤质。无香精、无防腐剂、不致痘。48g，$40。",
    },
    "cleanser": {
        "zh": "洗面奶", "en": "facial cleanser",
        "brand": "CeraVe", "product_en": "Foaming Facial Cleanser", "product_zh": "泡沫洁面乳",
        "desc_en": "A foaming gel cleanser with ceramides, hyaluronic acid, and niacinamide. Effectively removes excess oil and dirt without disrupting the skin barrier. Oil-free, non-comedogenic, fragrance-free. Suitable for normal to oily skin. 236ml, $16.",
        "desc_zh": "泡沫凝胶洁面乳，含神经酰胺、透明质酸和烟酰胺。有效去除多余油脂和污垢，不破坏肌肤屏障。无油、不致痘、无香精。适合中性至油性肤质。236ml，$16。",
    },
}

PERSONA = {
    "zh": "我是25岁男性程序员，油痘肌，脸部和头皮容易出油，居住在上海。",
    "en": "I'm a 25-year-old male programmer with oily, acne-prone skin. My face and scalp get greasy easily. I live in California.",
}

# ╔══════════════════════════════════════════════════════════════╗
# ║                    API & PROMPT                              ║
# ╚══════════════════════════════════════════════════════════════╝

def call_gemini(prompt, system=None):
    """Call Gemini API with increased max_tokens."""
    import google.generativeai as genai
    genai.configure(api_key=CONFIG["google_api_key"])
    model = genai.GenerativeModel(
        CONFIG["model_id"],
        generation_config=genai.GenerationConfig(
            temperature=CONFIG["temperature"],
            max_output_tokens=CONFIG["max_tokens"],
        ),
        system_instruction=system,
    )
    resp = model.generate_content(prompt)
    return resp.text


def call_with_retry(prompt, system=None):
    """Call with retry logic."""
    if CONFIG["dry_run"]:
        return f"[DRY RUN] prompt_len={len(prompt)}"
    for attempt in range(CONFIG["retry_attempts"]):
        if _STOP.is_set():
            return "[INTERRUPTED]"
        try:
            return call_gemini(prompt, system)
        except Exception as e:
            print(f"  [WARN] attempt {attempt+1} failed: {e}")
            if attempt < CONFIG["retry_attempts"] - 1:
                time.sleep(CONFIG["retry_delay_seconds"] * (attempt + 1))
    return f"[ERROR] All {CONFIG['retry_attempts']} attempts failed"


def build_prompt_alpha_ranking_first(subcat_key, lang, products):
    """
    Protocol α with RANKING-FIRST instruction for Gemini.
    Gemini outputs ranking on the FIRST line to avoid truncation.
    """
    sc = SUBCATEGORIES[subcat_key]
    cat = sc["zh"] if lang == "zh" else sc["en"]
    persona = PERSONA[lang]

    # Format product list
    if lang == "zh":
        lines = [f"产品{p['id']}：{p['name']}\n  描述：{p['desc']}" for p in products]
    else:
        lines = [f"Product {p['id']}: {p['name']}\n  Description: {p['desc']}" for p in products]
    product_text = "\n\n".join(lines)

    if lang == "zh":
        prompt = (
            f"{persona}\n\n"
            f"以下是10款{cat}产品：\n\n{product_text}\n\n"
            f"请根据对我的适用性，将这10款产品从最推荐到最不推荐排序。\n"
            f"请严格按以下格式，在回答的第一行输出排名（用逗号分隔字母编号），然后再简要说明理由：\n"
            f"RANKING: [A, B, C, D, E, F, G, H, I, J]\n"
            f"（请确保包含全部10个产品的字母编号）"
        )
    else:
        prompt = (
            f"{persona}\n\n"
            f"Here are 10 {cat} products:\n\n{product_text}\n\n"
            f"Please rank all 10 products from most to least recommended for me.\n"
            f"Output strictly in this format on the FIRST line (comma-separated letters), then explain briefly:\n"
            f"RANKING: [A, B, C, D, E, F, G, H, I, J]\n"
            f"(Make sure to include all 10 product letters)"
        )
    return prompt


# ╔══════════════════════════════════════════════════════════════╗
# ║                       PARSING                               ║
# ╚══════════════════════════════════════════════════════════════╝

def parse_ranking(response_text):
    """7-strategy ranking parser."""
    lines = response_text.strip().split("\n")

    # Strategy 1: RANKING: [...] format (search forward for RANKING-first)
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

    # Strategy 3: numbered list
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


# ╔══════════════════════════════════════════════════════════════╗
# ║                      MAIN EXECUTION                         ║
# ╚══════════════════════════════════════════════════════════════╝

def load_fictional_brands():
    """Load fictional brand pool."""
    path = CONFIG["fictional_brands_path"]
    if not os.path.exists(path):
        print(f"✗ Fictional brand file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    brands = {}
    for subcat in SUBCATEGORIES:
        pool = data.get("candidates", {}).get(subcat, [])
        brands[subcat] = pool[:9]
    return brands


def build_product_pool(subcat_key, lang, fictional_brands):
    """Build product pool with random order."""
    sc = SUBCATEGORIES[subcat_key]
    desc = sc[f"desc_{lang}"]
    real_name = f"{sc['brand']} {sc[f'product_{lang}']}"
    fake_names = fictional_brands[subcat_key]

    products = [{"name": real_name, "desc": desc, "is_real": True}]
    for fname in fake_names:
        display = f"{fname} {sc['zh']}" if lang == "zh" else fname
        products.append({"name": display, "desc": desc, "is_real": False})

    random.shuffle(products)
    for i, p in enumerate(products):
        p["id"] = chr(65 + i)

    real_id = next(p["id"] for p in products if p["is_real"])
    return products, real_id


def main():
    parser = argparse.ArgumentParser(description="Exp 1a: Rerun Gemini alpha with RANKING-first")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run test")
    args = parser.parse_args()
    CONFIG["dry_run"] = args.dry_run

    print("╔════��═════════════════════════════════════════════════════╗")
    print("║  Exp 1a: Rerun Gemini Alpha (RANKING-first prompt)      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Dry run:    {CONFIG['dry_run']}")
    print(f"  Model:      {CONFIG['model_id']}")
    print(f"  Max tokens: {CONFIG['max_tokens']}")

    # Load CSV
    csv_path = Path(CONFIG["output_dir"]) / "exp1a_raw.csv"
    if not csv_path.exists():
        print(f"✗ File not found: {csv_path}")
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)

    # Find Gemini alpha parse_fail or DRY_RUN rows to rerun
    rerun_indices = []
    for i, r in enumerate(rows):
        if (r.get("model") == "gemini-flash" and
            r.get("protocol") == "alpha" and
            (r.get("is_parse_fail") == "True" or
             r.get("response", "").startswith("[DRY RUN]") or
             r.get("response", "").startswith("[ERROR]"))):
            rerun_indices.append(i)

    print(f"\n  Total rows: {len(rows)}")
    print(f"  Gemini alpha to rerun: {len(rerun_indices)}")

    if not rerun_indices:
        print("  ✓ Nothing to rerun!")
        return

    # Load fictional brands
    fictional_brands = load_fictional_brands()

    # Process reruns
    completed = 0
    stats = Counter()

    def process_one(idx):
        if _STOP.is_set():
            return None, idx
        r = rows[idx]
        subcat = r["subcat"]
        lang = r["lang"]

        # Build new product pool (new random order)
        products, real_id = build_product_pool(subcat, lang, fictional_brands)
        product_order = ",".join(p["id"] + "=" + ("REAL" if p["is_real"] else "FAKE") for p in products)

        # Build RANKING-first prompt
        prompt = build_prompt_alpha_ranking_first(subcat, lang, products)

        # Call API
        response = call_with_retry(prompt)

        # Parse
        ranking = parse_ranking(response)
        parsed_ranking = ""
        parsed_choice = ""
        real_rank = ""
        is_refusal = False
        is_parse_fail = False

        if isinstance(ranking, list):
            parsed_ranking = ",".join(ranking)
            parsed_choice = ranking[0]
            if real_id in ranking:
                real_rank = str(ranking.index(real_id) + 1)
            elif len(set(ranking)) >= 9:
                real_rank = str(min(len(set(ranking)) + 1, 10))
            else:
                real_rank = "not_found"
        elif ranking == "REFUSED":
            is_refusal = True
        else:
            is_parse_fail = True

        result = {
            "timestamp": datetime.now().isoformat(),
            "subcat": subcat,
            "model": "gemini-flash",
            "lang": lang,
            "protocol": "alpha",
            "repeat": r["repeat"],
            "product_order": product_order,
            "real_brand_id": real_id,
            "response": response,
            "parsed_choice": parsed_choice,
            "parsed_ranking": parsed_ranking,
            "real_rank": real_rank,
            "is_refusal": str(is_refusal),
            "is_parse_fail": str(is_parse_fail),
        }
        return result, idx

    # Execute in batches
    BATCH_SIZE = 30
    batches = [rerun_indices[i:i+BATCH_SIZE] for i in range(0, len(rerun_indices), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        if _STOP.is_set():
            break
        print(f"\n── Batch {batch_idx+1}/{len(batches)} ({len(batch)} tasks) ──")

        with ThreadPoolExecutor(max_workers=CONFIG["max_parallel_workers"]) as executor:
            futures = {executor.submit(process_one, idx): idx for idx in batch}
            for future in as_completed(futures):
                if _STOP.is_set():
                    break
                result, idx = future.result()
                if result:
                    rows[idx] = result
                    completed += 1

                    if result["is_parse_fail"] == "True":
                        stats["parse_fail"] += 1
                    elif result["is_refusal"] == "True":
                        stats["refusal"] += 1
                    elif result["real_rank"] and result["real_rank"].isdigit():
                        stats["valid"] += 1
                    else:
                        stats["not_found"] += 1

                    if completed % 10 == 0:
                        print(f"  Progress: {completed}/{len(rerun_indices)} "
                              f"(valid={stats['valid']}, pf={stats['parse_fail']})")

    # Report
    print(f"\n{'='*60}")
    print(f"RERUN RESULTS")
    print(f"{'='*60}")
    print(f"  Completed: {completed}/{len(rerun_indices)}")
    print(f"  Valid:      {stats['valid']}")
    print(f"  ParseFail:  {stats['parse_fail']}")
    print(f"  Refusal:    {stats['refusal']}")
    print(f"  NotFound:   {stats['not_found']}")
    if completed > 0:
        print(f"  Success rate: {stats['valid']/completed*100:.1f}%")

    # After rerun, also apply reparse logic to remaining parse_fails
    # (infer rank from partial rankings)
    secondary_recoveries = 0
    for i in rerun_indices:
        r = rows[i]
        if r.get("is_parse_fail") == "True" and r.get("response", ""):
            # Try one more time with the 7-strategy parser
            # (already done above, but let's also try inference)
            pass

    if CONFIG["dry_run"]:
        print("\n[DRY RUN] No files written.")
        return

    # Write back
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Updated {csv_path}")
    print(f"  Next: python exp1a_reparse.py  (to recover any remaining parse fails)")
    print(f"  Then: python exp1a_baseline.py --analyze-only")


if __name__ == "__main__":
    main()
