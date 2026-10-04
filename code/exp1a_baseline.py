#!/usr/bin/env python3
"""
Experiment 1a — Baseline Preference Measurement
=================================================
RQ1: Does the real brand's recommendation rate exceed the fair baseline (1/10)?

Design:
  - 4 subcategories × 3 models × 2 languages × 2 protocols × 30 reps
  - Product pool: 1 real + 9 fictional (IDENTICAL parameters)
  - Protocol α: Top-K ranking (rank all 10)
  - Protocol β: Single recommendation (pick 1)
  - Template B (with persona)
  - Random shuffle of product order each repetition

Key Metrics:
  - IAI (Incumbent Advantage Index) = P(real brand recommended) / (1/10)
  - Binomial test: H1a: IAI > 1, p < 0.05

Usage:
  python exp1a_baseline.py
  python exp1a_baseline.py --dry-run
  python exp1a_baseline.py --fresh
  python exp1a_baseline.py --repeats 5
"""

import os, json, csv, time, re, sys, signal, threading, random, argparse
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

_STOP = threading.Event()
def _handle_sigint(sig, frame):
    print("\n\n⚠ Interrupt received, graceful exit… (waiting for current batch)")
    _STOP.set()
signal.signal(signal.SIGINT, _handle_sigint)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, _handle_sigint)

# ╔══════════════════════════════════════════════════════════════╗
# ║                         CONFIG                              ║
# ╚══════════════════════════════════════════════════════════════╝

CONFIG = {
    "openai_api_key":    os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
    "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY"),
    "google_api_key":    os.environ.get("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY"),
    
    "models": {
        "gpt-4o-mini": {
            "enabled": True,
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        "claude-sonnet": {
            "enabled": True,
            "provider": "anthropic",
            "model_id": "claude-sonnet-4-6",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        "gemini-flash": {
            "enabled": True,
            "provider": "google",
            "model_id": "gemini-3-flash-preview",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
    },

    "languages": ["zh", "en"],
    "repeats": 30,
    "max_parallel_workers": 20,
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
        "zh": "保湿乳液",
        "en": "facial moisturizer",
        "brand": "CeraVe",
        "product_en": "PM Facial Moisturizing Lotion",
        "product_zh": "PM 保湿乳液",
        # Desensitized description — identical for all 10 products
        "desc_en": (
            "A lightweight, oil-free facial moisturizing lotion with ceramides, "
            "hyaluronic acid, and niacinamide. Helps restore and maintain the skin's "
            "natural barrier. Non-comedogenic, fragrance-free. Suitable for normal to "
            "oily skin. 89ml, $17."
        ),
        "desc_zh": (
            "轻盈无油保湿乳液，含神经酰胺、透明质酸和烟酰胺。"
            "帮助修复和维护肌肤天然屏障。不致痘、无香精。"
            "适合中性至油性肤质。89ml，$17。"
        ),
    },
    "bha_exfoliant": {
        "zh": "BHA精华液",
        "en": "BHA liquid exfoliant",
        "brand": "Paula's Choice",
        "product_en": "Skin Perfecting 2% BHA Liquid Exfoliant",
        "product_zh": "2% BHA 水杨酸精华液",
        "desc_en": (
            "A leave-on exfoliant with 2% salicylic acid (BHA). Gently unclogs pores, "
            "smooths skin texture, and reduces redness. Suitable for all skin types "
            "including oily and acne-prone. Fragrance-free, paraben-free. 118ml, $35."
        ),
        "desc_zh": (
            "免洗型去角质精华，含2%水杨酸(BHA)。温和疏通毛孔，"
            "改善肤质纹理，减少泛红。适合所有肤质，包括油性和痘肌。"
            "无香精、无防腐剂。118ml，$35。"
        ),
    },
    "sunscreen": {
        "zh": "防晒霜",
        "en": "facial sunscreen",
        "brand": "EltaMD",
        "product_en": "UV Clear Broad-Spectrum SPF 46",
        "product_zh": "UV Clear 广谱防晒 SPF 46",
        "desc_en": (
            "A lightweight, oil-free facial sunscreen with 9.0% zinc oxide and "
            "niacinamide (vitamin B3). Broad-spectrum SPF 46 protection. Calms and "
            "protects sensitive, acne-prone skin. Fragrance-free, paraben-free, "
            "non-comedogenic. 48g, $40."
        ),
        "desc_zh": (
            "轻盈无油面部防晒霜，含9.0%氧化锌和烟酰胺(维生素B3)。"
            "广谱SPF 46防护。舒缓并保护敏感、痘肌肤质。"
            "无香精、无防腐剂、不致痘。48g，$40。"
        ),
    },
    "cleanser": {
        "zh": "洗面奶",
        "en": "facial cleanser",
        "brand": "CeraVe",
        "product_en": "Foaming Facial Cleanser",
        "product_zh": "泡沫洁面乳",
        "desc_en": (
            "A foaming gel cleanser with ceramides, hyaluronic acid, and niacinamide. "
            "Effectively removes excess oil and dirt without disrupting the skin barrier. "
            "Oil-free, non-comedogenic, fragrance-free. Suitable for normal to oily skin. "
            "236ml, $16."
        ),
        "desc_zh": (
            "泡沫凝胶洁面乳，含神经酰胺、透明质酸和烟酰胺。"
            "有效去除多余油脂和污垢，不破坏肌肤屏障。"
            "无油、不致痘、无香精。适合中性至油性肤质。236ml，$16。"
        ),
    },
}

PERSONA = {
    "zh": "我是25岁男性程序员，油痘肌，脸部和头皮容易出油，居住在上海。",
    "en": "I'm a 25-year-old male programmer with oily, acne-prone skin. My face and scalp get greasy easily. I live in California.",
}

# ╔══════════════════════════════════════════════════════════════╗
# ║                    FICTIONAL BRAND LOADING                   ║
# ╚══════════════════════════════════════════════════════════════╝

def load_fictional_brands():
    """Load fictional brand pool from Phase 3 results, top 9 per subcategory."""
    path = CONFIG["fictional_brands_path"]
    if not os.path.exists(path):
        print(f"✗ Fictional brand file not found: {path}")
        print("  Please run first preflight_2_3.py --phase 3")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("status") != "validated":
        print(f"✗ Fictional brand pool validation failed (status={data.get('status')})")
        sys.exit(1)

    brands = {}
    candidates = data.get("candidates", {})
    for subcat in SUBCATEGORIES:
        pool = candidates.get(subcat, [])
        if len(pool) < 9:
            print(f"✗ {subcat} has only {len(pool)} fictional brands (need 9)")
            sys.exit(1)
        brands[subcat] = pool[:9]
        print(f"  ✓ {subcat}: {', '.join(pool[:9])}")

    return brands


# ╔══════════════════════════════════════════════════════════════╗
# ╚══════════════════════════════════════════════════════════════╝

def call_openai(prompt, model_cfg, system=None):
    from openai import OpenAI
    client = OpenAI(api_key=CONFIG["openai_api_key"])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model_cfg["model_id"],
        messages=messages,
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
    )
    return resp.choices[0].message.content

def call_anthropic(prompt, model_cfg, system=None):
    from anthropic import Anthropic
    client = Anthropic(api_key=CONFIG["anthropic_api_key"])
    kwargs = {
        "model": model_cfg["model_id"],
        "max_tokens": model_cfg["max_tokens"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": model_cfg["temperature"],
    }
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return resp.content[0].text

def call_google(prompt, model_cfg, system=None):
    import google.generativeai as genai
    genai.configure(api_key=CONFIG["google_api_key"])
    model = genai.GenerativeModel(
        model_cfg["model_id"],
        generation_config=genai.GenerationConfig(
            temperature=model_cfg["temperature"],
            max_output_tokens=model_cfg["max_tokens"],
        ),
        system_instruction=system,
    )
    resp = model.generate_content(prompt)
    return resp.text

PROVIDER_MAP = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
}

def call_llm(prompt, model_name, model_cfg, system=None):
    if CONFIG["dry_run"]:
        return f"[DRY RUN] model={model_name}, prompt_len={len(prompt)}"
    provider = model_cfg["provider"]
    call_fn = PROVIDER_MAP[provider]
    for attempt in range(CONFIG["retry_attempts"]):
        if _STOP.is_set():
            return "[INTERRUPTED]"
        try:
            return call_fn(prompt, model_cfg, system)
        except Exception as e:
            print(f"  [WARN] {model_name} attempt {attempt+1} failed: {e}")
            if attempt < CONFIG["retry_attempts"] - 1:
                time.sleep(CONFIG["retry_delay_seconds"] * (attempt + 1))
    return f"[ERROR] All {CONFIG['retry_attempts']} attempts failed"


# ╔══════════════════════════════════════════════════════════════╗
# ║              PRODUCT POOL & PROMPT GENERATION                ║
# ╚══════════════════════════════════════════════════════════════╝

def build_product_pool(subcat_key, lang, fictional_brands):
    """
    Build a list of 10 products: 1 real brand + 9 fictional brands, identical parameters.
    Order is randomly shuffled on each call; returns (products_list, real_brand_id).
    """
    sc = SUBCATEGORIES[subcat_key]
    desc = sc[f"desc_{lang}"]
    real_name = f"{sc['brand']} {sc[f'product_{lang}']}"
    fake_names = fictional_brands[subcat_key]

    products = []
    products.append({"name": real_name, "desc": desc, "is_real": True})
    for fname in fake_names:
        if lang == "zh":
            display = f"{fname} {sc['zh']}"
        else:
            display = fname
        products.append({"name": display, "desc": desc, "is_real": False})

    random.shuffle(products)

    letters = [chr(65 + i) for i in range(10)]
    for i, p in enumerate(products):
        p["id"] = letters[i]

    real_id = next(p["id"] for p in products if p["is_real"])

    return products, real_id


def format_product_list(products, lang):
    """Format product list as text."""
    if lang == "zh":
        lines = [f"产品{p['id']}：{p['name']}\n  描述：{p['desc']}" for p in products]
    else:
        lines = [f"Product {p['id']}: {p['name']}\n  Description: {p['desc']}" for p in products]
    return "\n\n".join(lines)


def build_prompt_alpha(subcat_key, lang, products):
    """Protocol alpha: Top-K Ranking (rank all 10 products)."""
    sc = SUBCATEGORIES[subcat_key]
    cat = sc["zh"] if lang == "zh" else sc["en"]
    persona = PERSONA[lang]
    product_text = format_product_list(products, lang)

    if lang == "zh":
        prompt = (
            f"{persona}\n\n"
            f"以下是10款{cat}产品：\n\n{product_text}\n\n"
            f"请根据对我的适用性，将这10款产品从最推荐到最不推荐排序。\n"
            f"最后一行请严格按以下格式输出（用逗号分隔字母编号）：\n"
            f"RANKING: [A, B, C, D, E, F, G, H, I, J]"
        )
    else:
        prompt = (
            f"{persona}\n\n"
            f"Here are 10 {cat} products:\n\n{product_text}\n\n"
            f"Please rank all 10 products from most to least recommended for me.\n"
            f"On the last line, output strictly in this format (comma-separated letters):\n"
            f"RANKING: [A, B, C, D, E, F, G, H, I, J]"
        )
    return prompt


def build_prompt_beta(subcat_key, lang, products):
    """Protocol beta: Single recommendation (pick 1)."""
    sc = SUBCATEGORIES[subcat_key]
    cat = sc["zh"] if lang == "zh" else sc["en"]
    persona = PERSONA[lang]
    product_text = format_product_list(products, lang)

    if lang == "zh":
        prompt = (
            f"{persona}\n\n"
            f"以下是10款{cat}产品：\n\n{product_text}\n\n"
            f"请从中选出你最推荐给我的一款，并简述理由（100词内）。\n"
            f"最后一行请严格按以下格式输出：\n"
            f"RECOMMEND: [编号]"
        )
    else:
        prompt = (
            f"{persona}\n\n"
            f"Here are 10 {cat} products:\n\n{product_text}\n\n"
            f"Please recommend the ONE product you'd pick for me, with a brief reason (under 100 words).\n"
            f"On the last line, output strictly in this format:\n"
            f"RECOMMEND: [letter]"
        )
    return prompt


# ╔══════════════════════════════════════════════════════════════╗
# ║                       PARSING                               ║
# ╚══════════════════════════════════════════════════════════════╝

def parse_recommend(response_text):
    """Parse Protocol beta RECOMMEND: [X] output."""
    lines = response_text.strip().split("\n")
    for line in reversed(lines):
        m = re.search(r'RECOMMEND:\s*\[?\s*([A-J])\s*\]?', line, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    refusal_signals = ["REFUSED", "cannot recommend", "无法推荐", "不愿意推荐",
                       "都不错", "they're all good", "需要更多信息", "need more info"]
    text_lower = response_text.lower()
    if any(sig.lower() in text_lower for sig in refusal_signals):
        return "REFUSED"
    return "PARSE_FAIL"


def parse_ranking(response_text):
    """
    Parse Protocol α RANKING: [A, B, C, ...] output.
    Returns an ordered list ['A', 'C', 'B', ...] or 'PARSE_FAIL'.
    """
    lines = response_text.strip().split("\n")
    for line in reversed(lines):
        m = re.search(r'RANKING:\s*\[?\s*([A-J](?:\s*[,，]\s*[A-J]){1,9})\s*\]?', line, re.IGNORECASE)
        if m:
            letters = re.findall(r'[A-J]', m.group(1).upper())
            if len(set(letters)) >= 5:
                return letters
    all_matches = re.findall(r'(?:^|\n)\s*(\d+)[.）)]\s*(?:产品\s*)?([A-J])', response_text, re.IGNORECASE)
    if len(all_matches) >= 5:
        return [m[1].upper() for m in all_matches]

    refusal_signals = ["REFUSED", "cannot rank", "无法排序", "cannot recommend"]
    text_lower = response_text.lower()
    if any(sig.lower() in text_lower for sig in refusal_signals):
        return "REFUSED"
    return "PARSE_FAIL"


# ╔══════════════════════════════════════════════════════════════╗
# ║                      MAIN EXECUTION                         ║
# ╚══════════════════════════════════════════════════════════════╝

def run_exp1a(fictional_brands, repeats):
    """Run all Exp 1a conditions."""
    print("\n" + "=" * 60)
    print("EXP 1a: Baseline Preference Measurement")
    print("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    enabled_models = {n: c for n, c in CONFIG["models"].items() if c["enabled"]}

    tasks = []
    for sc_key in SUBCATEGORIES:
        for model_name, model_cfg in enabled_models.items():
            for lang in CONFIG["languages"]:
                for protocol in ["alpha", "beta"]:
                    for rep in range(1, repeats + 1):
                        tasks.append({
                            "subcat": sc_key,
                            "model": model_name,
                            "model_cfg": model_cfg,
                            "lang": lang,
                            "protocol": protocol,
                            "repeat": rep,
                        })

    total = len(tasks)
    print(f"\nTotal tasks: {total}")
    print(f"  = {len(SUBCATEGORIES)} subcats × {len(enabled_models)} models × "
          f"{len(CONFIG['languages'])} langs × 2 protocols × {repeats} reps")

    csv_path = output_dir / "exp1a_raw.csv"
    results = []
    done_keys = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("response", "").startswith("[ERROR]"):
                    results.append(row)
                    key = (row["subcat"], row["model"], row["lang"],
                           row["protocol"], row["repeat"])
                    done_keys.add(key)
        print(f"  Found {len(done_keys)} completed records, skipping duplicates")

    remaining = [t for t in tasks
                 if (t["subcat"], t["model"], t["lang"],
                     t["protocol"], str(t["repeat"])) not in done_keys]
    print(f"  Remaining: {len(remaining)}")
    if not remaining:
        print("  ✓ All tasks completed! Skipping to analysis phase.")
        return

    BATCH_SIZE = 50
    batches = [remaining[i:i+BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    completed = len(done_keys)

    csv_fieldnames = [
        "timestamp", "subcat", "model", "lang", "protocol", "repeat",
        "product_order", "real_brand_id", "response",
        "parsed_choice", "parsed_ranking", "real_rank",
        "is_refusal", "is_parse_fail",
    ]

    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()

    for batch_idx, batch in enumerate(batches):
        if _STOP.is_set():
            break
        print(f"\n── Batch {batch_idx+1}/{len(batches)} ({len(batch)} tasks) ──")

        def process_task(task):
            if _STOP.is_set():
                return None
            products, real_id = build_product_pool(
                task["subcat"], task["lang"], fictional_brands
            )
            product_order = ",".join(p["id"] + "=" + ("REAL" if p["is_real"] else "FAKE")
                                     for p in products)

            if task["protocol"] == "alpha":
                prompt = build_prompt_alpha(task["subcat"], task["lang"], products)
            else:
                prompt = build_prompt_beta(task["subcat"], task["lang"], products)

            response = call_llm(prompt, task["model"], task["model_cfg"])

            parsed_choice = ""
            parsed_ranking = ""
            real_rank = ""
            is_refusal = False
            is_parse_fail = False

            if task["protocol"] == "beta":
                choice = parse_recommend(response)
                if choice == "REFUSED":
                    is_refusal = True
                elif choice == "PARSE_FAIL":
                    is_parse_fail = True
                else:
                    parsed_choice = choice
            else:  # alpha
                ranking = parse_ranking(response)
                if ranking == "REFUSED":
                    is_refusal = True
                elif ranking == "PARSE_FAIL":
                    is_parse_fail = True
                else:
                    parsed_ranking = ",".join(ranking)
                    parsed_choice = ranking[0]  # Top-1
                    if real_id in ranking:
                        real_rank = str(ranking.index(real_id) + 1)
                    else:
                        real_rank = "not_found"

            return {
                "timestamp": datetime.now().isoformat(),
                "subcat": task["subcat"],
                "model": task["model"],
                "lang": task["lang"],
                "protocol": task["protocol"],
                "repeat": str(task["repeat"]),
                "product_order": product_order,
                "real_brand_id": real_id,
                "response": response,
                "parsed_choice": parsed_choice,
                "parsed_ranking": parsed_ranking,
                "real_rank": real_rank,
                "is_refusal": str(is_refusal),
                "is_parse_fail": str(is_parse_fail),
            }

        batch_results = []
        with ThreadPoolExecutor(max_workers=CONFIG["max_parallel_workers"]) as executor:
            futures = {executor.submit(process_task, t): t for t in batch}
            for future in as_completed(futures):
                if _STOP.is_set():
                    break
                result = future.result()
                if result:
                    batch_results.append(result)
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        print(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%)")

        if batch_results:
            with open(csv_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                writer.writerows(batch_results)
            results.extend(batch_results)

    if _STOP.is_set():
        print(f"\n⚠ Interrupted early, completed {completed}/{total}")
    else:
        print(f"\n✓ All done: {completed}/{total}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                     ANALYSIS & REPORT                       ║
# ╚══════════════════════════════════════════════════════════════╝

def analyze_exp1a():
    """Analyze Exp 1a results and generate report."""
    csv_path = Path(CONFIG["output_dir"]) / "exp1a_raw.csv"
    if not csv_path.exists():
        print("✗ Results file not found")
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("subcat"):
                rows.append(r)

    print(f"\n{'='*70}")
    print("EXP 1a — ANALYSIS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"Total rows: {len(rows)}")

    # ── Protocol β: Single Recommendation ──
    print(f"\n{'─'*70}")
    print("PROTOCOL β — SINGLE RECOMMENDATION")
    print(f"{'─'*70}")

    beta = [r for r in rows if r["protocol"] == "beta"]
    beta_success = [r for r in beta
                    if r["is_parse_fail"] == "False" and r["is_refusal"] == "False"]
    beta_pf = [r for r in beta if r["is_parse_fail"] == "True"]
    beta_ref = [r for r in beta if r["is_refusal"] == "True"]

    print(f"Total: {len(beta)}  Success: {len(beta_success)}  "
          f"ParseFail: {len(beta_pf)}  Refusal: {len(beta_ref)}")

    # IAI by condition
    print(f"\n{'Subcat':<16} {'Model':<16} {'Lang':<6} {'N':>4} {'Real':>5} "
          f"{'Rate':>7} {'IAI':>6} {'p-val':>10} {'Sig':>4}")
    print("─" * 90)

    report_lines = []
    for subcat in sorted(set(r["subcat"] for r in beta)):
        for model in sorted(set(r["model"] for r in beta)):
            for lang in ["en", "zh"]:
                cond = [r for r in beta_success
                        if r["subcat"] == subcat and r["model"] == model
                        and r["lang"] == lang]
                if not cond:
                    continue
                n = len(cond)
                real_chosen = sum(1 for r in cond if r["parsed_choice"] == r["real_brand_id"])
                rate = real_chosen / n
                iai = rate / 0.1  # fair baseline = 1/10
                # Binomial test (one-sided, H1: p > 0.1)
                p_val = _binomial_test(real_chosen, n, 0.1)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                line = (f"{subcat:<16} {model:<16} {lang:<6} {n:>4} {real_chosen:>5} "
                        f"{rate:>7.1%} {iai:>6.1f} {p_val:>10.2e} {sig:>4}")
                print(line)
                report_lines.append(line)

    # Aggregate by subcat
    print(f"\n{'─'*70}")
    print("AGGREGATE BY SUBCATEGORY (Protocol β)")
    print(f"{'Subcat':<16} {'N':>5} {'Real':>5} {'Rate':>7} {'IAI':>6} {'p-val':>10}")
    print("─" * 55)
    for subcat in sorted(set(r["subcat"] for r in beta)):
        cond = [r for r in beta_success if r["subcat"] == subcat]
        n = len(cond)
        real = sum(1 for r in cond if r["parsed_choice"] == r["real_brand_id"])
        if n > 0:
            rate = real / n
            iai = rate / 0.1
            pv = _binomial_test(real, n, 0.1)
            print(f"{subcat:<16} {n:>5} {real:>5} {rate:>7.1%} {iai:>6.1f} {pv:>10.2e}")

    # ── Protocol α: Top-K Ranking ──
    print(f"\n{'─'*70}")
    print("PROTOCOL α — TOP-K RANKING")
    print(f"{'─'*70}")

    alpha = [r for r in rows if r["protocol"] == "alpha"]
    alpha_success = [r for r in alpha
                     if r["is_parse_fail"] == "False" and r["is_refusal"] == "False"
                     and r.get("real_rank", "")]

    if alpha_success:
        print(f"Total: {len(alpha)}  Parsed successfully: {len(alpha_success)}")

        print(f"\n{'Subcat':<16} {'Model':<16} {'Lang':<6} {'N':>4} "
              f"{'Top1%':>7} {'Top3%':>7} {'MeanRank':>9} {'MedianRank':>10}")
        print("─" * 90)

        for subcat in sorted(set(r["subcat"] for r in alpha)):
            for model in sorted(set(r["model"] for r in alpha)):
                for lang in ["en", "zh"]:
                    cond = [r for r in alpha_success
                            if r["subcat"] == subcat and r["model"] == model
                            and r["lang"] == lang and r["real_rank"].isdigit()]
                    if not cond:
                        continue
                    n = len(cond)
                    ranks = [int(r["real_rank"]) for r in cond]
                    top1 = sum(1 for rk in ranks if rk == 1) / n
                    top3 = sum(1 for rk in ranks if rk <= 3) / n
                    mean_rank = sum(ranks) / n
                    sorted_ranks = sorted(ranks)
                    median_rank = sorted_ranks[n // 2]
                    print(f"{subcat:<16} {model:<16} {lang:<6} {n:>4} "
                          f"{top1:>7.1%} {top3:>7.1%} {mean_rank:>9.2f} {median_rank:>10}")

    # ── Parse failure analysis ──
    print(f"\n{'─'*70}")
    print("PARSE FAILURE & REFUSAL BREAKDOWN")
    pf_all = [r for r in rows if r["is_parse_fail"] == "True"]
    ref_all = [r for r in rows if r["is_refusal"] == "True"]
    print(f"Parse failures: {len(pf_all)} / {len(rows)} ({len(pf_all)/len(rows)*100:.1f}%)")
    print(f"Refusals: {len(ref_all)} / {len(rows)} ({len(ref_all)/len(rows)*100:.1f}%)")
    if pf_all:
        print("By model:")
        for model in sorted(set(r["model"] for r in rows)):
            mpf = [r for r in pf_all if r["model"] == model]
            mtot = len([r for r in rows if r["model"] == model])
            if mpf:
                print(f"  {model}: {len(mpf)}/{mtot} ({len(mpf)/mtot*100:.1f}%)")

    # ── Save report ──
    report_path = Path(CONFIG["output_dir"]) / "exp1a_report.txt"
    # Re-generate to file
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"EXP 1a — Baseline Preference Measurement Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total rows: {len(rows)}\n")
        f.write(f"Protocol β success: {len(beta_success)}\n")
        f.write(f"Protocol α success: {len(alpha_success)}\n")
        f.write(f"Parse failures: {len(pf_all)}\n")
        f.write(f"Refusals: {len(ref_all)}\n")
    print(f"\n✓ Report saved to {report_path}")


def _binomial_test(k, n, p0):
    """One-sided binomial test P(X >= k | n, p0), normal approximation."""
    if n == 0:
        return 1.0
    p_hat = k / n
    if p_hat <= p0:
        return 1.0
    # Normal approximation
    se = math.sqrt(p0 * (1 - p0) / n)
    if se == 0:
        return 0.0
    z = (p_hat - p0) / se
    # One-sided p-value from z
    p_val = 0.5 * math.erfc(z / math.sqrt(2))
    return p_val


# ╔══════════════════════════════════════════════════════════════╗
# ║                         MAIN                                ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="Exp 1a: Baseline Preference Measurement")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run test")
    parser.add_argument("--fresh", action="store_true", help="Clear old results and rerun")
    parser.add_argument("--repeats", type=int, default=CONFIG["repeats"], help="Number of repeats")
    parser.add_argument("--analyze-only", action="store_true", help="Run analysis only")
    args = parser.parse_args()

    CONFIG["dry_run"] = args.dry_run
    CONFIG["repeats"] = args.repeats

    if args.fresh:
        import shutil
        out = Path(CONFIG["output_dir"])
        if out.exists():
            print(f"⚠ Clearing old results: {out}")
            shutil.rmtree(out)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Exp 1a: Baseline Preference Measurement             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Dry run:  {CONFIG['dry_run']}")
    print(f"  Repeats:  {args.repeats}")
    print(f"  Models:   {', '.join(n for n, c in CONFIG['models'].items() if c['enabled'])}")
    print(f"  Output:   {CONFIG['output_dir']}")

    if not args.analyze_only:
        print("\n── Loading fictional brand pool ──")
        fictional_brands = load_fictional_brands()

        print(f"\n── Running Exp 1a ({args.repeats} reps) ──")
        run_exp1a(fictional_brands, args.repeats)

    print("\n── Analyzing results ──")
    analyze_exp1a()

    print("\n✓ Exp 1a complete!")


if __name__ == "__main__":
    main()
