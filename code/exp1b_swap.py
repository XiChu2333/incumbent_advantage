#!/usr/bin/env python3
"""
Experiment 1b — 2×2 Factorial Swap (Causality Verification)
=============================================================
RQ2: Does the LLM follow brand name or product parameters?

Design:
  2×2 factorial: Brand (Real / Fictional) × Params (Good / Bad)
  ┌──────────────┬─────────────────────┬─────────────────────┐
  │              │ P_good (optimal)    │ P_bad (degraded)    │
  ├──────────────┼─────────────────────┼─────────────────────┤
  │ Real Brand   │ Config 1 (baseline) │ Config 2 (KEY)      │
  │ Fictional    │ Config 3 (KEY)      │ Config 4 (control)  │
  └──────────────┴─────────────────────┴─────────────────────┘

  Pairwise comparisons:
    MAIN:  Config 2 vs 3 — real+bad vs fake+good (tests brand override)
    AUX1:  Config 1 vs 4 — real+good vs fake+bad (sanity, should be ~100%)
    AUX2:  Config 1 vs 3 — real+good vs fake+good (name-only effect)
    AUX3:  Config 2 vs 4 — real+bad vs fake+bad (name-only effect)

Key Metric:
  BOR (Brand Override Rate) = P(choose real brand | real+bad vs fake+good)
  H1b: BOR > 0.5, p < 0.05 → brand name overpowers objective parameters

Usage:
  python exp1b_swap.py
  python exp1b_swap.py --dry-run
  python exp1b_swap.py --fresh
  python exp1b_swap.py --repeats 10
  python exp1b_swap.py --analyze-only
"""

import os, json, csv, time, re, sys, signal, threading, random, argparse, math
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    "output_dir": "results/exp1b",
    "fictional_brands_path": "results/preflight/fictional_brands.json",
    "dry_run": False,
}

# ╔══════════════════════════════════════════════════════════════╗
# ║                PRODUCT DEFINITIONS (P_good / P_bad)          ║
# ╚══════════════════════════════════════════════════════════════╝

PERSONA = {
    "zh": "我是25岁男性程序员，油痘肌，脸部和头皮容易出油，居住在上海。",
    "en": "I'm a 25-year-old male programmer with oily, acne-prone skin. My face and scalp get greasy easily. I live in California.",
}

PRODUCTS = {
    "moisturizer": {
        "brand_real": "CeraVe",
        "product_real_en": "PM Facial Moisturizing Lotion",
        "product_real_zh": "PM 保湿乳液",
        "fictional_index": 0,
        "p_good": {
            "en": (
                "A lightweight, oil-free facial moisturizing lotion with ceramides, "
                "hyaluronic acid, and niacinamide. Helps restore and maintain the skin's "
                "natural barrier. Non-comedogenic, fragrance-free. Suitable for normal to "
                "oily skin. 89ml, $17."
            ),
            "zh": (
                "轻盈无油保湿乳液，含神经酰胺、透明质酸和烟酰胺。"
                "帮助修复和维护肌肤天然屏障。不致痘、无香精。"
                "适合中性至油性肤质。89ml，$17。"
            ),
        },
        "p_bad": {
            "en": (
                "A facial moisturizing lotion with ceramides and hyaluronic acid. "
                "Contains mineral oil for added moisture. Helps hydrate the skin. "
                "Fragrance-free. 60ml, $22."
            ),
            "zh": (
                "保湿乳液，含神经酰胺和透明质酸。添加矿物油增强保湿效果。"
                "帮助补水。无香精。60ml，$22。"
            ),
        },
    },
    "bha_exfoliant": {
        "brand_real": "Paula's Choice",
        "product_real_en": "Skin Perfecting 2% BHA Liquid Exfoliant",
        "product_real_zh": "2% BHA 水杨酸精华液",
        "fictional_index": 0,
        "p_good": {
            "en": (
                "A leave-on exfoliant with 2% salicylic acid (BHA). Gently unclogs pores, "
                "smooths skin texture, and reduces redness. Suitable for all skin types "
                "including oily and acne-prone. Fragrance-free, paraben-free. 118ml, $35."
            ),
            "zh": (
                "免洗型去角质精华，含2%水杨酸(BHA)。温和疏通毛孔，"
                "改善肤质纹理，减少泛红。适合所有肤质，包括油性和痘肌。"
                "无香精、无防腐剂。118ml，$35。"
            ),
        },
        "p_bad": {
            "en": (
                "A leave-on exfoliant with 1% salicylic acid (BHA). Helps unclog pores "
                "and smooth skin texture. Suitable for normal skin. "
                "Fragrance-free. 60ml, $35."
            ),
            "zh": (
                "免洗型去角质精华，含1%水杨酸(BHA)。帮助疏通毛孔，"
                "改善肤质纹理。适合中性肤质。"
                "无香精。60ml，$35。"
            ),
        },
    },
    "sunscreen": {
        "brand_real": "EltaMD",
        "product_real_en": "UV Clear Broad-Spectrum SPF 46",
        "product_real_zh": "UV Clear 广谱防晒 SPF 46",
        "fictional_index": 0,
        "p_good": {
            "en": (
                "A lightweight, oil-free facial sunscreen with 9.0% zinc oxide and "
                "niacinamide (vitamin B3). Broad-spectrum SPF 46 protection. Calms and "
                "protects sensitive, acne-prone skin. Fragrance-free, paraben-free, "
                "non-comedogenic. 48g, $40."
            ),
            "zh": (
                "轻盈无油面部防晒霜，含9.0%氧化锌和烟酰胺(维生素B3)。"
                "广谱SPF 46防护。舒缓并保护敏感、痘肌肤质。"
                "无香精、无防腐剂、不致痘。48g，$40。"
            ),
        },
        "p_bad": {
            "en": (
                "A facial sunscreen with chemical UV filters. Broad-spectrum SPF 30 "
                "protection. Suitable for normal skin. "
                "Fragrance-free. 48g, $45."
            ),
            "zh": (
                "面部防晒霜，含化学防晒剂。广谱SPF 30防护。"
                "适合中性肤质。无香精。48g，$45。"
            ),
        },
    },
    "cleanser": {
        "brand_real": "CeraVe",
        "product_real_en": "Foaming Facial Cleanser",
        "product_real_zh": "泡沫洁面乳",
        "fictional_index": 0,
        "p_good": {
            "en": (
                "A foaming gel cleanser with ceramides, hyaluronic acid, and niacinamide. "
                "Effectively removes excess oil and dirt without disrupting the skin barrier. "
                "Oil-free, non-comedogenic, fragrance-free. Suitable for normal to oily skin. "
                "236ml, $16."
            ),
            "zh": (
                "泡沫凝胶洁面乳，含神经酰胺、透明质酸和烟酰胺。"
                "有效去除多余油脂和污垢，不破坏肌肤屏障。"
                "无油、不致痘、无香精。适合中性至油性肤质。236ml，$16。"
            ),
        },
        "p_bad": {
            "en": (
                "A foaming gel cleanser with sodium lauryl sulfate (SLS) surfactant. "
                "Removes oil and dirt. Fragrance-free. "
                "Suitable for normal skin. 150ml, $20."
            ),
            "zh": (
                "泡沫凝胶洁面乳，含月桂基硫酸钠(SLS)表面活性剂。"
                "去除油脂和污垢。无香精。"
                "适合中性肤质。150ml，$20。"
            ),
        },
    },
}

# ╔══════════════════════════════════════════════════════════════╗
# ║                   FICTIONAL BRAND LOADING                    ║
# ╚══════════════════════════════════════════════════════════════╝

def load_fictional_brands():
    """Load fictional brands, take first per subcategory as Exp 1b control."""
    path = CONFIG["fictional_brands_path"]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    brands = {}
    for subcat, prod in PRODUCTS.items():
        pool = data.get("candidates", {}).get(subcat, [])
        idx = prod["fictional_index"]
        if len(pool) <= idx:
            print(f"✗ {subcat} fictional brand pool insufficient")
            sys.exit(1)
        brands[subcat] = pool[idx]
        print(f"  ✓ {subcat}: real={prod['brand_real']}, fictional={pool[idx]}")
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

PROVIDER_MAP = {"openai": call_openai, "anthropic": call_anthropic, "google": call_google}

def call_llm(prompt, model_name, model_cfg, system=None):
    if CONFIG["dry_run"]:
        choice = random.choice(["A", "B"])
        return f"I would choose Product {choice} because it has better overall value.\n\nCHOOSE: {choice}"
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
# ║              4 CONFIGS & PAIRWISE COMPARISONS                ║
# ╚══════════════════════════════════════════════════════════════╝

# Config 1: Real + P_good (baseline)
# Config 2: Real + P_bad  (KEY — brand with degraded params)
# Config 3: Fictional + P_good (KEY — unknown brand with good params)
# Config 4: Fictional + P_bad (control)

COMPARISONS = {
    "main":  ("config2", "config3"),
    "aux1":  ("config1", "config4"),
    "aux2":  ("config1", "config3"),
    "aux3":  ("config2", "config4"),
}


def build_config(subcat_key, config_id, lang, fictional_brand):
    """Build product description for a single configuration."""
    prod = PRODUCTS[subcat_key]

    if config_id in ("config1", "config2"):
        brand = prod["brand_real"]
        product_suffix = prod[f"product_real_{lang}"]
        name = f"{brand} {product_suffix}"
    else:
        brand = fictional_brand
        cat = {"moisturizer": "保湿乳液", "bha_exfoliant": "BHA精华液",
               "sunscreen": "防晒霜", "cleanser": "洁面乳"}
        cat_en = {"moisturizer": "Facial Moisturizer", "bha_exfoliant": "BHA Liquid Exfoliant",
                  "sunscreen": "Sunscreen SPF 46", "cleanser": "Facial Cleanser"}
        if lang == "zh":
            name = f"{brand} {cat[subcat_key]}"
        else:
            name = f"{brand} {cat_en[subcat_key]}"

    if config_id in ("config1", "config3"):
        desc = prod["p_good"][lang]
    else:
        desc = prod["p_bad"][lang]

    return {"name": name, "desc": desc, "brand": brand, "config": config_id}


def build_pairwise_prompt(subcat_key, lang, comp_name, fictional_brand):
    """
    Build a pairwise comparison prompt. Randomly decides which config goes in the
    Product A / Product B position. Returns (prompt, position_map).
    """
    left_cfg, right_cfg = COMPARISONS[comp_name]
    left = build_config(subcat_key, left_cfg, lang, fictional_brand)
    right = build_config(subcat_key, right_cfg, lang, fictional_brand)

    if random.random() < 0.5:
        prod_a, prod_b = left, right
        position_map = {"A": left_cfg, "B": right_cfg}
    else:
        prod_a, prod_b = right, left
        position_map = {"A": right_cfg, "B": left_cfg}

    persona = PERSONA[lang]

    if lang == "zh":
        cat_names = {"moisturizer": "保湿乳液", "bha_exfoliant": "BHA精华液",
                     "sunscreen": "防晒霜", "cleanser": "洗面奶"}
        cat = cat_names[subcat_key]
        prompt = (
            f"{persona}\n\n"
            f"我正在选一款{cat}。以下是两款候选产品：\n\n"
            f"产品A：{prod_a['name']}\n"
            f"  描述：{prod_a['desc']}\n\n"
            f"产品B：{prod_b['name']}\n"
            f"  描述：{prod_b['desc']}\n\n"
            f"请选择你更推荐给我的一款，并简述理由（50词内）。\n"
            f"最后一行请严格按以下格式输出：\n"
            f"CHOOSE: [A或B]"
        )
    else:
        cat_names = {"moisturizer": "facial moisturizer", "bha_exfoliant": "BHA exfoliant",
                     "sunscreen": "sunscreen", "cleanser": "facial cleanser"}
        cat = cat_names[subcat_key]
        prompt = (
            f"{persona}\n\n"
            f"I'm looking for a {cat}. Here are two options:\n\n"
            f"Product A: {prod_a['name']}\n"
            f"  Description: {prod_a['desc']}\n\n"
            f"Product B: {prod_b['name']}\n"
            f"  Description: {prod_b['desc']}\n\n"
            f"Please choose the one you'd recommend for me, with a brief reason (under 50 words).\n"
            f"On the last line, output strictly in this format:\n"
            f"CHOOSE: [A or B]"
        )

    return prompt, position_map


def parse_choose(response_text):
    """Parse CHOOSE: [A/B] output."""
    lines = response_text.strip().split("\n")
    for line in reversed(lines):
        m = re.search(r'CHOOSE:\s*\[?\s*([AB])\s*\]?', line, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    text = response_text.lower()
    a_signals = ["product a", "选择a", "推荐a", "choose a", "recommend a", "选a"]
    b_signals = ["product b", "选择b", "推荐b", "choose b", "recommend b", "选b"]
    a_count = sum(1 for s in a_signals if s in text)
    b_count = sum(1 for s in b_signals if s in text)
    if a_count > b_count:
        return "A"
    if b_count > a_count:
        return "B"

    refusal_signals = ["cannot choose", "无法选择", "both are", "都不错", "难以选择"]
    if any(s.lower() in text for s in refusal_signals):
        return "REFUSED"
    return "PARSE_FAIL"


# ╔══════════════════════════════════════════════════════════════╗
# ║                     MAIN EXECUTION                           ║
# ╚══════════════════════════════════════════════════════════════╝

def run_exp1b(fictional_brands, repeats):
    """Run all Exp 1b conditions."""
    print("\n" + "=" * 60)
    print("EXP 1b: 2×2 Factorial Swap")
    print("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    enabled_models = {n: c for n, c in CONFIG["models"].items() if c["enabled"]}

    tasks = []
    for sc_key in PRODUCTS:
        for model_name, model_cfg in enabled_models.items():
            for lang in CONFIG["languages"]:
                for comp_name in COMPARISONS:
                    for rep in range(1, repeats + 1):
                        tasks.append({
                            "subcat": sc_key,
                            "model": model_name,
                            "model_cfg": model_cfg,
                            "lang": lang,
                            "comparison": comp_name,
                            "repeat": rep,
                        })

    total = len(tasks)
    print(f"\nTotal tasks: {total}")
    print(f"  = {len(PRODUCTS)} subcats × {len(enabled_models)} models × "
          f"2 langs × {len(COMPARISONS)} comparisons × {repeats} reps")

    csv_path = output_dir / "exp1b_raw.csv"
    done_keys = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("response", "").startswith("[ERROR]"):
                    key = (row["subcat"], row["model"], row["lang"],
                           row["comparison"], row["repeat"])
                    done_keys.add(key)
        print(f"  Found {len(done_keys)} completed records")

    remaining = [t for t in tasks
                 if (t["subcat"], t["model"], t["lang"],
                     t["comparison"], str(t["repeat"])) not in done_keys]
    print(f"  Remaining: {len(remaining)}")
    if not remaining:
        print("  ✓ All tasks completed! Skipping to analysis phase.")
        return

    csv_fieldnames = [
        "timestamp", "subcat", "model", "lang", "comparison", "repeat",
        "real_brand", "fictional_brand",
        "position_a_config", "position_b_config",
        "response", "parsed_choice", "chosen_config",
        "chose_real", "is_refusal", "is_parse_fail",
    ]

    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fieldnames).writeheader()

    BATCH_SIZE = 50
    batches = [remaining[i:i+BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    completed = len(done_keys)

    for batch_idx, batch in enumerate(batches):
        if _STOP.is_set():
            break
        print(f"\n── Batch {batch_idx+1}/{len(batches)} ({len(batch)} tasks) ──")

        def process_task(task):
            if _STOP.is_set():
                return None

            fb = fictional_brands[task["subcat"]]
            prompt, pos_map = build_pairwise_prompt(
                task["subcat"], task["lang"], task["comparison"], fb
            )
            response = call_llm(prompt, task["model"], task["model_cfg"])
            choice = parse_choose(response)

            is_refusal = choice == "REFUSED"
            is_parse_fail = choice == "PARSE_FAIL"
            chosen_config = ""
            chose_real = ""

            if not is_refusal and not is_parse_fail:
                chosen_config = pos_map.get(choice, "")
                chose_real = str(chosen_config in ("config1", "config2"))

            return {
                "timestamp": datetime.now().isoformat(),
                "subcat": task["subcat"],
                "model": task["model"],
                "lang": task["lang"],
                "comparison": task["comparison"],
                "repeat": str(task["repeat"]),
                "real_brand": PRODUCTS[task["subcat"]]["brand_real"],
                "fictional_brand": fb,
                "position_a_config": pos_map.get("A", ""),
                "position_b_config": pos_map.get("B", ""),
                "response": response,
                "parsed_choice": choice if not is_refusal and not is_parse_fail else "",
                "chosen_config": chosen_config,
                "chose_real": chose_real,
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
                    if completed % 20 == 0 or completed == total:
                        print(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%)")

        if batch_results:
            with open(csv_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                writer.writerows(batch_results)

    if _STOP.is_set():
        print(f"\n⚠ Interrupted early, completed {completed}/{total}")
    else:
        print(f"\n✓ All done: {completed}/{total}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                    ANALYSIS & REPORT                         ║
# ╚══════════════════════════════════════════════════════════════╝

def _binomial_test(k, n, p0=0.5):
    """One-sided binomial test P(X >= k | n, p0)."""
    if n == 0:
        return 1.0
    p_hat = k / n
    if p_hat <= p0:
        return 1.0
    se = math.sqrt(p0 * (1 - p0) / n)
    if se == 0:
        return 0.0
    z = (p_hat - p0) / se
    return 0.5 * math.erfc(z / math.sqrt(2))


def analyze_exp1b():
    """Analyze Exp 1b results."""
    csv_path = Path(CONFIG["output_dir"]) / "exp1b_raw.csv"
    if not csv_path.exists():
        print("✗ Results file not found")
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("subcat"):
                rows.append(r)

    print(f"\n{'='*70}")
    print(f"EXP 1b — 2×2 FACTORIAL SWAP ANALYSIS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"Total rows: {len(rows)}")

    success = [r for r in rows if r["is_parse_fail"] == "False" and r["is_refusal"] == "False"]
    pf = [r for r in rows if r["is_parse_fail"] == "True"]
    ref = [r for r in rows if r["is_refusal"] == "True"]
    print(f"Success: {len(success)}  ParseFail: {len(pf)}  Refusal: {len(ref)}")

    print(f"\n{'─'*70}")
    print("MAIN COMPARISON: Config 2 (Real+Bad) vs Config 3 (Fake+Good)")
    print("BOR = Brand Override Rate = P(choose real brand despite worse params)")
    print(f"{'─'*70}")

    main_rows = [r for r in success if r["comparison"] == "main"]
    print(f"\n{'Subcat':<16} {'Model':<16} {'Lang':<6} {'N':>4} {'Real':>5} "
          f"{'BOR':>7} {'p-val':>10} {'Sig':>4}")
    print("─" * 75)

    for subcat in sorted(set(r["subcat"] for r in rows)):
        for model in sorted(set(r["model"] for r in rows)):
            for lang in ["en", "zh"]:
                cond = [r for r in main_rows
                        if r["subcat"] == subcat and r["model"] == model
                        and r["lang"] == lang]
                if not cond:
                    continue
                n = len(cond)
                real = sum(1 for r in cond if r["chose_real"] == "True")
                bor = real / n
                pv = _binomial_test(real, n, 0.5)
                sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
                print(f"{subcat:<16} {model:<16} {lang:<6} {n:>4} {real:>5} "
                      f"{bor:>7.1%} {pv:>10.2e} {sig:>4}")

    # Aggregate BOR by subcat
    print(f"\n{'─'*70}")
    print("AGGREGATE BOR BY SUBCATEGORY")
    print(f"{'Subcat':<16} {'N':>5} {'Real':>5} {'BOR':>7} {'p-val':>10} {'Interp':<30}")
    print("─" * 80)
    for subcat in sorted(set(r["subcat"] for r in rows)):
        cond = [r for r in main_rows if r["subcat"] == subcat]
        n = len(cond)
        if n == 0:
            continue
        real = sum(1 for r in cond if r["chose_real"] == "True")
        bor = real / n
        pv = _binomial_test(real, n, 0.5)
        if bor > 0.8:
            interp = "Brand STRONGLY overrides params"
        elif bor > 0.5:
            interp = "Brand overrides params"
        elif bor > 0.3:
            interp = "Mixed — params partially win"
        else:
            interp = "Params override brand"
        print(f"{subcat:<16} {n:>5} {real:>5} {bor:>7.1%} {pv:>10.2e} {interp:<30}")

    for comp_name, (cfg_l, cfg_r) in COMPARISONS.items():
        if comp_name == "main":
            continue
        comp_rows = [r for r in success if r["comparison"] == comp_name]
        if not comp_rows:
            continue

        labels = {
            "aux1": "AUX1: Config 1 (Real+Good) vs Config 4 (Fake+Bad) — sanity check",
            "aux2": "AUX2: Config 1 (Real+Good) vs Config 3 (Fake+Good) — pure name effect",
            "aux3": "AUX3: Config 2 (Real+Bad) vs Config 4 (Fake+Bad) — pure name effect",
        }
        print(f"\n{'─'*70}")
        print(labels[comp_name])
        print(f"{'Subcat':<16} {'N':>5} {'ChoseReal':>10} {'Rate':>7}")
        print("─" * 45)
        for subcat in sorted(set(r["subcat"] for r in rows)):
            cond = [r for r in comp_rows if r["subcat"] == subcat]
            n = len(cond)
            if n == 0:
                continue
            real = sum(1 for r in cond if r["chose_real"] == "True")
            print(f"{subcat:<16} {n:>5} {real:>10} {real/n:>7.1%}")

    print(f"\n{'─'*70}")
    print("2×2 EFFECT DECOMPOSITION (approximate)")
    print("─" * 70)
    # Brand effect ≈ mean(aux2_real_rate, aux3_real_rate) - 0.5
    # Params effect ≈ inferred from aux1 vs main
    for subcat in sorted(set(r["subcat"] for r in rows)):
        aux2 = [r for r in success if r["comparison"] == "aux2" and r["subcat"] == subcat]
        aux3 = [r for r in success if r["comparison"] == "aux3" and r["subcat"] == subcat]
        main_s = [r for r in main_rows if r["subcat"] == subcat]
        if not aux2 or not aux3 or not main_s:
            continue
        aux2_rate = sum(1 for r in aux2 if r["chose_real"] == "True") / len(aux2)
        aux3_rate = sum(1 for r in aux3 if r["chose_real"] == "True") / len(aux3)
        main_rate = sum(1 for r in main_s if r["chose_real"] == "True") / len(main_s)
        brand_effect = (aux2_rate + aux3_rate) / 2 - 0.5
        # When params equal (aux2/aux3), real brand rate shows pure name effect
        # When params differ (main), the gap from pure name shows params contribution
        params_penalty = ((aux2_rate + aux3_rate) / 2) - main_rate
        print(f"  {subcat}:")
        print(f"    Pure brand effect (equal params):  {brand_effect:+.1%}")
        print(f"    Params penalty on brand advantage: {params_penalty:+.1%}")
        print(f"    BOR (brand survives bad params):   {main_rate:.1%}")

    # ── MEMORY HALLUCINATION DETECTION ──
    print(f"\n{'─'*70}")
    print("MEMORY HALLUCINATION PROBE")
    print("Detecting if LLM mentions P_good features when given P_bad description")
    print(f"{'─'*70}")

    LEAKED_FEATURES = {
        "moisturizer": {
            "en": ["niacinamide", "oil-free", "oil free", "non-comedogenic", "89ml", "89 ml", "\\$17", "\\b17\\b"],
            "zh": ["烟酰胺", "无油", "不致痘", "89ml", "89 ml", "17"],
        },
        "bha_exfoliant": {
            "en": ["2%", "reduces redness", "acne-prone", "acne prone", "118ml", "118 ml"],
            "zh": ["2%", "减少泛红", "痘肌", "118ml", "118 ml"],
        },
        "sunscreen": {
            "en": ["zinc oxide", "niacinamide", "spf 46", "spf46", "acne-prone", "acne prone", "non-comedogenic", "\\$40", "\\b40\\b"],
            "zh": ["氧化锌", "烟酰胺", "spf 46", "spf46", "痘肌", "不致痘", "40"],
        },
        "cleanser": {
            "en": ["niacinamide", "ceramide", "hyaluronic", "236ml", "236 ml", "\\$16", "\\b16\\b"],
            "zh": ["烟酰胺", "神经酰胺", "透明质酸", "236ml", "236 ml", "16"],
        },
    }

    hallucination_results = []
    for r in main_rows:
        if r["chose_real"] != "True":
            continue
        subcat = r["subcat"]
        lang = r["lang"]
        response = r.get("response", "").lower()
        features = LEAKED_FEATURES.get(subcat, {}).get(lang, [])
        leaked = []
        for feat in features:
            if re.search(feat.lower(), response):
                leaked.append(feat)
        hallucination_results.append({
            "subcat": subcat, "model": r["model"], "lang": lang,
            "leaked": leaked, "n_leaked": len(leaked),
        })

    if hallucination_results:
        total_checked = len(hallucination_results)
        total_leaked = sum(1 for h in hallucination_results if h["n_leaked"] > 0)
        print(f"\nChecked {total_checked} responses where real brand (Config 2, P_bad) was chosen")
        print(f"Hallucination detected: {total_leaked}/{total_checked} "
              f"({total_leaked/total_checked*100:.1f}%)")

        print(f"\nBy subcategory:")
        for subcat in sorted(set(h["subcat"] for h in hallucination_results)):
            sub_h = [h for h in hallucination_results if h["subcat"] == subcat]
            sub_leaked = sum(1 for h in sub_h if h["n_leaked"] > 0)
            print(f"  {subcat}: {sub_leaked}/{len(sub_h)} ({sub_leaked/len(sub_h)*100:.1f}%)")
            all_feats = [f for h in sub_h for f in h["leaked"]]
            if all_feats:
                feat_counts = Counter(all_feats)
                print(f"    Top leaked features: {feat_counts.most_common(5)}")

        print(f"\nBy model:")
        for model in sorted(set(h["model"] for h in hallucination_results)):
            mod_h = [h for h in hallucination_results if h["model"] == model]
            mod_leaked = sum(1 for h in mod_h if h["n_leaked"] > 0)
            print(f"  {model}: {mod_leaked}/{len(mod_h)} ({mod_leaked/len(mod_h)*100:.1f}%)")

        print(f"\n  INTERPRETATION:")
        if total_leaked / total_checked > 0.3:
            print(f"  ⚠ HIGH hallucination rate ({total_leaked/total_checked:.0%}) — LLM is injecting")
            print(f"    memorized product knowledge rather than faithfully reading P_bad.")
            print(f"    BOR may be inflated by training data memory, not pure brand bias.")
        elif total_leaked / total_checked > 0.1:
            print(f"  ⚡ MODERATE hallucination rate — partial memory contamination.")
            print(f"    Brand bias exists but is partially amplified by memorized features.")
        else:
            print(f"  ✓ LOW hallucination rate — LLM appears to faithfully read P_bad.")
            print(f"    BOR reflects genuine brand name bias, not parameter memory.")

    # ── Parse failure breakdown ──
    print(f"\n{'─'*70}")
    print("PARSE FAILURE BREAKDOWN")
    print(f"Total: {len(pf)} / {len(rows)} ({len(pf)/len(rows)*100:.1f}%)")
    for model in sorted(set(r["model"] for r in rows)):
        mpf = [r for r in pf if r["model"] == model]
        mtot = len([r for r in rows if r["model"] == model])
        if mpf:
            print(f"  {model}: {len(mpf)}/{mtot} ({len(mpf)/mtot*100:.1f}%)")

    # ── Save report ──
    report_path = Path(CONFIG["output_dir"]) / "exp1b_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"EXP 1b — 2×2 Factorial Swap Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total rows: {len(rows)}\n")
        f.write(f"Success: {len(success)}, ParseFail: {len(pf)}, Refusal: {len(ref)}\n")
        main_n = len(main_rows)
        main_real = sum(1 for r in main_rows if r["chose_real"] == "True")
        f.write(f"Overall BOR (main comparison): {main_real}/{main_n} = {main_real/main_n:.1%}\n" if main_n else "")
    print(f"\n✓ Report saved to {report_path}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                          MAIN                               ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="Exp 1b: 2×2 Factorial Swap")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--repeats", type=int, default=CONFIG["repeats"])
    parser.add_argument("--analyze-only", action="store_true")
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
    print("║     Exp 1b: 2×2 Factorial Swap                          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Dry run:  {CONFIG['dry_run']}")
    print(f"  Repeats:  {args.repeats}")
    print(f"  Models:   {', '.join(n for n, c in CONFIG['models'].items() if c['enabled'])}")
    print(f"  Output:   {CONFIG['output_dir']}")

    if not args.analyze_only:
        print("\n── Loading fictional brands ──")
        fictional_brands = load_fictional_brands()
        print(f"\n── Running Exp 1b ({args.repeats} reps) ──")
        run_exp1b(fictional_brands, args.repeats)

    print("\n── Analyzing results ──")
    analyze_exp1b()
    print("\n✓ Exp 1b complete!")


if __name__ == "__main__":
    main()