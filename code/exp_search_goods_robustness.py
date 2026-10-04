#!/usr/bin/env python3
"""
Search Goods Robustness Check — Boundary Condition for Conditional Monopoly
===========================================================================
Tests whether Exp 1 findings (Conditional Monopoly, step-function transition,
variance decomposition) replicate on search goods where quality is objectively
comparable, as opposed to experience goods (skincare).

Design mirrors Exp 1a/1c/1d exactly, using the same:
  - 3 models × 2 languages × 20 reps
  - Protocol β (single recommendation) for S1 (baseline IAI)
  - Pairwise CHOOSE protocol for S2 (gradient) and S2' (decomposition)
  - Same gradient dimensions (rating, price, reviews) — ingredient excluded
    because search goods don't have ambiguous ingredients
  - Same persona (tech-savvy user buying for personal use)

Search goods categories:
  1. USB-C charging cable (incumbent: Anker)
  2. AA batteries (incumbent: Duracell)

Total calls: ~1,920
  S1: 2 subcats × 3 models × 2 langs × 1 protocol × 20 reps = 240 (baseline)
  S2: 2 subcats × 3 models × 2 langs × 3 dims × 5 levels × 20 reps = 3,600
      (but pairwise, same structure as exp1c)
  S2': uses S2 data (no extra calls)

Usage:
  python exp_search_goods_robustness.py                # run all
  python exp_search_goods_robustness.py --dry-run      # test without API
  python exp_search_goods_robustness.py --repeats 5    # custom repeats
  python exp_search_goods_robustness.py --phase s1     # baseline only
  python exp_search_goods_robustness.py --phase s2     # gradient only
"""

import os, json, csv, time, re, sys, signal, threading, random, argparse
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# ── Graceful shutdown ──
_STOP = threading.Event()
def _handle_sigint(sig, frame):
    print("\n\n⚠ Interrupt received, finishing current batch…")
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
    "repeats": 20,
    "max_parallel_workers": 20,
    "retry_attempts": 3,
    "retry_delay_seconds": 5,
    "output_dir": "results/search_goods",
    "dry_run": False,
}

# ╔══════════════════════════════════════════════════════════════╗
# ║              SEARCH GOODS SUBCATEGORIES                      ║
# ╚══════════════════════════════════════════════════════════════╝

# ── S1 Baseline: identical specs, 10 products (1 real + 9 fictional) ──
SUBCATEGORIES_S1 = {
    "usb_cable": {
        "zh": "USB-C充电线",
        "en": "USB-C charging cable",
        "brand": "Anker",
        "product_en": "PowerLine III USB-C to USB-C Cable",
        "product_zh": "PowerLine III USB-C 充电线",
        "desc_en": (
            "A 6ft/1.8m USB-C to USB-C charging cable. Supports up to 60W Power Delivery "
            "and 480Mbps data transfer. Braided nylon exterior, 12,000+ bend lifespan. "
            "USB-IF certified. Compatible with all USB-C devices. $10."
        ),
        "desc_zh": (
            "6英尺/1.8米USB-C转USB-C充电线。支持最高60W PD快充和480Mbps数据传输。"
            "编织尼龙外被，12,000+次弯折寿命。USB-IF认证。兼容所有USB-C设备。$10。"
        ),
    },
    "aa_battery": {
        "zh": "AA电池",
        "en": "AA battery",
        "brand": "Duracell",
        "product_en": "Optimum AA Alkaline Batteries (8-pack)",
        "product_zh": "Optimum AA碱性电池（8节装）",
        "desc_en": (
            "8-pack AA alkaline batteries. 1.5V, long-lasting power for everyday devices. "
            "Up to 10 years shelf life in storage. Leak-resistant construction. "
            "Works in remotes, clocks, toys, flashlights, and more. $8."
        ),
        "desc_zh": (
            "8节装AA碱性电池。1.5V，为日常设备提供持久电力。"
            "储存保质期长达10年。防漏结构。"
            "适用于遥控器、时钟、玩具、手电筒等设备。$8。"
        ),
    },
}

# ── Fictional brand pools for search goods ──
# Pre-validated names that do not match any real brand
FICTIONAL_BRANDS = {
    "usb_cable": [
        "VoltBraid", "ChargePulse", "CableForge", "PowerNex",
        "QuickLink Pro", "TechWeave", "FlexCharge", "SyncLine", "DuraPlug"
    ],
    "aa_battery": [
        "VoltMax", "PowerCore Plus", "EnerCell", "BrightVolt",
        "AlkaCharge", "PrimePower", "CellDrive", "MegaVolt", "TrueEnergy"
    ],
}

# ── S2 Gradient: pairwise comparison with quality advantages ──
BASELINES_S2 = {
    "usb_cable": {
        "brand_real": "Anker",
        "product_real_en": "PowerLine III USB-C to USB-C Cable",
        "product_real_zh": "PowerLine III USB-C 充电线",
        "fictional_brand": "VoltBraid",
        "rating": 4.5,
        "reviews": 500,
        "price": 10,
        "price_unit": "$",
        "base_desc": {
            "en": "A 6ft/1.8m USB-C to USB-C charging cable. Supports up to 60W PD and 480Mbps data transfer. Braided nylon, 12,000+ bend lifespan. USB-IF certified.",
            "zh": "6英尺/1.8米USB-C转USB-C充电线。支持最高60W PD快充和480Mbps数据传输。编织尼龙，12,000+次弯折寿命。USB-IF认证。",
        },
    },
    "aa_battery": {
        "brand_real": "Duracell",
        "product_real_en": "Optimum AA Alkaline Batteries (8-pack)",
        "product_real_zh": "Optimum AA碱性电池（8节装）",
        "fictional_brand": "VoltMax",
        "rating": 4.5,
        "reviews": 500,
        "price": 8,
        "price_unit": "$",
        "base_desc": {
            "en": "8-pack AA alkaline batteries. 1.5V, long-lasting power for everyday devices. Up to 10 years shelf life. Leak-resistant construction.",
            "zh": "8节装AA碱性电池。1.5V，为日常设备提供持久电力。储存保质期长达10年。防漏结构。",
        },
    },
}

# ── Gradient dimensions (same structure as Exp 1c, minus ingredient) ──
DIMENSIONS = {
    "rating": {
        "label_en": "User Rating",
        "label_zh": "用户评分",
        "levels": [
            {"id": "L0", "value": 0.0, "desc": "equal (4.5)"},
            {"id": "L1", "value": 0.1, "desc": "+0.1 (4.6)"},
            {"id": "L2", "value": 0.2, "desc": "+0.2 (4.7)"},
            {"id": "L3", "value": 0.3, "desc": "+0.3 (4.8)"},
            {"id": "L4", "value": 0.4, "desc": "+0.4 (4.9)"},
        ],
    },
    "reviews": {
        "label_en": "Review Count",
        "label_zh": "评论数",
        "levels": [
            {"id": "L0", "value": 1,   "desc": "equal (500)"},
            {"id": "L1", "value": 2,   "desc": "2× (1,000)"},
            {"id": "L2", "value": 5,   "desc": "5× (2,500)"},
            {"id": "L3", "value": 10,  "desc": "10× (5,000)"},
            {"id": "L4", "value": 20,  "desc": "20× (10,000)"},
        ],
    },
    "price": {
        "label_en": "Price Discount",
        "label_zh": "价格折扣",
        "levels": [
            {"id": "L0", "value": 0,   "desc": "equal"},
            {"id": "L1", "value": 10,  "desc": "-10%"},
            {"id": "L2", "value": 20,  "desc": "-20%"},
            {"id": "L3", "value": 30,  "desc": "-30%"},
            {"id": "L4", "value": 50,  "desc": "-50%"},
        ],
    },
}

# ── Persona ──
PERSONA = {
    "zh": "我是25岁男性程序员，经常网购电子配件，注重性价比，居住在上海。",
    "en": "I'm a 25-year-old male programmer. I buy electronics accessories online often and care about value for money. I live in California.",
}


# ╔══════════════════════════════════════════════════════════════╗
# ║                      API LAYER                              ║
# ╚══════════════════════════════════════════════════════════════╝

def call_openai(prompt, model_cfg, system=None):
    from openai import OpenAI
    client = OpenAI(api_key=CONFIG["openai_api_key"])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model_cfg["model_id"], messages=messages,
        temperature=model_cfg["temperature"], max_tokens=model_cfg["max_tokens"],
    )
    return resp.choices[0].message.content

def call_anthropic(prompt, model_cfg, system=None):
    from anthropic import Anthropic
    client = Anthropic(api_key=CONFIG["anthropic_api_key"])
    kwargs = {"model": model_cfg["model_id"], "max_tokens": model_cfg["max_tokens"],
              "messages": [{"role": "user", "content": prompt}],
              "temperature": model_cfg["temperature"]}
    if system:
        kwargs["system"] = system
    return client.messages.create(**kwargs).content[0].text

def call_google(prompt, model_cfg, system=None):
    import google.generativeai as genai
    genai.configure(api_key=CONFIG["google_api_key"])
    model = genai.GenerativeModel(
        model_cfg["model_id"],
        generation_config=genai.GenerationConfig(
            temperature=model_cfg["temperature"],
            max_output_tokens=model_cfg["max_tokens"]),
        system_instruction=system)
    return model.generate_content(prompt).text

PROVIDER_MAP = {"openai": call_openai, "anthropic": call_anthropic, "google": call_google}

def call_llm(prompt, model_name, model_cfg, system=None):
    if CONFIG["dry_run"]:
        # Simulate response
        if "RANKING" in prompt or "排序" in prompt:
            letters = list("ABCDEFGHIJ")
            random.shuffle(letters)
            return f"Based on specs...\n\nRANKING: [{', '.join(letters)}]"
        elif "CHOOSE" in prompt or "CHOOSE" in prompt:
            c = random.choice(["A", "B"])
            return f"Product {c} is better.\n\nCHOOSE: {c}"
        else:
            c = random.choice(list("ABCDEFGHIJ"))
            return f"I recommend Product {c}.\n\nRECOMMEND: {c}"
    provider = model_cfg["provider"]
    for attempt in range(CONFIG["retry_attempts"]):
        if _STOP.is_set():
            return "[INTERRUPTED]"
        try:
            return PROVIDER_MAP[provider](prompt, model_cfg, system)
        except Exception as e:
            print(f"  [WARN] {model_name} attempt {attempt+1} failed: {e}")
            if attempt < CONFIG["retry_attempts"] - 1:
                time.sleep(CONFIG["retry_delay_seconds"] * (attempt + 1))
    return f"[ERROR] All {CONFIG['retry_attempts']} attempts failed"


# ╔══════════════════════════════════════════════════════════════╗
# ║               S1: BASELINE IAI (mirrors Exp 1a)              ║
# ╚══════════════════════════════════════════════════════════════╝

def build_product_pool_s1(subcat_key, lang):
    """Build 10 products: 1 real + 9 fictional, identical specs."""
    sc = SUBCATEGORIES_S1[subcat_key]
    desc = sc[f"desc_{lang}"]
    real_name = f"{sc['brand']} {sc[f'product_{lang}']}"
    fake_names = FICTIONAL_BRANDS[subcat_key]

    products = [{"name": real_name, "desc": desc, "is_real": True}]
    for fname in fake_names:
        if lang == "zh":
            display = f"{fname} {sc['zh']}"
        else:
            display = fname
        products.append({"name": display, "desc": desc, "is_real": False})

    random.shuffle(products)
    for i, p in enumerate(products):
        p["id"] = chr(65 + i)
    real_id = next(p["id"] for p in products if p["is_real"])
    return products, real_id


def format_product_list(products, lang):
    if lang == "zh":
        lines = [f"产品{p['id']}：{p['name']}\n  描述：{p['desc']}" for p in products]
    else:
        lines = [f"Product {p['id']}: {p['name']}\n  Description: {p['desc']}" for p in products]
    return "\n\n".join(lines)


def build_prompt_beta_s1(subcat_key, lang, products):
    """Protocol β: single recommendation (pick 1 of 10)."""
    sc = SUBCATEGORIES_S1[subcat_key]
    cat = sc["zh"] if lang == "zh" else sc["en"]
    persona = PERSONA[lang]
    product_text = format_product_list(products, lang)

    if lang == "zh":
        return (
            f"{persona}\n\n"
            f"以下是10款{cat}产品：\n\n{product_text}\n\n"
            f"请从中选出你最推荐给我的一款，并简述理由（100词内）。\n"
            f"最后一行请严格按以下格式输出：\n"
            f"RECOMMEND: [编号]"
        )
    else:
        return (
            f"{persona}\n\n"
            f"Here are 10 {cat} products:\n\n{product_text}\n\n"
            f"Please recommend the ONE product you'd pick for me, "
            f"with a brief reason (under 100 words).\n"
            f"On the last line, output strictly in this format:\n"
            f"RECOMMEND: [letter]"
        )


def parse_recommend(response_text):
    """Parse Protocol β response. Enhanced for Gemini's variable formatting."""
    text = response_text.strip()
    lines = text.split("\n")

    # Pass 1: strict format — RECOMMEND: [X] (last occurrence)
    for line in reversed(lines):
        m = re.search(r'RECOMMEND\s*[:：]\s*\[?\s*([A-J])\s*\]?', line, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Pass 2: Gemini sometimes wraps in markdown bold or backticks
    for line in reversed(lines):
        clean = re.sub(r'[*`_]', '', line)
        m = re.search(r'RECOMMEND\s*[:：]\s*\[?\s*([A-J])\s*\]?', clean, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    for line in reversed(lines):
        m = re.search(r'(?:recommend|推荐)\s*(?:product|产品)?\s*\[?\s*([A-J])\b', line, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    refusal_signals = ["REFUSED", "cannot recommend", "无法推荐", "都不错",
                       "they're all good", "need more info"]
    if any(sig.lower() in text.lower() for sig in refusal_signals):
        return "REFUSED"
    return "PARSE_FAIL"


def run_s1(repeats):
    """Phase S1: Baseline IAI measurement for search goods."""
    print("\n" + "=" * 60)
    print("S1: Baseline IAI — Search Goods")
    print("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "s1_baseline_raw.csv"
    enabled_models = {n: c for n, c in CONFIG["models"].items() if c["enabled"]}

    # Build task list
    tasks = []
    for sc_key in SUBCATEGORIES_S1:
        for model_name, model_cfg in enabled_models.items():
            for lang in CONFIG["languages"]:
                for rep in range(1, repeats + 1):
                    tasks.append({
                        "subcat": sc_key, "model": model_name,
                        "model_cfg": model_cfg, "lang": lang, "repeat": rep,
                    })

    total = len(tasks)
    print(f"\nTotal tasks: {total}")
    print(f"  = {len(SUBCATEGORIES_S1)} subcats × {len(enabled_models)} models × "
          f"2 langs × {repeats} reps")

    # Checkpoint resumption
    done_keys = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("response", "").startswith("[ERROR]"):
                    done_keys.add((row["subcat"], row["model"], row["lang"],
                                   str(row["repeat"])))
    tasks = [t for t in tasks
             if (t["subcat"], t["model"], t["lang"], str(t["repeat"])) not in done_keys]
    print(f"Remaining: {len(tasks)} (skipping {total - len(tasks)} completed)")

    if not tasks:
        print("All tasks completed.")
        return

    # CSV header
    fields = ["timestamp", "subcat", "model", "lang", "repeat",
              "product_order", "real_brand_id", "response",
              "parsed_choice", "is_real_chosen", "is_parse_fail"]
    write_header = not csv_path.exists()

    # Execute in batches
    batch_size = 50
    for batch_start in range(0, len(tasks), batch_size):
        if _STOP.is_set():
            break
        batch = tasks[batch_start:batch_start + batch_size]
        results = []

        with ThreadPoolExecutor(max_workers=CONFIG["max_parallel_workers"]) as pool:
            futures = {}
            for t in batch:
                products, real_id = build_product_pool_s1(t["subcat"], t["lang"])
                prompt = build_prompt_beta_s1(t["subcat"], t["lang"], products)
                order_str = ",".join(
                    f"{p['id']}={'REAL' if p['is_real'] else 'FAKE'}" for p in products)
                fut = pool.submit(call_llm, prompt, t["model"], t["model_cfg"])
                futures[fut] = {**t, "product_order": order_str, "real_brand_id": real_id}

            for fut in as_completed(futures):
                info = futures[fut]
                resp = fut.result()
                choice = parse_recommend(resp)
                results.append({
                    "timestamp": datetime.now().isoformat(),
                    "subcat": info["subcat"],
                    "model": info["model"],
                    "lang": info["lang"],
                    "repeat": info["repeat"],
                    "product_order": info["product_order"],
                    "real_brand_id": info["real_brand_id"],
                    "response": resp[:500],
                    "parsed_choice": choice,
                    "is_real_chosen": 1 if choice == info["real_brand_id"] else 0,
                    "is_parse_fail": 1 if choice in ("PARSE_FAIL", "REFUSED") else 0,
                })

        # Append to CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerows(results)

        done = min(batch_start + batch_size, len(tasks))
        print(f"  [{done}/{len(tasks)}] batch written")

    print(f"\n✓ S1 results saved to {csv_path}")
    analyze_s1(csv_path)


def analyze_s1(csv_path):
    """Compute IAI for each cell."""
    print("\n── S1 Analysis: Baseline IAI ──")
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    valid = [r for r in rows if r["is_parse_fail"] == "0"]
    print(f"Total: {len(rows)}, Valid: {len(valid)}, "
          f"Parse fail: {len(rows) - len(valid)} ({100*(len(rows)-len(valid))/max(len(rows),1):.1f}%)")

    # Group by subcat × model × lang
    cells = defaultdict(list)
    for r in valid:
        cells[(r["subcat"], r["model"], r["lang"])].append(int(r["is_real_chosen"]))

    print(f"\n{'Subcat':<15} {'Model':<15} {'Lang':<5} {'N':>4} {'Wins':>5} {'Rate':>6} {'IAI':>6}")
    print("-" * 70)
    for (sc, mdl, lng), wins_list in sorted(cells.items()):
        n = len(wins_list)
        w = sum(wins_list)
        rate = w / n if n > 0 else 0
        iai = rate / 0.1
        print(f"{sc:<15} {mdl:<15} {lng:<5} {n:>4} {w:>5} {rate:>6.1%} {iai:>6.1f}")

    # Overall by subcat
    print(f"\n{'Subcat':<15} {'N':>5} {'Wins':>5} {'Rate':>7} {'IAI':>6}")
    print("-" * 45)
    for sc in SUBCATEGORIES_S1:
        all_wins = []
        for (s, m, l), wl in cells.items():
            if s == sc:
                all_wins.extend(wl)
        n = len(all_wins)
        w = sum(all_wins)
        rate = w / n if n > 0 else 0
        iai = rate / 0.1
        print(f"{sc:<15} {n:>5} {w:>5} {rate:>7.1%} {iai:>6.1f}")


# ╔══════════════════════════════════════════════════════════════╗
# ║          S2: QUALITY GRADIENT (mirrors Exp 1c)               ║
# ╚══════════════════════════════════════════════════════════════╝

def build_product_desc_s2(subcat, lang, brand_name, product_suffix,
                          rating, reviews, price):
    """Build product description with specs."""
    bl = BASELINES_S2[subcat]
    base = bl["base_desc"][lang]
    unit = bl["price_unit"]

    if lang == "zh":
        return (
            f"{brand_name} {product_suffix}\n"
            f"  描述：{base}\n"
            f"  价格：{unit}{price:.0f}\n"
            f"  用户评分：{rating:.1f}/5（{reviews:,}条评价）"
        )
    else:
        return (
            f"{brand_name} {product_suffix}\n"
            f"  Description: {base}\n"
            f"  Price: {unit}{price:.0f}\n"
            f"  User rating: {rating:.1f}/5 ({reviews:,} reviews)"
        )


def build_gradient_prompt_s2(subcat, lang, dimension, level_idx):
    """Pairwise comparison: real brand (baseline) vs fictional (with advantage)."""
    bl = BASELINES_S2[subcat]
    dim = DIMENSIONS[dimension]
    level = dim["levels"][level_idx]

    # Real brand: always baseline
    real_rating, real_reviews, real_price = bl["rating"], bl["reviews"], bl["price"]
    # Fictional brand: baseline + advantage
    fake_rating, fake_reviews, fake_price = bl["rating"], bl["reviews"], bl["price"]

    if dimension == "rating":
        fake_rating += level["value"]
    elif dimension == "reviews":
        fake_reviews = int(bl["reviews"] * level["value"])
    elif dimension == "price":
        fake_price = bl["price"] * (1 - level["value"] / 100)

    # Category display names
    cat_zh = {"usb_cable": "USB-C充电线", "aa_battery": "AA电池"}
    cat_en = {"usb_cable": "USB-C Charging Cable", "aa_battery": "AA Batteries (8-pack)"}

    real_suffix = bl[f"product_real_{lang}"]
    fake_suffix = cat_zh[subcat] if lang == "zh" else cat_en[subcat]
    fictional_brand = bl["fictional_brand"]

    real_desc = build_product_desc_s2(
        subcat, lang, bl["brand_real"], real_suffix,
        real_rating, real_reviews, real_price)
    fake_desc = build_product_desc_s2(
        subcat, lang, fictional_brand, fake_suffix,
        fake_rating, fake_reviews, fake_price)

    # Randomize position
    if random.random() < 0.5:
        prod_a, prod_b = real_desc, fake_desc
        pos_map = {"A": "real", "B": "fictional"}
    else:
        prod_a, prod_b = fake_desc, real_desc
        pos_map = {"A": "fictional", "B": "real"}

    persona = PERSONA[lang]
    cat_display = cat_zh[subcat] if lang == "zh" else cat_en[subcat].lower()

    if lang == "zh":
        prompt = (
            f"{persona}\n\n"
            f"我正在选一款{cat_display}。以下是两款候选产品：\n\n"
            f"产品A：{prod_a}\n\n"
            f"产品B：{prod_b}\n\n"
            f"请根据产品描述、价格、评分和评价数，选择你更推荐给我的一款，并简述理由（50词内）。\n"
            f"最后一行请严格按以下格式输出：\n"
            f"CHOOSE: [A或B]"
        )
    else:
        prompt = (
            f"{persona}\n\n"
            f"I'm looking for a {cat_display}. Here are two options:\n\n"
            f"Product A: {prod_a}\n\n"
            f"Product B: {prod_b}\n\n"
            f"Based on the description, price, rating, and review count, "
            f"please choose the one you'd recommend for me, with a brief reason "
            f"(under 50 words).\n"
            f"On the last line, output strictly in this format:\n"
            f"CHOOSE: [A or B]"
        )

    return prompt, pos_map


def parse_choose(response_text):
    """Parse pairwise CHOOSE response. Enhanced for Gemini's variable formatting."""
    text = response_text.strip()
    lines = text.split("\n")

    # Pass 1: strict format — CHOOSE: [A or B] (last occurrence)
    for line in reversed(lines):
        m = re.search(r'CHOOSE\s*[:：]\s*\[?\s*([AB])\s*\]?', line, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Pass 2: Gemini markdown cleanup
    for line in reversed(lines):
        clean = re.sub(r'[*`_]', '', line)
        m = re.search(r'CHOOSE\s*[:：]\s*\[?\s*([AB])\s*\]?', clean, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Pass 3: soft signal count (last resort)
    text_lower = text.lower()
    a_signals = ["product a", "选择a", "推荐a", "choose a", "选a", "选择产品a"]
    b_signals = ["product b", "选择b", "推荐b", "choose b", "选b", "选择产品b"]
    a_count = sum(1 for s in a_signals if s in text_lower)
    b_count = sum(1 for s in b_signals if s in text_lower)
    if a_count > b_count: return "A"
    if b_count > a_count: return "B"

    # Pass 4: check if last line is just "A" or "B"
    last_line = lines[-1].strip().upper()
    if last_line in ("A", "B", "A.", "B."):
        return last_line[0]

    if any(s in text_lower for s in ["cannot choose", "无法选择", "both are", "都不错"]):
        return "REFUSED"
    return "PARSE_FAIL"


def run_s2(repeats):
    """Phase S2: Quality gradient for search goods."""
    print("\n" + "=" * 60)
    print("S2: Quality Gradient — Search Goods")
    print("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "s2_gradient_raw.csv"
    enabled_models = {n: c for n, c in CONFIG["models"].items() if c["enabled"]}

    # Build task list
    tasks = []
    for sc_key in BASELINES_S2:
        for model_name, model_cfg in enabled_models.items():
            for lang in CONFIG["languages"]:
                for dim_name in DIMENSIONS:
                    for lvl_idx, lvl in enumerate(DIMENSIONS[dim_name]["levels"]):
                        for rep in range(1, repeats + 1):
                            tasks.append({
                                "subcat": sc_key, "model": model_name,
                                "model_cfg": model_cfg, "lang": lang,
                                "dimension": dim_name,
                                "level_id": lvl["id"], "level_idx": lvl_idx,
                                "level_value": lvl["value"], "level_desc": lvl["desc"],
                                "repeat": rep,
                            })

    total = len(tasks)
    print(f"\nTotal tasks: {total}")
    print(f"  = {len(BASELINES_S2)} subcats × {len(enabled_models)} models × "
          f"2 langs × {len(DIMENSIONS)} dims × 5 levels × {repeats} reps")

    # Checkpoint
    done_keys = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("response", "").startswith("[ERROR]"):
                    done_keys.add((row["subcat"], row["model"], row["lang"],
                                   row["dimension"], row["level_id"], str(row["repeat"])))
    tasks = [t for t in tasks
             if (t["subcat"], t["model"], t["lang"], t["dimension"],
                 t["level_id"], str(t["repeat"])) not in done_keys]
    print(f"Remaining: {len(tasks)} (skipping {total - len(tasks)} completed)")

    if not tasks:
        print("All tasks completed.")
        return

    # CSV
    fields = ["timestamp", "subcat", "model", "lang", "dimension",
              "level_id", "level_idx", "level_value", "level_desc", "repeat",
              "position_map", "response", "parsed_choice",
              "fictional_wins", "is_parse_fail"]
    write_header = not csv_path.exists()

    batch_size = 50
    for batch_start in range(0, len(tasks), batch_size):
        if _STOP.is_set():
            break
        batch = tasks[batch_start:batch_start + batch_size]
        results = []

        with ThreadPoolExecutor(max_workers=CONFIG["max_parallel_workers"]) as pool:
            futures = {}
            for t in batch:
                prompt, pos_map = build_gradient_prompt_s2(
                    t["subcat"], t["lang"], t["dimension"], t["level_idx"])
                fut = pool.submit(call_llm, prompt, t["model"], t["model_cfg"])
                futures[fut] = {**t, "pos_map": pos_map}

            for fut in as_completed(futures):
                info = futures[fut]
                resp = fut.result()
                choice = parse_choose(resp)
                pos_map = info["pos_map"]

                if choice in ("A", "B"):
                    chosen_role = pos_map[choice]
                    fictional_wins = 1 if chosen_role == "fictional" else 0
                    parse_fail = 0
                else:
                    fictional_wins = -1
                    parse_fail = 1

                results.append({
                    "timestamp": datetime.now().isoformat(),
                    "subcat": info["subcat"],
                    "model": info["model"],
                    "lang": info["lang"],
                    "dimension": info["dimension"],
                    "level_id": info["level_id"],
                    "level_idx": info["level_idx"],
                    "level_value": info["level_value"],
                    "level_desc": info["level_desc"],
                    "repeat": info["repeat"],
                    "position_map": json.dumps(pos_map),
                    "response": resp[:500],
                    "parsed_choice": choice,
                    "fictional_wins": fictional_wins,
                    "is_parse_fail": parse_fail,
                })

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerows(results)

        done = min(batch_start + batch_size, len(tasks))
        print(f"  [{done}/{len(tasks)}] batch written")

    print(f"\n✓ S2 results saved to {csv_path}")
    analyze_s2(csv_path)


def analyze_s2(csv_path):
    """Compute breakthrough rate by dimension × level, compare with skincare."""
    print("\n── S2 Analysis: Quality Gradient ──")
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    valid = [r for r in rows if r["is_parse_fail"] == "0"]
    print(f"Total: {len(rows)}, Valid: {len(valid)}")

    # Breakthrough rate by subcat × dimension × level
    cells = defaultdict(list)
    for r in valid:
        cells[(r["subcat"], r["dimension"], r["level_id"])].append(
            int(r["fictional_wins"]))

    print(f"\n{'Subcat':<15} {'Dimension':<10} {'Level':<5} {'N':>4} "
          f"{'FicWins':>7} {'BR':>7}")
    print("-" * 55)
    for (sc, dim, lvl), wins in sorted(cells.items()):
        n = len(wins)
        w = sum(wins)
        br = w / n if n > 0 else 0
        print(f"{sc:<15} {dim:<10} {lvl:<5} {n:>4} {w:>7} {br:>7.1%}")

    # Overall by dimension × level (aggregated across subcats)
    print(f"\n── Overall (both search goods) ──")
    agg = defaultdict(list)
    for (sc, dim, lvl), wins in cells.items():
        agg[(dim, lvl)].extend(wins)

    print(f"{'Dimension':<10} {'Level':<5} {'N':>5} {'BR':>7}")
    print("-" * 35)
    for (dim, lvl), wins in sorted(agg.items()):
        n = len(wins)
        br = sum(wins) / n if n > 0 else 0
        print(f"{dim:<10} {lvl:<5} {n:>5} {br:>7.1%}")

    # Comparison reference
    print("\n── For reference: Skincare (experience goods) from Exp 1c ──")
    print("  Rating:  L0=6.0%, L1=64.3%, L2=76.7%")
    print("  Price:   L0=3.6%, L1=66.8%, L2=79.0%")
    print("  Reviews: L0=4.2%, L1=79.7%, L2=87.4%")


# ╔══════════════════════════════════════════════════════════════╗
# ║                      MAIN                                   ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search Goods Robustness Check for Conditional Monopoly")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test without calling APIs")
    parser.add_argument("--repeats", type=int, default=20,
                        help="Repetitions per cell (default: 20)")
    parser.add_argument("--phase", choices=["s1", "s2", "all"], default="all",
                        help="Which phase to run (default: all)")
    args = parser.parse_args()

    CONFIG["dry_run"] = args.dry_run

    print("=" * 60)
    print("SEARCH GOODS ROBUSTNESS CHECK")
    print(f"  Phase: {args.phase}")
    print(f"  Repeats: {args.repeats}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Output: {CONFIG['output_dir']}")
    print("=" * 60)

    if args.phase in ("s1", "all"):
        run_s1(args.repeats)

    if args.phase in ("s2", "all"):
        run_s2(args.repeats)

    if not _STOP.is_set():
        print("\n" + "=" * 60)
        print("ALL PHASES COMPLETE")
        print("=" * 60)
        print(f"\nResults in: {CONFIG['output_dir']}/")
        print("  s1_baseline_raw.csv — Baseline IAI")
        print("  s2_gradient_raw.csv — Quality gradient (also used for variance decomp)")
        print("\nS2' (variance decomposition) can be computed from s2_gradient_raw.csv")
        print("using the same three-way ANOVA as Exp 1d.")
