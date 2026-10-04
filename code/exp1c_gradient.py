#!/usr/bin/env python3
"""
Experiment 1c — Multi-Dimensional Advantage Gradient
======================================================
RQ3: What level of objective advantage does a fictional brand need
     to overcome the incumbent's brand name advantage?

Design:
  Single-dimension gradient sweeps (pairwise comparison):
  Real brand (fixed baseline) vs Fictional brand (baseline + graduated advantage)

  4 Dimensions × 5 Levels × 4 Subcats × 3 Models × 2 Langs × 20 Reps = 9,600 calls

  Dimensions:
    1. rating    — Fictional brand gets higher user rating (4.5→4.6→4.7→4.8→4.9)
    2. reviews   — Fictional brand gets more reviews (500→1k→2.5k→5k→10k)
    3. price     — Fictional brand is cheaper (0%→-10%→-20%→-30%→-50%)
    4. ingredient— Fictional brand gets better formula (5 graduated levels)

  Key Output:
    Sigmoid curve: P(fictional wins) = 1 / (1 + exp(-k*(x - x_0)))
    x_0 = breakthrough threshold where P = 0.5

Usage:
  python exp1c_gradient.py
  python exp1c_gradient.py --dry-run
  python exp1c_gradient.py --fresh
  python exp1c_gradient.py --repeats 5
  python exp1c_gradient.py --analyze-only
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
    "repeats": 20,
    "max_parallel_workers": 20,
    "retry_attempts": 3,
    "retry_delay_seconds": 5,
    "output_dir": "results/exp1c",
    "fictional_brands_path": "results/preflight/fictional_brands.json",
    "dry_run": False,
}

PERSONA = {
    "zh": "我是25岁男性程序员，油痘肌，脸部和头皮容易出油，居住在上海。",
    "en": "I'm a 25-year-old male programmer with oily, acne-prone skin. My face and scalp get greasy easily. I live in California.",
}

# ╔══════════════════════════════════════════════════════════════╗
# ║                 PRODUCT BASELINES & GRADIENTS                ║
# ╚══════════════════════════════════════════════════════════════╝

BASELINES = {
    "moisturizer": {
        "brand_real": "CeraVe",
        "product_real_en": "PM Facial Moisturizing Lotion",
        "product_real_zh": "PM 保湿乳液",
        "fictional_index": 1,
        "rating": 4.5,
        "reviews": 500,
        "price": 17,
        "price_unit": "$",
        "base_desc": {
            "en": "A lightweight, oil-free facial moisturizing lotion with ceramides, hyaluronic acid, and niacinamide. Helps restore the skin's natural barrier. Non-comedogenic, fragrance-free. 89ml.",
            "zh": "轻盈无油保湿乳液，含神经酰胺、透明质酸和烟酰胺。帮助修复肌肤天然屏障。不致痘、无香精。89ml。",
        },
    },
    "bha_exfoliant": {
        "brand_real": "Paula's Choice",
        "product_real_en": "Skin Perfecting 2% BHA Liquid Exfoliant",
        "product_real_zh": "2% BHA 水杨酸精华液",
        "fictional_index": 1,
        "rating": 4.5,
        "reviews": 500,
        "price": 35,
        "price_unit": "$",
        "base_desc": {
            "en": "A leave-on exfoliant with 2% salicylic acid (BHA). Unclogs pores, smooths texture, reduces redness. Suitable for oily and acne-prone skin. Fragrance-free. 118ml.",
            "zh": "免洗型去角质精华，含2%水杨酸(BHA)。疏通毛孔，改善纹理，减少泛红。适合油性和痘肌。无香精。118ml。",
        },
    },
    "sunscreen": {
        "brand_real": "EltaMD",
        "product_real_en": "UV Clear Broad-Spectrum SPF 46",
        "product_real_zh": "UV Clear 广谱防晒 SPF 46",
        "fictional_index": 1,
        "rating": 4.5,
        "reviews": 500,
        "price": 40,
        "price_unit": "$",
        "base_desc": {
            "en": "A lightweight, oil-free facial sunscreen with 9.0% zinc oxide and niacinamide. Broad-spectrum SPF 46. Calms acne-prone skin. Fragrance-free, non-comedogenic. 48g.",
            "zh": "轻盈无油面部防晒霜，含9.0%氧化锌和烟酰胺。广谱SPF 46。舒缓痘肌。无香精、不致痘。48g。",
        },
    },
    "cleanser": {
        "brand_real": "CeraVe",
        "product_real_en": "Foaming Facial Cleanser",
        "product_real_zh": "泡沫洁面乳",
        "fictional_index": 1,
        "rating": 4.5,
        "reviews": 500,
        "price": 16,
        "price_unit": "$",
        "base_desc": {
            "en": "A foaming gel cleanser with ceramides, hyaluronic acid, and niacinamide. Removes excess oil without disrupting the skin barrier. Oil-free, non-comedogenic. 236ml.",
            "zh": "泡沫凝胶洁面乳，含神经酰胺、透明质酸和烟酰胺。去除多余油脂，不破坏肌肤屏障。无油、不致痘。236ml。",
        },
    },
}

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
    "ingredient": {
        "label_en": "Ingredient Upgrade",
        "label_zh": "成分升级",
        "levels": [
            {"id": "L0", "value": 0, "desc": "equal"},
            {"id": "L1", "value": 1, "desc": "slight+"},
            {"id": "L2", "value": 2, "desc": "moderate+"},
            {"id": "L3", "value": 3, "desc": "strong+"},
            {"id": "L4", "value": 4, "desc": "major+"},
        ],
    },
}

INGREDIENT_UPGRADES = {
    "moisturizer": {
        "en": [
            # L0: equal (same as base)
            "",
            # L1: slight — add peptides
            " Enhanced with peptide complex for additional anti-aging benefits.",
            # L2: moderate — add peptides + higher HA concentration
            " Enhanced with peptide complex and 2% hyaluronic acid (vs standard 1%). Clinically shown to improve hydration by 48%.",
            # L3: strong — above + squalane + clinical data
            " Enhanced with peptide complex, 2% hyaluronic acid, and plant-derived squalane. Clinically shown to improve hydration by 48% and reduce fine lines by 23% in 8 weeks.",
            # L4: major — above + patented tech + dermatologist endorsement
            " Enhanced with peptide complex, 2% hyaluronic acid, plant-derived squalane, and patented MoistureLock™ delivery system. Clinically shown to improve hydration by 72% and reduce fine lines by 38% in 4 weeks. Recommended by 9 out of 10 dermatologists in a blind trial.",
        ],
        "zh": [
            "",
            " 添加多肽复合物，额外抗衰老功效。",
            " 添加多肽复合物和2%透明质酸（高于标准1%）。临床证明保湿提升48%。",
            " 添加多肽复合物、2%透明质酸和植物角鲨烷。临床证明保湿提升48%，8周内细纹减少23%。",
            " 添加多肽复合物、2%透明质酸、植物角鲨烷和专利MoistureLock™递送系统。临床证明保湿提升72%，4周内细纹减少38%。9/10皮肤科医生盲测推荐。",
        ],
    },
    "bha_exfoliant": {
        "en": [
            "",
            " Also contains green tea extract for antioxidant protection.",
            " Also contains green tea extract and 0.5% niacinamide. Clinically shown to reduce pore size by 20%.",
            " Also contains green tea extract, 0.5% niacinamide, and willow bark extract. Clinically shown to reduce pore size by 35% and clear acne by 40% in 6 weeks.",
            " Also contains green tea extract, 0.5% niacinamide, willow bark extract, and patented PoreClear™ technology. Clinically shown to reduce pore size by 52% and clear acne by 65% in 4 weeks. Award-winning formula.",
        ],
        "zh": [
            "",
            " 添加绿茶提取物，抗氧化保护。",
            " 添加绿茶提取物和0.5%烟酰胺。临床证明毛孔缩小20%。",
            " 添加绿茶提取物、0.5%烟酰胺和柳树皮提取物。临床证明毛孔缩小35%，6周痘痘减少40%。",
            " 添加绿茶提取物、0.5%烟酰胺、柳树皮提取物和专利PoreClear™技术。临床证明毛孔缩小52%，4周痘痘减少65%。获奖配方。",
        ],
    },
    "sunscreen": {
        "en": [
            "",
            " Also contains vitamin E for additional antioxidant protection.",
            " Also contains vitamin E and centella asiatica extract. Clinically tested to be non-irritating on 98% of sensitive skin users.",
            " Also contains vitamin E, centella asiatica extract, and hyaluronic acid for hydration. Clinically tested: non-irritating on 98% of sensitive skin users, no white cast in 95% of users.",
            " Also contains vitamin E, centella asiatica extract, hyaluronic acid, and patented UVShield+™ broad-spectrum technology. Clinically tested: non-irritating on 99% of sensitive skin, invisible on all skin tones. Water-resistant 80 min. Dermatologist award winner.",
        ],
        "zh": [
            "",
            " 添加维生素E，额外抗氧化保护。",
            " 添加维生素E和积雪草提取物。临床测试98%敏感肌无刺激。",
            " 添加维生素E、积雪草提取物和透明质酸保湿。临床测试98%敏感肌无刺激，95%用户无泛白。",
            " 添加维生素E、积雪草提取物、透明质酸和专利UVShield+™广谱技术。临床测试99%敏感肌无刺激，所有肤色无泛白。防水80分钟。皮肤科获奖产品。",
        ],
    },
    "cleanser": {
        "en": [
            "",
            " Also contains centella asiatica for soothing benefits.",
            " Also contains centella asiatica and salicylic acid (0.5%) for gentle exfoliation. Clinically shown to reduce oiliness by 30%.",
            " Also contains centella asiatica, salicylic acid (0.5%), and tea tree oil. Clinically shown to reduce oiliness by 45% and prevent breakouts by 35% in 4 weeks.",
            " Also contains centella asiatica, salicylic acid (0.5%), tea tree oil, and patented OilControl™ micro-foam technology. Clinically shown to reduce oiliness by 60% and prevent breakouts by 55% in 2 weeks. Dermatologist blind-test winner.",
        ],
        "zh": [
            "",
            " 添加积雪草，舒缓修护。",
            " 添加积雪草和0.5%水杨酸温和去角质。临床证明控油提升30%。",
            " 添加积雪草、0.5%水杨酸和茶树油。临床证明控油提升45%，4周内减少35%痘痘。",
            " 添加积雪草、0.5%水杨酸、茶树油和专利OilControl™微泡技术。临床证明控油提升60%，2周内减少55%痘痘。皮肤科盲测获奖。",
        ],
    },
}


# ╔══════════════════════════════════════════════════════════════╗
# ║                   FICTIONAL BRAND LOADING                    ║
# ╚══════════════════════════════════════════════════════════════╝

def load_fictional_brands():
    path = CONFIG["fictional_brands_path"]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    brands = {}
    for subcat, bl in BASELINES.items():
        pool = data.get("candidates", {}).get(subcat, [])
        idx = bl["fictional_index"]
        brands[subcat] = pool[idx]
        print(f"  ✓ {subcat}: real={bl['brand_real']}, fictional={pool[idx]}")
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
        choice = random.choice(["A", "B"])
        return f"Product {choice} is better.\n\nCHOOSE: {choice}"
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
# ║                 PROMPT CONSTRUCTION                          ║
# ╚══════════════════════════════════════════════════════════════╝

def build_product_desc(subcat, lang, brand_name, product_suffix,
                       rating, reviews, price, ingredient_extra=""):
    """Build full product description (with rating, review count, price, and ingredient info)."""
    bl = BASELINES[subcat]
    base = bl["base_desc"][lang]
    unit = bl["price_unit"]

    if lang == "zh":
        full = (
            f"{brand_name} {product_suffix}\n"
            f"  描述：{base}{ingredient_extra}\n"
            f"  价格：{unit}{price:.0f}\n"
            f"  用户评分：{rating:.1f}/5（{reviews:,}条评价）"
        )
    else:
        full = (
            f"{brand_name} {product_suffix}\n"
            f"  Description: {base}{ingredient_extra}\n"
            f"  Price: {unit}{price:.0f}\n"
            f"  User rating: {rating:.1f}/5 ({reviews:,} reviews)"
        )
    return full


def build_gradient_prompt(subcat, lang, dimension, level_idx, fictional_brand):
    """
    Build a gradient comparison prompt.
    The real brand always uses baseline parameters; the fictional brand gets a
    level_idx-sized advantage on the specified dimension.
    """
    bl = BASELINES[subcat]
    dim = DIMENSIONS[dimension]
    level = dim["levels"][level_idx]

    real_rating = bl["rating"]
    real_reviews = bl["reviews"]
    real_price = bl["price"]
    real_ingredient = ""

    fake_rating = bl["rating"]
    fake_reviews = bl["reviews"]
    fake_price = bl["price"]
    fake_ingredient = ""

    if dimension == "rating":
        fake_rating = bl["rating"] + level["value"]
    elif dimension == "reviews":
        fake_reviews = int(bl["reviews"] * level["value"])
    elif dimension == "price":
        fake_price = bl["price"] * (1 - level["value"] / 100)
    elif dimension == "ingredient":
        lvl = level["value"]
        fake_ingredient = INGREDIENT_UPGRADES[subcat][lang][lvl]

    real_suffix = bl[f"product_real_{lang}"]
    cat_zh = {"moisturizer": "保湿乳液", "bha_exfoliant": "BHA精华液",
              "sunscreen": "防晒霜", "cleanser": "洁面乳"}
    cat_en = {"moisturizer": "Facial Moisturizer", "bha_exfoliant": "BHA Liquid Exfoliant",
              "sunscreen": "Sunscreen SPF 46", "cleanser": "Facial Cleanser"}
    fake_suffix = cat_zh[subcat] if lang == "zh" else cat_en[subcat]

    real_desc = build_product_desc(
        subcat, lang, bl["brand_real"], real_suffix,
        real_rating, real_reviews, real_price, real_ingredient)
    fake_desc = build_product_desc(
        subcat, lang, fictional_brand, fake_suffix,
        fake_rating, fake_reviews, fake_price, fake_ingredient)

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
            f"please choose the one you'd recommend for me, with a brief reason (under 50 words).\n"
            f"On the last line, output strictly in this format:\n"
            f"CHOOSE: [A or B]"
        )

    return prompt, pos_map


def parse_choose(response_text):
    lines = response_text.strip().split("\n")
    for line in reversed(lines):
        m = re.search(r'CHOOSE:\s*\[?\s*([AB])\s*\]?', line, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    text = response_text.lower()
    a_signals = ["product a", "选择a", "推荐a", "choose a", "recommend a"]
    b_signals = ["product b", "选择b", "推荐b", "choose b", "recommend b"]
    a_count = sum(1 for s in a_signals if s in text)
    b_count = sum(1 for s in b_signals if s in text)
    if a_count > b_count: return "A"
    if b_count > a_count: return "B"
    refusal_signals = ["cannot choose", "无法选择", "both are", "都不错"]
    if any(s in text for s in refusal_signals): return "REFUSED"
    return "PARSE_FAIL"


# ╔══════════════════════════════════════════════════════════════╗
# ║                      MAIN EXECUTION                         ║
# ╚══════════════════════════════════════════════════════════════╝

def run_exp1c(fictional_brands, repeats):
    print("\n" + "=" * 60)
    print("EXP 1c: Multi-Dimensional Advantage Gradient")
    print("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    enabled_models = {n: c for n, c in CONFIG["models"].items() if c["enabled"]}

    tasks = []
    for sc_key in BASELINES:
        for model_name, model_cfg in enabled_models.items():
            for lang in CONFIG["languages"]:
                for dim_name in DIMENSIONS:
                    for lvl_idx, lvl in enumerate(DIMENSIONS[dim_name]["levels"]):
                        for rep in range(1, repeats + 1):
                            tasks.append({
                                "subcat": sc_key,
                                "model": model_name,
                                "model_cfg": model_cfg,
                                "lang": lang,
                                "dimension": dim_name,
                                "level_id": lvl["id"],
                                "level_idx": lvl_idx,
                                "level_value": lvl["value"],
                                "level_desc": lvl["desc"],
                                "repeat": rep,
                            })

    total = len(tasks)
    print(f"\nTotal tasks: {total}")
    print(f"  = {len(BASELINES)} subcats × {len(enabled_models)} models × "
          f"2 langs × {len(DIMENSIONS)} dims × 5 levels × {repeats} reps")

    csv_path = output_dir / "exp1c_raw.csv"
    done_keys = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("response", "").startswith("[ERROR]"):
                    key = (row["subcat"], row["model"], row["lang"],
                           row["dimension"], row["level_id"], row["repeat"])
                    done_keys.add(key)
        print(f"  Found {len(done_keys)} completed records")

    remaining = [t for t in tasks
                 if (t["subcat"], t["model"], t["lang"],
                     t["dimension"], t["level_id"], str(t["repeat"])) not in done_keys]
    print(f"  Remaining: {len(remaining)}")
    if not remaining:
        print("  ✓ All tasks completed!")
        return

    csv_fieldnames = [
        "timestamp", "subcat", "model", "lang",
        "dimension", "level_id", "level_value", "level_desc", "repeat",
        "position_a", "position_b", "response",
        "parsed_choice", "chose_fictional",
        "is_refusal", "is_parse_fail",
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
            prompt, pos_map = build_gradient_prompt(
                task["subcat"], task["lang"],
                task["dimension"], task["level_idx"], fb)
            response = call_llm(prompt, task["model"], task["model_cfg"])
            choice = parse_choose(response)
            is_refusal = choice == "REFUSED"
            is_parse_fail = choice == "PARSE_FAIL"
            chose_fictional = ""
            if not is_refusal and not is_parse_fail:
                chose_fictional = str(pos_map.get(choice) == "fictional")
            return {
                "timestamp": datetime.now().isoformat(),
                "subcat": task["subcat"],
                "model": task["model"],
                "lang": task["lang"],
                "dimension": task["dimension"],
                "level_id": task["level_id"],
                "level_value": str(task["level_value"]),
                "level_desc": task["level_desc"],
                "repeat": str(task["repeat"]),
                "position_a": pos_map.get("A", ""),
                "position_b": pos_map.get("B", ""),
                "response": response,
                "parsed_choice": choice if not is_refusal and not is_parse_fail else "",
                "chose_fictional": chose_fictional,
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
                    if completed % 50 == 0 or completed == total:
                        print(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%)")

        if batch_results:
            with open(csv_path, "a", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_fieldnames).writerows(batch_results)

    if _STOP.is_set():
        print(f"\n⚠ Interrupted early, completed {completed}/{total}")
    else:
        print(f"\n✓ All done: {completed}/{total}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                    ANALYSIS & REPORT                         ║
# ╚══════════════════════════════════════════════════════════════╝

def _sigmoid(x, k, x0):
    """Sigmoid: P = 1 / (1 + exp(-k*(x-x0))), overflow-safe."""
    z = -k * (x - x0)
    if z > 500:
        return 0.0
    if z < -500:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))

def _fit_sigmoid(xs, ys):
    """Simple grid search to fit sigmoid parameters k and x0."""
    best_err = float('inf')
    best_k, best_x0 = 1.0, 0.5
    x_range = max(xs) - min(xs) if len(xs) > 1 else 1
    for k in [v * 0.5 for v in range(1, 41)]:  # k: 0.5 to 20
        for x0 in [min(xs) + x_range * i / 20 for i in range(21)]:
            err = sum((y - _sigmoid(x, k, x0))**2 for x, y in zip(xs, ys))
            if err < best_err:
                best_err = err
                best_k, best_x0 = k, x0
    return best_k, best_x0, best_err


def analyze_exp1c():
    csv_path = Path(CONFIG["output_dir"]) / "exp1c_raw.csv"
    if not csv_path.exists():
        print("✗ Results file not found")
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("subcat"):
                rows.append(r)

    print(f"\n{'='*70}")
    print("EXP 1c — ADVANTAGE GRADIENT ANALYSIS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"Total rows: {len(rows)}")

    success = [r for r in rows if r["is_parse_fail"] == "False" and r["is_refusal"] == "False"]
    pf = [r for r in rows if r["is_parse_fail"] == "True"]
    print(f"Success: {len(success)}  ParseFail: {len(pf)}  Refusal: {len([r for r in rows if r['is_refusal'] == 'True'])}")

    for dim_name, dim_info in DIMENSIONS.items():
        print(f"\n{'─'*70}")
        print(f"DIMENSION: {dim_info['label_en']} ({dim_info['label_zh']})")
        print(f"{'─'*70}")
        print(f"{'Level':<12} {'Desc':<20} {'N':>5} {'FakeWins':>9} {'Rate':>7}")
        print("─" * 58)

        dim_rows = [r for r in success if r["dimension"] == dim_name]
        curve_x, curve_y = [], []

        for lvl in dim_info["levels"]:
            lvl_rows = [r for r in dim_rows if r["level_id"] == lvl["id"]]
            n = len(lvl_rows)
            if n == 0:
                continue
            fake_wins = sum(1 for r in lvl_rows if r["chose_fictional"] == "True")
            rate = fake_wins / n
            print(f"{lvl['id']:<12} {lvl['desc']:<20} {n:>5} {fake_wins:>9} {rate:>7.1%}")
            curve_x.append(float(lvl["value"]))
            curve_y.append(rate)

        # Sigmoid fit
        if len(curve_x) >= 3 and max(curve_y) > 0.1:
            k, x0, err = _fit_sigmoid(curve_x, curve_y)
            print(f"\n  Sigmoid fit: k={k:.2f}, x0={x0:.3f} (breakthrough threshold)")
            print(f"  Fit error (SSE): {err:.4f}")
            if x0 <= max(curve_x):
                print(f"  → Fictional brand reaches 50% win rate at advantage level ≈ {x0:.2f}")
            else:
                print(f"  → Fictional brand does NOT reach 50% within tested range")
        else:
            print(f"\n  (Insufficient variation for sigmoid fit)")

        # Breakdown by model
        print(f"\n  By model × language:")
        print(f"  {'Model':<16} {'Lang':<6} ", end="")
        for lvl in dim_info["levels"]:
            print(f"{lvl['id']:>8}", end="")
        print()
        print("  " + "─" * 60)
        for model in sorted(set(r["model"] for r in rows)):
            for lang in ["en", "zh"]:
                print(f"  {model:<16} {lang:<6} ", end="")
                for lvl in dim_info["levels"]:
                    cond = [r for r in dim_rows
                            if r["model"] == model and r["lang"] == lang
                            and r["level_id"] == lvl["id"]]
                    if cond:
                        fw = sum(1 for r in cond if r["chose_fictional"] == "True")
                        print(f"{fw/len(cond):>7.0%} ", end="")
                    else:
                        print(f"{'—':>8}", end="")
                print()

    # ── Cross-dimension comparison ──
    print(f"\n{'─'*70}")
    print("CROSS-DIMENSION SUMMARY")
    print(f"{'─'*70}")
    print(f"{'Dimension':<16} {'L0 (equal)':>10} {'L1':>10} {'L2':>10} {'L3':>10} {'L4':>10}")
    print("─" * 70)
    for dim_name, dim_info in DIMENSIONS.items():
        dim_rows = [r for r in success if r["dimension"] == dim_name]
        vals = []
        for lvl in dim_info["levels"]:
            cond = [r for r in dim_rows if r["level_id"] == lvl["id"]]
            if cond:
                fw = sum(1 for r in cond if r["chose_fictional"] == "True")
                vals.append(f"{fw/len(cond):>9.1%}")
            else:
                vals.append(f"{'—':>10}")
        print(f"{dim_name:<16} {'  '.join(vals)}")

    # Parse failures
    print(f"\n{'─'*70}")
    print("PARSE FAILURE BREAKDOWN")
    print(f"Total: {len(pf)} / {len(rows)} ({len(pf)/len(rows)*100:.1f}%)")
    for model in sorted(set(r["model"] for r in rows)):
        mpf = [r for r in pf if r["model"] == model]
        mtot = len([r for r in rows if r["model"] == model])
        if mpf:
            print(f"  {model}: {len(mpf)}/{mtot} ({len(mpf)/mtot*100:.1f}%)")

    # Save
    report_path = Path(CONFIG["output_dir"]) / "exp1c_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"EXP 1c — Advantage Gradient Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total rows: {len(rows)}, Success: {len(success)}\n")
    print(f"\n✓ Report saved to {report_path}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                          MAIN                               ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="Exp 1c: Advantage Gradient")
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
            shutil.rmtree(out)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Exp 1c: Multi-Dimensional Advantage Gradient         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Dry run:  {CONFIG['dry_run']}")
    print(f"  Repeats:  {args.repeats}")
    print(f"  Models:   {', '.join(n for n, c in CONFIG['models'].items() if c['enabled'])}")
    print(f"  Output:   {CONFIG['output_dir']}")

    if not args.analyze_only:
        print("\n── Loading fictional brands ──")
        fictional_brands = load_fictional_brands()
        print(f"\n── Running Exp 1c ({args.repeats} reps) ──")
        run_exp1c(fictional_brands, args.repeats)

    print("\n── Analyzing results ──")
    analyze_exp1c()
    print("\n✓ Exp 1c complete!")


if __name__ == "__main__":
    main()
