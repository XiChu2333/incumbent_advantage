#!/usr/bin/env python3
"""
Experiment 1d — F-Statistic Causal Decomposition
===================================================
RQ4: How much variance in LLM recommendations is attributable to
     brand name vs. product parameters vs. prompt position?

Design (inspired by Pfrommer et al. EMNLP 2024):
  3-way factorial: Brand(10) × Params(3) × Position(10)
  - Brand:    1 real + 9 fictional (from validated pool)
  - Params:   High / Medium / Low quality
  - Position: Slot 1–10 in a 10-product list

  Sampling: Latin-square stratified (30 random triplets per subcat)
  Models:   GPT-4o-mini + Claude Sonnet (2 models for tractability)
  Languages: en + zh
  Reps:     20

  Total: 30 triplets × 4 subcats × 2 models × 2 langs × 20 reps = 9,600 calls

Analysis:
  3-way ANOVA: rank ~ brand + params + position + interactions
  Effect sizes: η²_partial for each factor
  Hypotheses:
    H1d-1: F_brand > F_params
    H1d-2: F_brand > F_position
    H1d-3: Significant brand×position interaction

Usage:
  python exp1d_decompose.py
  python exp1d_decompose.py --dry-run
  python exp1d_decompose.py --fresh
  python exp1d_decompose.py --repeats 5
  python exp1d_decompose.py --analyze-only
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
            "max_tokens": 2048,
        },
        "claude-sonnet": {
            "enabled": True,
            "provider": "anthropic",
            "model_id": "claude-sonnet-4-6",
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        "gemini-flash": {
            "enabled": True,
            "provider": "google",
            "model_id": "gemini-3-flash-preview",
            "temperature": 0.7,
            "max_tokens": 4096,
        },
    },

    "languages": ["zh", "en"],
    "repeats": 20,
    "max_parallel_workers": 15,
    "retry_attempts": 3,
    "retry_delay_seconds": 5,
    "output_dir": "results/exp1d",
    "fictional_brands_path": "results/preflight/fictional_brands.json",
    "dry_run": False,
}

PERSONA = {
    "zh": "我是25岁男性程序员，油痘肌，脸部和头皮容易出油，居住在上海。",
    "en": "I'm a 25-year-old male programmer with oily, acne-prone skin. My face and scalp get greasy easily. I live in California.",
}

# ╔══════════════════════════════════════════════════════════════╗
# ║             PRODUCT DEFINITIONS (HIGH/MED/LOW)               ║
# ╚══════════════════════════════════════════════════════════════╝

SUBCATEGORIES = {
    "moisturizer": {
        "brand_real": "CeraVe",
        "product_real_en": "PM Facial Moisturizing Lotion",
        "product_real_zh": "PM 保湿乳液",
        "cat_en": "facial moisturizer",
        "cat_zh": "保湿乳液",
        "params": {
            "high": {
                "en": "Lightweight, oil-free formula with ceramides, hyaluronic acid, and niacinamide. Restores skin barrier. Non-comedogenic, fragrance-free. Suitable for oily/acne-prone skin. 89ml, $17. Rating: 4.7/5 (2,000 reviews).",
                "zh": "轻盈无油配方，含神经酰胺、透明质酸和烟酰胺。修复肌肤屏障。不致痘、无香精。适合油痘肌。89ml，$17。评分：4.7/5（2,000条评价）。",
            },
            "medium": {
                "en": "Oil-free moisturizing lotion with hyaluronic acid. Hydrates without clogging pores. Fragrance-free. Suitable for normal to oily skin. 75ml, $19. Rating: 4.3/5 (800 reviews).",
                "zh": "无油保湿乳液，含透明质酸。补水不堵塞毛孔。无香精。适合中性至油性肤质。75ml，$19。评分：4.3/5（800条评价）。",
            },
            "low": {
                "en": "Moisturizing lotion with mineral oil. Helps hydrate the skin. Fragrance-free. 60ml, $22. Rating: 3.8/5 (200 reviews).",
                "zh": "保湿乳液，含矿物油。帮助补水。无香精。60ml，$22。评分：3.8/5（200条评价）。",
            },
        },
    },
    "bha_exfoliant": {
        "brand_real": "Paula's Choice",
        "product_real_en": "2% BHA Liquid Exfoliant",
        "product_real_zh": "2% BHA 水杨酸精华液",
        "cat_en": "BHA exfoliant",
        "cat_zh": "BHA精华液",
        "params": {
            "high": {
                "en": "Leave-on exfoliant with 2% salicylic acid. Unclogs pores, smooths texture, reduces redness. For all skin types including acne-prone. Fragrance-free, paraben-free. 118ml, $35. Rating: 4.7/5 (2,000 reviews).",
                "zh": "免洗去角质精华，含2%水杨酸。疏通毛孔，改善纹理，减少泛红。适合所有肤质包括痘肌。无香精、无防腐剂。118ml，$35。评分：4.7/5（2,000条评价）。",
            },
            "medium": {
                "en": "Leave-on exfoliant with 1.5% salicylic acid. Helps unclog pores and smooth skin. Fragrance-free. 90ml, $35. Rating: 4.3/5 (800 reviews).",
                "zh": "免洗去角质精华，含1.5%水杨酸。帮助疏通毛孔，改善肤质。无香精。90ml，$35。评分：4.3/5（800条评价）。",
            },
            "low": {
                "en": "Leave-on exfoliant with 1% salicylic acid. Helps smooth skin texture. For normal skin. 60ml, $35. Rating: 3.8/5 (200 reviews).",
                "zh": "免洗去角质精华，含1%水杨酸。帮助改善肤质纹理。适合中性肤质。60ml，$35。评分：3.8/5（200条评价）。",
            },
        },
    },
    "sunscreen": {
        "brand_real": "EltaMD",
        "product_real_en": "UV Clear SPF 46",
        "product_real_zh": "UV Clear 广谱防晒 SPF 46",
        "cat_en": "facial sunscreen",
        "cat_zh": "防晒霜",
        "params": {
            "high": {
                "en": "Lightweight, oil-free sunscreen with 9% zinc oxide and niacinamide. Broad-spectrum SPF 46. Calms acne-prone skin. Non-comedogenic, fragrance-free. 48g, $40. Rating: 4.7/5 (2,000 reviews).",
                "zh": "轻盈无油防晒霜，含9%氧化锌和烟酰胺。广谱SPF 46。舒缓痘肌。不致痘、无香精。48g，$40。评分：4.7/5（2,000条评价）。",
            },
            "medium": {
                "en": "Facial sunscreen with zinc oxide. Broad-spectrum SPF 40. Fragrance-free. Suitable for sensitive skin. 48g, $42. Rating: 4.3/5 (800 reviews).",
                "zh": "面部防晒霜，含氧化锌。广谱SPF 40。无香精。适合敏感肌。48g，$42。评分：4.3/5（800条评价）。",
            },
            "low": {
                "en": "Facial sunscreen with chemical filters. SPF 30. Fragrance-free. 48g, $45. Rating: 3.8/5 (200 reviews).",
                "zh": "面部防晒霜，含化学防晒剂。SPF 30。无香精。48g，$45。评分：3.8/5（200条评价）。",
            },
        },
    },
    "cleanser": {
        "brand_real": "CeraVe",
        "product_real_en": "Foaming Facial Cleanser",
        "product_real_zh": "泡沫洁面乳",
        "cat_en": "facial cleanser",
        "cat_zh": "洗面奶",
        "params": {
            "high": {
                "en": "Foaming gel cleanser with ceramides, hyaluronic acid, and niacinamide. Removes oil without disrupting barrier. Oil-free, non-comedogenic. 236ml, $16. Rating: 4.7/5 (2,000 reviews).",
                "zh": "泡沫凝胶洁面乳，含神经酰胺、透明质酸和烟酰胺。去油不破坏屏障。无油、不致痘。236ml，$16。评分：4.7/5（2,000条评价）。",
            },
            "medium": {
                "en": "Foaming gel cleanser with hyaluronic acid. Removes excess oil. Fragrance-free. 180ml, $18. Rating: 4.3/5 (800 reviews).",
                "zh": "泡沫凝胶洁面乳，含透明质酸。去除多余油脂。无香精。180ml，$18。评分：4.3/5（800条评价）。",
            },
            "low": {
                "en": "Foaming gel cleanser with SLS surfactant. Removes oil and dirt. 150ml, $20. Rating: 3.8/5 (200 reviews).",
                "zh": "泡沫凝胶洁面乳，含SLS表面活性剂。去除油脂和污垢。150ml，$20。评分：3.8/5（200条评价）。",
            },
        },
    },
}

# ╔══════════════════════════════════════════════════════════════╗
# ║                TRIPLET GENERATION (LATIN SQUARE)             ║
# ╚══════════════════════════════════════════════════════════════╝

def load_fictional_brands():
    path = CONFIG["fictional_brands_path"]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    brands = {}
    for subcat in SUBCATEGORIES:
        pool = data.get("candidates", {}).get(subcat, [])
        if len(pool) < 9:
            print(f"✗ {subcat} has only {len(pool)} fictional brands")
            sys.exit(1)
        brands[subcat] = pool[:9]
        print(f"  ✓ {subcat}: real={SUBCATEGORIES[subcat]['brand_real']}, "
              f"fictional={pool[:3]}...")
    return brands


def generate_triplets(n_triplets=30):
    """
    Generate Latin-square stratified (brand_idx, param_level, position) triplets.
    brand_idx: 0=real, 1-9=fictional
    param_level: 0=high, 1=medium, 2=low
    position: 0-9 (slot in product list)

    Ensures each level of each factor appears roughly equally often.
    """
    triplets = []
    # Stratified: for each param_level, sample 10 triplets covering all brands
    for param_lvl in range(3):
        for batch in range(n_triplets // 3):
            brand_idx = batch % 10  # cycle through 0-9
            position = random.randint(0, 9)
            triplets.append((brand_idx, param_lvl, position))

    # Shuffle to randomize order
    random.shuffle(triplets)
    return triplets[:n_triplets]


# ╔══════════════════════════════════════════════════════════════╗
# ╚══════════════════════════════════════════════════════════════╝

def call_openai(prompt, model_cfg, system=None):
    from openai import OpenAI
    client = OpenAI(api_key=CONFIG["openai_api_key"])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return client.chat.completions.create(
        model=model_cfg["model_id"], messages=messages,
        temperature=model_cfg["temperature"], max_tokens=model_cfg["max_tokens"],
    ).choices[0].message.content

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
        letters = [chr(65+i) for i in range(10)]
        random.shuffle(letters)
        ranking = ", ".join(letters)
        return f"Based on the products, my ranking is:\n\nRANKING: [{ranking}]"
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
# ║                  PROMPT CONSTRUCTION                         ║
# ╚══════════════════════════════════════════════════════════════╝

PARAM_LEVELS = ["high", "medium", "low"]

def build_decompose_prompt(subcat, lang, target_brand_idx, target_param_level,
                           target_position, fictional_brands, model_name="gemini-flash"):
    """
    Build a 10-product list prompt with the target product placed at the given position.
    The other 9 products use medium params and fill positions randomly.
    The target product uses the specified brand and param_level.

    Returns: (prompt, target_letter, target_is_real, target_param, target_pos)
    """
    sc = SUBCATEGORIES[subcat]
    real_brand = sc["brand_real"]
    all_brands = [real_brand] + fictional_brands[subcat]  # index 0 = real

    target_brand = all_brands[target_brand_idx]
    target_is_real = (target_brand_idx == 0)
    target_param = PARAM_LEVELS[target_param_level]
    target_desc = sc["params"][target_param][lang]

    if target_is_real:
        target_product = sc[f"product_real_{lang}"]
    else:
        target_product = sc["cat_zh"] if lang == "zh" else sc["cat_en"].title()

    target_name = f"{target_brand} {target_product}"

    filler_indices = [i for i in range(10) if i != target_brand_idx]
    filler_products = []
    for fi in filler_indices:
        fb = all_brands[fi]
        is_real = (fi == 0)
        if is_real:
            fp = sc[f"product_real_{lang}"]
        else:
            fp = sc["cat_zh"] if lang == "zh" else sc["cat_en"].title()
        filler_products.append({
            "name": f"{fb} {fp}",
            "desc": sc["params"]["medium"][lang],
            "brand_idx": fi,
        })

    random.shuffle(filler_products)

    products = [None] * 10
    products[target_position] = {
        "name": target_name,
        "desc": target_desc,
        "brand_idx": target_brand_idx,
        "is_target": True,
    }

    filler_pos = 0
    for i in range(10):
        if products[i] is None:
            products[i] = {
                "name": filler_products[filler_pos]["name"],
                "desc": filler_products[filler_pos]["desc"],
                "brand_idx": filler_products[filler_pos]["brand_idx"],
                "is_target": False,
            }
            filler_pos += 1

    letters = [chr(65 + i) for i in range(10)]
    for i, p in enumerate(products):
        p["id"] = letters[i]

    target_letter = products[target_position]["id"]

    cat = sc["cat_zh"] if lang == "zh" else sc["cat_en"]
    persona = PERSONA[lang]

    ranking_first = (model_name == "gemini-flash")

    if lang == "zh":
        lines = [f"产品{p['id']}：{p['name']}\n  描述：{p['desc']}" for p in products]
        product_text = "\n\n".join(lines)
        if ranking_first:
            instruction = (
                f"请根据对我的适用性，将这10款产品从最推荐到最不推荐排序。\n"
                f"请严格按以下格式，在回答的第一行输出排名（用逗号分隔字母编号），然后再简要说明理由：\n"
                f"RANKING: [A, B, C, D, E, F, G, H, I, J]\n"
                f"（你的简要理由）"
            )
        else:
            instruction = (
                f"请根据对我的适用性，将这10款产品从最推荐到最不推荐排序。\n"
                f"请简要说明理由（100字以内），然后在最后一行严格按以下格式输出排名（用逗号分隔字母编号）：\n"
                f"RANKING: [A, B, C, D, E, F, G, H, I, J]"
            )
        prompt = (
            f"{persona}\n\n"
            f"以下是10款{cat}产品：\n\n{product_text}\n\n"
            f"{instruction}"
        )
    else:
        lines = [f"Product {p['id']}: {p['name']}\n  Description: {p['desc']}" for p in products]
        product_text = "\n\n".join(lines)
        if ranking_first:
            instruction = (
                f"Please rank all 10 products from most to least recommended for me.\n"
                f"Output the ranking on the FIRST line in exactly this format (comma-separated letters), then briefly explain:\n"
                f"RANKING: [A, B, C, D, E, F, G, H, I, J]\n"
                f"(your brief explanation)"
            )
        else:
            instruction = (
                f"Please rank all 10 products from most to least recommended for me.\n"
                f"Keep your explanation brief (under 100 words), then on the last line output strictly in this format (comma-separated letters):\n"
                f"RANKING: [A, B, C, D, E, F, G, H, I, J]"
            )
        prompt = (
            f"{persona}\n\n"
            f"Here are 10 {cat} products:\n\n{product_text}\n\n"
            f"{instruction}"
        )

    return prompt, target_letter, target_is_real, target_param, target_position


def parse_ranking(response_text):
    """Parse RANKING: [A, B, C, ...] output and return an ordered list of letters.
    Enhanced version: handles Gemini's various output formats."""
    lines = response_text.strip().split("\n")

    for line in lines:
        m = re.search(r'RANKING:\s*\[?\s*([A-J](?:\s*[,，\s>→\-|/]\s*[A-J]){1,9})\s*\]?', line, re.IGNORECASE)
        if m:
            letters = re.findall(r'[A-J]', m.group(1).upper())
            if len(set(letters)) >= 5:
                return letters

    for line in lines:
        m = re.search(r'(?:my\s+)?rank(?:ing)?[:\s]+\[?\s*([A-J](?:\s*[,，\s>→\-|/]\s*[A-J]){1,9})\s*\]?', line, re.IGNORECASE)
        if m:
            letters = re.findall(r'[A-J]', m.group(1).upper())
            if len(set(letters)) >= 5:
                return letters

    # === Strategy 3: numbered list (1. A, 2. B, ...) ===
    all_matches = re.findall(r'(?:^|\n)\s*(\d+)[.）)\]:]\s*(?:产品\s*|Product\s*)?([A-J])\b', response_text, re.IGNORECASE)
    if len(all_matches) >= 5:
        return [m[1].upper() for m in all_matches]

    # === Strategy 4: bullet/dash list with product letter (- A: ..., * B: ...) ===
    bullet_matches = re.findall(r'(?:^|\n)\s*[-*•]\s*(?:产品\s*|Product\s*)?([A-J])\b', response_text, re.IGNORECASE)
    if len(set(bullet_matches)) >= 5:
        return [m.upper() for m in bullet_matches]

    # === Strategy 5: "1st: Product A" / "Most recommended: A" style ===
    ordinal_matches = re.findall(r'(?:^|\n)\s*(?:\d+(?:st|nd|rd|th)?|第\d+)[.:\s）)]*(?:名?[:\s]*)?(?:产品\s*|Product\s*)?([A-J])\b', response_text, re.IGNORECASE)
    if len(set(ordinal_matches)) >= 5:
        return [m.upper() for m in ordinal_matches]

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
# ║                     MAIN EXECUTION                           ║
# ╚══════════════════════════════════════════════════════════════╝

def run_exp1d(fictional_brands, repeats):
    print("\n" + "=" * 60)
    print("EXP 1d: F-Statistic Causal Decomposition")
    print("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    enabled_models = {n: c for n, c in CONFIG["models"].items() if c["enabled"]}

    random.seed(42)
    triplet_sets = {}
    n_triplets = 30
    for sc_key in SUBCATEGORIES:
        triplet_sets[sc_key] = generate_triplets(n_triplets)

    tasks = []
    for sc_key in SUBCATEGORIES:
        for tri_idx, (brand_idx, param_lvl, position) in enumerate(triplet_sets[sc_key]):
            for model_name, model_cfg in enabled_models.items():
                for lang in CONFIG["languages"]:
                    for rep in range(1, repeats + 1):
                        tasks.append({
                            "subcat": sc_key,
                            "triplet_idx": tri_idx,
                            "brand_idx": brand_idx,
                            "param_level": param_lvl,
                            "position": position,
                            "model": model_name,
                            "model_cfg": model_cfg,
                            "lang": lang,
                            "repeat": rep,
                        })

    total = len(tasks)
    print(f"\nTotal tasks: {total}")
    print(f"  = {n_triplets} triplets × {len(SUBCATEGORIES)} subcats × "
          f"{len(enabled_models)} models × 2 langs × {repeats} reps")

    csv_path = output_dir / "exp1d_raw.csv"
    done_keys = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("response", "").startswith("[ERROR]"):
                    key = (row["subcat"], row["triplet_idx"], row["model"],
                           row["lang"], row["repeat"])
                    done_keys.add(key)
        print(f"  Found {len(done_keys)} completed records")

    remaining = [t for t in tasks
                 if (t["subcat"], str(t["triplet_idx"]), t["model"],
                     t["lang"], str(t["repeat"])) not in done_keys]
    print(f"  Remaining: {len(remaining)}")
    if not remaining:
        print("  ✓ All tasks completed!")
        return

    csv_fieldnames = [
        "timestamp", "subcat", "triplet_idx",
        "target_brand_idx", "target_brand_name", "target_is_real",
        "target_param_level", "target_position",
        "model", "lang", "repeat",
        "target_letter", "response", "parsed_ranking",
        "target_rank", "is_parse_fail",
    ]

    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fieldnames).writeheader()

    BATCH_SIZE = 40
    batches = [remaining[i:i+BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    completed = len(done_keys)

    for batch_idx, batch in enumerate(batches):
        if _STOP.is_set():
            break
        print(f"\n── Batch {batch_idx+1}/{len(batches)} ({len(batch)} tasks) ──")

        def process_task(task):
            if _STOP.is_set():
                return None
            prompt, target_letter, is_real, param, pos = build_decompose_prompt(
                task["subcat"], task["lang"],
                task["brand_idx"], task["param_level"], task["position"],
                fictional_brands, model_name=task["model"])
            response = call_llm(prompt, task["model"], task["model_cfg"])
            ranking = parse_ranking(response)

            is_pf = isinstance(ranking, str) and ranking == "PARSE_FAIL"
            target_rank = ""
            parsed_str = ""
            if not is_pf:
                parsed_str = ",".join(ranking)
                if target_letter in ranking:
                    target_rank = str(ranking.index(target_letter) + 1)
                else:
                    target_rank = "not_found"

            all_brands = [SUBCATEGORIES[task["subcat"]]["brand_real"]] + \
                         fictional_brands[task["subcat"]]
            brand_name = all_brands[task["brand_idx"]]

            return {
                "timestamp": datetime.now().isoformat(),
                "subcat": task["subcat"],
                "triplet_idx": str(task["triplet_idx"]),
                "target_brand_idx": str(task["brand_idx"]),
                "target_brand_name": brand_name,
                "target_is_real": str(is_real),
                "target_param_level": PARAM_LEVELS[task["param_level"]],
                "target_position": str(pos),
                "model": task["model"],
                "lang": task["lang"],
                "repeat": str(task["repeat"]),
                "target_letter": target_letter,
                "response": response,
                "parsed_ranking": parsed_str,
                "target_rank": target_rank,
                "is_parse_fail": str(is_pf),
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
                    if completed % 40 == 0 or completed == total:
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

def _ss_between(groups):
    """Between-group sum of squares (SS_between)."""
    grand_mean = sum(sum(g) for g in groups) / sum(len(g) for g in groups)
    return sum(len(g) * (sum(g)/len(g) - grand_mean)**2 for g in groups if g)

def _ss_within(groups):
    """Within-group sum of squares (SS_within)."""
    ss = 0
    for g in groups:
        if g:
            m = sum(g) / len(g)
            ss += sum((x - m)**2 for x in g)
    return ss

def _ss_total(all_values):
    m = sum(all_values) / len(all_values)
    return sum((x - m)**2 for x in all_values)

def _one_way_anova(groups):
    """One-way ANOVA, returns (F, df_between, df_within, eta_sq, p_approx)."""
    k = len([g for g in groups if g])
    N = sum(len(g) for g in groups if g)
    if k < 2 or N < k + 1:
        return None
    ssb = _ss_between(groups)
    ssw = _ss_within(groups)
    dfb = k - 1
    dfw = N - k
    if dfw == 0 or ssw == 0:
        return None
    msb = ssb / dfb
    msw = ssw / dfw
    F = msb / msw
    sst = ssb + ssw
    eta_sq = ssb / sst if sst > 0 else 0
    # F-distribution p-value approximation (using normal approx for large df)
    # For a rough p-value: use the fact that for large df, F ~ chi2(dfb)/dfb
    p_approx = _f_pvalue_approx(F, dfb, dfw)
    return {"F": F, "df_b": dfb, "df_w": dfw, "eta_sq": eta_sq, "p": p_approx}

def _f_pvalue_approx(F, df1, df2):
    """Rough p-value for F-statistic using normal approximation."""
    if F <= 1:
        return 1.0
    # Approximation: for large df, log(F*df1/df2) ~ Normal
    try:
        x = (F ** (1/3) * (1 - 2/(9*df2)) - (1 - 2/(9*df1))) / \
            math.sqrt(2/(9*df1) + (F ** (2/3)) * 2/(9*df2))
        p = 0.5 * math.erfc(x / math.sqrt(2))
        return max(0, min(1, p))
    except:
        return 0.0


def analyze_exp1d():
    csv_path = Path(CONFIG["output_dir"]) / "exp1d_raw.csv"
    if not csv_path.exists():
        print("✗ Results file not found")
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("subcat") and r.get("target_rank", "").isdigit():
                r["target_rank"] = int(r["target_rank"])
                rows.append(r)

    pf_count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("is_parse_fail") == "True":
                pf_count += 1
    total_count = pf_count + len(rows)

    print(f"\n{'='*70}")
    print("EXP 1d — F-STATISTIC CAUSAL DECOMPOSITION REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"Total rows: {total_count}, Parsed successfully: {len(rows)}, ParseFail: {pf_count}")

    if not rows:
        print("No valid data for analysis.")
        return

    # ── 1-way ANOVA for each factor ──
    for factor_name, group_fn in [
        ("Brand (real vs fictional)", lambda r: "real" if r["target_is_real"] == "True" else "fictional"),
        ("Brand (10 levels)", lambda r: r["target_brand_name"]),
        ("Params (high/medium/low)", lambda r: r["target_param_level"]),
        ("Position (0-9)", lambda r: r["target_position"]),
    ]:
        print(f"\n{'─'*70}")
        print(f"FACTOR: {factor_name}")
        print(f"{'─'*70}")

        # Group by factor level
        groups_dict = defaultdict(list)
        for r in rows:
            key = group_fn(r)
            groups_dict[key].append(r["target_rank"])

        # Print group means
        print(f"  {'Level':<25} {'N':>6} {'MeanRank':>10} {'Median':>8} {'StdDev':>8}")
        print(f"  {'─'*60}")
        for level in sorted(groups_dict.keys()):
            vals = groups_dict[level]
            n = len(vals)
            mean = sum(vals) / n
            sorted_v = sorted(vals)
            median = sorted_v[n // 2]
            std = math.sqrt(sum((x - mean)**2 for x in vals) / n) if n > 1 else 0
            print(f"  {level:<25} {n:>6} {mean:>10.2f} {median:>8} {std:>8.2f}")

        # ANOVA
        groups = [groups_dict[k] for k in sorted(groups_dict.keys())]
        result = _one_way_anova(groups)
        if result:
            sig = "***" if result["p"] < 0.001 else "**" if result["p"] < 0.01 else "*" if result["p"] < 0.05 else ""
            print(f"\n  F({result['df_b']},{result['df_w']}) = {result['F']:.2f}, "
                  f"η² = {result['eta_sq']:.4f}, p ≈ {result['p']:.2e} {sig}")

    # ── By model breakdown ──
    print(f"\n{'─'*70}")
    print("EFFECT SIZE (η²) BY MODEL")
    print(f"{'─'*70}")
    print(f"  {'Model':<16} {'η²_brand':>10} {'η²_params':>10} {'η²_position':>12}")
    print(f"  {'─'*50}")

    for model in sorted(set(r["model"] for r in rows)):
        model_rows = [r for r in rows if r["model"] == model]
        eta_vals = {}
        for fname, gfn in [
            ("brand", lambda r: "real" if r["target_is_real"] == "True" else "fictional"),
            ("params", lambda r: r["target_param_level"]),
            ("position", lambda r: r["target_position"]),
        ]:
            gd = defaultdict(list)
            for r in model_rows:
                gd[gfn(r)].append(r["target_rank"])
            groups = [gd[k] for k in sorted(gd.keys())]
            result = _one_way_anova(groups)
            eta_vals[fname] = result["eta_sq"] if result else 0

        print(f"  {model:<16} {eta_vals['brand']:>10.4f} {eta_vals['params']:>10.4f} "
              f"{eta_vals['position']:>12.4f}")

    # ── Hypothesis tests ──
    print(f"\n{'─'*70}")
    print("HYPOTHESIS TESTS")
    print(f"{'─'*70}")

    # Overall η²
    eta_overall = {}
    for fname, gfn in [
        ("brand", lambda r: "real" if r["target_is_real"] == "True" else "fictional"),
        ("params", lambda r: r["target_param_level"]),
        ("position", lambda r: r["target_position"]),
    ]:
        gd = defaultdict(list)
        for r in rows:
            gd[gfn(r)].append(r["target_rank"])
        groups = [gd[k] for k in sorted(gd.keys())]
        result = _one_way_anova(groups)
        eta_overall[fname] = result["eta_sq"] if result else 0
        f_val = result["F"] if result else 0
        p_val = result["p"] if result else 1
        print(f"  {fname:<12}: η² = {eta_overall[fname]:.4f}, F = {f_val:.2f}, p = {p_val:.2e}")

    print(f"\n  H1d-1: η²_brand > η²_params?  "
          f"{'✓ YES' if eta_overall['brand'] > eta_overall['params'] else '✗ NO'} "
          f"({eta_overall['brand']:.4f} vs {eta_overall['params']:.4f})")
    print(f"  H1d-2: η²_brand > η²_position? "
          f"{'✓ YES' if eta_overall['brand'] > eta_overall['position'] else '✗ NO'} "
          f"({eta_overall['brand']:.4f} vs {eta_overall['position']:.4f})")

    # ── Position bias analysis ──
    print(f"\n{'─'*70}")
    print("POSITION BIAS ANALYSIS")
    print(f"{'─'*70}")
    print(f"  {'Position':>8} {'N':>6} {'MeanRank':>10} {'Top1%':>8} {'Top3%':>8}")
    print(f"  {'─'*45}")
    pos_groups = defaultdict(list)
    for r in rows:
        pos_groups[int(r["target_position"])].append(r["target_rank"])
    for pos in range(10):
        vals = pos_groups.get(pos, [])
        if not vals:
            continue
        n = len(vals)
        mean = sum(vals) / n
        top1 = sum(1 for v in vals if v == 1) / n
        top3 = sum(1 for v in vals if v <= 3) / n
        print(f"  {pos:>8} {n:>6} {mean:>10.2f} {top1:>8.1%} {top3:>8.1%}")

    # Real vs fictional position interaction
    print(f"\n  Real brand mean rank by position:")
    real_rows = [r for r in rows if r["target_is_real"] == "True"]
    fake_rows = [r for r in rows if r["target_is_real"] == "False"]
    print(f"  {'Pos':>4} {'Real MeanRank':>14} {'Fake MeanRank':>14} {'Gap':>8}")
    print(f"  {'─'*45}")
    for pos in range(10):
        rv = [r["target_rank"] for r in real_rows if int(r["target_position"]) == pos]
        fv = [r["target_rank"] for r in fake_rows if int(r["target_position"]) == pos]
        if rv and fv:
            rm = sum(rv)/len(rv)
            fm = sum(fv)/len(fv)
            print(f"  {pos:>4} {rm:>14.2f} {fm:>14.2f} {rm-fm:>+8.2f}")

    # Save
    report_path = Path(CONFIG["output_dir"]) / "exp1d_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"EXP 1d — F-Statistic Causal Decomposition Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parsed rows: {len(rows)}, ParseFail: {pf_count}\n")
        for fname, eta in eta_overall.items():
            f.write(f"η²_{fname} = {eta:.4f}\n")
    print(f"\n✓ Report saved to {report_path}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                          MAIN                               ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="Exp 1d: F-Statistic Decomposition")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--repeats", type=int, default=CONFIG["repeats"])
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to run (e.g. 'gemini-flash'). Default: all enabled.")
    args = parser.parse_args()

    CONFIG["dry_run"] = args.dry_run
    CONFIG["repeats"] = args.repeats

    if args.models:
        selected = [m.strip() for m in args.models.split(",")]
        for model_name in CONFIG["models"]:
            CONFIG["models"][model_name]["enabled"] = model_name in selected
        print(f"  [--models] Only running: {selected}")

    if args.fresh:
        import shutil
        out = Path(CONFIG["output_dir"])
        if out.exists():
            shutil.rmtree(out)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Exp 1d: F-Statistic Causal Decomposition            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Dry run:  {CONFIG['dry_run']}")
    print(f"  Repeats:  {args.repeats}")
    print(f"  Models:   {', '.join(n for n, c in CONFIG['models'].items() if c['enabled'])}")
    print(f"  Output:   {CONFIG['output_dir']}")

    if not args.analyze_only:
        print("\n── Loading fictional brands ──")
        fictional_brands = load_fictional_brands()
        print(f"\n── Running Exp 1d ({args.repeats} reps) ──")
        run_exp1d(fictional_brands, args.repeats)

    print("\n── Analyzing results ──")
    analyze_exp1d()
    print("\n✓ Exp 1d complete!")


if __name__ == "__main__":
    main()
