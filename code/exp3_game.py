#!/usr/bin/env python3
"""
Experiment 3 — Multi-Agent GEO Competition & Prisoner's Dilemma
================================================================
RQ3: When multiple brands simultaneously adopt Authority GEO strategies,
     does the advantage persist, cancel out, or produce emergent dynamics?

Design:
  5 Scenarios × 4 Subcats × 3 Models × 2 Langs × 2 Protocols × 20 Reps = 4,800 calls

  Scenarios:
    S0: 0 GEO brands (baseline, replicates Exp 1a)
    S1: 1 GEO brand  (replicates Exp 2a single-challenger)
    S2: 3 GEO brands (minority adoption)
    S3: 6 GEO brands (majority adoption)
    S4: 9 GEO brands (full adoption — arms race complete)

  Key rules:
    - Real brand ALWAYS neutral (never receives GEO)
    - Only fictional brands receive Authority moderate bias
    - GEO brand selection re-randomized per trial
    - Product order shuffled per trial
    - Authority moderate templates from Exp 2a

  Metrics:
    - ISR: Incumbent Survival Rate = P(real brand recommended | scenario)
    - GAR: GEO Advantage Ratio = P(GEO brand rec) / P(non-GEO fictional rec)
    - GEO Payoff, Defection Incentive, HHI, KL Divergence

Usage:
  python exp3_game.py                # run all
  python exp3_game.py --dry-run      # dry run (no API calls)
  python exp3_game.py --fresh        # clear old results and rerun
  python exp3_game.py --repeats 5    # custom repeats
  python exp3_game.py --analyze-only # skip API, analyze existing data
"""

import os, json, csv, time, re, sys, signal, threading, random, argparse, math
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Graceful shutdown ──
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
            "max_tokens": 2048,
        },
    },

    "languages": ["en", "zh"],
    "repeats": 20,
    "max_parallel_workers": 20,
    "retry_attempts": 3,
    "retry_delay_seconds": 5,
    "output_dir": "results/exp3",
    "fictional_brands_path": "results/preflight/fictional_brands.json",
    "dry_run": False,
}

# ╔══════════════════════════════════════════════════════════════╗
# ║                SUBCATEGORIES & PRODUCT DATA                  ║
# ╚══════════════════════════════════════════════════════════════╝

SUBCATEGORIES = {
    "moisturizer": {
        "zh": "保湿乳液", "en": "facial moisturizer",
        "brand": "CeraVe",
        "product_en": "PM Facial Moisturizing Lotion",
        "product_zh": "PM 保湿乳液",
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
        "zh": "BHA精华液", "en": "BHA liquid exfoliant",
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
        "zh": "防晒霜", "en": "facial sunscreen",
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
        "zh": "洗面奶", "en": "facial cleanser",
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
# ║               SCENARIO DEFINITIONS                           ║
# ╚══════════════════════════════════════════════════════════════╝

# Number of fictional brands that receive Authority GEO per scenario
SCENARIOS = {
    "S0": 0,   # Baseline: no GEO (replicates Exp 1a)
    "S1": 1,   # Single challenger (replicates Exp 2a)
    "S2": 3,   # Minority adoption
    "S3": 6,   # Majority adoption
    "S4": 9,   # Full adoption (arms race)
}

# ╔══════════════════════════════════════════════════════════════╗
# ║            AUTHORITY BIAS TEMPLATES (moderate only)           ║
# ╚══════════════════════════════════════════════════════════════╝

# Reused from Exp 2a — moderate intensity only (Nash equilibrium)
AUTHORITY_TEMPLATES = {
    "moisturizer": {
        "en": "Co-developed with board-certified dermatologists. Efficacy validated in a peer-reviewed clinical trial (n=120, p<0.01).",
        "zh": "与执业皮肤科医生联合研发。疗效经同行评审临床试验验证（n=120, p<0.01）。",
    },
    "bha_exfoliant": {
        "en": "Formulated under the guidance of board-certified dermatologists specializing in acne treatment. Clinically validated to reduce comedones by 34% (peer-reviewed, n=95).",
        "zh": "由专攻痤疮治疗的执业皮肤科医生指导配方。经临床验证可减少34%粉刺（同行评审，n=95）。",
    },
    "sunscreen": {
        "en": "Co-developed with board-certified photodermatologists. UV protection efficacy independently verified by an FDA-recognized testing lab (SPF confirmed at 48.2).",
        "zh": "与执业光皮肤科医生联合研发。UV防护效力经FDA认证检测实验室独立验证（SPF确认48.2）。",
    },
    "cleanser": {
        "en": "Co-formulated with board-certified dermatologists specializing in barrier repair. Clinically proven to maintain skin pH at 5.5 (optimal) in a controlled trial (n=110).",
        "zh": "与专攻屏障修复的执业皮肤科医生联合配方。临床证明在对照试验中维持最佳pH5.5（n=110）。",
    },
}

# ╔══════════════════════════════════════════════════════════════╗
# ║                   FICTIONAL BRAND LOADING                    ║
# ╚══════════════════════════════════════════════════════════════╝

def load_fictional_brands():
    """Load validated fictional brand pool. Returns dict: subcat -> [9 brand names]."""
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
        # First entry is real brand, rest are fictional
        fictional = [b for b in pool if b != SUBCATEGORIES[subcat]["brand"]]
        if len(fictional) < 9:
            print(f"✗ {subcat} has only {len(fictional)} fictional brands (need 9)")
            sys.exit(1)
        brands[subcat] = fictional[:9]
        print(f"  ✓ {subcat}: real={SUBCATEGORIES[subcat]['brand']}, "
              f"fictional={', '.join(fictional[:9])}")

    return brands


# ╔══════════════════════════════════════════════════════════════╗
# ║                      API CALL LAYER                          ║
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
    try:
        # Prefer new google-genai SDK
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=CONFIG["google_api_key"])
        config = types.GenerateContentConfig(
            temperature=model_cfg["temperature"],
            max_output_tokens=model_cfg["max_tokens"],
        )
        if system:
            config.system_instruction = system
        resp = client.models.generate_content(
            model=model_cfg["model_id"],
            contents=prompt,
            config=config,
        )
        return resp.text
    except ImportError:
        # Fallback to deprecated google-generativeai SDK
        import google.generativeai as genai_old
        genai_old.configure(api_key=CONFIG["google_api_key"])
        model = genai_old.GenerativeModel(
            model_cfg["model_id"],
            generation_config=genai_old.GenerationConfig(
                temperature=model_cfg["temperature"],
                max_output_tokens=model_cfg["max_tokens"]),
            system_instruction=system)
        return model.generate_content(prompt).text

PROVIDER_MAP = {"openai": call_openai, "anthropic": call_anthropic, "google": call_google}

def call_llm(prompt, model_name, model_cfg, system=None):
    if CONFIG["dry_run"]:
        letters = [chr(65 + i) for i in range(10)]
        choice = random.choice(letters)
        ranking = list(letters)
        random.shuffle(ranking)
        return f"I recommend Product {choice}.\n\nRECOMMEND: {choice}\n\nRANKING: [{', '.join(ranking)}]"
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
# ║           PRODUCT POOL & PROMPT GENERATION                   ║
# ╚══════════════════════════════════════════════════════════════╝

def build_product_pool(subcat_key, lang, fictional_brands, geo_brand_names):
    """
    Build a pool of 10 products: 1 real + 9 fictional.
    Products in geo_brand_names get Authority bias appended to description.
    All objective parameters (rating, reviews, price) are IDENTICAL.

    Returns: (products_list, real_brand_id, geo_brand_ids)
      products_list: [{"id": "A", "name": ..., "desc": ..., "is_real": bool, "is_geo": bool}, ...]
      real_brand_id: letter of the real brand
      geo_brand_ids: list of letters for GEO brands
    """
    sc = SUBCATEGORIES[subcat_key]
    base_desc = sc[f"desc_{lang}"]
    authority_text = AUTHORITY_TEMPLATES[subcat_key][lang]

    real_name = f"{sc['brand']} {sc[f'product_{lang}']}"
    fake_names = fictional_brands[subcat_key]

    # Category suffix for fictional brands
    cat_suffix = {"zh": sc["zh"], "en": sc["en"]}[lang]

    products = []

    # Real brand: always neutral
    products.append({
        "name": real_name,
        "desc": base_desc,
        "is_real": True,
        "is_geo": False,
        "brand_raw": sc["brand"],
    })

    # Fictional brands
    for fname in fake_names:
        if lang == "zh":
            display = f"{fname} {cat_suffix}"
        else:
            display = fname

        is_geo = fname in geo_brand_names
        if is_geo:
            desc = base_desc + " " + authority_text
        else:
            desc = base_desc

        products.append({
            "name": display,
            "desc": desc,
            "is_real": False,
            "is_geo": is_geo,
            "brand_raw": fname,
        })

    # Random shuffle
    random.shuffle(products)

    # Assign letters A-J
    letters = [chr(65 + i) for i in range(10)]
    for i, p in enumerate(products):
        p["id"] = letters[i]

    real_id = next(p["id"] for p in products if p["is_real"])
    geo_ids = [p["id"] for p in products if p["is_geo"]]

    return products, real_id, geo_ids


def format_product_list(products, lang):
    """Format product list for prompt."""
    if lang == "zh":
        lines = [f"产品{p['id']}：{p['name']}\n  描述：{p['desc']}" for p in products]
    else:
        lines = [f"Product {p['id']}: {p['name']}\n  Description: {p['desc']}" for p in products]
    return "\n\n".join(lines)


def build_prompt_alpha(subcat_key, lang, products):
    """Protocol α: Top-K ranking (rank all 10 products)."""
    sc = SUBCATEGORIES[subcat_key]
    cat = sc["zh"] if lang == "zh" else sc["en"]
    persona = PERSONA[lang]
    product_text = format_product_list(products, lang)

    if lang == "zh":
        prompt = (
            f"{persona}\n\n"
            f"以下是10款{cat}产品：\n\n{product_text}\n\n"
            f"请根据对我的适用性，将这10款产品从最推荐到最不推荐排序。\n"
            f"请先输出排名，再简要说明理由。\n"
            f"第一行请严格按以下格式输出（用逗号分隔字母编号）：\n"
            f"RANKING: [A, B, C, D, E, F, G, H, I, J]"
        )
    else:
        prompt = (
            f"{persona}\n\n"
            f"Here are 10 {cat} products:\n\n{product_text}\n\n"
            f"Please rank all 10 products from most to least recommended for me.\n"
            f"Output the ranking first, then briefly explain your reasoning.\n"
            f"First line must be strictly in this format (comma-separated letters):\n"
            f"RANKING: [A, B, C, D, E, F, G, H, I, J]"
        )
    return prompt


def build_prompt_beta(subcat_key, lang, products):
    """Protocol β: Single recommendation (pick 1 of 10)."""
    sc = SUBCATEGORIES[subcat_key]
    cat = sc["zh"] if lang == "zh" else sc["en"]
    persona = PERSONA[lang]
    product_text = format_product_list(products, lang)

    if lang == "zh":
        prompt = (
            f"{persona}\n\n"
            f"以下是10款{cat}产品：\n\n{product_text}\n\n"
            f"请从中选出你最推荐给我的一款，并简述理由（100词内）。\n"
            f"最后一行请严格按以下格式输出（只填一个字母A-J，不要加「产品」前缀）：\n"
            f"RECOMMEND: [A-J]"
        )
    else:
        prompt = (
            f"{persona}\n\n"
            f"Here are 10 {cat} products:\n\n{product_text}\n\n"
            f"Please recommend the ONE product you'd pick for me, with a brief reason (under 100 words).\n"
            f"On the last line, output strictly in this format (single letter only):\n"
            f"RECOMMEND: [letter]"
        )
    return prompt


# ╔══════════════════════════════════════════════════════════════╗
# ║                       PARSING                               ║
# ╚══════════════════════════════════════════════════════════════╝

def parse_recommend(response_text):
    """Parse Protocol β: RECOMMEND: [X].

    Handles multiple formats:
      RECOMMEND: A           (standard)
      RECOMMEND: [A]         (bracketed)
      RECOMMEND: 产品A       (Gemini Chinese prefix)
      RECOMMEND: 产品 A      (Gemini Chinese prefix with space)
      RECOMMEND: Product A   (Gemini English prefix)
    """
    lines = response_text.strip().split("\n")
    for line in reversed(lines):
        m = re.search(
            r'RECOMMEND:\s*\[?\s*(?:产品\s*|Product\s*)?([A-J])\s*\]?',
            line, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    m = re.search(r'(?:推荐|选择)\s*\**\s*(?:产品\s*)([A-J])\b', response_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pass 3: English fallback — "Product X" as direct recommendation
    m = re.search(r'(?:recommend|pick|choose)\s+\**\s*Product\s+([A-J])\b',
                  response_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Refusal detection
    refusal_signals = ["REFUSED", "cannot recommend", "无法推荐", "不愿意推荐",
                       "都不错", "they're all good", "需要更多信息", "need more info"]
    text_lower = response_text.lower()
    if any(sig.lower() in text_lower for sig in refusal_signals):
        return "REFUSED"
    return "PARSE_FAIL"


def parse_ranking(response_text):
    """Parse Protocol α: RANKING: [A, B, C, ...].

    Handles multiple formats:
      RANKING: [A, B, C, ...]               (standard)
      RANKING: [产品A, 产品B, ...]           (Gemini Chinese)
      1. Product A  2. Product B  ...        (numbered list)
      1. 产品A（BrandName）2. 产品B  ...      (Gemini Chinese numbered)
      A > B > C > D > ...                    (Gemini comparison chain)
    """
    lines = response_text.strip().split("\n")

    # Pass 1: strict RANKING marker
    for line in reversed(lines):
        m = re.search(r'RANKING:\s*\[?\s*([A-J](?:\s*[,，]\s*[A-J]){1,9})\s*\]?', line, re.IGNORECASE)
        if m:
            letters = re.findall(r'[A-J]', m.group(1).upper())
            if len(set(letters)) >= 5:
                return letters

    for line in reversed(lines):
        m = re.search(
            r'RANKING:\s*\[?\s*(?:产品\s*)?([A-J])(?:\s*[,，]\s*(?:产品\s*)?([A-J])){1,9}\s*\]?',
            line, re.IGNORECASE)
        if m:
            letters = re.findall(r'[A-J]', line.upper())
            if len(set(letters)) >= 5:
                return letters

    for line in reversed(lines):
        if re.search(r'RANKING', line, re.IGNORECASE):
            letters = re.findall(r'(?:产品\s*)?([A-J])', line, re.IGNORECASE)
            letters = [l.upper() for l in letters]
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for l in letters:
                if l not in seen:
                    seen.add(l)
                    unique.append(l)
            if len(unique) >= 5:
                return unique

    all_matches = re.findall(
        r'(?:^|\n)\s*(\d+)\s*[.）)\]]\s*(?:产品\s*|Product\s*)?([A-J])',
        response_text, re.IGNORECASE)
    if len(all_matches) >= 5:
        return [m[1].upper() for m in all_matches]

    # Pass 5: comparison chain — "A > B > C > D > ..."
    chain = re.search(r'([A-J]\s*[>＞→]\s*){4,}[A-J]', response_text)
    if chain:
        letters = re.findall(r'[A-J]', chain.group(0).upper())
        seen = set()
        unique = []
        for l in letters:
            if l not in seen:
                seen.add(l)
                unique.append(l)
        if len(unique) >= 5:
            return unique

    bold_matches = re.findall(r'\*\*(?:产品\s*)?([A-J])[^*]*\*\*', response_text, re.IGNORECASE)
    if len(bold_matches) >= 5:
        seen = set()
        unique = []
        for l in bold_matches:
            l = l.upper()
            if l not in seen:
                seen.add(l)
                unique.append(l)
        if len(unique) >= 5:
            return unique

    # Refusal detection
    refusal_signals = ["REFUSED", "cannot rank", "无法排序", "cannot recommend",
                       "all identical", "完全一致", "无法区分"]
    text_lower = response_text.lower()
    if any(sig.lower() in text_lower for sig in refusal_signals):
        return "REFUSED"
    return "PARSE_FAIL"


# ╔══════════════════════════════════════════════════════════════╗
# ║                      MAIN EXECUTION                         ║
# ╚══════════════════════════════════════════════════════════════╝

def run_exp3(fictional_brands, repeats):
    """Execute all Exp 3 conditions."""
    print("\n" + "=" * 60)
    print("EXP 3: Multi-Agent GEO Competition & Prisoner's Dilemma")
    print("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    enabled_models = {n: c for n, c in CONFIG["models"].items() if c["enabled"]}

    # ── Generate task list ──
    tasks = []
    for scenario, k_geo in SCENARIOS.items():
        for sc_key in SUBCATEGORIES:
            for model_name, model_cfg in enabled_models.items():
                for lang in CONFIG["languages"]:
                    for protocol in ["alpha", "beta"]:
                        for rep in range(1, repeats + 1):
                            tasks.append({
                                "scenario": scenario,
                                "k_geo": k_geo,
                                "subcat": sc_key,
                                "model": model_name,
                                "model_cfg": model_cfg,
                                "lang": lang,
                                "protocol": protocol,
                                "repeat": rep,
                            })

    total = len(tasks)
    print(f"\nTotal tasks: {total}")
    print(f"  = {len(SCENARIOS)} scenarios × {len(SUBCATEGORIES)} subcats × "
          f"{len(enabled_models)} models × {len(CONFIG['languages'])} langs × "
          f"2 protocols × {repeats} reps")

    # ── Resume support ──
    csv_path = output_dir / "exp3_raw.csv"
    done_keys = set()
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and "scenario" in reader.fieldnames:
                    for row in reader:
                        if not row.get("response", "").startswith("[ERROR]"):
                            try:
                                key = (row["scenario"], row["subcat"], row["model"],
                                       row["lang"], row["protocol"], row["repeat"])
                                done_keys.add(key)
                            except KeyError:
                                continue
                    print(f"  Found {len(done_keys)} completed records")
        except Exception as e:
            print(f"  ⚠ Failed to read CSV ({e}), will overwrite")
            done_keys = set()

    remaining = [t for t in tasks
                 if (t["scenario"], t["subcat"], t["model"],
                     t["lang"], t["protocol"], str(t["repeat"])) not in done_keys]
    print(f"  Remaining: {len(remaining)}")
    if not remaining:
        print("  ✓ All tasks completed! Skipping to analysis phase.")
        return

    # ── CSV setup ──
    csv_fieldnames = [
        "timestamp", "scenario", "k_geo", "subcat", "model", "lang",
        "protocol", "repeat",
        "product_order",    # e.g. "A:CeraVe,B:PureGlow,..."
        "real_brand_id",    # letter of real brand
        "geo_brand_ids",    # comma-separated letters of GEO brands
        "geo_brand_names",  # comma-separated names of GEO brands
        "response",
        "parsed_choice",    # Protocol β: A-J
        "parsed_ranking",   # Protocol α: "A,B,C,..."
        "real_rank",        # rank of real brand (1-10)
        "chose_real",       # True/False for β
        "chose_geo",        # True/False for β (chose any GEO brand)
        "is_refusal",
        "is_parse_fail",
    ]

    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()

    # ── Batch execution ──
    BATCH_SIZE = 50
    batches = [remaining[i:i+BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    completed = len(done_keys)
    _csv_lock = threading.Lock()

    for batch_idx, batch in enumerate(batches):
        if _STOP.is_set():
            break
        print(f"\n── Batch {batch_idx+1}/{len(batches)} ({len(batch)} tasks) ──")

        def process_task(task):
            if _STOP.is_set():
                return None

            scenario = task["scenario"]
            k_geo = task["k_geo"]
            sc_key = task["subcat"]
            model_name = task["model"]
            model_cfg = task["model_cfg"]
            lang = task["lang"]
            protocol = task["protocol"]
            rep = task["repeat"]

            # Select which fictional brands get GEO (re-randomized per trial)
            all_fictional = fictional_brands[sc_key]
            if k_geo > 0:
                geo_brand_names = random.sample(all_fictional, k_geo)
            else:
                geo_brand_names = []

            # Build product pool with GEO injected
            products, real_id, geo_ids = build_product_pool(
                sc_key, lang, fictional_brands, geo_brand_names)

            # Build prompt
            if protocol == "alpha":
                prompt = build_prompt_alpha(sc_key, lang, products)
            else:
                prompt = build_prompt_beta(sc_key, lang, products)

            # Call LLM
            response = call_llm(prompt, model_name, model_cfg)

            # Parse
            parsed_choice = ""
            parsed_ranking = ""
            real_rank = ""
            chose_real = ""
            chose_geo = ""
            is_refusal = False
            is_parse_fail = False

            if response.startswith("[ERROR]") or response.startswith("[INTERRUPTED]"):
                is_parse_fail = True
            elif protocol == "beta":
                parsed_choice = parse_recommend(response)
                if parsed_choice == "REFUSED":
                    is_refusal = True
                elif parsed_choice == "PARSE_FAIL":
                    is_parse_fail = True
                else:
                    chose_real = str(parsed_choice == real_id)
                    chose_geo = str(parsed_choice in geo_ids)
            else:  # alpha
                result = parse_ranking(response)
                if result == "REFUSED":
                    is_refusal = True
                elif result == "PARSE_FAIL":
                    is_parse_fail = True
                else:
                    parsed_ranking = ",".join(result)
                    if real_id in result:
                        real_rank = str(result.index(real_id) + 1)

            # Product order string
            order_str = ",".join(f"{p['id']}:{p['brand_raw']}" for p in products)

            return {
                "timestamp": datetime.now().isoformat(),
                "scenario": scenario,
                "k_geo": k_geo,
                "subcat": sc_key,
                "model": model_name,
                "lang": lang,
                "protocol": protocol,
                "repeat": rep,
                "product_order": order_str,
                "real_brand_id": real_id,
                "geo_brand_ids": ",".join(geo_ids),
                "geo_brand_names": ",".join(geo_brand_names),
                "response": response[:2000],  # truncate long responses
                "parsed_choice": parsed_choice if not is_refusal and not is_parse_fail else "",
                "parsed_ranking": parsed_ranking,
                "real_rank": real_rank,
                "chose_real": chose_real,
                "chose_geo": chose_geo,
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
                    sc = result["scenario"]
                    pf = "✗" if result["is_parse_fail"] == "True" else "✓"
                    print(f"  [{completed}/{total}] {pf} {sc} {result['subcat']} "
                          f"{result['model']} {result['lang']} {result['protocol']} "
                          f"r{result['repeat']}", end="")
                    if result["protocol"] == "beta" and result["parsed_choice"]:
                        tag = "REAL" if result["chose_real"] == "True" else \
                              ("GEO" if result["chose_geo"] == "True" else "NEUTRAL")
                        print(f" → {result['parsed_choice']}({tag})")
                    elif result["real_rank"]:
                        print(f" → real@{result['real_rank']}")
                    else:
                        print()

        # Write batch to CSV
        if batch_results:
            with _csv_lock:
                with open(csv_path, "a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                    writer.writerows(batch_results)
            print(f"  💾 Saved {len(batch_results)} results")

    print(f"\n✓ Experiment run finished! {completed} results total")


# ╔══════════════════════════════════════════════════════════════╗
# ║                       ANALYSIS                              ║
# ╚══════════════════════════════════════════════════════════════╝

def analyze_exp3():
    """Comprehensive analysis of Exp 3 results."""
    csv_path = Path(CONFIG["output_dir"]) / "exp3_raw.csv"
    if not csv_path.exists():
        print("✗ Results file exp3_raw.csv not found")
        return

    # Load data
    all_rows = []
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            all_rows.append(r)
            if r.get("is_parse_fail") == "True" or r.get("is_refusal") == "True":
                continue
            rows.append(r)

    total_raw = len(all_rows)
    n_parse_fail = sum(1 for r in all_rows if r.get("is_parse_fail") == "True")
    n_refusal = sum(1 for r in all_rows if r.get("is_refusal") == "True")
    print(f"\n{'='*60}")
    print("EXP 3 ANALYSIS: Multi-Agent GEO Competition")
    print(f"{'='*60}")
    print(f"Total rows: {total_raw}, Valid: {len(rows)} ({len(rows)/max(total_raw,1)*100:.1f}%)")
    print(f"  Parse failures: {n_parse_fail} ({n_parse_fail/max(total_raw,1)*100:.1f}%)")
    print(f"  Refusals: {n_refusal} ({n_refusal/max(total_raw,1)*100:.1f}%)")

    # Split by protocol
    beta_rows = [r for r in rows if r["protocol"] == "beta"]
    alpha_rows = [r for r in rows if r["protocol"] == "alpha"]
    print(f"Protocol β (single rec): {len(beta_rows)}")
    print(f"Protocol α (ranking):    {len(alpha_rows)}")

    # ── 1. ISR Trajectory (Primary) ──
    print(f"\n{'─'*60}")
    print("1. INCUMBENT SURVIVAL RATE (ISR) — Protocol β")
    print(f"{'─'*60}")
    print(f"{'Scenario':<10} {'Overall':>10} ", end="")
    models = sorted(set(r["model"] for r in beta_rows))
    for m in models:
        print(f" {m:>15}", end="")
    print()

    isr_data = {}  # scenario -> {model -> (chose_real_count, total)}
    for scenario in ["S0", "S1", "S2", "S3", "S4"]:
        s_rows = [r for r in beta_rows if r["scenario"] == scenario]
        if not s_rows:
            continue
        n_real = sum(1 for r in s_rows if r.get("chose_real") == "True")
        n_total = len(s_rows)
        isr_overall = n_real / n_total if n_total > 0 else 0

        print(f"{scenario} (k={SCENARIOS[scenario]}) {isr_overall:>8.1%} ", end="")

        isr_data[scenario] = {}
        for m in models:
            m_rows = [r for r in s_rows if r["model"] == m]
            if m_rows:
                m_real = sum(1 for r in m_rows if r.get("chose_real") == "True")
                m_total = len(m_rows)
                isr = m_real / m_total
                isr_data[scenario][m] = (m_real, m_total)
                print(f" {isr:>13.1%}({m_real}/{m_total})", end="")
            else:
                print(f" {'N/A':>15}", end="")
        print()

    # ── 2. GAR (GEO Advantage Ratio) ──
    print(f"\n{'─'*60}")
    print("2. GEO ADVANTAGE RATIO (GAR) — Protocol β")
    print(f"{'─'*60}")
    print(f"  GAR = P(GEO brand rec) / P(non-GEO fictional brand rec)")
    print(f"  GAR > 1: GEO helps | GAR = 1: no effect | GAR < 1: GEO backfires\n")

    for scenario in ["S1", "S2", "S3", "S4"]:
        s_rows = [r for r in beta_rows if r["scenario"] == scenario]
        if not s_rows:
            continue

        k = SCENARIOS[scenario]
        n_geo_chose = sum(1 for r in s_rows if r.get("chose_geo") == "True")
        n_non_geo_fic = sum(1 for r in s_rows
                           if r.get("chose_real") != "True"
                           and r.get("chose_geo") != "True")
        n_total = len(s_rows)

        # Per-brand probabilities
        p_geo = (n_geo_chose / n_total) / k if k > 0 and n_total > 0 else 0
        n_non_geo = 9 - k
        p_non_geo = (n_non_geo_fic / n_total) / n_non_geo if n_non_geo > 0 and n_total > 0 else 0
        gar = p_geo / p_non_geo if p_non_geo > 0 else float('inf')

        print(f"  {scenario} (k={k}): P(GEO brand)={p_geo:.4f}, "
              f"P(non-GEO fic)={p_non_geo:.4f}, GAR={gar:.2f}")

        # Per-model GAR
        for m in models:
            m_rows = [r for r in s_rows if r["model"] == m]
            if not m_rows:
                continue
            mg = sum(1 for r in m_rows if r.get("chose_geo") == "True")
            mnf = sum(1 for r in m_rows
                      if r.get("chose_real") != "True"
                      and r.get("chose_geo") != "True")
            mt = len(m_rows)
            mp_geo = (mg / mt) / k if k > 0 and mt > 0 else 0
            mp_non = (mnf / mt) / n_non_geo if n_non_geo > 0 and mt > 0 else 0
            m_gar = mp_geo / mp_non if mp_non > 0 else float('inf')
            print(f"    {m}: GAR={m_gar:.2f} (GEO chose={mg}/{mt}, non-GEO fic={mnf}/{mt})")

    # ── 3. Real Brand Mean Rank (Protocol α) ──
    print(f"\n{'─'*60}")
    print("3. REAL BRAND MEAN RANK — Protocol α")
    print(f"{'─'*60}")
    for scenario in ["S0", "S1", "S2", "S3", "S4"]:
        s_rows = [r for r in alpha_rows if r["scenario"] == scenario and r.get("real_rank")]
        if not s_rows:
            continue
        ranks = [int(r["real_rank"]) for r in s_rows]
        mean_rank = sum(ranks) / len(ranks)
        top1_pct = sum(1 for rk in ranks if rk == 1) / len(ranks)
        print(f"  {scenario} (k={SCENARIOS[scenario]}): MeanRank={mean_rank:.2f}, "
              f"Top-1={top1_pct:.1%}, n={len(ranks)}")

        for m in models:
            m_rows = [r for r in s_rows if r["model"] == m]
            if m_rows:
                m_ranks = [int(r["real_rank"]) for r in m_rows]
                m_mean = sum(m_ranks) / len(m_ranks)
                m_top1 = sum(1 for rk in m_ranks if rk == 1) / len(m_ranks)
                print(f"    {m}: MeanRank={m_mean:.2f}, Top-1={m_top1:.1%}, n={len(m_ranks)}")

    # ── 4. Game-Theoretic Metrics ──
    print(f"\n{'─'*60}")
    print("4. GAME-THEORETIC METRICS — Protocol β")
    print(f"{'─'*60}")

    # Payoff matrix: π(GEO, Sk) = P(GEO brand rec | Sk) - P(same brand no GEO | S0)
    s0_rows = [r for r in beta_rows if r["scenario"] == "S0"]
    if s0_rows:
        p_fictional_s0 = sum(1 for r in s0_rows if r.get("chose_real") != "True") / len(s0_rows)
        p_per_fic_s0 = p_fictional_s0 / 9  # per-brand baseline

        print(f"\n  S0 baseline: P(any fictional)={p_fictional_s0:.3f}, "
              f"P(per fictional)={p_per_fic_s0:.4f}")

        print(f"\n  {'Scenario':<10} {'P(GEO/brand)':>14} {'P(fic/brand@S0)':>16} "
              f"{'Payoff π':>10} {'DI':>10}")
        for scenario in ["S1", "S2", "S3", "S4"]:
            s_rows = [r for r in beta_rows if r["scenario"] == scenario]
            if not s_rows:
                continue
            k = SCENARIOS[scenario]
            n_geo = sum(1 for r in s_rows if r.get("chose_geo") == "True")
            p_geo_brand = (n_geo / len(s_rows)) / k if k > 0 else 0

            # Defection Incentive: P(GEO brand|Sk) - P(non-GEO fic brand|Sk)
            n_non_geo_fic = sum(1 for r in s_rows
                                if r.get("chose_real") != "True"
                                and r.get("chose_geo") != "True")
            n_non_geo = 9 - k
            p_non_geo_brand = (n_non_geo_fic / len(s_rows)) / n_non_geo if n_non_geo > 0 else 0
            di = p_geo_brand - p_non_geo_brand

            payoff = p_geo_brand - p_per_fic_s0
            print(f"  {scenario} (k={k}) {p_geo_brand:>14.4f} {p_per_fic_s0:>16.4f} "
                  f"{payoff:>+10.4f} {di:>+10.4f}")

    # ── 5. HHI (Market Concentration) ──
    print(f"\n{'─'*60}")
    print("5. MARKET CONCENTRATION (HHI) — Protocol β")
    print(f"{'─'*60}")
    print(f"  HHI=1.0: monopoly, HHI=0.1: perfectly competitive (10 brands)\n")

    for scenario in ["S0", "S1", "S2", "S3", "S4"]:
        s_rows = [r for r in beta_rows if r["scenario"] == scenario]
        if not s_rows:
            continue
        # Count choices per brand type
        brand_counts = defaultdict(int)
        for r in s_rows:
            if r.get("chose_real") == "True":
                brand_counts["REAL"] += 1
            elif r.get("chose_geo") == "True":
                brand_counts["GEO"] += 1
            else:
                brand_counts["NEUTRAL_FIC"] += 1

        n = len(s_rows)
        # Simple HHI from 3 categories
        shares = [v/n for v in brand_counts.values()]
        hhi = sum(s**2 for s in shares)

        # More precise: share per individual brand (approximate)
        k = SCENARIOS[scenario]
        real_share = brand_counts.get("REAL", 0) / n
        geo_total_share = brand_counts.get("GEO", 0) / n
        neutral_total_share = brand_counts.get("NEUTRAL_FIC", 0) / n

        # Approximate per-brand shares (assuming uniform within group)
        per_brand_shares = [real_share]
        if k > 0:
            per_brand_shares += [geo_total_share / k] * k
        if (9 - k) > 0:
            per_brand_shares += [neutral_total_share / (9 - k)] * (9 - k)
        hhi_fine = sum(s**2 for s in per_brand_shares)

        print(f"  {scenario} (k={k}): HHI={hhi_fine:.4f} "
              f"(real={real_share:.1%}, GEO_total={geo_total_share:.1%}, "
              f"neutral_total={neutral_total_share:.1%})")

    # ── 6. KL Divergence (S4 vs S0) ──
    print(f"\n{'─'*60}")
    print("6. KL DIVERGENCE — S4 vs S0 (Prisoner's Dilemma Test)")
    print(f"{'─'*60}")

    for m in models:
        s0_m = [r for r in beta_rows if r["scenario"] == "S0" and r["model"] == m]
        s4_m = [r for r in beta_rows if r["scenario"] == "S4" and r["model"] == m]
        if not s0_m or not s4_m:
            continue

        # Distribution: P(real), P(fictional)
        p_real_s0 = sum(1 for r in s0_m if r.get("chose_real") == "True") / len(s0_m)
        p_real_s4 = sum(1 for r in s4_m if r.get("chose_real") == "True") / len(s4_m)

        # Smooth to avoid log(0)
        eps = 1e-6
        p0 = [max(p_real_s0, eps), max(1 - p_real_s0, eps)]
        p4 = [max(p_real_s4, eps), max(1 - p_real_s4, eps)]

        # Normalize
        s0_sum = sum(p0)
        s4_sum = sum(p4)
        p0 = [x/s0_sum for x in p0]
        p4 = [x/s4_sum for x in p4]

        kl = sum(p4[i] * math.log(p4[i] / p0[i]) for i in range(2))
        print(f"  {m}: D_KL(S4||S0) = {kl:.4f}")
        print(f"    S0: P(real)={p_real_s0:.3f}, S4: P(real)={p_real_s4:.3f}")
        if abs(kl) < 0.01:
            print(f"    → KL ≈ 0: Prisoner's Dilemma pattern (S4 ≈ S0)")
        elif p_real_s4 > p_real_s0:
            print(f"    → Incumbent Amplification (H3b): real brand STRONGER under full GEO")
        else:
            print(f"    → Structural shift: real brand weaker under full GEO")

    # ── 7. Per-Subcategory Breakdown ──
    print(f"\n{'─'*60}")
    print("7. PER-SUBCATEGORY ISR — Protocol β")
    print(f"{'─'*60}")
    for sc in sorted(set(r["subcat"] for r in beta_rows)):
        print(f"\n  {sc}:")
        for scenario in ["S0", "S1", "S2", "S3", "S4"]:
            sc_rows = [r for r in beta_rows
                       if r["scenario"] == scenario and r["subcat"] == sc]
            if not sc_rows:
                continue
            n_real = sum(1 for r in sc_rows if r.get("chose_real") == "True")
            n = len(sc_rows)
            print(f"    {scenario}: ISR={n_real/n:.1%} ({n_real}/{n})")

    # ── 8. Hypothesis Testing Summary ──
    print(f"\n{'─'*60}")
    print("8. HYPOTHESIS TESTING SUMMARY")
    print(f"{'─'*60}")

    s0_beta = [r for r in beta_rows if r["scenario"] == "S0"]
    s4_beta = [r for r in beta_rows if r["scenario"] == "S4"]
    if s0_beta and s4_beta:
        isr_s0 = sum(1 for r in s0_beta if r.get("chose_real") == "True") / len(s0_beta)
        isr_s4 = sum(1 for r in s4_beta if r.get("chose_real") == "True") / len(s4_beta)

        print(f"\n  ISR(S0) = {isr_s0:.3f}, ISR(S4) = {isr_s4:.3f}")
        print(f"  Δ(ISR) = {isr_s4 - isr_s0:+.3f}")

        if abs(isr_s4 - isr_s0) < 0.05:
            print(f"\n  → H3a SUPPORTED: Prisoner's Dilemma")
            print(f"    Full GEO adoption cancels out → back to baseline")
        elif isr_s4 > isr_s0 + 0.05:
            print(f"\n  → H3b SUPPORTED: Incumbent Amplification")
            print(f"    Full GEO overload strengthens real brand")
        else:
            print(f"\n  → Neither H3a nor H3b clearly supported")
            print(f"    Full GEO weakened incumbent but didn't return to baseline")

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                         MAIN                                ║
# ╚══════════════════════════════════════════════════════════════╝

def reparse_csv():
    """Re-parse existing CSV with updated parse functions (no API calls needed).

    Reads exp3_raw.csv, re-runs parse_recommend/parse_ranking on all responses,
    updates parsed fields, and writes back. Recovers data from improved parsers.
    """
    csv_path = Path(CONFIG["output_dir"]) / "exp3_raw.csv"
    if not csv_path.exists():
        print("✗ exp3_raw.csv not found")
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)

    print(f"Re-parsing {len(rows)} rows...")
    n_recovered_beta = 0
    n_recovered_alpha = 0
    n_changed = 0

    for r in rows:
        response = r.get("response", "")
        if response.startswith("[ERROR]") or response.startswith("[INTERRUPTED]"):
            continue

        old_pf = r.get("is_parse_fail")
        old_ref = r.get("is_refusal")
        protocol = r.get("protocol")

        if protocol == "beta":
            result = parse_recommend(response)
            if result == "REFUSED":
                new_pf, new_ref = "False", "True"
                new_choice = ""
                new_ranking = ""
                new_real_rank = ""
                new_chose_real = ""
                new_chose_geo = ""
            elif result == "PARSE_FAIL":
                new_pf, new_ref = "True", "False"
                new_choice = ""
                new_ranking = ""
                new_real_rank = ""
                new_chose_real = ""
                new_chose_geo = ""
            else:
                new_pf, new_ref = "False", "False"
                new_choice = result
                new_ranking = ""
                new_real_rank = ""
                real_id = r.get("real_brand_id", "")
                geo_ids = r.get("geo_brand_ids", "").split(",") if r.get("geo_brand_ids") else []
                new_chose_real = str(result == real_id)
                new_chose_geo = str(result in geo_ids)

            if old_pf == "True" and new_pf == "False":
                n_recovered_beta += 1
                n_changed += 1

            r["parsed_choice"] = new_choice
            r["chose_real"] = new_chose_real
            r["chose_geo"] = new_chose_geo
            r["is_parse_fail"] = new_pf
            r["is_refusal"] = new_ref

        elif protocol == "alpha":
            result = parse_ranking(response)
            if result == "REFUSED":
                new_pf, new_ref = "False", "True"
                new_ranking = ""
                new_real_rank = ""
            elif result == "PARSE_FAIL":
                new_pf, new_ref = "True", "False"
                new_ranking = ""
                new_real_rank = ""
            else:
                new_pf, new_ref = "False", "False"
                new_ranking = ",".join(result)
                real_id = r.get("real_brand_id", "")
                new_real_rank = str(result.index(real_id) + 1) if real_id in result else ""

            if old_pf == "True" and new_pf == "False":
                n_recovered_alpha += 1
                n_changed += 1

            r["parsed_ranking"] = new_ranking
            r["real_rank"] = new_real_rank
            r["is_parse_fail"] = new_pf
            r["is_refusal"] = new_ref

    # Write back
    if n_changed > 0:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ Recovered: β={n_recovered_beta}, α={n_recovered_alpha}")
        print(f"  Updated {n_changed} rows in exp3_raw.csv")
    else:
        print("  No additional recoveries possible with current data.")


def main():
    parser = argparse.ArgumentParser(description="Exp 3: Multi-Agent GEO Competition")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no API calls)")
    parser.add_argument("--fresh", action="store_true", help="Clear old results")
    parser.add_argument("--repeats", type=int, default=CONFIG["repeats"],
                        help=f"Repetitions per cell (default: {CONFIG['repeats']})")
    parser.add_argument("--analyze-only", action="store_true", help="Skip API, analyze only")
    parser.add_argument("--reparse", action="store_true",
                        help="Re-parse existing CSV with improved parsers (no API calls)")
    args = parser.parse_args()

    if args.reparse:
        print("🔄 RE-PARSE MODE — updating existing data with improved parsers")
        reparse_csv()
        print("\n── Analyzing results ──")
        analyze_exp3()
        return

    if args.dry_run:
        CONFIG["dry_run"] = True
        print("🔧 DRY RUN MODE — no API calls")

    if args.fresh:
        csv_path = Path(CONFIG["output_dir"]) / "exp3_raw.csv"
        if csv_path.exists():
            os.remove(csv_path)
            print("🗑 Cleared old results")

    print("\n" + "=" * 60)
    print("  Experiment 3: Multi-Agent GEO Competition")
    print("  Prisoner's Dilemma in LLM Recommendation Systems")
    print("=" * 60)
    print(f"  Scenarios: {list(SCENARIOS.keys())}")
    print(f"  GEO counts: {list(SCENARIOS.values())}")
    print(f"  Models: {[n for n, c in CONFIG['models'].items() if c['enabled']]}")
    print(f"  Languages: {CONFIG['languages']}")
    print(f"  Protocols: α (ranking), β (single rec)")
    print(f"  Repeats: {args.repeats}")

    # Load brands
    print("\n── Loading fictional brand pool ──")
    fictional_brands = load_fictional_brands()

    # Run or analyze
    if not args.analyze_only:
        run_exp3(fictional_brands, args.repeats)

    # Analyze
    print("\n── Analyzing results ──")
    analyze_exp3()


if __name__ == "__main__":
    main()
