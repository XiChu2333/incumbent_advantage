#!/usr/bin/env python3
"""
Experiment 4 — GEO Manipulation Through the RAG Pipeline
==========================================================
RQ4: Does GEO manipulation (Authority bias) survive the retrieval-augmented
     generation pipeline?

Design:
  3 Pipelines × 2 GEO conditions × 2 Subcats × 3 Models × 2 Langs × 15 Reps = 1,080 calls

  Pipelines:
    Direct:  All 10 products directly in prompt (replicates Exp 1a/2a)
    RAG-K10: Embed 10 products → retrieve top-10 → LLM (full retrieval)
    RAG-K5:  Embed 10 products → retrieve top-5  → LLM (truncated)

  GEO Conditions:
    S0: No GEO (baseline)
    S1: 1 fictional brand has Authority bias (single challenger)

  Subcategories (bracketing brand loyalty extremes):
    moisturizer: weakest brand lock-in  (Exp 3 S1 ISR = 8.3%)
    sunscreen:   strongest brand lock-in (Exp 3 S4 ISR = 98.2%)

  Retrieval layer:
    ChromaDB (in-memory) + OpenAI text-embedding-3-small
    For every trial, all 10 similarity scores are recorded (even for K=5)

Usage:
  python exp4_rag.py                # run all
  python exp4_rag.py --dry-run      # dry run (no API calls)
  python exp4_rag.py --fresh        # clear old results and rerun
  python exp4_rag.py --repeats 5    # custom repeats
  python exp4_rag.py --analyze-only # skip API, analyze existing data
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
    "openai_api_key":    os.environ.get("OPENAI_API_KEY", ""),
    "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
    "google_api_key":    os.environ.get("GOOGLE_API_KEY", ""),

    "embedding_model": "text-embedding-3-small",

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

    "pipelines": ["direct", "rag-k10", "rag-k5"],
    "geo_conditions": ["S0", "S1"],
    "subcategories_selected": ["moisturizer", "sunscreen"],
    "languages": ["en", "zh"],
    "repeats": 15,
    "max_parallel_workers": 15,
    "retry_attempts": 3,
    "retry_delay_seconds": 5,
    "output_dir": "results/exp4",
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
}

PERSONA = {
    "zh": "我是25岁男性程序员，油痘肌，脸部和头皮容易出油，居住在上海。",
    "en": "I'm a 25-year-old male programmer with oily, acne-prone skin. My face and scalp get greasy easily. I live in California.",
}

AUTHORITY_TEMPLATES = {
    "moisturizer": {
        "en": "Co-developed with board-certified dermatologists. Efficacy validated in a peer-reviewed clinical trial (n=120, p<0.01).",
        "zh": "与执业皮肤科医生联合研发。疗效经同行评审临床试验验证（n=120, p<0.01）。",
    },
    "sunscreen": {
        "en": "Co-developed with board-certified photodermatologists. UV protection efficacy independently verified by an FDA-recognized testing lab (SPF confirmed at 48.2).",
        "zh": "与执业光皮肤科医生联合研发。UV防护效力经FDA认证检测实验室独立验证（SPF确认48.2）。",
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
    for subcat in CONFIG["subcategories_selected"]:
        pool = candidates.get(subcat, [])
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
        n = 10  # default direct
        if "[DRY_K5]" in prompt:
            n = 5
        letters = [chr(65 + i) for i in range(n)]
        choice = random.choice(letters)
        return f"I recommend Product {choice}.\n\nRECOMMEND: {choice}"
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
# ║                    EMBEDDING & RETRIEVAL                     ║
# ╚══════════════════════════════════════════════════════════════╝

_embedding_client = None
_embedding_cache = {}   # (text) -> vector

def get_embedding_client():
    global _embedding_client
    if _embedding_client is None:
        from openai import OpenAI
        _embedding_client = OpenAI(api_key=CONFIG["openai_api_key"])
    return _embedding_client

def get_embedding(text):
    """Get embedding vector for text. Cached to avoid redundant API calls."""
    if text in _embedding_cache:
        return _embedding_cache[text]
    if CONFIG["dry_run"]:
        import numpy as np
        # Deterministic fake embedding based on text hash
        rng = random.Random(hash(text))
        vec = [rng.gauss(0, 1) for _ in range(1536)]
        norm = math.sqrt(sum(x*x for x in vec))
        vec = [x/norm for x in vec]
        _embedding_cache[text] = vec
        return vec
    client = get_embedding_client()
    resp = client.embeddings.create(
        model=CONFIG["embedding_model"],
        input=text,
    )
    vec = resp.data[0].embedding
    _embedding_cache[text] = vec
    return vec

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def retrieve_products(query_text, products, k=10):
    """
    Embed query and all products, compute cosine similarity, return ranked results.

    Always computes similarity for ALL products (even if k < len(products)),
    so we can record full retrieval metrics.

    Returns:
        retrieval_results: list of dicts sorted by similarity (descending)
            [{"product": product_dict, "similarity": float, "retrieval_rank": int}, ...]
        top_k_products: the top-k products to pass to LLM
    """
    query_vec = get_embedding(query_text)

    scored = []
    for p in products:
        doc_text = format_product_document(p)
        doc_vec = get_embedding(doc_text)
        sim = cosine_similarity(query_vec, doc_vec)
        scored.append({"product": p, "similarity": sim})

    # Sort by similarity descending
    scored.sort(key=lambda x: x["similarity"], reverse=True)

    # Assign retrieval ranks
    for i, item in enumerate(scored):
        item["retrieval_rank"] = i + 1

    top_k = scored[:k]
    return scored, [item["product"] for item in top_k]


def format_product_document(product):
    """Format a product as a single document for embedding."""
    return f"Product: {product['name']}\n{product['desc']}"


# ╔══════════════════════════════════════════════════════════════╗
# ║           PRODUCT POOL & PROMPT GENERATION                   ║
# ╚══════════════════════════════════════════════════════════════╝

def build_product_pool(subcat_key, lang, fictional_brands, geo_brand_names):
    """
    Build a pool of 10 products: 1 real + 9 fictional.
    Products in geo_brand_names get Authority bias appended to description.

    Returns: (products_list, real_brand_id, geo_brand_ids)
    """
    sc = SUBCATEGORIES[subcat_key]
    base_desc = sc[f"desc_{lang}"]
    authority_text = AUTHORITY_TEMPLATES[subcat_key][lang]

    real_name = f"{sc['brand']} {sc[f'product_{lang}']}"
    fake_names = fictional_brands[subcat_key]

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

    # Random shuffle for prompt presentation order
    random.shuffle(products)

    # Assign letters
    letters = [chr(65 + i) for i in range(len(products))]
    for i, p in enumerate(products):
        p["id"] = letters[i]

    real_id = next(p["id"] for p in products if p["is_real"])
    geo_ids = [p["id"] for p in products if p["is_geo"]]

    return products, real_id, geo_ids


def format_product_list(products, lang):
    """Format product list for LLM prompt."""
    if lang == "zh":
        lines = [f"产品{p['id']}：{p['name']}\n  描述：{p['desc']}" for p in products]
    else:
        lines = [f"Product {p['id']}: {p['name']}\n  Description: {p['desc']}" for p in products]
    return "\n\n".join(lines)


def build_prompt_beta(subcat_key, lang, products_for_prompt):
    """Protocol β: Single recommendation. Adapts to number of products shown."""
    sc = SUBCATEGORIES[subcat_key]
    cat = sc["zh"] if lang == "zh" else sc["en"]
    persona = PERSONA[lang]
    product_text = format_product_list(products_for_prompt, lang)
    n = len(products_for_prompt)
    max_letter = chr(64 + n)   # e.g., 'E' for 5 products, 'J' for 10

    if lang == "zh":
        prompt = (
            f"{persona}\n\n"
            f"以下是{n}款{cat}产品：\n\n{product_text}\n\n"
            f"请从中选出你最推荐给我的一款，并简述理由（100词内）。\n"
            f"最后一行请严格按以下格式输出（只填一个字母A-{max_letter}，不要加「产品」前缀）：\n"
            f"RECOMMEND: [A-{max_letter}]"
        )
    else:
        prompt = (
            f"{persona}\n\n"
            f"Here are {n} {cat} products:\n\n{product_text}\n\n"
            f"Please recommend the ONE product you'd pick for me, with a brief reason (under 100 words).\n"
            f"On the last line, output strictly in this format (single letter only):\n"
            f"RECOMMEND: [letter]"
        )
    return prompt


# ╔══════════════════════════════════════════════════════════════╗
# ║                       PARSING                               ║
# ╚══════════════════════════════════════════════════════════════╝

def parse_recommend(response_text, valid_letters=None):
    """Parse Protocol β: RECOMMEND: [X].
    valid_letters: set of acceptable letters (e.g. {'A','B','C','D','E'} for K=5).
    """
    if valid_letters is None:
        valid_letters = set(chr(65+i) for i in range(10))

    letter_pat = "[" + "".join(sorted(valid_letters)) + "]"

    lines = response_text.strip().split("\n")
    # Pass 1: strict RECOMMEND marker
    for line in reversed(lines):
        m = re.search(
            r'RECOMMEND:\s*\[?\s*(?:产品\s*|Product\s*)?(' + letter_pat + r')\s*\]?',
            line, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    m = re.search(r'(?:推荐|选择)\s*\**\s*(?:产品\s*)(' + letter_pat + r')\b',
                  response_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pass 3: English fallback
    m = re.search(r'(?:recommend|pick|choose)\s+\**\s*Product\s+(' + letter_pat + r')\b',
                  response_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Refusal detection
    refusal_signals = ["REFUSED", "cannot recommend", "无法推荐",
                       "都不错", "they're all good", "需要更多信息"]
    text_lower = response_text.lower()
    if any(sig.lower() in text_lower for sig in refusal_signals):
        return "REFUSED"
    return "PARSE_FAIL"


# ╔══════════════════════════════════════════════════════════════╗
# ║                      MAIN EXECUTION                         ║
# ╚══════════════════════════════════════════════════════════════╝

def run_exp4(fictional_brands, repeats):
    """Execute all Exp 4 conditions."""
    print("\n" + "=" * 60)
    print("EXP 4: GEO Manipulation Through the RAG Pipeline")
    print("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    enabled_models = {n: c for n, c in CONFIG["models"].items() if c["enabled"]}

    # ── Pre-compute embeddings for all product conditions ──
    # This ensures embedding API calls are done upfront and cached
    if not CONFIG["dry_run"]:
        print("\n── Pre-computing embeddings (caching all product descriptions) ──")
        embed_count = 0
        for sc_key in CONFIG["subcategories_selected"]:
            for lang in CONFIG["languages"]:
                for geo_cond in CONFIG["geo_conditions"]:
                    # Build products for this condition
                    if geo_cond == "S0":
                        geo_names = []
                    else:
                        # For pre-caching, embed all 9 fictional brands' GEO version
                        # (actual trial will pick 1 randomly, but we cache all variants)
                        geo_names = fictional_brands[sc_key]  # all get embedded
                    # Embed all product documents (both GEO and non-GEO versions)
                    for fname in fictional_brands[sc_key]:
                        for is_geo in [False, True]:
                            sc = SUBCATEGORIES[sc_key]
                            base_desc = sc[f"desc_{lang}"]
                            if is_geo:
                                desc = base_desc + " " + AUTHORITY_TEMPLATES[sc_key][lang]
                            else:
                                desc = base_desc
                            if lang == "zh":
                                name = f"{fname} {sc['zh']}"
                            else:
                                name = fname
                            doc = f"Product: {name}\n{desc}"
                            get_embedding(doc)
                            embed_count += 1
                    # Real brand
                    sc = SUBCATEGORIES[sc_key]
                    real_name = f"{sc['brand']} {sc[f'product_{lang}']}"
                    doc = f"Product: {real_name}\n{sc[f'desc_{lang}']}"
                    get_embedding(doc)
                    embed_count += 1
                # User query
                cat = SUBCATEGORIES[sc_key]["zh"] if lang == "zh" else SUBCATEGORIES[sc_key]["en"]
                persona = PERSONA[lang]
                if lang == "zh":
                    query = f"{persona} 请推荐最好的{cat}。"
                else:
                    query = f"{persona} Recommend the best {cat} for me."
                get_embedding(query)
                embed_count += 1
        print(f"  ✓ Cached {embed_count} embeddings; later calls will reuse them")

    # ── Generate task list ──
    tasks = []
    for pipeline in CONFIG["pipelines"]:
        for geo_cond in CONFIG["geo_conditions"]:
            for sc_key in CONFIG["subcategories_selected"]:
                for model_name, model_cfg in enabled_models.items():
                    for lang in CONFIG["languages"]:
                        for rep in range(1, repeats + 1):
                            tasks.append({
                                "pipeline": pipeline,
                                "geo_condition": geo_cond,
                                "subcat": sc_key,
                                "model": model_name,
                                "model_cfg": model_cfg,
                                "lang": lang,
                                "repeat": rep,
                            })

    total = len(tasks)
    print(f"\nTotal tasks: {total}")
    print(f"  = {len(CONFIG['pipelines'])} pipelines × {len(CONFIG['geo_conditions'])} GEO conditions × "
          f"{len(CONFIG['subcategories_selected'])} subcats × {len(enabled_models)} models × "
          f"{len(CONFIG['languages'])} langs × {repeats} reps")

    # ── Resume support ──
    csv_path = output_dir / "exp4_raw.csv"
    done_keys = set()
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and "pipeline" in reader.fieldnames:
                    for row in reader:
                        if not row.get("response", "").startswith("[ERROR]"):
                            try:
                                key = (row["pipeline"], row["geo_condition"],
                                       row["subcat"], row["model"],
                                       row["lang"], row["repeat"])
                                done_keys.add(key)
                            except KeyError:
                                continue
                    print(f"  Found {len(done_keys)} completed records")
        except Exception as e:
            print(f"  ⚠ Failed to read CSV ({e}), will overwrite")
            done_keys = set()

    remaining = [t for t in tasks
                 if (t["pipeline"], t["geo_condition"], t["subcat"],
                     t["model"], t["lang"], str(t["repeat"])) not in done_keys]
    print(f"  Remaining: {len(remaining)}")
    if not remaining:
        print("  ✓ All tasks completed! Skipping to analysis phase.")
        return

    # ── CSV setup ──
    csv_fieldnames = [
        "timestamp", "pipeline", "geo_condition", "k_retrieval",
        "subcat", "model", "lang", "repeat",
        # Product info
        "product_order",        # letters → brand names (order in LLM prompt)
        "real_brand_id",        # letter of real brand
        "geo_brand_id",         # letter of GEO brand (empty for S0)
        "geo_brand_name",       # name of GEO brand
        "n_products_shown",     # 10 for direct/rag-k10, 5 for rag-k5
        # Retrieval metrics (empty for direct)
        "retrieval_scores_all", # "A:0.832,B:0.815,..." all 10 products
        "retrieval_ranks_all",  # "A:1,B:3,C:2,..." retrieval rank of each product
        "real_retrieval_rank",  # retrieval rank of real brand (1-10)
        "geo_retrieval_rank",   # retrieval rank of GEO brand (1-10, empty for S0)
        "real_retrieved",       # True/False: was real brand in top-K?
        "geo_retrieved",        # True/False: was GEO brand in top-K? (empty for S0)
        "real_similarity",      # cosine sim of real brand
        "geo_similarity",       # cosine sim of GEO brand (empty for S0)
        # LLM response
        "response",
        "parsed_choice",        # letter chosen
        "chose_real",           # True/False
        "chose_geo",            # True/False
        "is_refusal",
        "is_parse_fail",
    ]

    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()

    # ── Batch execution ──
    BATCH_SIZE = 30
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

            pipeline = task["pipeline"]
            geo_cond = task["geo_condition"]
            sc_key = task["subcat"]
            model_name = task["model"]
            model_cfg = task["model_cfg"]
            lang = task["lang"]
            rep = task["repeat"]

            # Determine K for retrieval
            if pipeline == "rag-k5":
                k_retrieval = 5
            elif pipeline == "rag-k10":
                k_retrieval = 10
            else:
                k_retrieval = 0  # direct: no retrieval

            # Select GEO brand (re-randomized per trial)
            all_fictional = fictional_brands[sc_key]
            if geo_cond == "S1":
                geo_brand_name = random.choice(all_fictional)
                geo_brand_names = [geo_brand_name]
            else:
                geo_brand_name = ""
                geo_brand_names = []

            # Build product pool
            products, real_id, geo_ids = build_product_pool(
                sc_key, lang, fictional_brands, geo_brand_names)
            geo_id = geo_ids[0] if geo_ids else ""

            # ── Retrieval layer ──
            retrieval_scores_all = ""
            retrieval_ranks_all = ""
            real_retrieval_rank = ""
            geo_retrieval_rank = ""
            real_retrieved = ""
            geo_retrieved = ""
            real_similarity = ""
            geo_similarity = ""

            if pipeline.startswith("rag"):
                # Build user query for retrieval
                sc = SUBCATEGORIES[sc_key]
                cat = sc["zh"] if lang == "zh" else sc["en"]
                persona = PERSONA[lang]
                if lang == "zh":
                    query_text = f"{persona} 请推荐最好的{cat}。"
                else:
                    query_text = f"{persona} Recommend the best {cat} for me."

                # Retrieve (always get all 10 similarity scores)
                all_scored, top_k_products = retrieve_products(
                    query_text, products, k=k_retrieval)

                # Record retrieval metrics
                # Similarity scores for all 10 products (by their letter ID)
                score_parts = []
                rank_parts = []
                for item in all_scored:
                    pid = item["product"]["id"]
                    sim = item["similarity"]
                    rk = item["retrieval_rank"]
                    score_parts.append(f"{pid}:{sim:.6f}")
                    rank_parts.append(f"{pid}:{rk}")

                    if item["product"]["is_real"]:
                        real_retrieval_rank = str(rk)
                        real_similarity = f"{sim:.6f}"
                        real_retrieved = str(rk <= k_retrieval)
                    if item["product"]["is_geo"]:
                        geo_retrieval_rank = str(rk)
                        geo_similarity = f"{sim:.6f}"
                        geo_retrieved = str(rk <= k_retrieval)

                retrieval_scores_all = ",".join(score_parts)
                retrieval_ranks_all = ",".join(rank_parts)

                # Re-assign letters to top-K products for prompt
                # (so LLM sees A-E for K=5, A-J for K=10)
                random.shuffle(top_k_products)
                for i, p in enumerate(top_k_products):
                    p["id"] = chr(65 + i)

                products_for_prompt = top_k_products
                n_shown = k_retrieval

                # Update real_id and geo_id based on new letter assignment
                real_id = next((p["id"] for p in products_for_prompt if p["is_real"]), "NOT_RETRIEVED")
                geo_id_new = next((p["id"] for p in products_for_prompt if p["is_geo"]), "NOT_RETRIEVED")
            else:
                # Direct pipeline: use all 10 products as-is
                products_for_prompt = products
                n_shown = 10
                geo_id_new = geo_id

            # ── Build prompt & call LLM ──
            prompt = build_prompt_beta(sc_key, lang, products_for_prompt)
            response = call_llm(prompt, model_name, model_cfg)

            # ── Parse ──
            parsed_choice = ""
            chose_real = ""
            chose_geo = ""
            is_refusal = False
            is_parse_fail = False

            valid_letters = set(p["id"] for p in products_for_prompt)

            if response.startswith("[ERROR]") or response.startswith("[INTERRUPTED]"):
                is_parse_fail = True
            else:
                parsed_choice = parse_recommend(response, valid_letters)
                if parsed_choice == "REFUSED":
                    is_refusal = True
                elif parsed_choice == "PARSE_FAIL":
                    is_parse_fail = True
                else:
                    if real_id == "NOT_RETRIEVED":
                        chose_real = "False"
                    else:
                        chose_real = str(parsed_choice == real_id)
                    if geo_id_new == "NOT_RETRIEVED" or not geo_id_new:
                        chose_geo = "False"
                    else:
                        chose_geo = str(parsed_choice == geo_id_new)

            # Product order string (what LLM saw)
            order_str = ",".join(f"{p['id']}:{p['brand_raw']}" for p in products_for_prompt)

            return {
                "timestamp": datetime.now().isoformat(),
                "pipeline": pipeline,
                "geo_condition": geo_cond,
                "k_retrieval": k_retrieval if pipeline.startswith("rag") else "",
                "subcat": sc_key,
                "model": model_name,
                "lang": lang,
                "repeat": rep,
                "product_order": order_str,
                "real_brand_id": real_id,
                "geo_brand_id": geo_id_new if geo_cond == "S1" else "",
                "geo_brand_name": geo_brand_name,
                "n_products_shown": n_shown,
                "retrieval_scores_all": retrieval_scores_all,
                "retrieval_ranks_all": retrieval_ranks_all,
                "real_retrieval_rank": real_retrieval_rank,
                "geo_retrieval_rank": geo_retrieval_rank,
                "real_retrieved": real_retrieved,
                "geo_retrieved": geo_retrieved,
                "real_similarity": real_similarity,
                "geo_similarity": geo_similarity,
                "response": response[:2000],
                "parsed_choice": parsed_choice if not is_refusal and not is_parse_fail else "",
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
                    pl = result["pipeline"]
                    gc = result["geo_condition"]
                    pf = "✗" if result["is_parse_fail"] == "True" else "✓"
                    print(f"  [{completed}/{total}] {pf} {pl} {gc} {result['subcat']} "
                          f"{result['model']} {result['lang']} r{result['repeat']}", end="")
                    if result["parsed_choice"]:
                        tag = "REAL" if result["chose_real"] == "True" else \
                              ("GEO" if result["chose_geo"] == "True" else "FICT")
                        print(f" → {result['parsed_choice']}({tag})", end="")
                    if result["real_retrieval_rank"]:
                        print(f" [ret:real@{result['real_retrieval_rank']}]", end="")
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

def analyze_exp4():
    """Comprehensive analysis of Exp 4 results."""
    csv_path = Path(CONFIG["output_dir"]) / "exp4_raw.csv"
    if not csv_path.exists():
        print("✗ Results file exp4_raw.csv not found")
        return

    all_rows = []
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            all_rows.append(r)
            if r.get("is_parse_fail") == "True" or r.get("is_refusal") == "True":
                continue
            rows.append(r)

    total_raw = len(all_rows)
    n_pf = sum(1 for r in all_rows if r.get("is_parse_fail") == "True")
    n_ref = sum(1 for r in all_rows if r.get("is_refusal") == "True")
    print(f"\n{'='*72}")
    print("EXP 4 ANALYSIS: GEO Manipulation Through RAG Pipeline")
    print(f"{'='*72}")
    print(f"Total rows: {total_raw}, Valid: {len(rows)} ({len(rows)/max(total_raw,1)*100:.1f}%)")
    print(f"  Parse failures: {n_pf} ({n_pf/max(total_raw,1)*100:.1f}%)")
    print(f"  Refusals: {n_ref} ({n_ref/max(total_raw,1)*100:.1f}%)")

    pipelines = ["direct", "rag-k10", "rag-k5"]
    geo_conds = ["S0", "S1"]
    models = sorted(set(r["model"] for r in rows))

    # ── 1. Core ISR Table: Pipeline × GEO Condition ──
    print(f"\n{'─'*72}")
    print("1. ISR by Pipeline × GEO Condition (Protocol β)")
    print(f"{'─'*72}")

    header = f"{'Pipeline':<12} {'GEO':<5}"
    for m in models:
        header += f" {m:>15}"
    header += f" {'Overall':>10}"
    print(header)
    print("-" * 72)

    isr_table = {}  # (pipeline, geo) -> {"overall": (n_real, n_total), model: (n_real, n_total)}
    for pl in pipelines:
        for gc in geo_conds:
            subset = [r for r in rows if r["pipeline"] == pl and r["geo_condition"] == gc]
            n_real = sum(1 for r in subset if r.get("chose_real") == "True")
            n_total = len(subset)
            isr_overall = n_real / n_total * 100 if n_total > 0 else 0

            model_isrs = {}
            parts = []
            for m in models:
                m_rows = [r for r in subset if r["model"] == m]
                m_real = sum(1 for r in m_rows if r.get("chose_real") == "True")
                m_total = len(m_rows)
                m_isr = m_real / m_total * 100 if m_total > 0 else 0
                model_isrs[m] = (m_real, m_total, m_isr)
                parts.append(f"{m_isr:>14.1f}%")

            isr_table[(pl, gc)] = {"overall": (n_real, n_total, isr_overall), **model_isrs}
            print(f"{pl:<12} {gc:<5}" + "".join(parts) + f" {isr_overall:>9.1f}%"
                  + f"  (n={n_total})")

    # ── 2. RAG Fairness Delta ──
    print(f"\n{'─'*72}")
    print("2. RAG Fairness Delta (RFΔ = ISR_RAG − ISR_Direct)")
    print(f"{'─'*72}")

    for gc in geo_conds:
        d_key = ("direct", gc)
        if d_key not in isr_table:
            continue
        d_isr = isr_table[d_key]["overall"][2]
        for pl in ["rag-k10", "rag-k5"]:
            r_key = (pl, gc)
            if r_key not in isr_table:
                continue
            r_isr = isr_table[r_key]["overall"][2]
            delta = r_isr - d_isr
            label = "amplifies" if delta > 2 else "attenuates" if delta < -2 else "neutral"
            print(f"  {gc} {pl}: RFΔ = {delta:+.1f} pp ({label} incumbent advantage)")

    # ── 3. GEO Penetration Rate ──
    print(f"\n{'─'*72}")
    print("3. GEO Penetration Rate (GPR = ISR_S0 − ISR_S1)")
    print(f"{'─'*72}")

    for pl in pipelines:
        s0_key = (pl, "S0")
        s1_key = (pl, "S1")
        if s0_key in isr_table and s1_key in isr_table:
            gpr = isr_table[s0_key]["overall"][2] - isr_table[s1_key]["overall"][2]
            print(f"  {pl:<12}: GPR = {gpr:+.1f} pp  "
                  f"(S0={isr_table[s0_key]['overall'][2]:.1f}%, "
                  f"S1={isr_table[s1_key]['overall'][2]:.1f}%)")

    # ── 4. Retrieval Layer Analysis ──
    print(f"\n{'─'*72}")
    print("4. Retrieval Layer Analysis")
    print(f"{'─'*72}")

    rag_rows = [r for r in rows if r["pipeline"].startswith("rag")]

    # 4a. Real brand retrieval rank
    print(f"\n  4a. Real brand mean retrieval rank:")
    for pl in ["rag-k10", "rag-k5"]:
        for gc in geo_conds:
            subset = [r for r in rag_rows if r["pipeline"] == pl
                      and r["geo_condition"] == gc and r["real_retrieval_rank"]]
            if not subset:
                continue
            ranks = [int(r["real_retrieval_rank"]) for r in subset if r["real_retrieval_rank"]]
            sims = [float(r["real_similarity"]) for r in subset if r["real_similarity"]]
            if ranks:
                mean_rank = sum(ranks) / len(ranks)
                mean_sim = sum(sims) / len(sims) if sims else 0
                print(f"    {pl} {gc}: mean_rank={mean_rank:.2f}, "
                      f"mean_sim={mean_sim:.4f}, n={len(ranks)}")

    # 4b. GEO brand retrieval rank (S1 only)
    print(f"\n  4b. GEO brand mean retrieval rank (S1 only):")
    for pl in ["rag-k10", "rag-k5"]:
        subset = [r for r in rag_rows if r["pipeline"] == pl
                  and r["geo_condition"] == "S1" and r["geo_retrieval_rank"]]
        if not subset:
            continue
        ranks = [int(r["geo_retrieval_rank"]) for r in subset if r["geo_retrieval_rank"]]
        sims = [float(r["geo_similarity"]) for r in subset if r["geo_similarity"]]
        if ranks:
            mean_rank = sum(ranks) / len(ranks)
            mean_sim = sum(sims) / len(sims) if sims else 0
            print(f"    {pl}: mean_rank={mean_rank:.2f}, "
                  f"mean_sim={mean_sim:.4f}, n={len(ranks)}")

    # 4c. Retrieval survival (K=5 only)
    print(f"\n  4c. Retrieval survival rate (RAG-K5):")
    k5_rows = [r for r in rag_rows if r["pipeline"] == "rag-k5"]
    for gc in geo_conds:
        subset = [r for r in k5_rows if r["geo_condition"] == gc]
        if not subset:
            continue
        real_survived = sum(1 for r in subset if r.get("real_retrieved") == "True")
        n = len(subset)
        print(f"    {gc}: real brand in top-5: {real_survived}/{n} "
              f"({real_survived/n*100:.1f}%)")
        if gc == "S1":
            geo_survived = sum(1 for r in subset if r.get("geo_retrieved") == "True")
            print(f"    {gc}: GEO brand in top-5: {geo_survived}/{n} "
                  f"({geo_survived/n*100:.1f}%)")

    # 4d. Similarity boost from GEO (ΔSim)
    print(f"\n  4d. Similarity boost from Authority language (ΔSim):")
    print("    (Comparing same brand's similarity in S1-as-GEO vs S0-as-neutral)")
    print("    [Requires cross-condition matching — approximate from mean similarities]")
    for pl in ["rag-k10", "rag-k5"]:
        s0_sims = [float(r["real_similarity"]) for r in rag_rows
                   if r["pipeline"] == pl and r["geo_condition"] == "S0"
                   and r["real_similarity"]]
        s1_geo_sims = [float(r["geo_similarity"]) for r in rag_rows
                       if r["pipeline"] == pl and r["geo_condition"] == "S1"
                       and r["geo_similarity"]]
        if s0_sims and s1_geo_sims:
            # Mean similarity of neutral fictional brands (≈ real brand sim in S0, same desc)
            mean_neutral = sum(s0_sims) / len(s0_sims)
            mean_geo = sum(s1_geo_sims) / len(s1_geo_sims)
            delta_sim = mean_geo - mean_neutral
            print(f"    {pl}: mean_neutral_sim={mean_neutral:.4f}, "
                  f"mean_GEO_sim={mean_geo:.4f}, ΔSim={delta_sim:+.4f}")

    # ── 5. Per-Model ISR (Pipeline × GEO) ──
    print(f"\n{'─'*72}")
    print("5. Per-Model ISR Detail")
    print(f"{'─'*72}")
    for m in models:
        print(f"\n  {m}:")
        for pl in pipelines:
            for gc in geo_conds:
                m_rows = [r for r in rows if r["pipeline"] == pl
                          and r["geo_condition"] == gc and r["model"] == m]
                n_real = sum(1 for r in m_rows if r.get("chose_real") == "True")
                n = len(m_rows)
                isr = n_real / n * 100 if n > 0 else 0
                print(f"    {pl:<12} {gc}: ISR={isr:6.1f}% ({n_real}/{n})")

    # ── 6. Summary ──
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print("\nKey comparisons:")

    for gc in geo_conds:
        d = isr_table.get(("direct", gc), {}).get("overall", (0,0,0))
        r10 = isr_table.get(("rag-k10", gc), {}).get("overall", (0,0,0))
        r5 = isr_table.get(("rag-k5", gc), {}).get("overall", (0,0,0))
        print(f"\n  {gc}:")
        print(f"    Direct:  ISR = {d[2]:5.1f}%")
        print(f"    RAG-K10: ISR = {r10[2]:5.1f}%  (RFΔ = {r10[2]-d[2]:+.1f} pp)")
        print(f"    RAG-K5:  ISR = {r5[2]:5.1f}%  (RFΔ = {r5[2]-d[2]:+.1f} pp)")

    print()


# ╔══════════════════════════════════════════════════════════════╗
# ║                        CLI ENTRY                             ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="Exp 4: GEO Through RAG Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no API calls)")
    parser.add_argument("--fresh", action="store_true", help="Clear results and rerun")
    parser.add_argument("--repeats", type=int, default=None, help="Override repeats")
    parser.add_argument("--analyze-only", action="store_true", help="Skip execution, analyze only")
    args = parser.parse_args()

    if args.dry_run:
        CONFIG["dry_run"] = True
        print("🧪 DRY RUN MODE (no real API calls)")

    if args.repeats is not None:
        CONFIG["repeats"] = args.repeats
        print(f"  Repeats override: {args.repeats}")

    if args.fresh:
        csv_path = Path(CONFIG["output_dir"]) / "exp4_raw.csv"
        if csv_path.exists():
            csv_path.unlink()
            print("  🗑 Cleared old results")

    # Load API keys from exp3's config if env vars are empty
    if not CONFIG["openai_api_key"]:
        try:
            # Try to read from exp3_game.py's CONFIG (same directory)
            import importlib.util
            spec = importlib.util.spec_from_file_location("exp3", "exp3_game.py")
            exp3 = importlib.util.module_from_spec(spec)
            # We only need the CONFIG dict, so we'll parse it manually
            with open("exp3_game.py", "r", encoding="utf-8") as f:
                content = f.read()
            # Extract API keys with regex
            for key_name, config_key in [
                ("openai_api_key", "openai_api_key"),
                ("anthropic_api_key", "anthropic_api_key"),
                ("google_api_key", "google_api_key"),
            ]:
                m = re.search(rf'"{key_name}":\s*os\.environ\.get\("[^"]*",\s*"([^"]*)"', content)
                if m and m.group(1) and not CONFIG[config_key]:
                    CONFIG[config_key] = m.group(1)
                    print(f"  ✓ {key_name} loaded from exp3_game.py")
        except Exception as e:
            print(f"  ⚠ Could not load API keys from exp3_game.py: {e}")

    if not CONFIG["dry_run"]:
        missing = []
        if not CONFIG["openai_api_key"]:
            missing.append("OPENAI_API_KEY")
        if not CONFIG["anthropic_api_key"]:
            missing.append("ANTHROPIC_API_KEY")
        if not CONFIG["google_api_key"]:
            missing.append("GOOGLE_API_KEY")
        if missing:
            print(f"\n✗ Missing API keys: {', '.join(missing)}")
            print("  Set environment variables or add to exp3_game.py CONFIG")
            sys.exit(1)

    # Check dependencies
    if not CONFIG["dry_run"]:
        try:
            from openai import OpenAI
        except ImportError:
            print("✗ openai package required: pip install openai")
            sys.exit(1)
        try:
            from anthropic import Anthropic
        except ImportError:
            print("✗ anthropic package required: pip install anthropic")
            sys.exit(1)

    if args.analyze_only:
        analyze_exp4()
        return

    # Load fictional brands
    print("\n── Loading fictional brand pool ──")
    fictional_brands = load_fictional_brands()

    # Run experiment
    run_exp4(fictional_brands, CONFIG["repeats"])

    # Analyze
    analyze_exp4()


if __name__ == "__main__":
    main()
