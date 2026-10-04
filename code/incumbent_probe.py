"""
Incumbent Awareness Probe — Exp 1 Pre-flight Experiment 2.1
===========================================================
Verifies the candidate real brands' "incumbent" status in LLMs.

Usage:
  1. pip install openai anthropic google-generativeai together pandas
  2. Edit the CONFIG section below and fill in API keys
  3. python incumbent_probe.py
  4. Results are written to the results/ directory

Budget estimate: ~$5-7 (5 subcategories × 7 candidate brands × 4 models × 2 languages × 30 reps × 3 questions)
"""

import os, json, csv, time, re, sys, signal, threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

_STOP = threading.Event()

def _handle_sigint(sig, frame):
    print("\n\n⚠ Interrupt received, graceful exit (completed results will be saved)...")
    _STOP.set()

signal.signal(signal.SIGINT, _handle_sigint)
if hasattr(signal, "SIGBREAK"):          # Windows: Ctrl+Break
    signal.signal(signal.SIGBREAK, _handle_sigint)

# ╔══════════════════════════════════════════════════════════════╗
# ╚══════════════════════════════════════════════════════════════╝

CONFIG = {
    "openai_api_key":    os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
    "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY"),
    "google_api_key":    os.environ.get("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY"),
    "together_api_key":  os.environ.get("TOGETHER_API_KEY", ""),

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
        "gemini-3-flash": {
            "enabled": True,
            "provider": "google",
            "model_id": "gemini-3-flash-preview",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        "llama-3.1-70b": {
            "enabled": False,
            "provider": "together",
            "model_id": "meta-llama/Llama-3.1-70B-Instruct-Turbo",
            "temperature": 0.7,
            "max_tokens": 1024,
        },
    },

    "languages": ["zh", "en"],
    "repeats_per_condition": 30,
    "max_parallel_workers": 20,
    "retry_attempts": 3,               # Number of retries on API failure
    "retry_delay_seconds": 5,

    "output_dir": "results/incumbent_probe",

    "dry_run": False,
}

# ╔══════════════════════════════════════════════════════════════╗
# ╚══════════════════════════════════════════════════════════════╝

SUBCATEGORIES = {
    "moisturizer": {
        "zh_name": "保湿乳液",
        "en_name": "facial moisturizer",
        "status": "locked",            # locked = anchor confirmed
        "candidates": [
            {
                "brand": "CeraVe",
                "product": "PM Facial Moisturizing Lotion",
                "zh_brand": "CeraVe（适乐肤）",
                "keywords": ["cerave", "适乐肤", "pm lotion", "pm facial"],
            }
        ],
    },
    "bha_exfoliant": {
        "zh_name": "水杨酸精华/BHA精华",
        "en_name": "BHA exfoliant",
        "status": "locked",
        "candidates": [
            {
                "brand": "Paula's Choice",
                "product": "2% BHA Liquid Exfoliant",
                "zh_brand": "宝拉珍选",
                "keywords": ["paula", "宝拉", "bha liquid", "2% bha"],
            }
        ],
    },
    "sunscreen": {
        "zh_name": "防晒霜",
        "en_name": "sunscreen",
        "status": "locked",
        "candidates": [
            {
                "brand": "EltaMD",
                "product": "UV Clear SPF 46",
                "zh_brand": "EltaMD",
                "keywords": ["eltamd", "uv clear", "spf 46"],
            }
        ],
    },
    "shampoo": {
        "zh_name": "控油洗发水",
        "en_name": "oil-control shampoo",
        "status": "candidate",         # candidate = pending preflight screening
        "candidates": [
            {
                "brand": "Head & Shoulders",
                "product": "Clinical Strength Dandruff Shampoo",
                "zh_brand": "海飞丝",
                "keywords": ["head & shoulders", "head and shoulders", "海飞丝", "h&s", "clinical strength"],
            },
            {
                "brand": "Nizoral",
                "product": "A-D Anti-Dandruff Shampoo",
                "zh_brand": "Nizoral（仁山利舒）",
                "keywords": ["nizoral", "仁山利舒", "ketoconazole", "酮康唑"],
            },
        ],
    },
    "cleanser": {
        "zh_name": "洗面奶",
        "en_name": "facial cleanser",
        "status": "candidate",
        "candidates": [
            {
                "brand": "CeraVe",
                "product": "Foaming Facial Cleanser",
                "zh_brand": "CeraVe（适乐肤）",
                "keywords": ["cerave", "适乐肤", "foaming cleanser", "泡沫洁面"],
            },
            {
                "brand": "La Roche-Posay",
                "product": "Toleriane Purifying Foaming Cleanser",
                "zh_brand": "理肤泉",
                "keywords": ["la roche", "理肤泉", "toleriane", "purifying"],
            },
        ],
    },
}

# ╔══════════════════════════════════════════════════════════════╗
# ╚══════════════════════════════════════════════════════════════╝

def build_prompts(subcat_key, brand_info, lang):
    """Generate Q1/Q2/Q3 prompt sets for one subcategory x brand x language."""
    sc = SUBCATEGORIES[subcat_key]
    cat_zh = sc["zh_name"]
    cat_en = sc["en_name"]
    brand = brand_info["brand"]
    brand_zh = brand_info["zh_brand"]

    if lang == "zh":
        q1 = (
            "请列举你所知道的护肤品牌和个人洗护品牌 Top 20，按你的熟悉度从高到低排序。\n"
            "请严格按以下格式输出，每行一个：\n"
            "1. 品牌名\n2. 品牌名\n...\n20. 品牌名"
        )
        q2 = (
            f"如果有人问你推荐{cat_zh}，你最推荐哪 5 个品牌？\n"
            f"请严格按以下格式输出：\n"
            f"1. 品牌名 - 一句话理由\n2. 品牌名 - 一句话理由\n...\n5. 品牌名 - 一句话理由"
        )
        q3 = (
            f"请简要介绍一下 {brand_zh} 这个品牌。你对它了解多少？\n"
            f"请从以下几个方面回答（如果你知道的话）：\n"
            f"1. 品牌定位和主打产品\n"
            f"2. 核心成分或技术\n"
            f"3. 目标用户群\n"
            f"4. 市场口碑\n"
            f"如果你不太了解这个品牌，请直接说'我不太了解这个品牌'。"
        )
    else:  # en
        q1 = (
            "List your top 20 most familiar skincare and personal care brands, "
            "ranked from most to least familiar.\n"
            "Format:\n1. Brand Name\n2. Brand Name\n...\n20. Brand Name"
        )
        q2 = (
            f"If someone asked you to recommend {cat_en} products, what are your top 5 brand recommendations?\n"
            f"Format:\n1. Brand - one-line reason\n2. Brand - one-line reason\n...\n5. Brand - one-line reason"
        )
        q3 = (
            f"Please briefly describe the brand {brand}. How much do you know about it?\n"
            f"Cover these aspects (if you know):\n"
            f"1. Brand positioning and flagship products\n"
            f"2. Key ingredients or technology\n"
            f"3. Target audience\n"
            f"4. Market reputation\n"
            f"If you don't know much about this brand, just say 'I'm not very familiar with this brand'."
        )

    return {"Q1": q1, "Q2": q2, "Q3": q3}


# ╔══════════════════════════════════════════════════════════════╗
# ╚══════════════════════════════════════════════════════════════╝

def call_openai(prompt, model_cfg):
    """Call OpenAI API (GPT-4o)."""
    from openai import OpenAI
    client = OpenAI(api_key=CONFIG["openai_api_key"])
    resp = client.chat.completions.create(
        model=model_cfg["model_id"],
        messages=[{"role": "user", "content": prompt}],
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
    )
    return resp.choices[0].message.content


def call_anthropic(prompt, model_cfg):
    """Call Anthropic API (Claude Sonnet)."""
    from anthropic import Anthropic
    client = Anthropic(api_key=CONFIG["anthropic_api_key"])
    resp = client.messages.create(
        model=model_cfg["model_id"],
        max_tokens=model_cfg["max_tokens"],
        messages=[{"role": "user", "content": prompt}],
        temperature=model_cfg["temperature"],
    )
    return resp.content[0].text


def call_google(prompt, model_cfg):
    """Call Google Gemini API."""
    import google.generativeai as genai
    genai.configure(api_key=CONFIG["google_api_key"])
    model = genai.GenerativeModel(
        model_cfg["model_id"],
        generation_config=genai.GenerationConfig(
            temperature=model_cfg["temperature"],
            max_output_tokens=model_cfg["max_tokens"],
        ),
    )
    resp = model.generate_content(prompt)
    return resp.text


def call_together(prompt, model_cfg):
    """Call Together AI API (LLaMA)."""
    from openai import OpenAI
    client = OpenAI(
        api_key=CONFIG["together_api_key"],
        base_url="https://api.together.xyz/v1",
    )
    resp = client.chat.completions.create(
        model=model_cfg["model_id"],
        messages=[{"role": "user", "content": prompt}],
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
    )
    return resp.choices[0].message.content


PROVIDER_MAP = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
    "together": call_together,
}


def call_llm(prompt, model_name, model_cfg):
    """Unified API entry point with retry logic."""
    if CONFIG["dry_run"]:
        return f"[DRY RUN] model={model_name}, prompt_len={len(prompt)}"

    provider = model_cfg["provider"]
    call_fn = PROVIDER_MAP[provider]

    for attempt in range(CONFIG["retry_attempts"]):
        try:
            return call_fn(prompt, model_cfg)
        except Exception as e:
            print(f"  [WARN] {model_name} attempt {attempt+1} failed: {e}")
            if attempt < CONFIG["retry_attempts"] - 1:
                time.sleep(CONFIG["retry_delay_seconds"] * (attempt + 1))
    return f"[ERROR] All {CONFIG['retry_attempts']} attempts failed for {model_name}"


# ╔══════════════════════════════════════════════════════════════╗
# ╚══════════════════════════════════════════════════════════════╝

def parse_ranked_list(response_text):
    """Extract the ranked list of brand names from an LLM response."""
    lines = response_text.strip().split("\n")
    brands = []
    for line in lines:
        m = re.match(r'^\s*\d+[\.\)\s]+\s*(.+?)(?:\s*[-–—:：].*)?$', line)
        if m:
            brand = m.group(1).strip().strip("*").strip()
            if brand:
                brands.append(brand)
    return brands


def score_q1(response_text, brand_info, top_n=20):
    """
    Q1 scoring: does the target brand appear in the Top 20 list?
    Returns: {found: bool, position: int|None, brands_listed: list}
    """
    brands = parse_ranked_list(response_text)
    keywords = [kw.lower() for kw in brand_info["keywords"]]
    brand_name = brand_info["brand"].lower()

    for i, b in enumerate(brands[:top_n]):
        b_lower = b.lower()
        if brand_name in b_lower or any(kw in b_lower for kw in keywords):
            return {"found": True, "position": i + 1, "brands_listed": brands[:top_n]}

    return {"found": False, "position": None, "brands_listed": brands[:top_n]}


def score_q2(response_text, brand_info, subcat_key, top_n=5):
    """
    Q2 scoring: does the target brand appear in the subcategory Top 5 recommendations?
    Returns: {found: bool, position: int|None, brands_listed: list}
    """
    brands = parse_ranked_list(response_text)
    keywords = [kw.lower() for kw in brand_info["keywords"]]
    brand_name = brand_info["brand"].lower()

    for i, b in enumerate(brands[:top_n]):
        b_lower = b.lower()
        if brand_name in b_lower or any(kw in b_lower for kw in keywords):
            return {"found": True, "position": i + 1, "brands_listed": brands[:top_n]}

    return {"found": False, "position": None, "brands_listed": brands[:top_n]}


def score_q3(response_text, brand_info):
    """
    Q3 scoring: can the model give an accurate brand description?
    Simple scoring rules (can later be refined with GPT-4o as a judge):
      - 3: detailed and accurate (mentions specific products/ingredients/positioning)
      - 2: basic knowledge (mentions the brand but details are vague)
      - 1: knows the name only
      - 0: explicitly states unfamiliarity, or information entirely wrong
    Returns: {score: int, explanation: str}
    """
    text = response_text.lower()
    brand_name = brand_info["brand"].lower()

    unfamiliar_signals = [
        "不太了解", "不太熟悉", "没有太多信息", "i'm not very familiar",
        "i don't know much", "not familiar", "i'm not sure",
        "无法确认", "not aware", "cannot confirm",
    ]
    if any(sig in text for sig in unfamiliar_signals):
        return {"score": 0, "explanation": "Model indicates unfamiliarity"}

    knowledge_keywords = [kw.lower() for kw in brand_info["keywords"]]
    hits = sum(1 for kw in knowledge_keywords if kw in text)

    if hits >= 3 and len(response_text) > 200:
        return {"score": 3, "explanation": f"Detailed description, {hits} keyword matches"}
    elif hits >= 2 or len(response_text) > 150:
        return {"score": 2, "explanation": f"Basic knowledge, {hits} keyword matches"}
    elif hits >= 1 or brand_name in text:
        return {"score": 1, "explanation": f"Minimal knowledge, {hits} keyword matches"}
    else:
        return {"score": 0, "explanation": "No relevant knowledge detected"}


# ╔══════════════════════════════════════════════════════════════╗
# ╚══════════════════════════════════════════════════════════════╝

def build_task_list():
    """Build all probe tasks (subcategory × brand × model × language × question type × repeats)."""
    tasks = []
    enabled_models = {
        name: cfg for name, cfg in CONFIG["models"].items() if cfg["enabled"]
    }

    for sc_key, sc_info in SUBCATEGORIES.items():
        for brand_info in sc_info["candidates"]:
            for model_name, model_cfg in enabled_models.items():
                for lang in CONFIG["languages"]:
                    prompts = build_prompts(sc_key, brand_info, lang)
                    for q_type, prompt in prompts.items():
                        for rep in range(CONFIG["repeats_per_condition"]):
                            tasks.append({
                                "subcat": sc_key,
                                "brand": brand_info["brand"],
                                "brand_info": brand_info,
                                "model": model_name,
                                "model_cfg": model_cfg,
                                "lang": lang,
                                "q_type": q_type,
                                "prompt": prompt,
                                "repeat": rep + 1,
                            })
    return tasks


def execute_single_task(task):
    """Run a single probe task and return a result row."""
    response = call_llm(task["prompt"], task["model"], task["model_cfg"])

    if task["q_type"] == "Q1":
        score = score_q1(response, task["brand_info"])
    elif task["q_type"] == "Q2":
        score = score_q2(response, task["brand_info"], task["subcat"])
    else:  # Q3
        score = score_q3(response, task["brand_info"])

    return {
        "timestamp": datetime.now().isoformat(),
        "subcat": task["subcat"],
        "subcat_status": SUBCATEGORIES[task["subcat"]]["status"],
        "brand": task["brand"],
        "model": task["model"],
        "lang": task["lang"],
        "q_type": task["q_type"],
        "repeat": task["repeat"],
        "response": response[:500],
        "response_full_length": len(response),
        "found": score.get("found"),
        "position": score.get("position"),
        "q3_score": score.get("score"),
        "q3_explanation": score.get("explanation"),
    }


def run_probe():
    """Run the full Incumbent Awareness Probe."""
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_tasks = build_task_list()

    csv_path = output_dir / "incumbent_awareness_raw.csv"
    results = []
    done_keys = set()

    rerun_q2 = "--rerun-q2" in sys.argv

    if csv_path.exists() and not CONFIG["dry_run"]:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("response", "").startswith("[ERROR]") or row.get("response", "").startswith("[EXCEPTION]"):
                    continue
                if rerun_q2 and row.get("q_type") == "Q2":
                    continue
                results.append(row)
                key = (row["subcat"], row["brand"], row["model"],
                       row["lang"], row["q_type"], row["repeat"])
                done_keys.add(key)
    if rerun_q2:
        print("⚡ --rerun-q2 mode: old Q2 data discarded; Q2 will be rerun with the new prompt (with user persona)")

    tasks = []
    for t in all_tasks:
        key = (t["subcat"], t["brand"], t["model"],
               t["lang"], t["q_type"], str(t["repeat"]))
        if key not in done_keys:
            tasks.append(t)

    skipped = len(all_tasks) - len(tasks)
    total = len(tasks)

    print(f"=" * 60)
    print(f"Incumbent Awareness Probe")
    print(f"=" * 60)
    if skipped > 0:
        print(f"⚡ Resume from checkpoint: loaded {len(results)} historical results, skipping {skipped} completed tasks")
    print(f"Remaining tasks: {total}  (total planned: {len(all_tasks)})")
    print(f"Subcategories: {len(SUBCATEGORIES)}")
    print(f"Candidate brands: {sum(len(sc['candidates']) for sc in SUBCATEGORIES.values())}")
    print(f"Models: {[m for m, c in CONFIG['models'].items() if c['enabled']]}")
    print(f"Languages: {CONFIG['languages']}")
    print(f"Repeats: {CONFIG['repeats_per_condition']}")
    print(f"Dry run: {CONFIG['dry_run']}")
    print(f"Output: {output_dir}")

    est_tokens = total * 300
    est_cost = est_tokens / 1_000_000 * 15
    print(f"Estimated remaining tokens: ~{est_tokens:,}")
    print(f"Estimated remaining cost: ~${est_cost:.2f}")
    print(f"=" * 60)

    if total == 0:
        print("\n✓ All tasks completed, nothing to rerun. Generating report...")
        generate_report(results, output_dir)
        return results

    if not CONFIG["dry_run"]:
        confirm = input("\nConfirm execution? (y/N) > ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return

    completed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=CONFIG["max_parallel_workers"]) as executor:
        batch_size = CONFIG["max_parallel_workers"] * 2
        task_batches = [tasks[i:i+batch_size] for i in range(0, len(tasks), batch_size)]

        for batch in task_batches:
            if _STOP.is_set():
                print("\n⚠ User interrupt, skipping remaining tasks...")
                break

            futures = {executor.submit(execute_single_task, t): t for t in batch}
            for future in as_completed(futures):
                if _STOP.is_set():
                    break
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    task = futures[future]
                    print(f"  [ERROR] {task['model']}/{task['lang']}/{task['q_type']}: {e}")
                    results.append({
                        "timestamp": datetime.now().isoformat(),
                        "subcat": task["subcat"],
                        "brand": task["brand"],
                        "model": task["model"],
                        "lang": task["lang"],
                        "q_type": task["q_type"],
                        "repeat": task["repeat"],
                        "response": f"[EXCEPTION] {str(e)[:200]}",
                        "found": None,
                        "position": None,
                        "q3_score": None,
                })
            completed += 1
            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"  Progress: {completed}/{total} ({completed*100//total}%) "
                      f"| {rate:.1f} tasks/s | ETA: {eta:.0f}s")

    elapsed_total = time.time() - start_time
    if _STOP.is_set():
        print(f"\n⚠ Interrupted early, completed {completed}/{total} tasks ({elapsed_total:.1f}s)")
        print(f"  The {len(results)} collected results will still be saved and analyzed.")
    else:
        print(f"\nAll tasks completed in {elapsed_total:.1f}s")

    csv_path = output_dir / "incumbent_awareness_raw.csv"
    if results:
        fieldnames = results[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Raw results saved to: {csv_path}")

    generate_report(results, output_dir)

    return results


# ╔══════════════════════════════════════════════════════════════╗
# ╚══════════════════════════════════════════════════════════════╝

def generate_report(results, output_dir):
    """Generate the pass/fail decision table and a readable report."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("INCUMBENT AWARENESS PROBE — SUMMARY REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)

    for r in results:
        # found: "True"/"False"/""  → bool/None
        f = r.get("found")
        if isinstance(f, str):
            r["found"] = True if f == "True" else (False if f == "False" else None)
        # position: "3"/""  → int/None
        p = r.get("position")
        if isinstance(p, str):
            r["position"] = int(p) if p and p not in ("None", "") else None
        # q3_score: "2"/""  → int/None
        q = r.get("q3_score")
        if isinstance(q, str):
            r["q3_score"] = int(q) if q and q not in ("None", "") else None

    groups = defaultdict(list)
    for r in results:
        key = (r["subcat"], r["brand"], r["model"], r["lang"], r["q_type"])
        groups[key].append(r)

    report_lines.append("\n" + "─" * 70)
    report_lines.append("Q1: Brand in Top-20 Familiarity List")
    report_lines.append("─" * 70)
    report_lines.append(f"{'Subcat':<15} {'Brand':<22} {'Model':<18} {'Lang':<6} {'Hit%':<8} {'AvgPos':<8} {'Pass?'}")

    q1_scores = {}  # (subcat, brand) → list of hit rates

    for key, rows in sorted(groups.items()):
        sc, brand, model, lang, q_type = key
        if q_type != "Q1":
            continue
        hits = sum(1 for r in rows if r.get("found"))
        total = len(rows)
        hit_rate = hits / total if total > 0 else 0
        positions = [r["position"] for r in rows if r.get("position")]
        avg_pos = sum(positions) / len(positions) if positions else float("nan")
        passed = "✓" if hit_rate >= 1.0 else "✗"
        report_lines.append(
            f"{sc:<15} {brand:<22} {model:<18} {lang:<6} {hit_rate*100:>5.1f}%  {avg_pos:>5.1f}   {passed}"
        )
        q1_scores.setdefault((sc, brand), []).append(hit_rate)

    report_lines.append("\n" + "─" * 70)
    report_lines.append("Q2: Brand in Category Top-5 Recommendations")
    report_lines.append("─" * 70)
    report_lines.append(f"{'Subcat':<15} {'Brand':<22} {'Model':<18} {'Lang':<6} {'Hit%':<8} {'AvgPos':<8} {'Pass?'}")

    q2_scores = {}

    for key, rows in sorted(groups.items()):
        sc, brand, model, lang, q_type = key
        if q_type != "Q2":
            continue
        hits = sum(1 for r in rows if r.get("found"))
        total = len(rows)
        hit_rate = hits / total if total > 0 else 0
        positions = [r["position"] for r in rows if r.get("position")]
        avg_pos = sum(positions) / len(positions) if positions else float("nan")
        passed = "✓" if hit_rate >= 0.80 else "✗"
        report_lines.append(
            f"{sc:<15} {brand:<22} {model:<18} {lang:<6} {hit_rate*100:>5.1f}%  {avg_pos:>5.1f}   {passed}"
        )
        q2_scores.setdefault((sc, brand), []).append(hit_rate)

    report_lines.append("\n" + "─" * 70)
    report_lines.append("Q3: Brand Knowledge Depth (0=unknown, 3=detailed)")
    report_lines.append("─" * 70)
    report_lines.append(f"{'Subcat':<15} {'Brand':<22} {'Model':<18} {'Lang':<6} {'AvgScore':<10} {'Pass?'}")

    q3_scores = {}

    for key, rows in sorted(groups.items()):
        sc, brand, model, lang, q_type = key
        if q_type != "Q3":
            continue
        scores = [r["q3_score"] for r in rows if r.get("q3_score") is not None]
        avg_score = sum(scores) / len(scores) if scores else 0
        passed = "✓" if avg_score >= 2.0 else "✗"
        report_lines.append(
            f"{sc:<15} {brand:<22} {model:<18} {lang:<6} {avg_score:>7.2f}    {passed}"
        )
        q3_scores.setdefault((sc, brand), []).append(avg_score)

    report_lines.append("\n" + "=" * 70)
    report_lines.append("FINAL DECISION TABLE — Subcategory × Brand Pass/Fail")
    report_lines.append("=" * 70)
    report_lines.append(
        f"{'Subcat':<15} {'Brand':<22} {'Status':<10} "
        f"{'Q1 avg':<10} {'Q2 avg':<10} {'Q3 avg':<10} {'VERDICT'}"
    )

    decisions = []
    for sc_key, sc_info in SUBCATEGORIES.items():
        for brand_info in sc_info["candidates"]:
            brand = brand_info["brand"]
            q1_avg = (sum(q1_scores.get((sc_key, brand), [0])) /
                      max(len(q1_scores.get((sc_key, brand), [1])), 1))
            q2_avg = (sum(q2_scores.get((sc_key, brand), [0])) /
                      max(len(q2_scores.get((sc_key, brand), [1])), 1))
            q3_avg = (sum(q3_scores.get((sc_key, brand), [0])) /
                      max(len(q3_scores.get((sc_key, brand), [1])), 1))

            q1_pass = q1_avg >= 1.0
            q2_pass = q2_avg >= 0.80
            q3_pass = q3_avg >= 2.0
            overall = "PASS ✓" if (q1_pass and q2_pass and q3_pass) else "FAIL ✗"

            report_lines.append(
                f"{sc_key:<15} {brand:<22} {sc_info['status']:<10} "
                f"{q1_avg*100:>6.1f}%   {q2_avg*100:>6.1f}%   {q3_avg:>6.2f}     {overall}"
            )
            decisions.append({
                "subcat": sc_key,
                "brand": brand,
                "status": sc_info["status"],
                "q1_avg": q1_avg,
                "q2_avg": q2_avg,
                "q3_avg": q3_avg,
                "verdict": overall,
            })

    report_lines.append("\n" + "=" * 70)
    report_lines.append("RECOMMENDED SUBCATEGORY SELECTION")
    report_lines.append("=" * 70)

    passed_subcats = set()
    for d in decisions:
        if "PASS" in d["verdict"]:
            passed_subcats.add(d["subcat"])

    locked_pass = [s for s in passed_subcats if SUBCATEGORIES[s]["status"] == "locked"]
    candidate_pass = [s for s in passed_subcats if SUBCATEGORIES[s]["status"] == "candidate"]

    report_lines.append(f"Locked subcategories passing: {locked_pass}")
    report_lines.append(f"Candidate subcategories passing: {candidate_pass}")
    report_lines.append(f"Total passing: {len(passed_subcats)}")

    if len(passed_subcats) <= 3:
        report_lines.append("→ Recommendation: STRICT path ($505 budget)")
    elif len(passed_subcats) == 4:
        report_lines.append("→ Recommendation: EXPANDED path (~$900 budget)")
    else:
        report_lines.append("→ Recommendation: FULL path (~$1,500 budget)")

    for sc_key in ["shampoo", "cleanser"]:
        sc_decisions = [d for d in decisions if d["subcat"] == sc_key]
        if len(sc_decisions) > 1:
            best = max(sc_decisions, key=lambda d: d["q2_avg"])
            report_lines.append(
                f"\n[{sc_key}] Winner: {best['brand']} "
                f"(Q2={best['q2_avg']*100:.1f}% vs "
                f"{[d for d in sc_decisions if d != best][0]['brand']}="
                f"{[d for d in sc_decisions if d != best][0]['q2_avg']*100:.1f}%)"
            )

    report_text = "\n".join(report_lines)
    report_path = output_dir / "incumbent_awareness_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")
    print("\n" + report_text)

    decision_path = output_dir / "final_subcategories.json"
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "decisions": decisions,
            "passed_subcats": list(passed_subcats),
            "recommendation": (
                "strict" if len(passed_subcats) <= 3 else
                "expanded" if len(passed_subcats) == 4 else "full"
            ),
        }, f, indent=2, ensure_ascii=False)
    print(f"Decision JSON saved to: {decision_path}")


# ╔══════════════════════════════════════════════════════════════╗
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        CONFIG["dry_run"] = True
    if "--repeats" in sys.argv:
        idx = sys.argv.index("--repeats")
        CONFIG["repeats_per_condition"] = int(sys.argv[idx + 1])

    run_probe()
