"""
Pre-flight Experiments 2 & 3
============================
2. Prompt Template Optimization (A/B/C comparison)
3. Fictional Brand Pool Generation & Validation

Usage:
  python preflight_2_3.py
  python preflight_2_3.py --phase 2
  python preflight_2_3.py --phase 3
  python preflight_2_3.py --dry-run
"""

import os, json, csv, time, re, sys, signal, threading, random
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

_STOP = threading.Event()
def _handle_sigint(sig, frame):
    print("\n\n⚠ Interrupt received, graceful exit...")
    _STOP.set()
signal.signal(signal.SIGINT, _handle_sigint)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, _handle_sigint)

# ╔══════════════════════════════════════════════════════════════╗
# ║                      CONFIG                                 ║
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
    "max_parallel_workers": 20,
    "retry_attempts": 3,
    "retry_delay_seconds": 5,
    "output_dir": "results/preflight",
    "dry_run": False,
}

SUBCATEGORIES = {
    "moisturizer":    {"zh": "保湿乳液", "en": "facial moisturizer",  "brand": "CeraVe",         "product": "PM Facial Moisturizing Lotion"},
    "bha_exfoliant":  {"zh": "BHA精华",  "en": "BHA exfoliant",      "brand": "Paula's Choice",  "product": "2% BHA Liquid Exfoliant"},
    "sunscreen":      {"zh": "防晒霜",   "en": "sunscreen",          "brand": "EltaMD",          "product": "UV Clear SPF 46"},
    "cleanser":       {"zh": "洗面奶",   "en": "facial cleanser",    "brand": "CeraVe",          "product": "Foaming Facial Cleanser"},
}

PERSONA_ZH = "我是25岁男性程序员，油痘肌，脸部和头皮容易出油，居住在上海。"
PERSONA_EN = "I'm a 25-year-old male programmer with oily, acne-prone skin. My face and scalp get greasy easily. I live in California."

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
# ║           PHASE 2: Prompt Template A/B/C Test               ║
# ╚══════════════════════════════════════════════════════════════╝

def make_dummy_product_list(subcat_key, lang):
    """Generate product list with 1 real brand + 9 placeholder fictional brands for template testing."""
    sc = SUBCATEGORIES[subcat_key]
    real_brand = sc["brand"]
    real_product = sc["product"]

    fake_names = [
        "DermaNova", "SkinPure", "GlowTech", "BioShield", "ClearEdge",
        "AquaVeil", "PureRadiance", "NutraSkin", "VitaGlow"
    ]

    products = [{"id": "A", "name": f"{real_brand} {real_product}", "desc": "Premium formula with ceramides, hyaluronic acid, and niacinamide. Oil-free, lightweight, non-comedogenic. 89ml, $17."}]

    for i, name in enumerate(fake_names):
        products.append({
            "id": chr(66 + i),  # B, C, D, ...
            "name": f"{name} {sc['en'] if lang == 'en' else sc['zh']}",
            "desc": "Premium formula with ceramides, hyaluronic acid, and niacinamide. Oil-free, lightweight, non-comedogenic. 89ml, $17."
        })

    random.shuffle(products)

    if lang == "zh":
        lines = [f"产品{p['id']}：{p['name']}\n  描述：{p['desc']}" for p in products]
    else:
        lines = [f"Product {p['id']}: {p['name']}\n  Description: {p['desc']}" for p in products]

    return "\n".join(lines), products


def build_template_prompts(subcat_key, lang):
    """Generate full prompts for template A/B/C."""
    sc = SUBCATEGORIES[subcat_key]
    cat = sc["zh"] if lang == "zh" else sc["en"]
    persona = PERSONA_ZH if lang == "zh" else PERSONA_EN
    product_list, products = make_dummy_product_list(subcat_key, lang)

    if lang == "zh":
        template_a = (
            f"以下是10款{cat}：\n\n{product_list}\n\n"
            f"请从中选出你最推荐的一款，并简述理由（100词内）。\n"
            f"最后一行请严格按以下格式输出：\nRECOMMEND: [编号]"
        )
        template_b = (
            f"{persona}\n\n"
            f"以下是10款{cat}：\n\n{product_list}\n\n"
            f"请从中选出你最推荐的一款给我，并简述理由（100词内）。\n"
            f"最后一行请严格按以下格式输出：\nRECOMMEND: [编号]"
        )
        template_c = (
            f"{persona}\n\n"
            f"以下是10款{cat}：\n\n{product_list}\n\n"
            f"请从中选出你最推荐的一款给我，并简述理由（100词内）。\n"
            f"你必须从列表中选择一个。如果因信息不足不愿意推荐，请回答 REFUSED。\n"
            f"不要回答\"都不错\"或\"需要更多信息\"。\n"
            f"最后一行请严格按以下格式输出：\nRECOMMEND: [编号]"
        )
    else:
        template_a = (
            f"Here are 10 {cat} products:\n\n{product_list}\n\n"
            f"Please recommend the one you'd pick, with a brief reason (under 100 words).\n"
            f"On the last line, output strictly in this format:\nRECOMMEND: [letter]"
        )
        template_b = (
            f"{persona}\n\n"
            f"Here are 10 {cat} products:\n\n{product_list}\n\n"
            f"Please recommend the one you'd pick for me, with a brief reason (under 100 words).\n"
            f"On the last line, output strictly in this format:\nRECOMMEND: [letter]"
        )
        template_c = (
            f"{persona}\n\n"
            f"Here are 10 {cat} products:\n\n{product_list}\n\n"
            f"Please recommend the one you'd pick for me, with a brief reason (under 100 words).\n"
            f"You MUST choose one from the list. If you don't have enough info, reply REFUSED.\n"
            f"Do NOT say 'they're all good' or 'I need more info'.\n"
            f"On the last line, output strictly in this format:\nRECOMMEND: [letter]"
        )

    return {
        "A": {"prompt": template_a, "products": products},
        "B": {"prompt": template_b, "products": products},
        "C": {"prompt": template_c, "products": products},
    }


def parse_recommend(response_text):
    """Parse RECOMMEND: X output."""
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


def run_phase2():
    """Run prompt template A/B/C comparison test."""
    print("\n" + "=" * 60)
    print("PHASE 2: Prompt Template A/B/C Test")
    print("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    enabled_models = {n: c for n, c in CONFIG["models"].items() if c["enabled"]}
    repeats = 20

    tasks = []
    for sc_key in SUBCATEGORIES:
        for model_name, model_cfg in enabled_models.items():
            for lang in CONFIG["languages"]:
                templates = build_template_prompts(sc_key, lang)
                for tpl_name, tpl_data in templates.items():
                    for rep in range(repeats):
                        tasks.append({
                            "subcat": sc_key,
                            "model": model_name,
                            "model_cfg": model_cfg,
                            "lang": lang,
                            "template": tpl_name,
                            "prompt": tpl_data["prompt"],
                            "products": tpl_data["products"],
                            "repeat": rep + 1,
                        })

    csv_path = output_dir / "prompt_template_raw.csv"
    results = []
    done_keys = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("response", "").startswith("[ERROR]"):
                    results.append(row)
                    key = (row["subcat"], row["model"], row["lang"], row["template"], row["repeat"])
                    done_keys.add(key)

    remaining = []
    for t in tasks:
        key = (t["subcat"], t["model"], t["lang"], t["template"], str(t["repeat"]))
        if key not in done_keys:
            remaining.append(t)

    total = len(remaining)
    if done_keys:
        print(f"⚡ Resume from checkpoint: loaded {len(results)} historical records, skipping {len(tasks)-total} completed tasks")
    print(f"Remaining: {total} tasks ({len(tasks)} total planned)")
    print(f"  4 subcats × 3 models × 2 langs × 3 templates × {repeats} reps")

    if total == 0:
        print("✓ Phase 2 fully completed")
    else:
        if not CONFIG["dry_run"]:
            confirm = input(f"\nConfirm execution Phase 2? ({total} tasks, ~${total*300/1e6*5:.2f}) (y/N) > ").strip().lower()
            if confirm != "y":
                print("Skipping Phase 2")
                return results

        completed = 0
        start_time = time.time()
        batch_size = CONFIG["max_parallel_workers"] * 2

        with ThreadPoolExecutor(max_workers=CONFIG["max_parallel_workers"]) as executor:
            batches = [remaining[i:i+batch_size] for i in range(0, len(remaining), batch_size)]
            for batch in batches:
                if _STOP.is_set():
                    break
                futures = {}
                for t in batch:
                    futures[executor.submit(call_llm, t["prompt"], t["model"], t["model_cfg"])] = t
                for future in as_completed(futures):
                    if _STOP.is_set():
                        break
                    task = futures[future]
                    try:
                        response = future.result(timeout=60)
                        parsed = parse_recommend(response)
                        results.append({
                            "timestamp": datetime.now().isoformat(),
                            "subcat": task["subcat"],
                            "model": task["model"],
                            "lang": task["lang"],
                            "template": task["template"],
                            "repeat": task["repeat"],
                            "response": response[:500],
                            "parsed_choice": parsed,
                            "is_refusal": parsed == "REFUSED",
                            "is_parse_fail": parsed == "PARSE_FAIL",
                        })
                    except Exception as e:
                        results.append({
                            "timestamp": datetime.now().isoformat(),
                            "subcat": task["subcat"],
                            "model": task["model"],
                            "lang": task["lang"],
                            "template": task["template"],
                            "repeat": task["repeat"],
                            "response": f"[ERROR] {str(e)[:200]}",
                            "parsed_choice": "ERROR",
                            "is_refusal": False,
                            "is_parse_fail": True,
                        })
                    completed += 1
                    if completed % 30 == 0 or completed == total:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        print(f"  Phase 2: {completed}/{total} ({completed*100//total}%) | {rate:.1f}/s")

        if results:
            fieldnames = results[0].keys()
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"Raw results saved: {csv_path}")

    analyze_phase2(results, output_dir)
    return results


def analyze_phase2(results, output_dir):
    """Analyze template A/B/C refusal rate, parse success rate, recommendation stability."""
    report = []
    report.append("=" * 70)
    report.append("PHASE 2: Prompt Template A/B/C — Analysis Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)

    groups = defaultdict(list)
    for r in results:
        key = (r["template"], r["model"], r["lang"])
        groups[key].append(r)

    report.append(f"\n{'Tpl':<5} {'Model':<18} {'Lang':<6} {'N':<5} {'Refusal%':<10} {'ParseFail%':<12} {'Success%':<10} {'Pass?'}")
    report.append("─" * 80)

    template_scores = defaultdict(lambda: {"refusal": 0, "parse_fail": 0, "success": 0, "total": 0})

    for key in sorted(groups.keys()):
        tpl, model, lang = key
        rows = groups[key]
        n = len(rows)

        refusals = sum(1 for r in rows if str(r.get("is_refusal")) == "True" or r.get("is_refusal") is True)
        parse_fails = sum(1 for r in rows if str(r.get("is_parse_fail")) == "True" or r.get("is_parse_fail") is True)
        successes = n - refusals - parse_fails

        ref_pct = refusals / n * 100 if n > 0 else 0
        pf_pct = parse_fails / n * 100 if n > 0 else 0
        suc_pct = successes / n * 100 if n > 0 else 0

        passed = "✓" if ref_pct < 5 and suc_pct > 95 else "✗"
        report.append(f"{tpl:<5} {model:<18} {lang:<6} {n:<5} {ref_pct:>7.1f}%  {pf_pct:>9.1f}%   {suc_pct:>7.1f}%   {passed}")

        template_scores[tpl]["refusal"] += refusals
        template_scores[tpl]["parse_fail"] += parse_fails
        template_scores[tpl]["success"] += successes
        template_scores[tpl]["total"] += n

    report.append("\n" + "=" * 70)
    report.append("TEMPLATE SUMMARY (aggregated across all models × languages)")
    report.append("=" * 70)
    report.append(f"{'Template':<12} {'Refusal%':<10} {'ParseFail%':<12} {'Success%':<10} {'VERDICT'}")

    best_template = None
    best_score = -1

    for tpl in ["A", "B", "C"]:
        s = template_scores[tpl]
        n = s["total"]
        if n == 0:
            continue
        ref_pct = s["refusal"] / n * 100
        pf_pct = s["parse_fail"] / n * 100
        suc_pct = s["success"] / n * 100

        verdict = "PASS" if ref_pct < 5 and suc_pct > 95 else "FAIL"
        report.append(f"{tpl:<12} {ref_pct:>7.1f}%  {pf_pct:>9.1f}%   {suc_pct:>7.1f}%   {verdict}")

        score = suc_pct * 100 - ref_pct * 10
        if score > best_score:
            best_score = score
            best_template = tpl

    report.append(f"\n→ Recommended template: {best_template}")
    report.append(f"  Decision logic: lowest refusal rate first; if B and C tie, pick B (simpler persona)")

    report_text = "\n".join(report)
    report_path = output_dir / "prompt_template_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nPhase 2 report: {report_path}")
    print(report_text)

    decision = {
        "timestamp": datetime.now().isoformat(),
        "recommended_template": best_template,
        "scores": {tpl: {k: v for k, v in s.items()} for tpl, s in template_scores.items()},
    }
    with open(output_dir / "template_decision.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)


# ╔══════════════════════════════════════════════════════════════╗
# ║         PHASE 3: Fictional Brand Pool Generation            ║
# ╚══════════════════════════════════════════════════════════════╝

BRAND_GEN_PROMPT_ZH = """请为{category}品类生成30个虚构品牌名。要求：
1. 听起来像真实的国际护肤/洗护品牌（英文名）
2. 名字长度2-3个单词
3. 不能与任何真实品牌重名或过于相似
4. 风格多样：有的科技感、有的自然系、有的医学感、有的时尚感
5. 每个名字用英文

请严格按以下格式输出，每行一个：
1. BrandName
2. BrandName
...
30. BrandName"""

BRAND_GEN_PROMPT_EN = """Generate 30 fictional brand names for the {category} product category. Requirements:
1. Sound like real international skincare/personal care brands
2. 2-3 words each
3. Must NOT match or closely resemble any real brand
4. Diverse styles: some tech-sounding, some natural, some medical, some trendy
5. English names only

Output strictly in this format, one per line:
1. BrandName
2. BrandName
...
30. BrandName"""

BRAND_CHECK_PROMPT = """Do you recognize "{brand_name}" as a real skincare, personal care, or cosmetics brand?
Answer with ONLY one of:
- KNOWN: if you recognize it as a real brand
- UNKNOWN: if you don't recognize it or are unsure
- SIMILAR: if it's very similar to a known brand (specify which)

Answer:"""


def run_phase3():
    """Generate and validate the fictional brand pool."""
    print("\n" + "=" * 60)
    print("PHASE 3: Fictional Brand Pool Generation")
    print("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    brands_path = output_dir / "fictional_brands.json"

    if brands_path.exists():
        with open(brands_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("status") == "validated":
            print(f"✓ Phase 3 completed, loaded existing brand pool: {len(existing.get('final_pool', {}))} subcategories")
            print(json.dumps(existing["final_pool"], indent=2, ensure_ascii=False))
            return existing
        print(f"⚡ Found an unfinished brand pool, continuing...")

    enabled_models = {n: c for n, c in CONFIG["models"].items() if c["enabled"]}
    model_names = list(enabled_models.keys())

    gen_model = model_names[0]
    gen_cfg = enabled_models[gen_model]

    print(f"\nStep 1: Generating candidate names with {gen_model}...")
    candidates = {}

    for sc_key, sc_info in SUBCATEGORIES.items():
        cat_name = sc_info["en"]
        prompt = BRAND_GEN_PROMPT_EN.format(category=cat_name)

        print(f"  Generating for {sc_key} ({cat_name})...")
        response = call_llm(prompt, gen_model, gen_cfg)

        names = []
        for line in response.strip().split("\n"):
            m = re.match(r'^\s*\d+[\.\)]\s*(.+)$', line)
            if m:
                name = m.group(1).strip().strip("*").strip('"').strip("'")
                if name and len(name) > 2:
                    names.append(name)

        candidates[sc_key] = names[:30]
        print(f"    Got {len(candidates[sc_key])} candidates")

    if _STOP.is_set():
        return None

    print(f"\nStep 2: Validating with {len(enabled_models)} models...")

    validation_results = defaultdict(lambda: defaultdict(list))  # brand → model → [KNOWN/UNKNOWN/SIMILAR]

    all_check_tasks = []
    for sc_key, names in candidates.items():
        for name in names:
            for model_name, model_cfg in enabled_models.items():
                all_check_tasks.append({
                    "subcat": sc_key,
                    "brand": name,
                    "model": model_name,
                    "model_cfg": model_cfg,
                    "prompt": BRAND_CHECK_PROMPT.format(brand_name=name),
                })

    print(f"  Total validation tasks: {len(all_check_tasks)}")

    if not CONFIG["dry_run"]:
        confirm = input(f"  Confirm validation run? (~${len(all_check_tasks)*100/1e6*5:.2f}) (y/N) > ").strip().lower()
        if confirm != "y":
            print("Skipping validation")
            return None

    completed = 0
    start_time = time.time()
    batch_size = CONFIG["max_parallel_workers"] * 2

    with ThreadPoolExecutor(max_workers=CONFIG["max_parallel_workers"]) as executor:
        batches = [all_check_tasks[i:i+batch_size] for i in range(0, len(all_check_tasks), batch_size)]
        for batch in batches:
            if _STOP.is_set():
                break
            futures = {}
            for t in batch:
                futures[executor.submit(call_llm, t["prompt"], t["model"], t["model_cfg"])] = t
            for future in as_completed(futures):
                if _STOP.is_set():
                    break
                task = futures[future]
                try:
                    response = future.result(timeout=30)
                    resp_upper = response.upper().strip()
                    if "KNOWN" in resp_upper and "UNKNOWN" not in resp_upper:
                        verdict = "KNOWN"
                    elif "SIMILAR" in resp_upper:
                        verdict = "SIMILAR"
                    else:
                        verdict = "UNKNOWN"
                    validation_results[(task["subcat"], task["brand"])][task["model"]].append(verdict)
                except Exception as e:
                    validation_results[(task["subcat"], task["brand"])][task["model"]].append("ERROR")

                completed += 1
                if completed % 50 == 0 or completed == len(all_check_tasks):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"  Validation: {completed}/{len(all_check_tasks)} | {rate:.1f}/s")

    print(f"\nStep 3: Filtering validated brands...")

    final_pool = {}
    validation_log = []

    for sc_key in SUBCATEGORIES:
        passed = []
        for name in candidates.get(sc_key, []):
            model_verdicts = validation_results.get((sc_key, name), {})
            all_unknown = True
            verdict_summary = {}
            for model_name in model_names:
                verdicts = model_verdicts.get(model_name, ["NO_DATA"])
                majority = Counter(verdicts).most_common(1)[0][0]
                verdict_summary[model_name] = majority
                if majority != "UNKNOWN":
                    all_unknown = False

            validation_log.append({
                "subcat": sc_key,
                "brand": name,
                "verdicts": verdict_summary,
                "passed": all_unknown,
            })

            if all_unknown:
                passed.append(name)

        selected = passed[:9] if len(passed) >= 9 else passed
        final_pool[sc_key] = selected
        print(f"  {sc_key}: {len(candidates.get(sc_key, []))} candidates → {len(passed)} passed → {len(selected)} selected")

        if len(selected) < 9:
            print(f"    ⚠ Fewer than 9! More need to be generated.")

    result = {
        "timestamp": datetime.now().isoformat(),
        "status": "validated",
        "candidates": candidates,
        "validation_log": validation_log,
        "final_pool": final_pool,
        "stats": {
            sc_key: {
                "candidates": len(candidates.get(sc_key, [])),
                "passed": len([v for v in validation_log if v["subcat"] == sc_key and v["passed"]]),
                "selected": len(final_pool.get(sc_key, [])),
            } for sc_key in SUBCATEGORIES
        },
    }

    with open(brands_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nFictional brands saved: {brands_path}")

    print("\n" + "=" * 60)
    print("FINAL FICTIONAL BRAND POOL")
    print("=" * 60)
    for sc_key, brands in final_pool.items():
        print(f"\n{sc_key} ({SUBCATEGORIES[sc_key]['brand']} + {len(brands)} fictional):")
        for i, b in enumerate(brands, 1):
            print(f"  {i}. {b}")

    return result


# ╔══════════════════════════════════════════════════════════════╗
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        CONFIG["dry_run"] = True

    if "--fresh" in sys.argv:
        import shutil
        p = Path(CONFIG["output_dir"])
        if p.exists():
            shutil.rmtree(p)
            print("🗑 Cleared old results directory, starting from scratch")

    phase = None
    if "--phase" in sys.argv:
        idx = sys.argv.index("--phase")
        phase = int(sys.argv[idx + 1])

    if phase is None or phase == 2:
        run_phase2()

    if phase is None or phase == 3:
        if not _STOP.is_set():
            run_phase3()

    print("\n✓ Pre-flight experiments complete!")
