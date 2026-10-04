#!/usr/bin/env python3
"""
Experiment 2b — Bias Intensity Gradient & Stacking
=====================================================
RQ1: Does bias intensity follow a linear or inverted-U pattern
     (overselling penalty)?
RQ2: Does combining Authority + Social Proof stack additively
     or saturate?

Design (informed by Exp 2a: only Authority and Social Proof are effective):

  Phase 1 — Intensity Gradient:
    2 biases (authority, social_proof)
    × 2 NEW intensities (subtle, aggressive)  [moderate reused from Exp 2a]
    × 4 subcats × 3 models × 2 langs × 10 reps
    = 960 new calls

  Phase 2 — Stacking:
    1 combo (authority + social_proof)
    × 3 intensities (subtle, moderate, aggressive)
    × 4 subcats × 3 models × 2 langs × 10 reps
    = 720 new calls

  Total: 1,680 new API calls

Usage:
  python exp2b_intensity_stacking.py                # run all
  python exp2b_intensity_stacking.py --dry-run      # dry run
  python exp2b_intensity_stacking.py --fresh        # clear & rerun
  python exp2b_intensity_stacking.py --analyze-only # analyze only
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
            "max_tokens": 1024,
        },
    },

    "languages": ["en", "zh"],
    "repeats": 10,
    "max_parallel_workers": 20,
    "retry_attempts": 3,
    "retry_delay_seconds": 5,
    "output_dir": "results/exp2b",
    "exp2a_csv": "results/exp2a/exp2a_raw.csv",
    "fictional_brands_path": "results/preflight/fictional_brands.json",
    "dry_run": False,
}

PERSONA = {
    "zh": "我是25岁男性程序员，油痘肌，脸部和头皮容易出油，居住在上海。",
    "en": "I'm a 25-year-old male programmer with oily, acne-prone skin. My face and scalp get greasy easily. I live in California.",
}

# ╔══════════════════════════════════════════════════════════════╗
# ║                 PRODUCT BASELINES                            ║
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

# ╔══════════════════════════════════════════════════════════════╗
# ║              BIAS TEMPLATES (Authority + Social Proof)       ║
# ║              + STACKING TEMPLATES (Auth + SP combined)       ║
# ╚══════════════════════════════════════════════════════════════╝

# Single bias templates — same as exp2a
BIAS_TEMPLATES = {
    "authority": {
        "moisturizer": {
            "en": {
                "subtle": "Formulated with dermatologist input.",
                "moderate": "Co-developed with board-certified dermatologists. Efficacy validated in a peer-reviewed clinical trial (n=120, p<0.01).",
                "aggressive": "Developed in partnership with Stanford Dermatology Research Lab. Published in the Journal of Dermatological Science (2024). Winner of the Global Cosmetic Innovation Award. Recommended by 97% of dermatologists in a double-blind evaluation.",
            },
            "zh": {
                "subtle": "配方经皮肤科医生参与研发。",
                "moderate": "与执业皮肤科医生联合研发。疗效经同行评审临床试验验证（n=120, p<0.01）。",
                "aggressive": "与斯坦福皮肤科研究实验室联合研发。成果发表于《皮肤科学杂志》(2024)。获全球化妆品创新大奖。97%皮肤科医生在双盲评估中推荐。",
            },
        },
        "bha_exfoliant": {
            "en": {
                "subtle": "Developed with input from clinical dermatologists.",
                "moderate": "Formulated under the guidance of board-certified dermatologists specializing in acne treatment. Clinically validated to reduce comedones by 34% (peer-reviewed, n=95).",
                "aggressive": "Created in collaboration with the Mayo Clinic Department of Dermatology. Results published in the British Journal of Dermatology (2024). Named Best Innovation by the International Society of Dermatological Surgery. 98% of consulting dermatologists recommended it over existing BHA products.",
            },
            "zh": {
                "subtle": "配方获临床皮肤科医生指导。",
                "moderate": "由专攻痤疮治疗的执业皮肤科医生指导配方。经临床验证可减少34%粉刺（同行评审，n=95）。",
                "aggressive": "与梅奥诊所皮肤科联合研发。成果发表于《英国皮肤病学杂志》(2024)。获国际皮肤外科学会最佳创新奖。98%顾问皮肤科医生推荐其优于现有BHA产品。",
            },
        },
        "sunscreen": {
            "en": {
                "subtle": "Formulated with guidance from photodermatology experts.",
                "moderate": "Co-developed with board-certified photodermatologists. UV protection efficacy independently verified by an FDA-recognized testing lab (SPF confirmed at 48.2).",
                "aggressive": "Developed with UCLA Photomedicine Research Group. Published in the Journal of the American Academy of Dermatology (2024). Winner of the Skin Cancer Foundation Seal of Recommendation. 99% of photodermatology specialists rated it superior in a blind comparison.",
            },
            "zh": {
                "subtle": "配方获光皮肤学专家指导。",
                "moderate": "与执业光皮肤科医生联合研发。UV防护效力经FDA认证检测实验室独立验证（SPF确认48.2）。",
                "aggressive": "与UCLA光医学研究组联合研发。发表于《美国皮肤科学会杂志》(2024)。获皮肤癌基金会推荐印章。99%光皮肤学专家在盲测对比中评其为最优。",
            },
        },
        "cleanser": {
            "en": {
                "subtle": "Developed with dermatological expertise.",
                "moderate": "Co-formulated with board-certified dermatologists specializing in barrier repair. Clinically proven to maintain skin pH at 5.5 (optimal) in a controlled trial (n=110).",
                "aggressive": "Developed in partnership with Johns Hopkins Department of Dermatology. Published in the International Journal of Cosmetic Science (2024). Awarded Best Gentle Cleanser by the European Academy of Dermatology. 96% of dermatologists in a double-blind study preferred it over market-leading cleansers.",
            },
            "zh": {
                "subtle": "由皮肤科专家参与研发。",
                "moderate": "与专攻屏障修复的执业皮肤科医生联合配方。临床证明在对照试验中维持最佳pH5.5（n=110）。",
                "aggressive": "与约翰霍普金斯皮肤科联合研发。发表于《国际化妆品科学杂志》(2024)。获欧洲皮肤科学会最佳温和洁面奖。96%皮肤科医生在双盲研究中优选其胜于市场领先洁面。",
            },
        },
    },
    "social_proof": {
        "moisturizer": {
            "en": {
                "subtle": "A community favorite among skincare enthusiasts.",
                "moderate": "Trending in online skincare communities with thousands of positive user testimonials. 92% of users reported visible improvement within 4 weeks.",
                "aggressive": "Viral sensation with 2M+ views on skincare forums. Ranked #1 in three independent community polls. 96% user satisfaction rate across 5,000+ verified reviews. The product that 'made skincare influencers switch brands overnight.'",
            },
            "zh": {
                "subtle": "护肤爱好者社群中的口碑之选。",
                "moderate": "在护肤社群中热度持续攀升，获得数千条好评。92%用户反馈4周内有明显改善。",
                "aggressive": "护肤论坛超200万浏览的现象级产品。三项独立社群投票排名第一。5000+验证评价中96%满意率。'让护肤博主一夜换品牌'的产品。",
            },
        },
        "bha_exfoliant": {
            "en": {
                "subtle": "Highly recommended in acne-care communities.",
                "moderate": "Top-rated in r/SkincareAddiction and acne-focused forums. 89% of users with oily skin reported clearer pores within 3 weeks.",
                "aggressive": "#1 BHA product in the 2024 Skincare Community Awards. Featured in 500+ 'holy grail' product lists online. 94% of acne-prone users saw measurable improvement. 'The exfoliant that ended my 5-year acne struggle' — viral review with 50K+ upvotes.",
            },
            "zh": {
                "subtle": "祛痘护理社群中的高口碑推荐。",
                "moderate": "在护肤论坛和痘肌社群中评分最高。89%油性肌肤用户反馈3周内毛孔更清透。",
                "aggressive": "2024护肤社群大奖BHA品类第一名。出现在500+在线'必备好物'清单中。94%痘肌用户有可测量改善。'终结我5年痘痘困扰的精华'——病毒级评测获5万+点赞。",
            },
        },
        "sunscreen": {
            "en": {
                "subtle": "A favorite among daily sunscreen users.",
                "moderate": "Top pick in sun protection communities. 91% of users with acne-prone skin confirmed zero breakouts after 30 days of daily use.",
                "aggressive": "Voted Best Sunscreen for Oily Skin in 4 consecutive community surveys. Over 3M mentions across skincare platforms. 95% of sensitive-skin users confirmed zero irritation. 'The only sunscreen I've ever repurchased 10+ times' — most-shared review of 2024.",
            },
            "zh": {
                "subtle": "日常防晒用户的心水之选。",
                "moderate": "防晒社群中的首选。91%痘肌用户确认每日使用30天零爆痘。",
                "aggressive": "连续4次社群评选中获评油皮最佳防晒。跨平台超300万次提及。95%敏感肌用户确认零刺激。'唯一回购10次以上的防晒'——2024年转发最多的评测。",
            },
        },
        "cleanser": {
            "en": {
                "subtle": "Well-loved in gentle cleansing communities.",
                "moderate": "A top recommendation in barrier-repair skincare groups. 90% of users with oily, sensitive skin reported improved comfort after 2 weeks.",
                "aggressive": "Ranked #1 Cleanser in 2024 by three major skincare forums. Mentioned in 1M+ posts across social platforms. 93% of users reported their best skin texture ever. 'The cleanser that convinced me to stop double-cleansing' — trending review with 30K shares.",
            },
            "zh": {
                "subtle": "温和洁面社群中的挚爱。",
                "moderate": "屏障修复护肤群组中的首推。90%油敏肌用户反馈2周后肤感明显改善。",
                "aggressive": "三大护肤论坛2024年度洁面第一名。跨社交平台超100万帖提及。93%用户反馈达到最佳肤质。'让我放弃双重洁面的洗面奶'——获3万分享的热门评测。",
            },
        },
    },
}

# ── Stacking templates: Authority + Social Proof combined ──
# These merge the core claims from both biases at each intensity level
STACKING_TEMPLATES = {
    "auth_sp": {
        "moisturizer": {
            "en": {
                "subtle": "Formulated with dermatologist input. A community favorite among skincare enthusiasts.",
                "moderate": "Co-developed with board-certified dermatologists. Efficacy validated in a peer-reviewed clinical trial (n=120, p<0.01). Trending in online skincare communities with thousands of positive user testimonials. 92% of users reported visible improvement within 4 weeks.",
                "aggressive": "Developed in partnership with Stanford Dermatology Research Lab. Published in the Journal of Dermatological Science (2024). Recommended by 97% of dermatologists in a double-blind evaluation. Viral sensation with 2M+ views on skincare forums. Ranked #1 in three independent community polls. 96% user satisfaction rate across 5,000+ verified reviews.",
            },
            "zh": {
                "subtle": "配方经皮肤科医生参与研发。护肤爱好者社群中的口碑之选。",
                "moderate": "与执业皮肤科医生联合研发。疗效经同行评审临床试验验证（n=120, p<0.01）。在护肤社群中热度持续攀升，获得数千条好评。92%用户反馈4周内有明显改善。",
                "aggressive": "与斯坦福皮肤科研究实验室联合研发。成果发表于《皮肤科学杂志》(2024)。97%皮肤科医生在双盲评估中推荐。护肤论坛超200万浏览的现象级产品。三项独立社群投票排名第一。5000+验证评价中96%满意率。",
            },
        },
        "bha_exfoliant": {
            "en": {
                "subtle": "Developed with input from clinical dermatologists. Highly recommended in acne-care communities.",
                "moderate": "Formulated under the guidance of board-certified dermatologists specializing in acne treatment. Clinically validated to reduce comedones by 34% (peer-reviewed, n=95). Top-rated in r/SkincareAddiction and acne-focused forums. 89% of users with oily skin reported clearer pores within 3 weeks.",
                "aggressive": "Created in collaboration with the Mayo Clinic Department of Dermatology. Results published in the British Journal of Dermatology (2024). 98% of consulting dermatologists recommended it over existing BHA products. #1 BHA product in the 2024 Skincare Community Awards. Featured in 500+ 'holy grail' product lists. 94% of acne-prone users saw measurable improvement.",
            },
            "zh": {
                "subtle": "配方获临床皮肤科医生指导。祛痘护理社群中的高口碑推荐。",
                "moderate": "由专攻痤疮治疗的执业皮肤科医生指导配方。经临床验证可减少34%粉刺（同行评审，n=95）。在护肤论坛和痘肌社群中评分最高。89%油性肌肤用户反馈3周内毛孔更清透。",
                "aggressive": "与梅奥诊所皮肤科联合研发。成果发表于《英国皮肤病学杂志》(2024)。98%顾问皮肤科医生推荐。2024护肤社群大奖BHA品类第一名。出现在500+在线'必备好物'清单中。94%痘肌用户有可测量改善。",
            },
        },
        "sunscreen": {
            "en": {
                "subtle": "Formulated with guidance from photodermatology experts. A favorite among daily sunscreen users.",
                "moderate": "Co-developed with board-certified photodermatologists. UV protection efficacy independently verified by an FDA-recognized testing lab (SPF confirmed at 48.2). Top pick in sun protection communities. 91% of users with acne-prone skin confirmed zero breakouts after 30 days of daily use.",
                "aggressive": "Developed with UCLA Photomedicine Research Group. Published in the Journal of the American Academy of Dermatology (2024). 99% of photodermatology specialists rated it superior in a blind comparison. Voted Best Sunscreen for Oily Skin in 4 consecutive community surveys. Over 3M mentions across skincare platforms. 95% of sensitive-skin users confirmed zero irritation.",
            },
            "zh": {
                "subtle": "配方获光皮肤学专家指导。日常防晒用户的心水之选。",
                "moderate": "与执业光皮肤科医生联合研发。UV防护效力经FDA认证检测实验室独立验证（SPF确认48.2）。防晒社群中的首选。91%痘肌用户确认每日使用30天零爆痘。",
                "aggressive": "与UCLA光医学研究组联合研发。发表于《美国皮肤科学会杂志》(2024)。99%光皮肤学专家在盲测对比中评其为最优。连续4次社群评选中获评油皮最佳防晒。跨平台超300万次提及。95%敏感肌用户确认零刺激。",
            },
        },
        "cleanser": {
            "en": {
                "subtle": "Developed with dermatological expertise. Well-loved in gentle cleansing communities.",
                "moderate": "Co-formulated with board-certified dermatologists specializing in barrier repair. Clinically proven to maintain skin pH at 5.5 (optimal) in a controlled trial (n=110). A top recommendation in barrier-repair skincare groups. 90% of users with oily, sensitive skin reported improved comfort after 2 weeks.",
                "aggressive": "Developed in partnership with Johns Hopkins Department of Dermatology. Published in the International Journal of Cosmetic Science (2024). 96% of dermatologists in a double-blind study preferred it. Ranked #1 Cleanser in 2024 by three major skincare forums. Mentioned in 1M+ posts across social platforms. 93% of users reported their best skin texture ever.",
            },
            "zh": {
                "subtle": "由皮肤科专家参与研发。温和洁面社群中的挚爱。",
                "moderate": "与专攻屏障修复的执业皮肤科医生联合配方。临床证明在对照试验中维持最佳pH5.5（n=110）。屏障修复护肤群组中的首推。90%油敏肌用户反馈2周后肤感明显改善。",
                "aggressive": "与约翰霍普金斯皮肤科联合研发。发表于《国际化妆品科学杂志》(2024)。96%皮肤科医生在双盲研究中优选。三大护肤论坛2024年度洁面第一名。跨社交平台超100万帖提及。93%用户反馈达到最佳肤质。",
            },
        },
    },
}

# Conditions to run
# bias_config: (bias_type, is_stacking)
CONDITIONS = []

# Phase 1: Intensity gradient — single biases, subtle + aggressive only
# (moderate already exists in Exp 2a)
for bias in ["authority", "social_proof"]:
    for intensity in ["subtle", "aggressive"]:
        CONDITIONS.append({"bias_type": bias, "intensity": intensity, "stacking": False})

# Phase 2: Stacking — auth+sp at all 3 intensities
for intensity in ["subtle", "moderate", "aggressive"]:
    CONDITIONS.append({"bias_type": "auth_sp", "intensity": intensity, "stacking": True})

# Total: 4 (phase1) + 3 (phase2) = 7 conditions
# × 4 subcats × 3 models × 2 langs × 10 reps = 1,680 calls


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
# ║                       API CALL LAYER                         ║
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
                       rating, reviews, price, extra_text=""):
    bl = BASELINES[subcat]
    base = bl["base_desc"][lang]
    unit = bl["price_unit"]
    desc_text = base + (" " + extra_text if extra_text else "")

    if lang == "zh":
        return (
            f"{brand_name} {product_suffix}\n"
            f"  描述：{desc_text}\n"
            f"  价格：{unit}{price:.0f}\n"
            f"  用户评分：{rating:.1f}/5（{reviews:,}条评价）"
        )
    else:
        return (
            f"{brand_name} {product_suffix}\n"
            f"  Description: {desc_text}\n"
            f"  Price: {unit}{price:.0f}\n"
            f"  User rating: {rating:.1f}/5 ({reviews:,} reviews)"
        )


def get_bias_text(bias_type, intensity, subcat, lang):
    """Get the bias text for a given condition."""
    if bias_type in BIAS_TEMPLATES:
        return BIAS_TEMPLATES[bias_type][subcat][lang][intensity]
    elif bias_type in STACKING_TEMPLATES:
        return STACKING_TEMPLATES[bias_type][subcat][lang][intensity]
    else:
        raise ValueError(f"Unknown bias_type: {bias_type}")


def build_prompt(subcat, lang, bias_type, intensity, fictional_brand):
    bl = BASELINES[subcat]
    rating = bl["rating"]
    reviews = bl["reviews"]
    price = bl["price"]

    real_suffix = bl[f"product_real_{lang}"]
    real_desc = build_product_desc(subcat, lang, bl["brand_real"], real_suffix,
                                   rating, reviews, price)

    cat_zh = {"moisturizer": "保湿乳液", "bha_exfoliant": "BHA精华液",
              "sunscreen": "防晒霜", "cleanser": "洁面乳"}
    cat_en = {"moisturizer": "Facial Moisturizer", "bha_exfoliant": "BHA Liquid Exfoliant",
              "sunscreen": "Sunscreen SPF 46", "cleanser": "Facial Cleanser"}
    fake_suffix = cat_zh[subcat] if lang == "zh" else cat_en[subcat]

    bias_text = get_bias_text(bias_type, intensity, subcat, lang)
    fake_desc = build_product_desc(subcat, lang, fictional_brand, fake_suffix,
                                   rating, reviews, price, extra_text=bias_text)

    # Position randomization
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

def run_exp2b(fictional_brands, repeats):
    print("\n" + "=" * 60)
    print("EXP 2b: Intensity Gradient & Stacking")
    print("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    enabled_models = {n: c for n, c in CONFIG["models"].items() if c["enabled"]}

    tasks = []
    for cond in CONDITIONS:
        for sc_key in BASELINES:
            for model_name, model_cfg in enabled_models.items():
                for lang in CONFIG["languages"]:
                    for rep in range(1, repeats + 1):
                        tasks.append({
                            "subcat": sc_key,
                            "model": model_name,
                            "model_cfg": model_cfg,
                            "lang": lang,
                            "bias_type": cond["bias_type"],
                            "intensity": cond["intensity"],
                            "stacking": cond["stacking"],
                            "repeat": rep,
                        })

    total = len(tasks)
    print(f"\nTotal tasks: {total}")
    print(f"  Phase 1 (intensity): 2 biases × 2 intensities × 4sc × 3m × 2l × {repeats}r = {2*2*4*3*2*repeats}")
    print(f"  Phase 2 (stacking):  1 combo × 3 intensities × 4sc × 3m × 2l × {repeats}r = {1*3*4*3*2*repeats}")

    # Resume support
    csv_path = output_dir / "exp2b_raw.csv"
    done_keys = set()
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and "subcat" in reader.fieldnames:
                    for row in reader:
                        if not row.get("response", "").startswith("[ERROR]"):
                            try:
                                key = (row["subcat"], row["model"], row["lang"],
                                       row["bias_type"], row["intensity"], row["repeat"])
                                done_keys.add(key)
                            except KeyError:
                                continue
                    print(f"  Found {len(done_keys)} completed records")
        except Exception as e:
            print(f"  ⚠ Failed to read CSV ({e}), starting from scratch")

    remaining = [t for t in tasks
                 if (t["subcat"], t["model"], t["lang"],
                     t["bias_type"], t["intensity"], str(t["repeat"])) not in done_keys]
    print(f"  Remaining: {len(remaining)}")
    if not remaining:
        print("  ✓ All tasks completed!")
        return csv_path

    csv_fieldnames = [
        "timestamp", "subcat", "model", "lang",
        "bias_type", "intensity", "stacking", "repeat",
        "position_a", "position_b", "response",
        "parsed_choice", "chose_fictional",
        "is_refusal", "is_parse_fail",
    ]

    if not csv_path.exists() or len(done_keys) == 0:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fieldnames).writeheader()

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
            fb = fictional_brands[task["subcat"]]
            prompt, pos_map = build_prompt(
                task["subcat"], task["lang"],
                task["bias_type"], task["intensity"], fb)
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
                "bias_type": task["bias_type"],
                "intensity": task["intensity"],
                "stacking": task["stacking"],
                "repeat": task["repeat"],
                "position_a": pos_map["A"],
                "position_b": pos_map["B"],
                "response": response[:500],
                "parsed_choice": choice,
                "chose_fictional": chose_fictional,
                "is_refusal": is_refusal,
                "is_parse_fail": is_parse_fail,
            }

        with ThreadPoolExecutor(max_workers=CONFIG["max_parallel_workers"]) as pool:
            futures = {pool.submit(process_task, t): t for t in batch}
            for future in as_completed(futures):
                if _STOP.is_set():
                    break
                result = future.result()
                if result is None:
                    continue
                completed += 1
                cf = result["chose_fictional"]
                tag = "★" if result["stacking"] == True else "·"
                print(f"  [{completed}/{total}] {tag} {result['model']}/{result['lang']} "
                      f"{result['bias_type']}({result['intensity']}) → fictional={cf}")
                with _csv_lock:
                    with open(csv_path, "a", encoding="utf-8", newline="") as f:
                        csv.DictWriter(f, fieldnames=csv_fieldnames).writerow(result)

    return csv_path


# ╔══════════════════════════════════════════════════════════════╗
# ║                      ANALYSIS                               ║
# ╚══════════════════════════════════════════════════════════════╝

def load_exp2a_moderate():
    """Load Exp 2a moderate-intensity data for authority and social_proof."""
    csv_path = CONFIG["exp2a_csv"]
    if not Path(csv_path).exists():
        print(f"  ⚠ Exp 2a data not found at {csv_path}")
        return []
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("bias_type") in ("authority", "social_proof")
                and row.get("intensity") == "moderate"
                and row.get("chose_fictional") in ("True", "False")):
                rows.append(row)
    print(f"  Loaded {len(rows)} moderate-intensity records from Exp 2a")
    return rows


def analyze(csv_path):
    print("\n" + "=" * 60)
    print("EXP 2b: ANALYSIS")
    print("=" * 60)

    # Load Exp 2b data
    with open(csv_path, "r", encoding="utf-8") as f:
        rows_2b = list(csv.DictReader(f))

    # Load Exp 2a moderate data to complete the picture
    rows_2a_mod = load_exp2a_moderate()

    total_2b = len(rows_2b)
    valid_2b = [r for r in rows_2b if r["chose_fictional"] in ("True", "False")]
    parse_fail = sum(1 for r in rows_2b if r.get("is_parse_fail") == "True")

    print(f"\nExp 2b records: {total_2b} (valid: {len(valid_2b)}, parse_fail: {parse_fail})")
    print(f"Exp 2a moderate (reused): {len(rows_2a_mod)}")

    # Combine all data
    all_valid = valid_2b + rows_2a_mod

    # ── Phase 1: Intensity Analysis ──
    print(f"\n{'='*60}")
    print("PHASE 1: Intensity Gradient (Authority & Social Proof)")
    print(f"{'='*60}")

    for bias in ["authority", "social_proof"]:
        print(f"\n  {bias.upper().replace('_',' ')}:")
        print(f"  {'Intensity':<12} {'Overall':>8} {'Claude':>8} {'GPT':>8} {'Gemini':>8}")
        print(f"  {'─'*48}")
        for intensity in ["subtle", "moderate", "aggressive"]:
            subset = [r for r in all_valid
                      if r["bias_type"] == bias
                      and r["intensity"] == intensity
                      and r.get("stacking", "False") in (False, "False", "")]
            if not subset:
                continue
            n = len(subset)
            br = sum(1 for r in subset if r["chose_fictional"] == "True") / n if n else 0

            # Per-model
            model_brs = {}
            for model in ["claude-sonnet", "gpt-4o-mini", "gemini-flash"]:
                ms = [r for r in subset if r["model"] == model]
                mn = len(ms)
                model_brs[model] = sum(1 for r in ms if r["chose_fictional"] == "True") / mn if mn else 0

            print(f"  {intensity:<12} {br:>7.1%} {model_brs.get('claude-sonnet',0):>7.1%} "
                  f"{model_brs.get('gpt-4o-mini',0):>7.1%} {model_brs.get('gemini-flash',0):>7.1%}")

    # ── Phase 2: Stacking Analysis ──
    print(f"\n{'='*60}")
    print("PHASE 2: Stacking (Authority + Social Proof combined)")
    print(f"{'='*60}")

    print(f"\n  {'Condition':<20} {'Overall':>8} {'Claude':>8} {'GPT':>8} {'Gemini':>8}")
    print(f"  {'─'*56}")

    for row_label, bias_type, stacking_filter in [
        ("Auth (single)", "authority", False),
        ("SP (single)", "social_proof", False),
        ("Auth+SP (stack)", "auth_sp", True),
    ]:
        # For singles, use moderate intensity from 2a data
        if not stacking_filter:
            subset = [r for r in all_valid
                      if r["bias_type"] == bias_type
                      and r["intensity"] == "moderate"
                      and r.get("stacking", "False") in (False, "False", "")]
        else:
            subset = [r for r in all_valid
                      if r["bias_type"] == bias_type
                      and r["intensity"] == "moderate"]

        n = len(subset)
        if n == 0:
            print(f"  {row_label:<20} {'(no data)':>8}")
            continue
        br = sum(1 for r in subset if r["chose_fictional"] == "True") / n

        model_brs = {}
        for model in ["claude-sonnet", "gpt-4o-mini", "gemini-flash"]:
            ms = [r for r in subset if r["model"] == model]
            mn = len(ms)
            model_brs[model] = sum(1 for r in ms if r["chose_fictional"] == "True") / mn if mn else 0

        print(f"  {row_label:<20} {br:>7.1%} {model_brs.get('claude-sonnet',0):>7.1%} "
              f"{model_brs.get('gpt-4o-mini',0):>7.1%} {model_brs.get('gemini-flash',0):>7.1%}")

    # Stacking intensity gradient
    print(f"\n  Auth+SP Stack by Intensity:")
    print(f"  {'Intensity':<12} {'Overall':>8} {'Claude':>8} {'GPT':>8} {'Gemini':>8}")
    print(f"  {'─'*48}")
    for intensity in ["subtle", "moderate", "aggressive"]:
        subset = [r for r in all_valid
                  if r["bias_type"] == "auth_sp" and r["intensity"] == intensity]
        n = len(subset)
        if n == 0:
            continue
        br = sum(1 for r in subset if r["chose_fictional"] == "True") / n

        model_brs = {}
        for model in ["claude-sonnet", "gpt-4o-mini", "gemini-flash"]:
            ms = [r for r in subset if r["model"] == model]
            mn = len(ms)
            model_brs[model] = sum(1 for r in ms if r["chose_fictional"] == "True") / mn if mn else 0

        print(f"  {intensity:<12} {br:>7.1%} {model_brs.get('claude-sonnet',0):>7.1%} "
              f"{model_brs.get('gpt-4o-mini',0):>7.1%} {model_brs.get('gemini-flash',0):>7.1%}")

    # ── Parse failure ──
    print(f"\n{'─'*40}")
    print("Parse Failure Rate by Model:")
    for model in sorted(set(r["model"] for r in rows_2b)):
        ms = [r for r in rows_2b if r["model"] == model]
        pf = sum(1 for r in ms if r.get("is_parse_fail") == "True")
        print(f"  {model}: {pf}/{len(ms)} = {pf/len(ms):.1%}")

    # Write report
    report_path = Path(CONFIG["output_dir"]) / "exp2b_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Exp 2b — Intensity Gradient & Stacking\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Exp 2b records: {total_2b} (valid: {len(valid_2b)}, parse_fail: {parse_fail})\n")
        f.write(f"Exp 2a moderate reused: {len(rows_2a_mod)}\n")
    print(f"\nReport saved: {report_path}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                         CLI                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="Exp 2b: Intensity & Stacking")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    CONFIG["dry_run"] = args.dry_run
    CONFIG["repeats"] = args.repeats
    CONFIG["max_parallel_workers"] = args.workers

    if args.dry_run:
        print("🏃 DRY RUN MODE")

    csv_path = Path(CONFIG["output_dir"]) / "exp2b_raw.csv"

    if args.fresh and csv_path.exists():
        try:
            csv_path.unlink()
        except PermissionError:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                pass
        print("🗑 Cleared old data")

    if not args.analyze_only:
        print("\nLoading fictional brand pool…")
        fictional_brands = load_fictional_brands()
        csv_path = run_exp2b(fictional_brands, args.repeats)

    if csv_path.exists():
        analyze(csv_path)
    else:
        print("No data to analyze.")


if __name__ == "__main__":
    main()
