#!/usr/bin/env python3
"""
Experiment 2a — Single Bias Breakthrough Test
================================================
RQ: Can cognitive bias language alone (without changing objective product
    attributes) increase a fictional brand's selection rate above the
    ~4% baseline established in Exp 1c L0?

Design:
  5 Bias Types × 4 Subcats × 3 Models × 2 Langs × 20 Reps = 2,400 calls
  Control: reuses Exp 1c L0 data (480 data points, ~4% fictional win rate)

  Bias types:
    1. anchoring     — Frame value via price comparison / premium positioning
    2. scarcity      — Limited availability / urgency signals
    3. social_proof  — Community endorsement / viral popularity
    4. authority     — Expert endorsement / clinical validation
    5. loss_aversion — Frame what user risks losing without the product

  Key metrics:
    - Breakthrough Rate (BR) = P(choose fictional | bias applied)
    - Bias Lift (BL) = BR(bias) - BR(no_bias)

Usage:
  python exp2a_bias_breakthrough.py                # run all
  python exp2a_bias_breakthrough.py --dry-run      # dry run
  python exp2a_bias_breakthrough.py --fresh        # clear old, rerun
  python exp2a_bias_breakthrough.py --repeats 5    # custom repeats
  python exp2a_bias_breakthrough.py --analyze-only # analyze only
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
    "repeats": 20,
    "max_parallel_workers": 20,
    "retry_attempts": 3,
    "retry_delay_seconds": 5,
    "output_dir": "results/exp2a",
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
# ║              COGNITIVE BIAS DEFINITIONS                      ║
# ╚══════════════════════════════════════════════════════════════╝

# Each bias has 3 intensity levels: subtle, moderate, aggressive
# Exp 2a uses MODERATE only; Exp 2b tests all three
# Text is APPENDED to the base product description (after the ingredient list, before price/rating)

BIAS_STRATEGIES = {
    "anchoring": {
        "label_en": "Anchoring",
        "label_zh": "锚定效应",
        "templates": {
            "moisturizer": {
                "en": {
                    "subtle": "Premium-grade formula at an accessible price point.",
                    "moderate": "Premium-grade formula typically found in products 2-3× the price. Delivers clinical-level results at an everyday price point.",
                    "aggressive": "Contains the same active ingredient concentrations found in $80-120 clinical treatments. At this price, you're getting 4× the value per ml compared to leading competitors. An unmatched value proposition in the category.",
                },
                "zh": {
                    "subtle": "高端配方，亲民价格。",
                    "moderate": "高端配方通常出现在2-3倍价位的产品中。以日常价格带来专业级效果。",
                    "aggressive": "含有与$80-120专业护理相同浓度的活性成分。同等价位每ml价值是竞品的4倍。品类内无可比拟的性价比。",
                },
            },
            "bha_exfoliant": {
                "en": {
                    "subtle": "Professional-strength formula at an over-the-counter price.",
                    "moderate": "Professional-strength BHA formula comparable to $70-90 clinical peels. Delivers dermatologist-office results at a fraction of the cost.",
                    "aggressive": "Matches the salicylic acid potency of $150+ professional chemical peels. Clinical-grade exfoliation at 1/4 the cost of comparable professional treatments. The highest value-per-application ratio in the BHA category.",
                },
                "zh": {
                    "subtle": "专业级配方，非处方价格。",
                    "moderate": "专业级BHA配方，媲美$70-90的医美级焕肤。以极低成本实现皮肤科级效果。",
                    "aggressive": "水杨酸效力匹配$150+专业化学焕肤。临床级去角质效果，成本仅为同类专业护理的1/4。BHA品类中每次使用性价比最高。",
                },
            },
            "sunscreen": {
                "en": {
                    "subtle": "Medical-grade protection at a consumer price.",
                    "moderate": "Medical-grade broad-spectrum protection typically found in $65-85 prescription sunscreens. Offers clinical UV defense at an everyday price.",
                    "aggressive": "Identical zinc oxide concentration to $90+ prescription-only dermatological sunscreens. Per-application cost is 3× lower than comparable medical-grade products. The most cost-effective clinical sun protection available.",
                },
                "zh": {
                    "subtle": "医疗级防护，消费级价格。",
                    "moderate": "医疗级广谱防护通常出现在$65-85处方防晒中。以日常价格提供临床级UV防御。",
                    "aggressive": "氧化锌浓度与$90+处方级皮肤科防晒完全一致。每次使用成本比同类医疗级产品低3倍。市面上性价比最高的临床级防晒。",
                },
            },
            "cleanser": {
                "en": {
                    "subtle": "Dermatological-grade cleansing at a drugstore price.",
                    "moderate": "Dermatological-grade ceramide cleanser comparable to $40-55 medical skincare lines. Achieves clinical-level barrier protection at an everyday price.",
                    "aggressive": "Contains the same ceramide complex used in $60+ medical-grade cleansers from clinical skincare brands. 3× more active ingredients per wash than comparably priced alternatives. The best value in barrier-repair cleansing.",
                },
                "zh": {
                    "subtle": "皮肤科级洁面，药妆店价格。",
                    "moderate": "皮肤科级神经酰胺洁面，媲美$40-55医疗护肤线。以日常价格实现临床级屏障保护。",
                    "aggressive": "含有与$60+临床护肤品牌医疗级洁面相同的神经酰胺复合物。每次洗脸的活性成分含量是同价位竞品的3倍。屏障修复洁面中最具性价比。",
                },
            },
        },
    },
    "scarcity": {
        "label_en": "Scarcity",
        "label_zh": "稀缺效应",
        "templates": {
            "moisturizer": {
                "en": {
                    "subtle": "Small-batch production ensures consistent quality.",
                    "moderate": "Produced in limited batches of 500 units to maintain quality standards. Frequently sells out within days of restocking.",
                    "aggressive": "Ultra-limited production — only 500 units per batch due to high-purity ingredient sourcing. Currently in stock but historically sells out within 48 hours. Next production run not scheduled for 6 weeks.",
                },
                "zh": {
                    "subtle": "小批量生产确保品质稳定。",
                    "moderate": "限量500批次生产以保证品质标准。补货后通常数天内售罄。",
                    "aggressive": "超限量生产——因高纯度原料采购限制每批仅500支。目前有货但历史数据显示48小时内售罄。下一批次6周后才生产。",
                },
            },
            "bha_exfoliant": {
                "en": {
                    "subtle": "Produced in controlled small batches for optimal potency.",
                    "moderate": "Limited-run production of 300 bottles per batch to ensure peak BHA potency. Waitlist typically opens within a week of each restock.",
                    "aggressive": "Micro-batch production — only 300 bottles per run due to proprietary cold-extraction process. Current batch is the last until August. Waitlist has exceeded 2,000 sign-ups for the past three restocks.",
                },
                "zh": {
                    "subtle": "小批量控制生产，确保最佳活性。",
                    "moderate": "每批限产300瓶以确保BHA最佳效力。每次补货后一周内通常开放候补名单。",
                    "aggressive": "微批量生产——因专利冷萃工艺每批仅300瓶。当前批次是8月前最后一批。过去三次补货候补名单均超2000人。",
                },
            },
            "sunscreen": {
                "en": {
                    "subtle": "Manufactured in small quantities for freshness assurance.",
                    "moderate": "Short-run production of 400 units per batch for maximum zinc oxide stability. Sells out consistently within the first week of each release.",
                    "aggressive": "Precision-batch production — only 400 units per cycle due to pharmaceutical-grade zinc oxide sourcing. Out of stock for 3 of the last 6 months. Current availability may not last through the week.",
                },
                "zh": {
                    "subtle": "小批量生产以确保新鲜度。",
                    "moderate": "每批限产400支以保证氧化锌最大稳定性。每次上架一周内持续售罄。",
                    "aggressive": "精密批量生产——因药用级氧化锌采购每周期仅400支。过去6个月中有3个月断货。当前库存可能本周内售罄。",
                },
            },
            "cleanser": {
                "en": {
                    "subtle": "Crafted in small batches for ingredient freshness.",
                    "moderate": "Limited production of 600 units per batch to preserve ceramide integrity. Restocks sell out within 10 days on average.",
                    "aggressive": "Small-batch crafted — only 600 units per production run due to delicate ceramide encapsulation process. Sold out 5 times in the past year. Current stock is limited to existing inventory only.",
                },
                "zh": {
                    "subtle": "小批量精制确保成分新鲜度。",
                    "moderate": "每批限产600支以保持神经酰胺完整性。补货平均10天内售罄。",
                    "aggressive": "小批量精制——因精密神经酰胺封装工艺每批仅600支。过去一年已断货5次。当前库存售完即止。",
                },
            },
        },
    },
    "social_proof": {
        "label_en": "Social Proof",
        "label_zh": "社会认同",
        "templates": {
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
    },
    "authority": {
        "label_en": "Authority",
        "label_zh": "权威效应",
        "templates": {
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
    },
    "loss_aversion": {
        "label_en": "Loss Aversion",
        "label_zh": "损失厌恶",
        "templates": {
            "moisturizer": {
                "en": {
                    "subtle": "Helps protect your skin from daily environmental stressors.",
                    "moderate": "Without proper moisturizing, daily environmental exposure causes cumulative barrier damage that becomes visible within months. This formula actively prevents the degradation process.",
                    "aggressive": "Every day without adequate barrier protection, you're losing ground: transepidermal water loss accelerates, micro-inflammation compounds, and visible aging begins years ahead of schedule. Once barrier damage passes a threshold, recovery takes 3-6 months of intensive treatment. This formula is your defense against damage you can't undo.",
                },
                "zh": {
                    "subtle": "帮助保护皮肤免受日常环境侵害。",
                    "moderate": "缺乏适当保湿时，日常环境暴露造成的累积屏障损伤会在数月内显现。本配方主动阻止退化过程。",
                    "aggressive": "每一天缺乏充分的屏障保护，你都在失去：经皮水分流失加速、微炎症累积、可见衰老提前数年出现。一旦屏障损伤超过阈值，恢复需要3-6个月的密集修护。本配方是你抵御不可逆损伤的防线。",
                },
            },
            "bha_exfoliant": {
                "en": {
                    "subtle": "Helps prevent pore congestion from building up over time.",
                    "moderate": "Without regular chemical exfoliation, dead skin cells and sebum accumulate daily, leading to enlarged pores and persistent breakouts that become harder to treat over time.",
                    "aggressive": "Every day without proper BHA exfoliation, pore-clogging buildup compounds: sebum hardens, dead cells compact, and bacterial colonization deepens. What starts as minor congestion can become chronic acne scarring — permanent texture damage that no moisturizer can reverse. This formula stops the cascade before it starts.",
                },
                "zh": {
                    "subtle": "帮助防止毛孔堵塞随时间累积。",
                    "moderate": "缺乏规律的化学去角质，死皮细胞和皮脂每天累积，导致毛孔粗大和持续性爆痘，随时间推移越来越难治疗。",
                    "aggressive": "每天不使用BHA去角质，毛孔堵塞都在恶化：皮脂硬化、死皮压实、细菌定植加深。从轻微堵塞到慢性痘疤——永久性纹理损伤，任何保湿霜都无法逆转。本配方在连锁反应开始前就将其阻断。",
                },
            },
            "sunscreen": {
                "en": {
                    "subtle": "Helps shield against everyday UV damage accumulation.",
                    "moderate": "Without daily broad-spectrum protection, UV damage accumulates invisibly — each unprotected hour adds to your lifetime skin cancer risk and accelerates photoaging that cannot be reversed.",
                    "aggressive": "Every hour without adequate SPF, you're absorbing UV damage that silently rewrites your skin's DNA. Photoaging from cumulative unprotected exposure is irreversible — once collagen breaks down and hyperpigmentation sets in, no serum or treatment can fully restore what's lost. This sunscreen is the single most impactful thing you can do to prevent visible skin decline.",
                },
                "zh": {
                    "subtle": "帮助抵御日常紫外线损伤累积。",
                    "moderate": "缺乏每日广谱防护，紫外线损伤在无形中累积——每一小时无防护暴露都增加终身皮肤癌风险，加速不可逆的光老化。",
                    "aggressive": "每一小时缺乏足够SPF防护，你都在吸收默默改写皮肤DNA的紫外线损伤。累积无防护暴露造成的光老化不可逆转——一旦胶原蛋白分解、色素沉着形成，任何精华或治疗都无法完全恢复。这款防晒是你能做的最有效的抗衰措施。",
                },
            },
            "cleanser": {
                "en": {
                    "subtle": "Helps prevent daily impurity buildup from compromising your skin.",
                    "moderate": "Without proper barrier-safe cleansing, daily oil, pollution, and residue accumulate and compromise your skin barrier — leading to increased sensitivity, breakouts, and reduced efficacy of every other product in your routine.",
                    "aggressive": "Every night you skip proper cleansing, a full day's worth of sebum, PM2.5 particles, and oxidized sunscreen sits on your skin, slowly degrading your moisture barrier. A compromised barrier means every product you apply afterward works less effectively — you're wasting money on serums and moisturizers that can't penetrate. This cleanser protects the foundation that everything else depends on.",
                },
                "zh": {
                    "subtle": "帮助防止日常杂质累积损害肌肤。",
                    "moderate": "缺乏适当的温和洁面，日常油脂、污染和残留物累积损害皮肤屏障——导致敏感度增加、爆痘，以及护肤步骤中所有其他产品效果减弱。",
                    "aggressive": "每晚跳过正确洁面，一整天的皮脂、PM2.5颗粒和氧化防晒残留都在侵蚀你的皮肤屏障。屏障受损意味着后续每一步护肤效果都打折——你花在精华和乳液上的钱都在浪费。这款洁面保护的是一切护肤的根基。",
                },
            },
        },
    },
}

BIAS_TYPES = list(BIAS_STRATEGIES.keys())


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
    """Build complete product description with optional bias text appended."""
    bl = BASELINES[subcat]
    base = bl["base_desc"][lang]
    unit = bl["price_unit"]

    # Bias text is appended AFTER the base description
    desc_text = base
    if extra_text:
        desc_text = base + " " + extra_text

    if lang == "zh":
        full = (
            f"{brand_name} {product_suffix}\n"
            f"  描述：{desc_text}\n"
            f"  价格：{unit}{price:.0f}\n"
            f"  用户评分：{rating:.1f}/5（{reviews:,}条评价）"
        )
    else:
        full = (
            f"{brand_name} {product_suffix}\n"
            f"  Description: {desc_text}\n"
            f"  Price: {unit}{price:.0f}\n"
            f"  User rating: {rating:.1f}/5 ({reviews:,} reviews)"
        )
    return full


def build_bias_prompt(subcat, lang, bias_type, intensity, fictional_brand):
    """
    Build pairwise comparison prompt with bias language injected into
    the fictional brand's description.

    Key principle: objective attributes (rating, reviews, price) are IDENTICAL.
    Only the description text differs via appended bias language.
    """
    bl = BASELINES[subcat]

    # Both brands use identical objective parameters
    rating = bl["rating"]
    reviews = bl["reviews"]
    price = bl["price"]

    # Real brand: no bias text
    real_suffix = bl[f"product_real_{lang}"]
    real_desc = build_product_desc(
        subcat, lang, bl["brand_real"], real_suffix,
        rating, reviews, price, extra_text="")

    # Fictional brand: bias text appended
    cat_zh = {"moisturizer": "保湿乳液", "bha_exfoliant": "BHA精华液",
              "sunscreen": "防晒霜", "cleanser": "洁面乳"}
    cat_en = {"moisturizer": "Facial Moisturizer", "bha_exfoliant": "BHA Liquid Exfoliant",
              "sunscreen": "Sunscreen SPF 46", "cleanser": "Facial Cleanser"}
    fake_suffix = cat_zh[subcat] if lang == "zh" else cat_en[subcat]

    bias_text = BIAS_STRATEGIES[bias_type]["templates"][subcat][lang][intensity]
    fake_desc = build_product_desc(
        subcat, lang, fictional_brand, fake_suffix,
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
    """Parse LLM response to extract A/B choice."""
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

def run_exp2a(fictional_brands, repeats, intensity="moderate"):
    """
    Run Exp 2a: test each bias type individually at the given intensity.
    Default intensity is 'moderate' for Exp 2a.
    """
    print("\n" + "=" * 60)
    print(f"EXP 2a: Single Bias Breakthrough Test (intensity={intensity})")
    print("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    enabled_models = {n: c for n, c in CONFIG["models"].items() if c["enabled"]}

    # Build task list
    tasks = []
    for sc_key in BASELINES:
        for model_name, model_cfg in enabled_models.items():
            for lang in CONFIG["languages"]:
                for bias_type in BIAS_TYPES:
                    for rep in range(1, repeats + 1):
                        tasks.append({
                            "subcat": sc_key,
                            "model": model_name,
                            "model_cfg": model_cfg,
                            "lang": lang,
                            "bias_type": bias_type,
                            "intensity": intensity,
                            "repeat": rep,
                        })

    total = len(tasks)
    print(f"\nTotal tasks: {total}")
    print(f"  = {len(BASELINES)} subcats × {len(enabled_models)} models × "
          f"2 langs × {len(BIAS_TYPES)} biases × {repeats} reps")

    # Resume support
    csv_path = output_dir / "exp2a_raw.csv"
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
                else:
                    print("  ⚠ CSV format mismatch, will overwrite")
                    done_keys = set()
        except Exception as e:
            print(f"  ⚠ Failed to read CSV ({e}), will overwrite")
            done_keys = set()

    remaining = [t for t in tasks
                 if (t["subcat"], t["model"], t["lang"],
                     t["bias_type"], t["intensity"], str(t["repeat"])) not in done_keys]
    print(f"  Remaining: {len(remaining)}")
    if not remaining:
        print("  ✓ All tasks completed!")
        return csv_path

    csv_fieldnames = [
        "timestamp", "subcat", "model", "lang",
        "bias_type", "intensity", "repeat",
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
    _csv_lock = threading.Lock()

    for batch_idx, batch in enumerate(batches):
        if _STOP.is_set():
            break
        print(f"\n── Batch {batch_idx+1}/{len(batches)} ({len(batch)} tasks) ──")

        def process_task(task):
            if _STOP.is_set():
                return None
            fb = fictional_brands[task["subcat"]]
            prompt, pos_map = build_bias_prompt(
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
                status = "✓" if result["chose_fictional"] else "·"
                cf = result["chose_fictional"]
                print(f"  [{completed}/{total}] {status} {result['model']}/{result['lang']} "
                      f"{result['bias_type']} → fictional={cf}")
                with _csv_lock:
                    with open(csv_path, "a", encoding="utf-8", newline="") as f:
                        csv.DictWriter(f, fieldnames=csv_fieldnames).writerow(result)

    return csv_path


# ╔══════════════════════════════════════════════════════════════╗
# ║                      ANALYSIS                               ║
# ╚══════════════════════════════════════════════════════════════╝

def analyze(csv_path):
    """Analyze Exp 2a results and print summary statistics."""
    print("\n" + "=" * 60)
    print("EXP 2a: ANALYSIS")
    print("=" * 60)

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    valid = [r for r in rows if r["chose_fictional"] in ("True", "False")]
    parse_fail = sum(1 for r in rows if r["is_parse_fail"] == "True")
    refused = sum(1 for r in rows if r["is_refusal"] == "True")
    errors = sum(1 for r in rows if r["response"].startswith("[ERROR]"))

    print(f"\nTotal records: {total}")
    print(f"  Valid: {len(valid)}, Parse fail: {parse_fail}, Refused: {refused}, Error: {errors}")

    # ── Overall BR per bias type ──
    print(f"\n{'─'*60}")
    print(f"{'Bias Type':<18} {'N':>5} {'Chose Fictional':>16} {'BR':>8} {'Lift vs 4%':>12}")
    print(f"{'─'*60}")

    baseline_br = 0.04  # from Exp 1c L0

    bias_results = {}
    for bias in BIAS_TYPES:
        subset = [r for r in valid if r["bias_type"] == bias]
        n = len(subset)
        chose_f = sum(1 for r in subset if r["chose_fictional"] == "True")
        br = chose_f / n if n > 0 else 0
        lift = br - baseline_br
        bias_results[bias] = {"n": n, "chose_f": chose_f, "br": br, "lift": lift}
        label = BIAS_STRATEGIES[bias]["label_en"]
        print(f"{label:<18} {n:>5} {chose_f:>16} {br:>7.1%} {lift:>+11.1%}")

    # ── BR per bias × model ──
    print(f"\n{'─'*70}")
    print(f"{'Bias × Model':<30} {'N':>5} {'BR':>8}")
    print(f"{'─'*70}")

    for bias in BIAS_TYPES:
        label = BIAS_STRATEGIES[bias]["label_en"]
        for model in sorted(set(r["model"] for r in valid)):
            subset = [r for r in valid if r["bias_type"] == bias and r["model"] == model]
            n = len(subset)
            chose_f = sum(1 for r in subset if r["chose_fictional"] == "True")
            br = chose_f / n if n > 0 else 0
            print(f"  {label:<16} {model:<12} {n:>5} {br:>7.1%}")

    # ── BR per bias × language ──
    print(f"\n{'─'*60}")
    print(f"{'Bias × Lang':<25} {'N':>5} {'BR':>8}")
    print(f"{'─'*60}")

    for bias in BIAS_TYPES:
        label = BIAS_STRATEGIES[bias]["label_en"]
        for lang in ["en", "zh"]:
            subset = [r for r in valid if r["bias_type"] == bias and r["lang"] == lang]
            n = len(subset)
            chose_f = sum(1 for r in subset if r["chose_fictional"] == "True")
            br = chose_f / n if n > 0 else 0
            print(f"  {label:<16} {lang:<6} {n:>5} {br:>7.1%}")

    # ── BR per bias × subcategory ──
    print(f"\n{'─'*60}")
    print(f"{'Bias × Subcat':<30} {'N':>5} {'BR':>8}")
    print(f"{'─'*60}")

    for bias in BIAS_TYPES:
        label = BIAS_STRATEGIES[bias]["label_en"]
        for sc in BASELINES:
            subset = [r for r in valid if r["bias_type"] == bias and r["subcat"] == sc]
            n = len(subset)
            chose_f = sum(1 for r in subset if r["chose_fictional"] == "True")
            br = chose_f / n if n > 0 else 0
            print(f"  {label:<16} {sc:<12} {n:>5} {br:>7.1%}")

    # ── Parse failure rate by model ──
    print(f"\n{'─'*40}")
    print("Parse Failure Rate by Model:")
    for model in sorted(set(r["model"] for r in rows)):
        subset = [r for r in rows if r["model"] == model]
        pf = sum(1 for r in subset if r["is_parse_fail"] == "True")
        print(f"  {model}: {pf}/{len(subset)} = {pf/len(subset):.1%}")

    # Write report
    report_path = Path(CONFIG["output_dir"]) / "exp2a_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Exp 2a — Single Bias Breakthrough Test\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Total records: {total}\n")
        f.write(f"Valid: {len(valid)}, Parse fail: {parse_fail}, Refused: {refused}, Error: {errors}\n\n")

        f.write(f"Overall BR per Bias Type (baseline ≈ 4%):\n")
        f.write(f"{'─'*50}\n")
        for bias in BIAS_TYPES:
            r = bias_results[bias]
            label = BIAS_STRATEGIES[bias]["label_en"]
            f.write(f"  {label:<18} BR={r['br']:.1%}  Lift={r['lift']:+.1%}  (n={r['n']})\n")

        # χ² test vs baseline
        f.write(f"\nχ² Tests vs Baseline (4%):\n")
        f.write(f"{'─'*50}\n")
        for bias in BIAS_TYPES:
            r = bias_results[bias]
            # Simple χ² for proportion test
            n = r["n"]
            observed = r["chose_f"]
            expected = n * baseline_br
            if expected > 0:
                chi2 = (observed - expected)**2 / expected + ((n - observed) - (n - expected))**2 / (n - expected)
            else:
                chi2 = float('inf')
            label = BIAS_STRATEGIES[bias]["label_en"]
            f.write(f"  {label:<18} χ²={chi2:.2f}  (n={n}, observed={observed}, expected={expected:.1f})\n")

    print(f"\nReport saved: {report_path}")
    return bias_results


# ╔══════════════════════════════════════════════════════════════╗
# ║                         CLI                                  ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="Exp 2a: Single Bias Breakthrough")
    parser.add_argument("--dry-run", action="store_true", help="Simulate API calls")
    parser.add_argument("--fresh", action="store_true", help="Clear old data and rerun")
    parser.add_argument("--repeats", type=int, default=20, help="Repetitions per condition")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing data")
    parser.add_argument("--workers", type=int, default=20, help="Max parallel workers")
    args = parser.parse_args()

    CONFIG["dry_run"] = args.dry_run
    CONFIG["repeats"] = args.repeats
    CONFIG["max_parallel_workers"] = args.workers

    if args.dry_run:
        print("🏃 DRY RUN MODE — no real API calls")

    csv_path = Path(CONFIG["output_dir"]) / "exp2a_raw.csv"

    if args.fresh and csv_path.exists():
        try:
            csv_path.unlink()
        except PermissionError:
            # Overwrite instead of delete
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                pass  # truncate
        print("🗑 Cleared old data")

    if not args.analyze_only:
        print("\nLoading fictional brand pool…")
        fictional_brands = load_fictional_brands()
        csv_path = run_exp2a(fictional_brands, args.repeats)

    if csv_path.exists():
        analyze(csv_path)
    else:
        print("No data to analyze.")


if __name__ == "__main__":
    main()
