#!/usr/bin/env python3
"""
Generate publication-quality summary figure for Experiment 3.
Reads from results/exp3/exp3_raw.csv.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)

# ── Style — TrueType fonts for proper PDF embedding ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,      # TrueType embedding
    "ps.fonttype": 42,
})

MODEL_ORDER = ["claude-sonnet", "gpt-4o-mini", "gemini-flash"]
MODEL_LABELS = {
    "claude-sonnet": "Claude",
    "gpt-4o-mini": "GPT",
    "gemini-flash": "Gemini",
}
MODEL_LABELS_FULL = {
    "claude-sonnet": "Claude Sonnet",
    "gpt-4o-mini": "GPT-4o-mini",
    "gemini-flash": "Gemini 3 Flash",
}
MODEL_COLORS = {
    "claude-sonnet": "#4e79a7",
    "gpt-4o-mini": "#e15759",
    "gemini-flash": "#59a14f",
}
MODEL_MARKERS = {"claude-sonnet": "o", "gpt-4o-mini": "s", "gemini-flash": "D"}
MODEL_LINESTYLES = {"claude-sonnet": "-", "gpt-4o-mini": "--", "gemini-flash": "-."}

SCENARIOS = ["S0", "S1", "S2", "S3", "S4"]
K_GEO = {"S0": 0, "S1": 1, "S2": 3, "S3": 6, "S4": 9}
SUBCATS = ["moisturizer", "sunscreen", "cleanser", "bha_exfoliant"]
SUBCAT_LABELS = ["Moist.", "Sunsc.", "Clean.", "BHA"]


def fig3_summary():
    # ── Load data ──
    df = pd.read_csv(os.path.join(RESULTS, "exp3", "exp3_raw.csv"))
    df = df[(df.is_parse_fail.astype(str) != "True") &
            (df.is_refusal.astype(str) != "True")].copy()

    beta = df[df.protocol == "beta"].copy()
    alpha = df[df.protocol == "alpha"].copy()
    alpha["real_rank"] = pd.to_numeric(alpha.real_rank, errors="coerce")
    alpha = alpha.dropna(subset=["real_rank"])

    # ── Pre-compute metrics ──
    # ISR per scenario × model (protocol beta)
    isr = {}
    for s in SCENARIOS:
        isr[s] = {}
        for m in MODEL_ORDER:
            sub = beta[(beta.scenario == s) & (beta.model == m)]
            isr[s][m] = sub.chose_real.mean() * 100 if len(sub) > 0 else 0
        sub_all = beta[beta.scenario == s]
        isr[s]["overall"] = sub_all.chose_real.mean() * 100

    # Mean rank per scenario × model (protocol alpha)
    ranks = {}
    for s in SCENARIOS:
        ranks[s] = {}
        for m in MODEL_ORDER:
            sub = alpha[(alpha.scenario == s) & (alpha.model == m)]
            ranks[s][m] = sub.real_rank.mean() if len(sub) > 0 else 5

    # Payoff per GEO brand per scenario
    payoff = {}
    for s in SCENARIOS:
        k = K_GEO[s]
        if k > 0:
            sub = beta[beta.scenario == s]
            geo_rate = sub.chose_geo.mean()
            payoff[s] = geo_rate / k

    # HHI per scenario
    hhi = {}
    for s in SCENARIOS:
        sub = beta[beta.scenario == s]
        k = K_GEO[s]
        isr_rate = sub.chose_real.mean()
        geo_rate = sub.chose_geo.mean()
        nongeo_rate = max(0, 1 - isr_rate - geo_rate)

        h = isr_rate ** 2
        if k > 0:
            h += k * (geo_rate / k) ** 2
        nongeo_count = 9 - k
        if nongeo_count > 0 and nongeo_rate > 0:
            h += nongeo_count * (nongeo_rate / nongeo_count) ** 2
        hhi[s] = h

    # KL divergence per model, D_KL(S4 || S0) over the two-category
    # distribution {P(incumbent), P(fictional)}, natural log, zero cells
    # floored at eps=1e-6 (same definition as exp3_game.py analyze_exp3).
    # NOTE: because P(fictional | S0) = 0, magnitudes depend on eps and
    # should be read only as an ordering.
    eps = 1e-6
    kl_values = {}
    for m in MODEL_ORDER:
        p4 = np.array([isr["S4"][m] / 100, 1 - isr["S4"][m] / 100])
        p0 = np.array([isr["S0"][m] / 100, 1 - isr["S0"][m] / 100])
        p4 = np.clip(p4, eps, None); p4 /= p4.sum()
        p0 = np.clip(p0, eps, None); p0 /= p0.sum()
        kl_values[m] = float(np.sum(p4 * np.log(p4 / p0)))

    # Subcategory ISR at S1 and S4
    subcat_isr = {}
    for s in ["S1", "S4"]:
        subcat_isr[s] = {}
        for sc in SUBCATS:
            sub = beta[(beta.scenario == s) & (beta.subcat == sc)]
            subcat_isr[s][sc] = sub.chose_real.mean() * 100 if len(sub) > 0 else 0

    # ── Validity stats for subtitle (compute from unfiltered data) ──
    df_raw = pd.read_csv(os.path.join(RESULTS, "exp3", "exp3_raw.csv"))
    n_total = len(df_raw)
    n_valid = len(df_raw[(df_raw.is_parse_fail.astype(str) != "True") &
                         (df_raw.is_refusal.astype(str) != "True")])
    validity = n_valid / n_total * 100 if n_total > 0 else 0

    # ── Figure layout: 2 rows × 3 columns ──
    fig = plt.figure(figsize=(14, 8.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32,
                           left=0.06, right=0.97, top=0.87, bottom=0.06)

    # ═══ Panel (a): U-Curve ISR (Protocol β) ═══
    ax1 = fig.add_subplot(gs[0, 0])

    x = np.arange(len(SCENARIOS))
    # Overall (gray dashed)
    overall_vals = [isr[s]["overall"] for s in SCENARIOS]
    ax1.plot(x, overall_vals, marker="x", color="#999999", linestyle="--",
             linewidth=1.5, markersize=6, label="Overall", zorder=2)

    for m in MODEL_ORDER:
        vals = [isr[s][m] for s in SCENARIOS]
        ax1.plot(x, vals, marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                 linestyle=MODEL_LINESTYLES[m], linewidth=2, markersize=7,
                 label=MODEL_LABELS[m], zorder=3)

    ax1.axhline(y=50, color="#cccccc", linestyle="--", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(SCENARIOS)
    ax1.set_ylabel("ISR %")
    ax1.set_ylim(-5, 108)
    ax1.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    ax1.set_title("(a) U-Curve ISR (Protocol β)",
                  fontsize=10, fontweight="bold", pad=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ═══ Panel (b): Incumbent Rank (Protocol α) ═══
    ax2 = fig.add_subplot(gs[0, 1])

    for m in MODEL_ORDER:
        vals = [ranks[s][m] for s in SCENARIOS]
        ax2.plot(x, vals, marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                 linestyle=MODEL_LINESTYLES[m], linewidth=2, markersize=7,
                 label=MODEL_LABELS[m], zorder=3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(SCENARIOS)
    ax2.set_ylabel("Mean Rank")
    ax2.invert_yaxis()
    ax2.set_ylim(9.5, -0.5)
    ax2.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    ax2.set_title("(b) Incumbent Rank (Protocol α)",
                  fontsize=10, fontweight="bold", pad=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ═══ Panel (c): Payoff Decay (DI) ═══
    ax3 = fig.add_subplot(gs[0, 2])

    payoff_scenarios = ["S1", "S2", "S3", "S4"]
    payoff_vals = [payoff[s] for s in payoff_scenarios]
    y_pos = np.arange(len(payoff_scenarios))

    # Color gradient: S1 bright → S4 faded
    payoff_colors = ["#f97316", "#fb923c", "#c9a960", "#a3a373"]
    bars = ax3.barh(y_pos, payoff_vals, color=payoff_colors,
                    edgecolor="white", height=0.6)

    for i, v in enumerate(payoff_vals):
        ax3.text(v + 0.015, i, f"+{v:.3f}", ha="left", va="center",
                 fontsize=9, fontweight="bold", color=payoff_colors[i])

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(payoff_scenarios)
    ax3.invert_yaxis()
    ax3.set_xlabel("GEO Payoff")
    ax3.set_xlim(0, 0.95)
    ax3.set_title("(c) Payoff Decay (DI)",
                  fontsize=10, fontweight="bold", pad=8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # ═══ Panel (d): Market Concentration (HHI) ═══
    ax4 = fig.add_subplot(gs[1, 0])

    hhi_vals = [hhi[s] for s in SCENARIOS]
    hhi_colors = ["#c0392b", "#f39c12", "#27ae60", "#f39c12", "#c0392b"]
    bars = ax4.bar(x, hhi_vals, color=hhi_colors, edgecolor="white", width=0.6)

    for i, v in enumerate(hhi_vals):
        ax4.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    ax4.set_xticks(x)
    ax4.set_xticklabels(SCENARIOS)
    ax4.set_ylabel("HHI")
    ax4.set_ylim(0, 1.15)
    ax4.set_title("(d) Market Concentration (HHI)",
                  fontsize=10, fontweight="bold", pad=8)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # ═══ Panel (e): Distribution Shift at S4 ═══
    ax5 = fig.add_subplot(gs[1, 1])

    kl_vals = [kl_values[m] for m in MODEL_ORDER]
    kl_bar_colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    x_kl = np.arange(len(MODEL_ORDER))

    bars = ax5.bar(x_kl, kl_vals, color=kl_bar_colors,
                   edgecolor="white", width=0.55)

    for i, v in enumerate(kl_vals):
        ax5.text(i, v + 0.03, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    ax5.set_xticks(x_kl)
    ax5.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax5.set_ylabel("KL(S4 ∥ S0)")
    ax5.set_ylim(0, 2.0)
    ax5.set_title("(e) Distribution Shift at S4",
                  fontsize=10, fontweight="bold", pad=8)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    # ═══ Panel (f): Subcategory Variation ═══
    ax6 = fig.add_subplot(gs[1, 2])

    x_sc = np.arange(len(SUBCATS))
    width = 0.32

    s1_vals = [subcat_isr["S1"][sc] for sc in SUBCATS]
    s4_vals = [subcat_isr["S4"][sc] for sc in SUBCATS]

    bars1 = ax6.bar(x_sc - width / 2, s1_vals, width,
                    label="S1 (max disruption)", color="#e15759",
                    edgecolor="white", alpha=0.85)
    bars2 = ax6.bar(x_sc + width / 2, s4_vals, width,
                    label="S4 (recovery)", color="#4e79a7",
                    edgecolor="white", alpha=0.85)

    ax6.set_xticks(x_sc)
    ax6.set_xticklabels(SUBCAT_LABELS)
    ax6.set_ylabel("ISR %")
    ax6.set_ylim(0, 115)
    ax6.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax6.set_title("(f) Subcategory Variation",
                  fontsize=10, fontweight="bold", pad=8)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    # ── Suptitle ──
    fig.suptitle(
        "Experiment 3: Multi-Agent GEO Competition — Summary\n"
        f"4,800 calls · {validity:.1f}% valid · "
        "5 scenarios × 4 subcategories × 3 models × 2 languages × 2 protocols × 20 reps",
        fontsize=12, fontweight="bold", y=0.97
    )

    # ── Save ──
    fig.savefig(os.path.join(OUT, "fig3_summary.pdf"))
    fig.savefig(os.path.join(OUT, "fig3_summary.png"))
    plt.close(fig)
    print("✓ fig3_summary")
    print(f"  HHI: {[f'{v:.3f}' for v in hhi_vals]}")
    print(f"  Payoff: {[f'{v:.3f}' for v in payoff_vals]}")
    print(f"  KL: {[f'{v:.3f}' for v in kl_vals]}")


if __name__ == "__main__":
    print("Generating Exp 3 figures...")
    fig3_summary()
    print(f"\nFigures saved to: {OUT}/")
