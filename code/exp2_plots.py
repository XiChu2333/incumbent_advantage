#!/usr/bin/env python3
"""
Generate publication-quality summary figure for Experiment 2.
Reads from results/exp2a, results/exp2b, results/exp2_bsv.
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
    "claude-sonnet": "Claude Sonnet",
    "gpt-4o-mini": "GPT-4o-mini",
    "gemini-flash": "Gemini 3 Flash",
}
BIAS_ORDER = ["authority", "social_proof", "anchoring", "scarcity", "loss_aversion"]
BIAS_LABELS = {
    "authority": "Authority",
    "social_proof": "Social Proof",
    "anchoring": "Anchoring",
    "scarcity": "Scarcity",
    "loss_aversion": "Loss Aversion",
}


def fig2_summary():
    # ── Load data ──
    df_a = pd.read_csv(os.path.join(RESULTS, "exp2a", "exp2a_raw.csv"))
    df_a = df_a[df_a.is_parse_fail.astype(str) != "True"].copy()

    df_b = pd.read_csv(os.path.join(RESULTS, "exp2b", "exp2b_raw.csv"))
    df_b = df_b[df_b.is_parse_fail.astype(str) != "True"].copy()

    bsv = pd.read_csv(os.path.join(RESULTS, "exp2_bsv", "bsv_table.csv"))

    # ── Figure layout ──
    fig = plt.figure(figsize=(10, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    # ═══ Panel (a): Single Bias BR by Model — Heatmap ═══
    ax1 = fig.add_subplot(gs[0, 0])

    mat = np.zeros((len(BIAS_ORDER), len(MODEL_ORDER)))
    for i, bias in enumerate(BIAS_ORDER):
        for j, model in enumerate(MODEL_ORDER):
            sub = df_a[(df_a.bias_type == bias) & (df_a.model == model)]
            mat[i, j] = sub.chose_fictional.mean() * 100 if len(sub) > 0 else 0

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("br_heat", [
        "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
        "#4292c6", "#2171b5", "#08519c", "#08306b"
    ], N=256)

    im = ax1.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=100)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            color = "white" if v > 55 else "black"
            ax1.text(j, i, f"{v:.0f}%", ha="center", va="center",
                     fontsize=10, fontweight="bold", color=color)

    ax1.set_xticks(range(len(MODEL_ORDER)))
    ax1.set_xticklabels([MODEL_LABELS[m].split()[0] for m in MODEL_ORDER])
    ax1.set_yticks(range(len(BIAS_ORDER)))
    ax1.set_yticklabels([BIAS_LABELS[b] for b in BIAS_ORDER], fontsize=9)
    ax1.set_title("(a) Single Bias BR by Model", fontsize=10, fontweight="bold", pad=8)

    # ═══ Panel (b): Authority Intensity Curve ═══
    ax2 = fig.add_subplot(gs[0, 1])

    intensities = ["subtle", "moderate", "aggressive"]
    intensity_labels = ["Subtle", "Moderate", "Aggressive"]
    colors = {"claude-sonnet": "#4e79a7", "gpt-4o-mini": "#e15759", "gemini-flash": "#59a14f"}
    markers = {"claude-sonnet": "o", "gpt-4o-mini": "s", "gemini-flash": "D"}
    linestyles = {"claude-sonnet": "-", "gpt-4o-mini": "--", "gemini-flash": "-."}

    # Combine exp2a (moderate) and exp2b (subtle/aggressive) for authority
    for model in MODEL_ORDER:
        vals = []
        for intensity in intensities:
            if intensity == "moderate":
                sub = df_a[(df_a.bias_type == "authority") & (df_a.model == model)]
            else:
                sub = df_b[(df_b.bias_type == "authority") & (df_b.model == model)
                           & (df_b.intensity == intensity)
                           & (df_b.stacking.astype(str) == "False")]
            vals.append(sub.chose_fictional.mean() * 100 if len(sub) > 0 else 0)

        ax2.plot(range(3), vals, marker=markers[model], color=colors[model],
                 linestyle=linestyles[model], linewidth=2, markersize=7,
                 label=MODEL_LABELS[model], zorder=3)

        # Annotate key points
        for k, v in enumerate(vals):
            if k == 1:  # moderate
                ax2.annotate(f"{v:.0f}%", (k, v), textcoords="offset points",
                             xytext=(-15, 8), fontsize=8, color=colors[model])

    ax2.set_xticks(range(3))
    ax2.set_xticklabels(intensity_labels)
    ax2.set_ylabel("BR (%)")
    ax2.set_ylim(-5, 115)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax2.set_title("(b) Authority Intensity Curve", fontsize=10, fontweight="bold", pad=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ═══ Panel (c): Stacking — Auth + Social Proof ═══
    ax3 = fig.add_subplot(gs[1, 0])

    # Auth moderate from exp2a, SP moderate from exp2a, Auth+SP moderate from exp2b
    categories = ["Auth", "SP", "Auth+SP"]
    x = np.arange(len(categories))
    width = 0.22

    model_colors = {"claude-sonnet": "#4e79a7", "gpt-4o-mini": "#e15759", "gemini-flash": "#59a14f"}

    for idx, model in enumerate(MODEL_ORDER):
        auth_val = df_a[(df_a.bias_type == "authority") & (df_a.model == model)].chose_fictional.mean() * 100
        sp_val = df_a[(df_a.bias_type == "social_proof") & (df_a.model == model)].chose_fictional.mean() * 100
        stacked = df_b[(df_b.model == model) & (df_b.stacking.astype(str) == "True")
                       & (df_b.intensity == "moderate")]
        stack_val = stacked.chose_fictional.mean() * 100 if len(stacked) > 0 else 0

        vals = [auth_val, sp_val, stack_val]
        offset = (idx - 1) * width
        bars = ax3.bar(x + offset, vals, width, label=MODEL_LABELS[model].split()[0],
                       color=model_colors[model], edgecolor="white", linewidth=0.5)
        for bar, v in enumerate(vals):
            ax3.text(x[bar] + offset, v + 2, f"{v:.0f}%", ha="center", va="bottom",
                     fontsize=7, fontweight="bold")

    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.set_ylabel("BR (%)")
    ax3.set_ylim(0, 115)
    ax3.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax3.set_title("(c) Stacking: Auth + Social Proof", fontsize=10, fontweight="bold", pad=8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # ═══ Panel (d): Bias Surplus Value ═══
    ax4 = fig.add_subplot(gs[1, 1])

    bsv_order = ["authority", "social_proof", "anchoring", "scarcity", "loss_aversion"]
    bsv_labels = ["Authority", "Social Proof", "Anchoring", "Scarcity", "Loss Aversion"]
    bsv_vals = []
    for bt in bsv_order:
        row = bsv[bsv.bias_type == bt]
        bsv_vals.append(row.bsv_rating_linear.values[0] if len(row) > 0 else 0)

    y_pos = np.arange(len(bsv_order))
    bar_colors = ["#2171b5", "#41ab5d", "#bdbdbd", "#bdbdbd", "#bdbdbd"]
    bars = ax4.barh(y_pos, bsv_vals, color=bar_colors, edgecolor="white", height=0.6)

    for i, v in enumerate(bsv_vals):
        if v > 0.02:
            ax4.text(v + 0.005, i, f"+{v:.2f}", ha="left", va="center",
                     fontsize=9, fontweight="bold", color=bar_colors[i])
        else:
            ax4.text(0.015, i, "≈ 0", ha="left", va="center",
                     fontsize=8, color="#999999", style="italic")

    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(bsv_labels, fontsize=9)
    ax4.invert_yaxis()
    ax4.set_xlabel("BSV (rating points)")
    ax4.set_xlim(0, 0.30)
    ax4.axvline(x=0.1, color="#cccccc", linestyle=":", linewidth=0.8)
    ax4.axvline(x=0.2, color="#cccccc", linestyle=":", linewidth=0.8)
    ax4.set_title("(d) Bias Surplus Value", fontsize=10, fontweight="bold", pad=8)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # ── Save ──
    fig.savefig(os.path.join(OUT, "fig2_summary.pdf"))
    fig.savefig(os.path.join(OUT, "fig2_summary.png"))
    plt.close(fig)
    print("✓ fig2_summary")
    print(f"  BSV Authority = +{bsv_vals[0]:.2f}, Social Proof = +{bsv_vals[1]:.2f}")


if __name__ == "__main__":
    print("Generating Exp 2 figures...")
    fig2_summary()
    print(f"\nFigures saved to: {OUT}/")
