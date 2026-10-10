#!/usr/bin/env python3
"""
Generate publication-quality figures for Experiment 1.
Inspired by Pfrommer et al. (EMNLP 2024) Figure 2b/2d heatmap style.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# ── Paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)

# ── Style ──
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

MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o-mini",
    "claude-sonnet": "Claude Sonnet",
    "gemini-flash": "Gemini 3 Flash",
}
LANG_LABELS = {"en": "EN", "zh": "ZH"}
SUBCAT_LABELS = {
    "moisturizer": "Moisturizer",
    "bha_exfoliant": "BHA Exfoliant",
    "sunscreen": "Sunscreen",
    "cleanser": "Cleanser",
}
SUBCAT_ORDER = ["moisturizer", "bha_exfoliant", "sunscreen", "cleanser"]
MODEL_ORDER = ["gpt-4o-mini", "claude-sonnet", "gemini-flash"]
DIM_LABELS = {"rating": "Rating", "reviews": "Reviews", "price": "Price", "ingredient": "Ingredient"}
DIM_ORDER = ["rating", "reviews", "price", "ingredient"]

# Color maps
CMAP_HEAT = LinearSegmentedColormap.from_list("brand_heat", [
    "#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"
], N=256)
CMAP_RANK = LinearSegmentedColormap.from_list("rank_heat", [
    "#1a0636", "#4a1486", "#7a33a8", "#b26ac5", "#e09de0", "#f5d5f5"
], N=256)


# ═══════════════════════════════════════
# FIGURE 1: Exp 1a — IAI Baseline Heatmap (Protocol α Top-1 rate)
# ═══════════════════════════════════════
def fig1_baseline():
    df = pd.read_csv(os.path.join(RESULTS, "exp1a", "exp1a_raw.csv"))
    # Filter protocol alpha, successfully parsed
    da = df[(df.protocol == "alpha") & (df.is_parse_fail.astype(str) != "True")].copy()

    # Top-1 = real_rank == 1
    da["top1"] = (da.real_rank == 1).astype(int)

    pivot = da.groupby(["model", "lang", "subcat"])["top1"].mean()

    # Build matrix: rows = model×lang, cols = subcat
    row_keys = [(m, l) for m in MODEL_ORDER for l in ["en", "zh"]]
    row_labels = [f"{MODEL_LABELS[m]} ({LANG_LABELS[l]})" for m, l in row_keys]
    col_keys = SUBCAT_ORDER
    col_labels = [SUBCAT_LABELS[s] for s in col_keys]

    mat = np.full((len(row_keys), len(col_keys)), np.nan)
    for i, (m, l) in enumerate(row_keys):
        for j, s in enumerate(col_keys):
            try:
                mat[i, j] = pivot.loc[(m, l, s)]
            except KeyError:
                pass

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    im = ax.imshow(mat * 100, cmap="YlOrRd", aspect="auto", vmin=50, vmax=100)

    # Annotate
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                color = "white" if v * 100 > 85 else "black"
                ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center", fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Horizontal lines to separate models
    for k in [2, 4]:
        ax.axhline(y=k - 0.5, color="white", linewidth=2)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Top-1 Hit Rate (%)", fontsize=9)

    ax.set_title("(a) Exp 1a: Protocol α Top-1 Hit Rate\n(Real brand ranked #1 out of 10 identical products)", fontsize=10, pad=8)

    fig.savefig(os.path.join(OUT, "fig1a_baseline_heatmap.pdf"))
    fig.savefig(os.path.join(OUT, "fig1a_baseline_heatmap.png"))
    plt.close(fig)
    print("✓ fig1a_baseline_heatmap")


# ═══════════════════════════════════════
# FIGURE 2: Exp 1b — 2×2 BOR grouped bar
# ═══════════════════════════════════════
def fig2_bor():
    df = pd.read_csv(os.path.join(RESULTS, "exp1b", "exp1b_raw.csv"))
    df = df[df.is_parse_fail.astype(str) != "True"].copy()

    comparisons = ["main", "aux1", "aux2", "aux3"]
    comp_labels = [
        "MAIN\n(Real+Bad vs\nFake+Good)",
        "AUX1\n(Real+Good vs\nFake+Bad)",
        "AUX2\n(Real+Good vs\nFake+Good)",
        "AUX3\n(Real+Bad vs\nFake+Bad)",
    ]

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(comparisons))
    width = 0.18
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

    for idx, subcat in enumerate(SUBCAT_ORDER):
        vals = []
        for comp in comparisons:
            sub = df[(df.subcat == subcat) & (df.comparison == comp)]
            if len(sub) > 0:
                vals.append(sub.chose_real.mean() * 100)
            else:
                vals.append(0)
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=SUBCAT_LABELS[subcat], color=colors[idx], edgecolor="white", linewidth=0.5)
        # Annotate MAIN bars
        for k, v in enumerate(vals):
            if k == 0:  # MAIN comparison
                ax.text(x[k] + offset, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels, fontsize=8)
    ax.set_ylabel("Real Brand Chosen (%)")
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9, ncol=2, fontsize=8)

    # Add annotation for MAIN
    ax.annotate("BOR ≈ 2–5%\n(brand name fails\nwhen quality differs)",
                xy=(0, 8), fontsize=8, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", edgecolor="#ffc107", alpha=0.9))

    ax.set_title("(b) Exp 1b: Brand Override Rate — 2×2 Factorial Swap", fontsize=10, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(os.path.join(OUT, "fig1b_bor_bars.pdf"))
    fig.savefig(os.path.join(OUT, "fig1b_bor_bars.png"))
    plt.close(fig)
    print("✓ fig1b_bor_bars")


# ═══════════════════════════════════════
# FIGURE 3: Exp 1c — Gradient Heatmap (KEY FIGURE)
# Model×Lang rows vs Level columns, one panel per dimension
# ═══════════════════════════════════════
def fig3_gradient_heatmap():
    df = pd.read_csv(os.path.join(RESULTS, "exp1c", "exp1c_raw.csv"))
    df = df[df.is_parse_fail.astype(str) != "True"].copy()

    row_keys = [(m, l) for m in MODEL_ORDER for l in ["en", "zh"]]
    row_labels = [f"{MODEL_LABELS[m]} {LANG_LABELS[l]}" for m, l in row_keys]
    levels = ["L0", "L1", "L2", "L3", "L4"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    fig.suptitle("(c) Exp 1c: Fictional Brand Win Rate by Advantage Level\n(Aggregated across subcategories)", fontsize=12, y=1.02)

    for panel_idx, dim in enumerate(DIM_ORDER):
        ax = axes[panel_idx // 2][panel_idx % 2]

        mat = np.full((len(row_keys), len(levels)), np.nan)
        for i, (m, l) in enumerate(row_keys):
            for j, lev in enumerate(levels):
                sub = df[(df.dimension == dim) & (df.model == m) & (df.lang == l) & (df.level_id == lev)]
                if len(sub) > 0:
                    mat[i, j] = sub.chose_fictional.mean() * 100

        im = ax.imshow(mat, cmap=CMAP_HEAT, aspect="auto", vmin=0, vmax=100)

        # Annotate cells
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isnan(v):
                    color = "white" if (v > 75 or v < 15) else "black"
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, fontweight="bold", color=color)

        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels if panel_idx % 2 == 0 else [], fontsize=8)

        # Dim-specific level descriptions
        desc_map = {
            "rating": ["+0.0", "+0.1", "+0.2", "+0.3", "+0.4"],
            "reviews": ["1×", "2×", "5×", "10×", "20×"],
            "price": ["0%", "−10%", "−20%", "−30%", "−50%"],
            "ingredient": ["Equal", "Slight+", "Mod+", "Strong+", "Major+"],
        }
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(range(len(levels)))
        ax2.set_xticklabels(desc_map[dim], fontsize=7, color="#666666")

        # Separator lines between models
        for k in [2, 4]:
            ax.axhline(y=k - 0.5, color="white", linewidth=1.5)

        # Mark L0→L1 step
        ax.axvline(x=0.5, color="red", linewidth=1.2, linestyle="--", alpha=0.6)

        ax.set_title(DIM_LABELS[dim], fontsize=11, fontweight="bold", pad=22)

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, location="bottom", aspect=40)
    cbar.set_label("Fictional Brand Win Rate (%)", fontsize=9)

    fig.savefig(os.path.join(OUT, "fig1c_gradient_heatmap.pdf"))
    fig.savefig(os.path.join(OUT, "fig1c_gradient_heatmap.png"))
    plt.close(fig)
    print("✓ fig1c_gradient_heatmap")


# ═══════════════════════════════════════
# FIGURE 4: Exp 1c — Step-function line plot (aggregate)
# ═══════════════════════════════════════
def fig4_stepfunction():
    df = pd.read_csv(os.path.join(RESULTS, "exp1c", "exp1c_raw.csv"))
    df = df[df.is_parse_fail.astype(str) != "True"].copy()

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True, constrained_layout=True)

    colors_model = {"gpt-4o-mini": "#e15759", "claude-sonnet": "#4e79a7", "gemini-flash": "#59a14f"}
    markers_model = {"gpt-4o-mini": "o", "claude-sonnet": "s", "gemini-flash": "D"}

    for panel_idx, dim in enumerate(DIM_ORDER):
        ax = axes[panel_idx]
        levels = ["L0", "L1", "L2", "L3", "L4"]
        x = range(5)

        for model in MODEL_ORDER:
            vals = []
            for lev in levels:
                sub = df[(df.dimension == dim) & (df.model == model) & (df.level_id == lev)]
                if len(sub) > 0:
                    vals.append(sub.chose_fictional.mean() * 100)
                else:
                    vals.append(np.nan)
            ax.plot(x, vals, marker=markers_model[model], color=colors_model[model],
                    linewidth=2, markersize=6, label=MODEL_LABELS[model], zorder=3)

        ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvspan(-0.5, 0.5, alpha=0.08, color="blue")  # L0 zone

        ax.set_xticks(x)
        desc_map = {
            "rating": ["+0.0", "+0.1", "+0.2", "+0.3", "+0.4"],
            "reviews": ["1×", "2×", "5×", "10×", "20×"],
            "price": ["0%", "−10%", "−20%", "−30%", "−50%"],
            "ingredient": ["=", "Sli+", "Mod+", "Str+", "Maj+"],
        }
        ax.set_xticklabels(desc_map[dim], fontsize=7.5)
        ax.set_title(DIM_LABELS[dim], fontsize=10, fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if panel_idx == 0:
            ax.set_ylabel("Fictional Brand Win Rate (%)")
        if panel_idx == 3:
            ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    fig.suptitle("(d) Exp 1c: Step-Function Transition — L0→L1 Abrupt Shift by Model", fontsize=11, y=1.04)

    fig.savefig(os.path.join(OUT, "fig1c_stepfunction.pdf"))
    fig.savefig(os.path.join(OUT, "fig1c_stepfunction.png"))
    plt.close(fig)
    print("✓ fig1c_stepfunction")


# ═══════════════════════════════════════
# FIGURE 5: Exp 1d — η² Decomposition (Pfrommer Figure 2d style)
# ═══════════════════════════════════════
def fig5_anova():
    # Read exp1d data and compute ANOVA
    df = pd.read_csv(os.path.join(RESULTS, "exp1d", "exp1d_raw.csv"))
    df = df[df.is_parse_fail.astype(str) != "True"].copy()
    df = df[pd.to_numeric(df.target_rank, errors="coerce").notna()].copy()
    df["target_rank"] = df["target_rank"].astype(int)

    # We need: target_rank ~ target_is_real (brand) + target_param_level (params) + target_position (position)
    # Compute η² per model

    def compute_eta2(data, group_col, value_col="target_rank"):
        grand_mean = data[value_col].mean()
        groups = data.groupby(group_col)[value_col]
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for _, g in groups)
        ss_total = ((data[value_col] - grand_mean)**2).sum()
        return ss_between / ss_total if ss_total > 0 else 0

    factors = [
        ("Params", "target_param_level"),
        ("Position", "target_position"),
        ("Brand", "target_is_real"),
    ]

    models_1d = ["gpt-4o-mini", "claude-sonnet", "gemini-flash"]
    model_labels_1d = {"gpt-4o-mini": "GPT-4o-mini", "claude-sonnet": "Claude Sonnet", "gemini-flash": "Gemini Flash"}

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), constrained_layout=True)

    # Panel 1: Overall η²
    ax = axes[0]
    eta2_all = [compute_eta2(df, col) for _, col in factors]
    bars = ax.bar(range(3), eta2_all, color=["#4e79a7", "#f28e2b", "#e15759"], edgecolor="white", width=0.6)
    for bar, v in zip(bars, eta2_all):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f[0] for f in factors], fontsize=9)
    ax.set_ylabel("η² (Proportion of Variance)")
    ax.set_title("Overall", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2 & 3: Per model
    for midx, model in enumerate(models_1d):
        ax = axes[midx + 1]
        dmodel = df[df.model == model]
        eta2_m = [compute_eta2(dmodel, col) for _, col in factors]
        bars = ax.bar(range(3), eta2_m, color=["#4e79a7", "#f28e2b", "#e15759"], edgecolor="white", width=0.6)
        for bar, v in zip(bars, eta2_m):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_xticklabels([f[0] for f in factors], fontsize=9)
        ax.set_title(model_labels_1d[model], fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("(e) Exp 1d: Variance Decomposition (η²) — rank ~ params + position + brand", fontsize=11, y=1.04)

    fig.savefig(os.path.join(OUT, "fig1d_anova_eta2.pdf"))
    fig.savefig(os.path.join(OUT, "fig1d_anova_eta2.png"))
    plt.close(fig)
    print("✓ fig1d_anova_eta2")


# ═══════════════════════════════════════
# FIGURE 6: Exp 1d — Brand Mean Rank Heatmap (Pfrommer Fig 2b style)
# ═══════════════════════════════════════
def fig6_brand_rank_heatmap():
    df = pd.read_csv(os.path.join(RESULTS, "exp1d", "exp1d_raw.csv"))
    df = df[df.is_parse_fail.astype(str) != "True"].copy()
    df = df[pd.to_numeric(df.target_rank, errors="coerce").notna()].copy()
    df["target_rank"] = df["target_rank"].astype(int)

    # Get all brands and their mean rank per subcategory
    brands = df.groupby(["target_brand_name", "subcat", "target_is_real"])["target_rank"].mean().reset_index()

    # Get overall mean rank per brand
    brand_mean = df.groupby("target_brand_name")["target_rank"].mean().sort_values()
    brand_order = brand_mean.index.tolist()

    # Mark real brands
    real_brands = set(df[df.target_is_real == True]["target_brand_name"].unique())

    # Build matrix
    subcats_present = [s for s in SUBCAT_ORDER if s in df.subcat.unique()]
    mat = np.full((len(brand_order), len(subcats_present)), np.nan)
    for i, brand in enumerate(brand_order):
        for j, subcat in enumerate(subcats_present):
            sub = brands[(brands.target_brand_name == brand) & (brands.subcat == subcat)]
            if len(sub) > 0:
                mat[i, j] = sub.target_rank.values[0]

    fig, ax = plt.subplots(figsize=(5.5, 7))

    im = ax.imshow(mat, cmap=CMAP_RANK, aspect="auto", vmin=1, vmax=10)

    # Annotate
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                color = "white" if v < 6 else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5, color=color)

    # Y-axis: brand names with real brands in bold
    ax.set_yticks(range(len(brand_order)))
    ylabels = []
    for b in brand_order:
        if b in real_brands:
            ylabels.append(f"* {b}")
        else:
            ylabels.append(f"   {b}")
    ax.set_yticklabels(ylabels, fontsize=8)

    # Bold real brands on y-axis
    for i, b in enumerate(brand_order):
        if b in real_brands:
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_color("#d62728")

    ax.set_xticks(range(len(subcats_present)))
    ax.set_xticklabels([SUBCAT_LABELS[s] for s in subcats_present], rotation=0, fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Mean Rank (1=best, 10=worst)", fontsize=9)

    ax.set_title("(f) Exp 1d: Brand Mean Rank by Subcategory\n(* = Real/Incumbent brand)", fontsize=10, pad=30)

    fig.savefig(os.path.join(OUT, "fig1d_brand_rank_heatmap.pdf"))
    fig.savefig(os.path.join(OUT, "fig1d_brand_rank_heatmap.png"))
    plt.close(fig)
    print("✓ fig1d_brand_rank_heatmap")


# ═══════════════════════════════════════
# FIGURE 7: Combined Exp 1 Summary — The "Conditional Monopoly" narrative
# ═══════════════════════════════════════
def fig7_summary():
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    # Panel 1a: IAI bar
    ax1 = fig.add_subplot(gs[0, 0])
    models_short = ["GPT", "Claude", "Gemini"]
    iai_vals = [10.0, 10.0, 10.0]
    ax1.bar(range(3), iai_vals, color=["#e15759", "#4e79a7", "#59a14f"], edgecolor="white", width=0.6)
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="Fair baseline (IAI=1)")
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(models_short, fontsize=8)
    ax1.set_ylabel("IAI")
    ax1.set_ylim(0, 12)
    ax1.set_title("1a: Baseline\nIAI = 10.0", fontsize=9, fontweight="bold")
    ax1.legend(fontsize=6.5, loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel 1b: BOR bar
    ax2 = fig.add_subplot(gs[0, 1])
    subcats_short = ["Moist.", "BHA", "Sun.", "Clean."]
    bor_vals = [4.6, 1.7, 1.7, 3.9]
    bars2 = ax2.bar(range(4), bor_vals, color="#f28e2b", edgecolor="white", width=0.6)
    for bar, v in zip(bars2, bor_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax2.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, label="50% (no bias)")
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(subcats_short, fontsize=7.5)
    ax2.set_ylabel("BOR (%)")
    ax2.set_ylim(0, 60)
    ax2.set_title("1b: Swap\nBOR ≈ 2–5%", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=6.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel 1c: Step function (aggregate)
    ax3 = fig.add_subplot(gs[0, 2])
    df_c = pd.read_csv(os.path.join(RESULTS, "exp1c", "exp1c_raw.csv"))
    df_c = df_c[df_c.is_parse_fail.astype(str) != "True"]

    for dim in DIM_ORDER:
        levels = ["L0", "L1", "L2", "L3", "L4"]
        vals = []
        for lev in levels:
            sub = df_c[(df_c.dimension == dim) & (df_c.level_id == lev)]
            vals.append(sub.chose_fictional.mean() * 100 if len(sub) > 0 else np.nan)
        style = {"rating": "-o", "reviews": "-s", "price": "-D", "ingredient": "-^"}
        ax3.plot(range(5), vals, style[dim], label=DIM_LABELS[dim], markersize=4, linewidth=1.5)

    ax3.axhline(y=50, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(["L0", "L1", "L2", "L3", "L4"], fontsize=8)
    ax3.set_ylabel("Fictional Win (%)")
    ax3.set_ylim(-5, 105)
    ax3.set_title("1c: Gradient\nStep L0→L1", fontsize=9, fontweight="bold")
    ax3.legend(fontsize=6, loc="center right")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Panel 1d: η² bars
    ax4 = fig.add_subplot(gs[0, 3])
    df_d = pd.read_csv(os.path.join(RESULTS, "exp1d", "exp1d_raw.csv"))
    df_d = df_d[df_d.is_parse_fail.astype(str) != "True"].copy()
    df_d = df_d[pd.to_numeric(df_d.target_rank, errors="coerce").notna()].copy()
    df_d["target_rank"] = df_d["target_rank"].astype(int)

    def eta2(data, col):
        gm = data.target_rank.mean()
        groups = data.groupby(col)["target_rank"]
        ss_b = sum(len(g) * (g.mean() - gm)**2 for _, g in groups)
        ss_t = ((data.target_rank - gm)**2).sum()
        return ss_b / ss_t if ss_t > 0 else 0

    eta_vals = [eta2(df_d, "target_param_level"), eta2(df_d, "target_position"), eta2(df_d, "target_is_real")]
    bars4 = ax4.bar(range(3), eta_vals, color=["#4e79a7", "#f28e2b", "#e15759"], edgecolor="white", width=0.6)
    for bar, v in zip(bars4, eta_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(["Params", "Position", "Brand"], fontsize=8)
    ax4.set_ylabel("η²")
    ax4.set_ylim(0, 1.0)
    ax4.set_title("1d: ANOVA\nParams >> Brand", fontsize=9, fontweight="bold")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    fig.suptitle("Experiment 1 Summary: Conditional Monopoly", fontsize=13, fontweight="bold", y=1.06)

    fig.savefig(os.path.join(OUT, "fig1_summary.pdf"))
    fig.savefig(os.path.join(OUT, "fig1_summary.png"))
    plt.close(fig)
    print("✓ fig1_summary")


# ═══════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════
if __name__ == "__main__":
    print("Generating Exp 1 figures...")
    fig1_baseline()
    fig2_bor()
    fig3_gradient_heatmap()
    fig4_stepfunction()
    fig5_anova()
    fig6_brand_rank_heatmap()
    fig7_summary()
    print(f"\nAll figures saved to: {OUT}/")
    print("Files:", ", ".join(sorted(os.listdir(OUT))))
