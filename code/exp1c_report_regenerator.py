"""
Exp 1c Report Regenerator (M-4 fix)
====================================

Regenerates `results/exp1c/exp1c_report.txt` with:
  - Per-dimension BR-vs-level table (overall and per-model)
  - 50% breakthrough thresholds via linear interpolation
  - 4-parameter logistic sigmoid fits where well-conditioned

Reads `results/exp1c/exp1c_raw.csv`. No new API calls.

Run:    python exp1c_report_regenerator.py
Output: overwrites results/exp1c/exp1c_report.txt
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path

from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "results" / "exp1c" / "exp1c_raw.csv"
OUT_PATH = ROOT / "results" / "exp1c" / "exp1c_report.txt"

DIMS = ["rating", "price", "reviews", "ingredient"]
TARGET = 0.5  # 50% breakthrough threshold


def logistic4(x, a, b, k, x0):
    return a + (b - a) / (1 + np.exp(-k * (x - x0)))


def linear_threshold(levels: pd.DataFrame, target: float):
    """Find x where the BR curve crosses target via linear interpolation."""
    levels = levels.sort_values("x").reset_index(drop=True)
    for i in range(len(levels) - 1):
        a, b = levels.iloc[i], levels.iloc[i + 1]
        if (a["br"] <= target <= b["br"]) or (b["br"] <= target <= a["br"]):
            denom = b["br"] - a["br"]
            if abs(denom) < 1e-9:
                return float(a["x"])
            return float(a["x"] + (target - a["br"]) / denom * (b["x"] - a["x"]))
    return None


def sigmoid_threshold(levels: pd.DataFrame, target: float):
    """Sigmoid x_50%, or None when fit is ill-conditioned."""
    levels = levels.sort_values("x").reset_index(drop=True)
    x_data = levels["x"].values
    y_data = levels["br"].values
    try:
        p0 = [float(y_data.min()), float(y_data.max()), 5.0, float(x_data.mean())]
        popt, _ = curve_fit(logistic4, x_data, y_data, p0=p0, maxfev=10000)
        a, b, k, x0 = popt
        if not (a < target < b):
            return None, popt
        ratio = (b - a) / (target - a) - 1
        if ratio <= 0:
            return None, popt
        x = x0 - np.log(ratio) / k
        if abs(x) > 100 * max(abs(x_data).max(), 1.0):
            return None, popt
        return float(x), popt
    except Exception:
        return None, None


def main():
    df = pd.read_csv(CSV_PATH)
    n_total = len(df)
    v = df[df["is_parse_fail"] == False].copy()
    n_valid = len(v)

    L = []
    add = L.append
    add("EXP 1c Step-Function Gradient Report (regenerated)")
    add("=" * 70)
    add(f"CSV source:           {CSV_PATH.name}")
    add(f"Total rows:           {n_total}")
    add(f"Parse failures:       {n_total - n_valid}")
    add(f"Valid rows:           {n_valid}")
    add("")

    add("Overall BR vs gradient level (per dimension)")
    add("-" * 70)
    add(f"{'dim':<11} {'L0':>10} {'L1':>10} {'L2':>10} {'L3':>10} {'L4':>10}")
    overall_curves = {}
    for dim in DIMS:
        d = v[v["dimension"] == dim]
        by_lv = d.groupby(["level_id", "level_value"])["chose_fictional"].mean().reset_index()
        by_lv = by_lv.sort_values("level_value").reset_index(drop=True)
        overall_curves[dim] = by_lv.rename(columns={"level_value": "x", "chose_fictional": "br"})[["x", "br"]]
        cells = {row["level_id"]: f"{row['chose_fictional']*100:6.1f}%@{row['level_value']}" for _, row in by_lv.iterrows()}
        add(f"{dim:<11} {cells.get('L0',''):>10} {cells.get('L1',''):>10} {cells.get('L2',''):>10} {cells.get('L3',''):>10} {cells.get('L4',''):>10}")

    # The L0 / L1 ranges quoted by the paper:
    l0_brs = []
    l1_brs = []
    for dim in DIMS:
        levels = overall_curves[dim]
        # First two rows
        if len(levels) >= 2:
            l0_brs.append(levels.iloc[0]["br"])
            l1_brs.append(levels.iloc[1]["br"])
    add("")
    add(f"L0 BR range across dimensions: {min(l0_brs)*100:.1f}% – {max(l0_brs)*100:.1f}%")
    add(f"L1 BR range across dimensions: {min(l1_brs)*100:.1f}% – {max(l1_brs)*100:.1f}%")
    add("")

    add("50% breakthrough threshold (overall, per dimension)")
    add("-" * 70)
    add(f"{'dim':<11} {'linear':>12} {'sigmoid':>12} {'L1 value':>10}")
    headlines = {}
    for dim in DIMS:
        levels = overall_curves[dim]
        lin = linear_threshold(levels, TARGET)
        sig, _ = sigmoid_threshold(levels, TARGET)
        l1 = float(levels.iloc[1]["x"]) if len(levels) > 1 else None
        headlines[dim] = {"linear": lin, "sigmoid": sig, "l1": l1}
        lin_s = "n/a" if lin is None else f"{lin:.3f}"
        sig_s = "n/a" if sig is None else f"{sig:.3f}"
        l1_s = "n/a" if l1 is None else f"{l1:.3f}"
        add(f"{dim:<11} {lin_s:>12} {sig_s:>12} {l1_s:>10}")
    add("")
    add("Headline thresholds (linear interpolation):")
    add(f"  rating:  +{headlines['rating']['linear']:.3f} stars  (L1 = +{headlines['rating']['l1']:.2f})")
    add(f"  price:   {headlines['price']['linear']:.2f}% discount  (L1 = {headlines['price']['l1']:.0f}%)")
    add(f"  reviews: {headlines['reviews']['linear']:.2f}x  (L1 = {headlines['reviews']['l1']:.1f}x)")
    add("")

    add("Per-model BR vs gradient level (rating, price, reviews)")
    add("-" * 70)
    for dim in ["rating", "price", "reviews"]:
        add(f"\n--- {dim} ---")
        for model in sorted(v["model"].unique()):
            d = v[(v["dimension"] == dim) & (v["model"] == model)]
            if len(d) == 0:
                continue
            by_lv = d.groupby(["level_id", "level_value"])["chose_fictional"].mean().reset_index()
            by_lv = by_lv.sort_values("level_value").reset_index(drop=True)
            cells = {row["level_id"]: f"{row['chose_fictional']*100:5.1f}%" for _, row in by_lv.iterrows()}
            add(f"  {model:<14} L0:{cells.get('L0',''):>7} L1:{cells.get('L1',''):>7} L2:{cells.get('L2',''):>7} L3:{cells.get('L3',''):>7} L4:{cells.get('L4',''):>7}")
    add("")

    add("Per-model 50% threshold (linear interpolation)")
    add("-" * 70)
    add(f"{'dim':<11} {'claude':>10} {'gpt':>10} {'gemini':>10}")
    for dim in ["rating", "price", "reviews"]:
        row_vals = {}
        for model in sorted(v["model"].unique()):
            d = v[(v["dimension"] == dim) & (v["model"] == model)]
            if len(d) == 0:
                row_vals[model] = "n/a"
                continue
            by_lv = d.groupby("level_value")["chose_fictional"].mean().reset_index()
            by_lv.columns = ["x", "br"]
            t = linear_threshold(by_lv, TARGET)
            if t is None:
                max_br = by_lv["br"].max() * 100
                row_vals[model] = f"NA<{max_br:.0f}%"
            else:
                row_vals[model] = f"{t:.3f}"
        add(f"{dim:<11} {row_vals.get('claude-sonnet',''):>10} {row_vals.get('gpt-4o-mini',''):>10} {row_vals.get('gemini-flash',''):>10}")
    add("")

    add("Sigmoid fit parameters (overall)")
    add("-" * 70)
    add(f"{'dim':<11} {'a':>8} {'b':>8} {'k':>10} {'x0':>10}")
    for dim in DIMS:
        levels = overall_curves[dim]
        _, popt = sigmoid_threshold(levels, TARGET)
        if popt is None:
            add(f"{dim:<11} fit failed")
        else:
            a, b, k, x0 = popt
            add(f"{dim:<11} {a:>8.3f} {b:>8.3f} {k:>10.2f} {x0:>10.3f}")
    add("")

    add("=" * 70)

    OUT_PATH.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print("\nHeadline thresholds (linear interpolation, overall):")
    for dim in ["rating", "price", "reviews"]:
        h = headlines[dim]
        print(f"  {dim}: linear={h['linear']}  sigmoid={h['sigmoid']}  L1={h['l1']}")


if __name__ == "__main__":
    main()
