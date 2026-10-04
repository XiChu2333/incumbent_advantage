"""
Bias Surplus Value (BSV) Derivation (C-2 fix)
==============================================

Computes the commercial-equivalent Bias Surplus Value for each cognitive
bias measured in Exp 2a, by reverse-mapping each bias's Breakthrough Rate
(BR) into the equivalent product-quality dimension level (rating, price,
reviews) calibrated by Exp 1c's gradient curves.

Definition (linear interpolation, the paper's "BSV"):
    Given Authority moderate BR_target observed in Exp 2a, find x such
    that Exp 1c's BR-vs-level curve in dimension D equals BR_target.
    Then BSV_D(authority) = x.

Optionally also reports the 4-parameter logistic sigmoid fit when it
yields a stable solution.

Run:    python exp2_bsv_derivation.py
Output: results/exp2_bsv/bsv_table.csv
        results/exp2_bsv/bsv_report.txt

Inputs: results/exp1c/exp1c_raw.csv  (gradient curves)
        results/exp2a/exp2a_raw.csv  (per-bias BR)
"""

import warnings
import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
EXP1C_CSV = ROOT / "results" / "exp1c" / "exp1c_raw.csv"
EXP2A_CSV = ROOT / "results" / "exp2a" / "exp2a_raw.csv"
OUT_DIR = ROOT / "results" / "exp2_bsv"
OUT_DIR.mkdir(exist_ok=True, parents=True)
TABLE_PATH = OUT_DIR / "bsv_table.csv"
REPORT_PATH = OUT_DIR / "bsv_report.txt"


def logistic4(x, a, b, k, x0):
    return a + (b - a) / (1 + np.exp(-k * (x - x0)))


def linear_bsv(levels: pd.DataFrame, target: float) -> Optional[float]:
    """Reverse-interpolate level value where BR == target.
    `levels` must have columns level_value (float) and br (in [0,1]).
    Returns None if target is not bracketed."""
    levels = levels.sort_values("level_value").reset_index(drop=True)
    for i in range(len(levels) - 1):
        a, b = levels.iloc[i], levels.iloc[i + 1]
        if (a["br"] <= target <= b["br"]) or (b["br"] <= target <= a["br"]):
            denom = b["br"] - a["br"]
            if abs(denom) < 1e-9:
                return float(a["level_value"])
            x = a["level_value"] + (target - a["br"]) / denom * (b["level_value"] - a["level_value"])
            return float(x)
    return None


def sigmoid_bsv(levels: pd.DataFrame, target: float) -> Optional[float]:
    if not HAS_SCIPY:
        return None
    levels = levels.sort_values("level_value").reset_index(drop=True)
    x_data = levels["level_value"].values
    y_data = levels["br"].values
    try:
        p0 = [float(y_data.min()), float(y_data.max()), 5.0, float(x_data.mean())]
        popt, _ = curve_fit(logistic4, x_data, y_data, p0=p0, maxfev=10000)
        a, b, k, x0 = popt
        if not (a < target < b):
            return None
        ratio = (b - a) / (target - a) - 1
        if ratio <= 0:
            return None
        x = x0 - math.log(ratio) / k
        # Reject implausibly extreme extrapolations
        if abs(x) > 100 * max(abs(x_data).max(), 1.0):
            return None
        return float(x)
    except Exception:
        return None


def bsv_for_target(gradient: pd.DataFrame, target_br: float, model: Optional[str] = None) -> dict:
    """Return BSV (linear and sigmoid) for each dimension at the given target BR."""
    out = {}
    for dim in ["rating", "price", "reviews"]:
        d = gradient[gradient["dimension"] == dim].copy()
        if model is not None:
            d = d[d["model"] == model]
        if len(d) == 0:
            out[dim] = {"linear": None, "sigmoid": None, "br_max": None, "br_min": None, "n": 0}
            continue
        levels = (
            d.groupby(["level_id", "level_value"])["chose_fictional"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "br"})
        )
        out[dim] = {
            "linear": linear_bsv(levels[["level_value", "br"]], target_br),
            "sigmoid": sigmoid_bsv(levels[["level_value", "br"]], target_br),
            "br_max": float(levels["br"].max()),
            "br_min": float(levels["br"].min()),
            "n": int(levels["count"].sum()),
        }
    return out


def main():
    g = pd.read_csv(EXP1C_CSV)
    g = g[g["is_parse_fail"] == False].copy()

    a = pd.read_csv(EXP2A_CSV)
    a = a[a["is_parse_fail"] == False].copy()

    # Authority moderate BR (overall + per-model)
    auth = a[(a["bias_type"].str.lower() == "authority") & (a["intensity"] == "moderate")]
    overall_br = float(auth["chose_fictional"].mean())
    per_model_br = {m: float(auth[auth["model"] == m]["chose_fictional"].mean())
                    for m in sorted(auth["model"].unique())}

    # All biases moderate BR (for the full BSV table)
    bias_br = (
        a[a["intensity"] == "moderate"]
        .groupby("bias_type")["chose_fictional"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "br_overall"})
        .reset_index()
    )

    # === Compute BSV for Authority (overall) — the headline numbers ===
    auth_bsv_overall = bsv_for_target(g, overall_br)

    # === Compute BSV for Authority per model ===
    auth_bsv_per_model = {m: bsv_for_target(g, br, model=m) for m, br in per_model_br.items()}

    # === Compute BSV for each bias type at moderate intensity (overall) ===
    full_table_rows = []
    for _, r in bias_br.iterrows():
        target = float(r["br_overall"])
        result = bsv_for_target(g, target)
        full_table_rows.append({
            "bias_type": r["bias_type"],
            "br_overall_pct": round(target * 100, 2),
            "n_obs": int(r["count"]),
            "bsv_rating_linear": result["rating"]["linear"],
            "bsv_price_pct_linear": result["price"]["linear"],
            "bsv_reviews_x_linear": result["reviews"]["linear"],
            "bsv_rating_sigmoid": result["rating"]["sigmoid"],
            "bsv_price_pct_sigmoid": result["price"]["sigmoid"],
            "bsv_reviews_x_sigmoid": result["reviews"]["sigmoid"],
        })
    table = pd.DataFrame(full_table_rows)
    table.to_csv(TABLE_PATH, index=False)

    # === Build report ===
    L = []
    add = L.append

    add("Bias Surplus Value (BSV) Derivation Report")
    add("=" * 70)
    add("Derivation method:")
    add("  Given a target BR observed in Exp 2a (cognitive-bias condition),")
    add("  compute the equivalent quality-dimension level x in Exp 1c that")
    add("  yields the same BR. Linear interpolation between bracketing")
    add("  levels is the primary method; 4-param logistic is reported when")
    add("  the fit is well-conditioned.")
    add("")
    add(f"Inputs:")
    add(f"  Exp 1c gradient CSV: {EXP1C_CSV.name}  ({len(g)} valid rows)")
    add(f"  Exp 2a bias CSV:     {EXP2A_CSV.name}  ({len(a)} valid rows)")
    add("")

    add("Authority moderate BR (Exp 2a) — target for BSV computation")
    add("-" * 70)
    add(f"  Overall:        {overall_br * 100:.2f}%  (n = {len(auth)})")
    for m, br in per_model_br.items():
        add(f"  {m:<14}  {br * 100:.2f}%  (n = {len(auth[auth['model']==m])})")
    add("")

    add("Authority BSV (overall, headline)")
    add("-" * 70)
    add(f"{'dimension':<10} {'linear':>12} {'sigmoid':>12} {'note':<24}")
    for dim, label, fmt in [
        ("rating", "+__ rating points", "+{:.3f}"),
        ("price", "__% price discount", "{:.2f}%"),
        ("reviews", "__× reviews", "{:.2f}x"),
    ]:
        r = auth_bsv_overall[dim]
        lin = fmt.format(r["linear"]) if r["linear"] is not None else "n/a"
        sig = fmt.format(r["sigmoid"]) if r["sigmoid"] is not None else "n/a"
        # Diagnostic: does the rating curve span the target?
        diag = ""
        if r["br_max"] is not None and overall_br > r["br_max"]:
            diag = f"target {overall_br*100:.1f}% > max {r['br_max']*100:.1f}%"
        add(f"{dim:<10} {lin:>12} {sig:>12}  {diag}")
    add("")

    add("Authority BSV (per model)")
    add("-" * 70)
    add(f"{'model':<14} {'rating':>14} {'price':>14} {'reviews':>14}")
    for m in sorted(per_model_br.keys()):
        r = auth_bsv_per_model[m]
        rat = f"{r['rating']['linear']:.3f}" if r['rating']['linear'] is not None else "n/a"
        pri = f"{r['price']['linear']:.2f}%" if r['price']['linear'] is not None else "n/a"
        rev = f"{r['reviews']['linear']:.2f}x" if r['reviews']['linear'] is not None else "n/a"
        # Annotate undefined cases
        notes = []
        if r['rating']['linear'] is None:
            notes.append(f"rating max BR={r['rating']['br_max']*100:.1f}% < target {per_model_br[m]*100:.1f}%")
        if r['price']['linear'] is None:
            notes.append(f"price max BR={r['price']['br_max']*100:.1f}% < target")
        if r['reviews']['linear'] is None:
            notes.append("reviews undefined")
        note_str = ("  // " + "; ".join(notes)) if notes else ""
        add(f"{m:<14} {rat:>14} {pri:>14} {rev:>14}{note_str}")
    add("")

    add("Full BSV table (every bias, moderate intensity, overall)")
    add("-" * 70)
    add(f"{'bias':<15} {'BR%':>6} {'n':>5} {'rating':>10} {'price':>10} {'reviews':>10}")
    for _, row in table.iterrows():
        rat = f"+{row['bsv_rating_linear']:.3f}" if pd.notna(row['bsv_rating_linear']) else "n/a"
        pri = f"{row['bsv_price_pct_linear']:.2f}%" if pd.notna(row['bsv_price_pct_linear']) else "n/a"
        rev = f"{row['bsv_reviews_x_linear']:.2f}x" if pd.notna(row['bsv_reviews_x_linear']) else "n/a"
        add(f"{row['bias_type']:<15} {row['br_overall_pct']:>6.1f} {int(row['n_obs']):>5} "
            f"{rat:>10} {pri:>10} {rev:>10}")
    add("")

    add("Notes")
    add("-" * 70)
    add("- Linear interpolation is conservative: it is well-defined whenever")
    add("  the target BR lies between two adjacent gradient levels.")
    add("- Sigmoid (4-param logistic) often fails to converge on dimensions")
    add("  whose response saturates within the experimental range; we report")
    add("  it only when it succeeds.")
    add("- Per-model BSVs are undefined when the model's gradient curve does")
    add("  not span the per-model Authority BR target (e.g. Claude's rating")
    add("  curve saturates at ~42%, but Claude's Authority BR is 55%).")
    add("- Marketing biases (Anchoring, Scarcity, Loss Aversion) hover near")
    add("  the L0 baseline (BR ≈ 4%), so their BSV ≈ 0 in every dimension.")
    add("")
    add("=" * 70)

    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {TABLE_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print()
    print("=== Headline (Authority overall BSV) ===")
    for dim in ["rating", "price", "reviews"]:
        v = auth_bsv_overall[dim]["linear"]
        print(f"  {dim}: {v}")


if __name__ == "__main__":
    main()
