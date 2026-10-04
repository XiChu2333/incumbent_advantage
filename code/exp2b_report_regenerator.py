"""
Exp 2b Report Regenerator (M-7 fix)
====================================

Regenerates `results/exp2b/exp2b_report.txt` with the full intensity
gradient and Stacking Paradox tables, using exp2a (moderate-only) and
exp2b (subtle/moderate/aggressive) raw CSVs combined.

Key sections:
  - Per-bias x intensity x model BR (Authority and Social Proof)
  - Model archetype verdict (Claude inverted-U vs GPT monotonic vs
    Gemini saturated)
  - Stacking Paradox: Authority+Social Proof BR vs each individually,
    per intensity and per model

Run:    python exp2b_report_regenerator.py
Output: overwrites results/exp2b/exp2b_report.txt
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
EXP2A_CSV = ROOT / "results" / "exp2a" / "exp2a_raw.csv"
EXP2B_CSV = ROOT / "results" / "exp2b" / "exp2b_raw.csv"
OUT_PATH = ROOT / "results" / "exp2b" / "exp2b_report.txt"


def main():
    df_a = pd.read_csv(EXP2A_CSV)
    df_b = pd.read_csv(EXP2B_CSV)

    n_a = len(df_a)
    n_b = len(df_b)
    a = df_a[df_a["is_parse_fail"] == False].copy()
    b = df_b[df_b["is_parse_fail"] == False].copy()
    a["stacking"] = False
    if "stacking" not in b.columns:
        b["stacking"] = False

    combined = pd.concat([a, b], ignore_index=True, sort=False)

    # === Single-bias intensity gradient (Authority, Social Proof) ===
    intensity_order = ["subtle", "moderate", "aggressive"]
    models = sorted(combined["model"].unique())

    L = []
    add = L.append

    add("EXP 2b Intensity Gradient & Stacking Paradox Report (regenerated)")
    add("=" * 70)
    add(f"Sources:")
    add(f"  exp2a_raw.csv (moderate, all 5 biases): {n_a} rows ({len(a)} valid)")
    add(f"  exp2b_raw.csv (subtle/aggressive + auth_sp): {n_b} rows ({len(b)} valid)")
    add(f"Combined valid records: {len(combined)}")
    add("")

    # --- Single-bias intensity gradient ---
    add("Intensity gradient — Authority and Social Proof BR (%) per model")
    add("-" * 70)
    for bias in ["authority", "social_proof"]:
        add(f"\n--- bias: {bias} ---")
        d = combined[combined["bias_type"] == bias]
        # Build the model x intensity matrix
        pivot = (
            d.groupby(["intensity", "model"])["chose_fictional"]
            .mean()
            .unstack()
            * 100
        )
        # Reorder rows
        pivot = pivot.reindex([i for i in intensity_order if i in pivot.index])
        n_pivot = (
            d.groupby(["intensity", "model"])["chose_fictional"]
            .count()
            .unstack()
            .reindex([i for i in intensity_order if i in pivot.index])
        )

        add(f"{'intensity':<12} {' '.join(f'{m:>14}' for m in pivot.columns)}")
        for intensity, row in pivot.iterrows():
            n_row = n_pivot.loc[intensity]
            cells = " ".join(f"{row[m]:>9.2f}% n={int(n_row[m]):>3}" for m in pivot.columns)
            add(f"{intensity:<12} {cells}")

    add("")

    # --- Model archetype verdict ---
    add("Model archetype verdict (Authority intensity gradient)")
    add("-" * 70)
    auth = combined[combined["bias_type"] == "authority"]
    auth_pivot = auth.groupby(["model", "intensity"])["chose_fictional"].mean() * 100
    for model in models:
        try:
            sub = auth_pivot.loc[(model, "subtle")]
            mod = auth_pivot.loc[(model, "moderate")]
            agg = auth_pivot.loc[(model, "aggressive")]
        except KeyError:
            add(f"  {model}: incomplete intensity coverage")
            continue
        if mod > sub and mod > agg:
            arche = "INVERTED-U (peaks at moderate)"
        elif sub <= mod <= agg and (agg - sub) > 5:
            arche = "MONOTONIC (rising with intensity)"
        elif min(mod, agg) > 90:
            arche = "SATURATED (≥90% from moderate onwards)"
        else:
            arche = "MIXED"
        add(f"  {model:<14} subtle={sub:5.1f}%  moderate={mod:5.1f}%  aggressive={agg:5.1f}%  -> {arche}")
    add("")

    # --- Stacking Paradox ---
    add("Stacking Paradox — Authority+Social Proof (auth_sp) vs single biases")
    add("-" * 70)
    add("Goal: verify the paradox where combining biases either drops below or")
    add("super-adds individual-bias BRs, depending on the model.")
    add("")

    for intensity in intensity_order:
        add(f"\n--- intensity: {intensity} ---")
        # Single bias BRs (Authority alone, Social Proof alone)
        # auth_sp combined
        try:
            auth_pv = (
                combined[(combined["bias_type"] == "authority") & (combined["intensity"] == intensity)]
                .groupby("model")["chose_fictional"]
                .mean()
                * 100
            )
            sp_pv = (
                combined[(combined["bias_type"] == "social_proof") & (combined["intensity"] == intensity)]
                .groupby("model")["chose_fictional"]
                .mean()
                * 100
            )
            cmb_pv = (
                combined[(combined["bias_type"] == "auth_sp") & (combined["intensity"] == intensity)]
                .groupby("model")["chose_fictional"]
                .mean()
                * 100
            )
        except Exception:
            continue

        add(f"  {'model':<14} {'Authority':>11} {'SocialProof':>13} {'Auth+SP':>10} {'verdict':>22}")
        for model in models:
            a_br = auth_pv.get(model, float("nan"))
            s_br = sp_pv.get(model, float("nan"))
            c_br = cmb_pv.get(model, float("nan"))
            verdict = ""
            if not pd.isna(a_br) and not pd.isna(c_br):
                if c_br < min(a_br, s_br):
                    verdict = "DROP below max"
                elif c_br > max(a_br, s_br):
                    verdict = "super-additive"
                else:
                    verdict = "between max/min"
            add(f"  {model:<14} {a_br:>10.2f}% {s_br:>12.2f}% {c_br:>9.2f}% {verdict:>22}")

    add("")
    add("Headline numbers (matches §4.2 of paper):")
    auth_mod = combined[(combined["bias_type"] == "authority") & (combined["intensity"] == "moderate")].groupby("model")["chose_fictional"].mean() * 100
    cmb_mod = combined[(combined["bias_type"] == "auth_sp") & (combined["intensity"] == "moderate")].groupby("model")["chose_fictional"].mean() * 100
    for m in models:
        a_v = auth_mod.get(m, float("nan"))
        c_v = cmb_mod.get(m, float("nan"))
        add(f"  {m}: Authority moderate = {a_v:.1f}%  Auth+SP moderate = {c_v:.1f}%")
    add("")
    add("=" * 70)

    OUT_PATH.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
