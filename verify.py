"""
Audit every numeric claim made in the dashboard against the source data.

For each claim, prints:
  CLAIM:     what's displayed in the app / writeup
  COMPUTED:  what the actual data says
  STATUS:    OK / DRIFT / FAIL
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from data_prep import build_dataset

ROOT  = Path(__file__).parent
OUT   = ROOT / "outputs"
CACHE = ROOT / "cache"


def check(label, claim, computed, tol=0.005):
    """tol = absolute tolerance for floats."""
    if isinstance(claim, (int, float)) and isinstance(computed, (int, float)):
        ok = abs(claim - computed) <= tol
    else:
        ok = str(claim).strip() == str(computed).strip()
    flag = "✓ OK   " if ok else "✗ DRIFT"
    print(f"  [{flag}] {label}")
    print(f"           claim:    {claim}")
    print(f"           computed: {computed}")


def section(title):
    print(f"\n\n{'=' * 78}\n  {title}\n{'=' * 78}")


# -------------------------------------------------------------------
# 1. Sample size
# -------------------------------------------------------------------
section("1.  Sample size")
resp, long, merged = build_dataset()
raw = pd.read_excel("/Users/samenhossain/Downloads/MKTG 2120  Redbull Conjoint (Responses).xlsx")

print(f"  Raw rows in xlsx: {len(raw)}")
print(f"  Respondents after balance filter: {len(resp)}")
print(f"  Long-format rows: {len(long)}")
check("Respondents kept",    64,  len(resp))
check("Long-format rows",    768, len(long))
check("Profiles per resp",   12,  len(long) // len(resp))


# -------------------------------------------------------------------
# 2. Replicate the deck's pooled OLS regression
# -------------------------------------------------------------------
section("2.  Pooled OLS (replicates the deck's slide 5 numbers)")
X_cols = ["redbull", "monster", "price_350", "price_450", "eng_medium", "eng_high"]
X = sm.add_constant(long[X_cols])
y = long["rating"]
ols = sm.OLS(y, X).fit()
print(ols.summary().tables[1])

print()
print("  Deck values (slide 5):")
deck = {"const": 5.49462, "redbull": -0.21935, "monster": -0.60323,
        "price_350": -0.94194, "price_450": -2.25161,
        "eng_medium": 0.37097, "eng_high": 0.50269}
for name, deck_val in deck.items():
    key = "const" if name == "const" else name
    actual = ols.params["const" if name == "const" else name]
    check(f"deck coef {name}", deck_val, float(actual), tol=0.001)


# -------------------------------------------------------------------
# 3. Attribute importance percentages (Price 67% / Brand 18% / Eng 15%)
# -------------------------------------------------------------------
section("3.  Attribute importance (deck slide 6)")
b = ols.params
brand_range = max(0, b["redbull"], b["monster"]) - min(0, b["redbull"], b["monster"])
price_range = max(0, b["price_350"], b["price_450"]) - min(0, b["price_350"], b["price_450"])
eng_range   = max(0, b["eng_medium"], b["eng_high"]) - min(0, b["eng_medium"], b["eng_high"])
total = brand_range + price_range + eng_range
print(f"  Brand range: {brand_range:.3f}")
print(f"  Price range: {price_range:.3f}")
print(f"  Engagement range: {eng_range:.3f}")
check("Price importance %",  67.1, round(100 * price_range / total, 1), tol=0.2)
check("Brand importance %",  18.0, round(100 * brand_range / total, 1), tol=0.2)
check("Eng importance %",    15.0, round(100 * eng_range   / total, 1), tol=0.2)
# Sanity: "Price dominates 4:1" — verify
ratio = price_range / eng_range
print(f"\n  Claim: Price dominates engagement 4:1")
print(f"  Computed price/engagement utility-range ratio: {ratio:.2f}:1")


# -------------------------------------------------------------------
# 4. PyMC WTP (the $0.28-$0.63 credible interval claim)
# -------------------------------------------------------------------
section("4.  PyMC posterior — WTP for engagement")
wtp_csv = pd.read_csv(OUT / "pymc_wtp.csv")
print(wtp_csv.to_string(index=False))

high_row = wtp_csv[wtp_csv["contrast"].str.contains("High")].iloc[0]
print()
print(f"  Claim in synthesis: WTP for High vs Light = $0.28 – $0.63 (95% CrI)")
print(f"  Actual from csv:    ${high_row['ci_2.5%']:.2f} – ${high_row['ci_97.5%']:.2f}")
check("WTP High lower",  0.28, float(high_row["ci_2.5%"]),  tol=0.02)
check("WTP High upper",  0.63, float(high_row["ci_97.5%"]), tol=0.02)
check("WTP High mean",   0.45, float(high_row["mean_$"]),    tol=0.02)


# -------------------------------------------------------------------
# 5. EconML LinearDML moderator t-statistics
# -------------------------------------------------------------------
section("5.  EconML — moderator t-statistics for engagement effect")
linear_csv = pd.read_csv(OUT / "econml_linear_dml.csv")
print(linear_csv.to_string(index=False))

# Claim: rb_cans_per_week t ≈ -3.3
cans_row = linear_csv[linear_csv["term"].str.contains("rb_cans_per_week")].iloc[0]
sm_row   = linear_csv[linear_csv["term"].str.contains("sm_engagement")].iloc[0]
print()
print(f"  Claim: rb_cans_per_week moderator t = -3.3 (negative)")
print(f"  Actual t-stat: {cans_row['t']:.2f}")
check("rb_cans_per_week t", -3.3, float(cans_row["t"]), tol=0.1)
check("rb_cans_per_week is negative", "negative",
      "negative" if cans_row["estimate"] < 0 else "positive")

print(f"\n  Claim: sm_engagement moderator t ≈ 1.6 (positive)")
print(f"  Actual t-stat: {sm_row['t']:.2f}")
check("sm_engagement t (rough)", 1.6, float(sm_row["t"]), tol=0.3)
check("sm_engagement is positive", "positive",
      "positive" if sm_row["estimate"] > 0 else "negative")


# -------------------------------------------------------------------
# 6. ATE 0.485 (95% CI 0.284 - 0.685) — from EconML run
# -------------------------------------------------------------------
section("6.  EconML ATE")
print("  Claim (synthesis & EconML tab): ATE = +0.485, 95% CI (+0.284, +0.685)")
# Recompute from cached linear estimator
with open(CACHE / "econml_linear.pkl", "rb") as f:
    linear = pickle.load(f)
import analysis_econml
T, Y, W, X, _ = analysis_econml.prepare(merged)
ate_inf = linear.ate_inference(X=X)
lo, hi = ate_inf.conf_int_mean()
print(f"  Actual: {ate_inf.mean_point:.3f}  CI ({lo:.3f}, {hi:.3f})")
check("ATE point",   0.485, float(ate_inf.mean_point), tol=0.005)
check("ATE CI lo",   0.284, float(lo), tol=0.005)
check("ATE CI hi",   0.685, float(hi), tol=0.005)


# -------------------------------------------------------------------
# 7. CATE quintiles — "0.10 → 0.91, ~9× spread"
# -------------------------------------------------------------------
section("7.  EconML CATE quintile spread")
quintile_csv = pd.read_csv(OUT / "econml_cate_quintiles.csv", index_col=0)
print(quintile_csv.to_string())
q_means = quintile_csv["mean"]
q_low   = q_means.iloc[0]
q_high  = q_means.iloc[-1]
print()
print(f"  Claim: CATE quintiles span +0.10 → +0.91")
print(f"  Actual lowest quintile mean:  {q_low:.3f}")
print(f"  Actual highest quintile mean: {q_high:.3f}")
check("Q1 mean", 0.10, float(q_low),  tol=0.02)
check("Q5 mean", 0.91, float(q_high), tol=0.02)
# "9× spread" claim
ratio = q_high / q_low if q_low != 0 else float("inf")
print(f"\n  Claim: ~9× spread (Q5 / Q1 ratio)")
print(f"  Computed Q5/Q1 ratio: {ratio:.2f}×")
check("9x spread", 9.0, ratio, tol=1.0)


# -------------------------------------------------------------------
# 8. CausalNex direct edges to purchase_intent_rb
# -------------------------------------------------------------------
section("8.  CausalNex — direct edges to purchase_intent_rb")
cnex_csv = pd.read_csv(OUT / "causalnex_edges.csv")
intent_edges = cnex_csv[cnex_csv["target"] == "purchase_intent_rb"]
print("  All learned edges pointing into purchase_intent_rb:")
print(intent_edges.to_string(index=False) if not intent_edges.empty else "  (none)")
print()
print(f"  Claim: 'no direct edge sm_engagement → purchase_intent after conditioning on consumption'")
sm_to_intent = intent_edges[intent_edges["source"] == "sm_engagement"]
if sm_to_intent.empty:
    print(f"  Confirmed: no sm_engagement → purchase_intent_rb edge in learned DAG (threshold 0.25)")
    check("no sm→intent edge",  True, True)
else:
    print(f"  CONTRADICTED: edge exists with weight {sm_to_intent['weight'].iloc[0]}")
    check("no sm→intent edge", True, False)


# -------------------------------------------------------------------
# 9. CausalNex parents of avg_rb_rating
# -------------------------------------------------------------------
section("9.  CausalNex parents of avg_rb_rating (synthesis cites these)")
rating_edges = cnex_csv[cnex_csv["target"] == "avg_rb_rating"]
print(rating_edges.to_string(index=False) if not rating_edges.empty else "  (none)")
print()
print("  Claim (from earlier synthesis): rb_cans_per_week (+), ed_per_week (-)")
cans = rating_edges[rating_edges["source"] == "rb_cans_per_week"]
ed   = rating_edges[rating_edges["source"] == "ed_per_week"]
if not cans.empty:
    print(f"  rb_cans_per_week → avg_rb_rating  weight = {cans['weight'].iloc[0]:+.3f}")
if not ed.empty:
    print(f"  ed_per_week      → avg_rb_rating  weight = {ed['weight'].iloc[0]:+.3f}")


# -------------------------------------------------------------------
# 10. DoWhy point estimate (+0.328)
# -------------------------------------------------------------------
section("10. DoWhy causal estimate")
dowhy_csv = pd.read_csv(OUT / "dowhy_refutations.csv")
print(dowhy_csv.to_string(index=False))
orig = float(dowhy_csv["original_effect"].iloc[0])
placebo = float(dowhy_csv[dowhy_csv["test"].str.contains("Placebo")]["new_effect"].iloc[0])
print()
print("  Claim (DoWhy tab hero): +0.328")
check("DoWhy point estimate", 0.328, orig, tol=0.005)
print("  Claim: placebo collapses to ~0")
check("Placebo ≈ 0", 0.0, placebo, tol=0.05)


print("\n\n" + "=" * 78)
print("  Verification complete.")
print("=" * 78)
