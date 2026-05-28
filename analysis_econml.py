"""
Heterogeneous treatment effects with EconML.

Question we answer that the deck's pooled regression CANNOT:
    "Does High social-media engagement convert better for heavy social-media users
     than for light ones?"

If High engagement converts uniformly across the Gen Z sample, the deck's recommendation
("Frictionless Conversion via shoppable social") works for the whole base. If the effect
is concentrated in already-engaged respondents, the recommendation needs a segmentation
tweak (target only people who actually consume your content) -- or a different strategy
for the disengaged tail.

Treatment T   = 1 if engagement == "High" else 0
Outcome   Y   = favorability rating (1-7)
Controls  W   = brand + price dummies (randomized within profile; included for variance)
Modifiers X   = respondent covariates (sm_engagement, price_importance, brand_image, ...)

Identification: the conjoint design randomizes engagement WITHIN profile, so
T is independent of X by construction. CATE is identified directly; DML is overkill
for confounding but useful for borrowing strength across many modifiers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML, LinearDML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV

from data_prep import build_dataset

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


MODIFIERS = [
    "sm_engagement",      # how engaged with Red Bull on social media (avg of 4 Likert items)
    "price_importance",   # 1-5 importance of price
    "rb_brand_image",     # 1-5 brand image rating
    "rb_cans_per_week",   # baseline RB consumption
    "purchase_intent_rb", # 0-1 stated intent to buy RB next time
    "rb_affordability",   # 1-5 perceived affordability
]


def prepare(merged: pd.DataFrame):
    df = merged.copy()
    # Fill rare missingness in modifiers with the column median (a couple of free-text cells)
    for col in MODIFIERS:
        df[col] = df[col].fillna(df[col].median())

    T = (df["engagement"] == "High").astype(int).to_numpy()
    Y = df["rating"].to_numpy().astype(float)
    W = df[["redbull", "monster", "price_350", "price_450"]].to_numpy().astype(float)
    X = df[MODIFIERS].to_numpy().astype(float)
    return T, Y, W, X, df


def fit_linear_dml(T, Y, W, X):
    """Linear CATE: ATE per unit change in each modifier."""
    est = LinearDML(
        model_y=GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
        model_t=GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
        random_state=42,
        discrete_treatment=True,
    )
    est.fit(Y=Y, T=T, X=X, W=W)
    return est


def fit_causal_forest(T, Y, W, X):
    est = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
        model_t=GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
        n_estimators=400,
        min_samples_leaf=10,
        random_state=42,
        discrete_treatment=True,
    )
    est.fit(Y=Y, T=T, X=X, W=W)
    return est


def summarize_linear(est, X) -> pd.DataFrame:
    """Coefficient table for the linear CATE model: how each modifier shifts the engagement effect."""
    coef_inf = est.coef__inference()
    point = np.atleast_1d(coef_inf.point_estimate).ravel()
    se    = np.atleast_1d(coef_inf.stderr).ravel()
    lo, hi = point - 1.96 * se, point + 1.96 * se

    rows = []
    # Intercept = ATE at X=0 (not meaningful with un-centered X, but reported for completeness)
    intc_inf = est.intercept__inference()
    rows.append({
        "term": "Intercept (ATE at modifier=0)",
        "estimate": float(np.atleast_1d(intc_inf.point_estimate).item()),
        "stderr":   float(np.atleast_1d(intc_inf.stderr).item()),
    })
    for name, p, s in zip(MODIFIERS, point, se):
        rows.append({"term": f"slope w.r.t. {name}", "estimate": p, "stderr": s})
    df = pd.DataFrame(rows)
    df["ci_lo"]  = df["estimate"] - 1.96 * df["stderr"]
    df["ci_hi"]  = df["estimate"] + 1.96 * df["stderr"]
    df["t"]      = df["estimate"] / df["stderr"]
    df["sig?"]   = np.where(np.abs(df["t"]) > 1.96, "*", "")
    return df.round(3)


def plot_cate_by_modifier(forest, df, modifier: str):
    """Plot CATE as a function of one modifier, holding others at their median."""
    x = np.linspace(df[modifier].quantile(0.05), df[modifier].quantile(0.95), 50)
    X_grid = np.tile(df[MODIFIERS].median().to_numpy(), (len(x), 1))
    col = MODIFIERS.index(modifier)
    X_grid[:, col] = x

    cate = forest.effect(X_grid)
    lb, ub = forest.effect_interval(X_grid, alpha=0.1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, cate, color="#cc0000", linewidth=2, label="CATE")
    ax.fill_between(x, lb, ub, alpha=0.2, color="#cc0000", label="90% CI")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(modifier)
    ax.set_ylabel("Effect of High vs. non-High engagement on favorability")
    ax.set_title(f"How the engagement effect varies with {modifier}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"econml_cate_{modifier}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def cate_decile_table(forest, X) -> pd.DataFrame:
    """Bucket respondent-profiles into deciles by predicted CATE."""
    cate = forest.effect(X)
    deciles = pd.qcut(cate, q=5, labels=[f"Q{i+1}" for i in range(5)])
    df = pd.DataFrame({"cate": cate, "bucket": deciles})
    summary = df.groupby("bucket", observed=False)["cate"].agg(["mean", "min", "max", "count"]).round(3)
    summary.index.name = "Quintile (low -> high CATE)"
    return summary


def run():
    _, _, merged = build_dataset()
    T, Y, W, X, df = prepare(merged)

    print(f"Sample: {len(df)} respondent-profile rows | T=1 share: {T.mean():.2%}")

    print("\n=== LinearDML: how respondent type moderates the engagement effect ===")
    linear = fit_linear_dml(T, Y, W, X)
    linear_tbl = summarize_linear(linear, X)
    print(linear_tbl.to_string(index=False))
    linear_tbl.to_csv(OUT / "econml_linear_dml.csv", index=False)

    print("\n=== Average treatment effect (ATE) of High engagement ===")
    ate_inf = linear.ate_inference(X=X)
    print(f"ATE: {ate_inf.mean_point:.3f}   (95% CI: "
          f"{ate_inf.conf_int_mean()[0]:.3f}, {ate_inf.conf_int_mean()[1]:.3f})")

    print("\n=== CausalForestDML: non-linear heterogeneity ===")
    forest = fit_causal_forest(T, Y, W, X)
    decile_tbl = cate_decile_table(forest, X)
    print(decile_tbl)
    decile_tbl.to_csv(OUT / "econml_cate_quintiles.csv")

    for mod in MODIFIERS:
        plot_cate_by_modifier(forest, df, mod)
    print(f"\nWrote per-modifier CATE plots to {OUT}/")

    return linear, forest


if __name__ == "__main__":
    run()
