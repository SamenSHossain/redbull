"""
Hierarchical Bayesian conjoint with PyMC.

Why this beats the pooled OLS in the deck:
1. Partial pooling -> stable individual-level part-worths (the pooled OLS treats every
   respondent as identical, which is the very assumption the deck's "engagement converts"
   question puts in doubt).
2. Posterior distributions -> credible intervals on every utility AND on WTP, instead of
   a single point estimate.
3. We get a posterior on the engagement -> dollar trade-off directly.
"""
from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from data_prep import build_dataset

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def fit_hierarchical_conjoint(long: pd.DataFrame, draws: int = 1500, tune: int = 1500):
    resp_codes, _ = pd.factorize(long["respondent_id"])
    n_resp = int(resp_codes.max() + 1)

    X_cols = ["redbull", "monster", "price_350", "price_450", "eng_medium", "eng_high"]
    X = long[X_cols].to_numpy()
    y = long["rating"].to_numpy().astype(float)
    k = X.shape[1]

    with pm.Model() as model:
        # population-level (group) means and dispersions for each part-worth
        mu_beta    = pm.Normal("mu_beta",    mu=0.0, sigma=2.0, shape=k)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1.5, shape=k)

        # individual-level part-worths (non-centered for sampling efficiency)
        z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=(n_resp, k))
        beta_i = pm.Deterministic("beta_i", mu_beta + z_beta * sigma_beta)

        # individual-level intercepts
        mu_alpha    = pm.Normal("mu_alpha", mu=5.0, sigma=1.5)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.5)
        alpha_i = pm.Normal("alpha_i", mu=mu_alpha, sigma=sigma_alpha, shape=n_resp)

        # observation noise
        sigma_y = pm.HalfNormal("sigma_y", sigma=1.5)

        mu = alpha_i[resp_codes] + (beta_i[resp_codes] * X).sum(axis=-1)
        pm.Normal("y", mu=mu, sigma=sigma_y, observed=y)

        idata = pm.sample(draws=draws, tune=tune, target_accept=0.92,
                          chains=4, cores=4, random_seed=[42, 43, 44, 45],
                          progressbar=False, return_inferencedata=True)

    idata.posterior.attrs["X_cols"] = X_cols
    return idata, X_cols


def summarize_population(idata, X_cols):
    summary = az.summary(idata, var_names=["mu_beta", "mu_alpha", "sigma_y"], round_to=3)
    # Rename mu_beta rows to attribute names
    new_index = []
    for ix in summary.index:
        if ix.startswith("mu_beta["):
            j = int(ix.split("[")[1].rstrip("]"))
            new_index.append(f"mu_beta[{X_cols[j]}]")
        else:
            new_index.append(ix)
    summary.index = new_index
    return summary


def wtp_engagement(idata, X_cols) -> pd.DataFrame:
    """
    Willingness-to-pay (in $) for moving from Light -> Medium and Light -> High engagement.

    We translate utility deltas into dollars using the price slope inferred from the
    $2.50 -> $4.50 contrast ($2 spread <-> mu_beta[price_450] utility drop).
    """
    posterior = idata.posterior
    # population-level coefficients (chain x draw)
    pb = posterior["mu_beta"].values.reshape(-1, len(X_cols))
    idx = {c: i for i, c in enumerate(X_cols)}

    # utility per dollar (negative; we make it positive via absolute value)
    util_per_dollar = pb[:, idx["price_450"]] / 2.0   # 2-dollar gap from $2.50 -> $4.50
    # Avoid division by ~zero
    util_per_dollar = np.where(np.abs(util_per_dollar) < 1e-3, np.nan, util_per_dollar)

    wtp_medium = pb[:, idx["eng_medium"]] / -util_per_dollar
    wtp_high   = pb[:, idx["eng_high"]]   / -util_per_dollar

    def summarize(name, samples):
        samples = samples[np.isfinite(samples)]
        return {
            "contrast":  name,
            "mean_$":    np.mean(samples).round(3),
            "median_$":  np.median(samples).round(3),
            "ci_2.5%":   np.quantile(samples, 0.025).round(3),
            "ci_97.5%":  np.quantile(samples, 0.975).round(3),
        }

    return pd.DataFrame([
        summarize("WTP for Medium vs Light engagement", wtp_medium),
        summarize("WTP for High vs Light engagement",   wtp_high),
    ])


def plot_individual_heterogeneity(idata, X_cols):
    beta_i = idata.posterior["beta_i"].mean(("chain", "draw")).values  # (n_resp, k)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    titles = ["Red Bull (vs Celsius)", "Monster (vs Celsius)",
              "Price $3.50 (vs $2.50)",  "Price $4.50 (vs $2.50)",
              "Medium engagement (vs Light)", "High engagement (vs Light)"]
    for j, (ax, title) in enumerate(zip(axes.flat, titles)):
        ax.hist(beta_i[:, j], bins=20, alpha=0.75, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.axvline(beta_i[:, j].mean(), color="black", linewidth=2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Posterior-mean part-worth")
    fig.suptitle("Individual-level part-worths (each bar = one respondent)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "pymc_individual_heterogeneity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_population_forest(idata, X_cols):
    fig, ax = plt.subplots(figsize=(9, 5))
    az.plot_forest(idata, var_names=["mu_beta"], combined=True, ax=ax,
                   hdi_prob=0.95)
    ax.set_yticklabels([f"mu_beta[{c}]" for c in reversed(X_cols)])
    ax.set_title("Population-level part-worths (95% HDI)")
    fig.tight_layout()
    fig.savefig(OUT / "pymc_population_forest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    _, long, _ = build_dataset()
    idata, X_cols = fit_hierarchical_conjoint(long)

    summary = summarize_population(idata, X_cols)
    wtp = wtp_engagement(idata, X_cols)

    print("\n=== Population-level posterior summary ===")
    print(summary)
    print("\n=== Engagement willingness-to-pay (dollars) ===")
    print(wtp.to_string(index=False))

    plot_individual_heterogeneity(idata, X_cols)
    plot_population_forest(idata, X_cols)

    summary.to_csv(OUT / "pymc_population_summary.csv")
    wtp.to_csv(OUT / "pymc_wtp.csv", index=False)
    try:
        idata.to_netcdf(OUT / "pymc_idata.nc")
    except Exception as e:
        print(f"(skipped saving idata: {e})")
    print(f"\nWrote outputs to {OUT}/")
    return idata, summary, wtp


if __name__ == "__main__":
    run()
