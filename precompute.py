"""
Precompute expensive artifacts so the Shiny app starts in seconds.

Run once before launching the app:
    .venv/bin/python precompute.py

Cached artifacts (in cache/):
  pymc_idata.nc          Bayesian conjoint posterior (~5 min to refit)
  econml_forest.pkl      Trained CausalForestDML for live CATE prediction
  econml_linear.pkl      Trained LinearDML for live coefficient display
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from data_prep import build_dataset
import analysis_econml
import analysis_pymc

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)


def precompute_econml():
    print("Training EconML estimators on the full sample...")
    _, _, merged = build_dataset()
    T, Y, W, X, _ = analysis_econml.prepare(merged)
    linear = analysis_econml.fit_linear_dml(T, Y, W, X)
    forest = analysis_econml.fit_causal_forest(T, Y, W, X)
    with open(CACHE / "econml_linear.pkl", "wb") as f:
        pickle.dump(linear, f)
    with open(CACHE / "econml_forest.pkl", "wb") as f:
        pickle.dump(forest, f)
    print(f"  saved econml_linear.pkl, econml_forest.pkl")


def precompute_pymc():
    print("Sampling PyMC hierarchical Bayesian conjoint (~5 min)...")
    _, long, _ = build_dataset()
    idata, X_cols = analysis_pymc.fit_hierarchical_conjoint(long)
    try:
        idata.to_netcdf(CACHE / "pymc_idata.nc")
        print(f"  saved pymc_idata.nc")
    except Exception as e:
        print(f"  netcdf save failed ({e}); falling back to pickle")
        with open(CACHE / "pymc_idata.pkl", "wb") as f:
            pickle.dump((idata, X_cols), f)
        print(f"  saved pymc_idata.pkl")


if __name__ == "__main__":
    precompute_econml()
    precompute_pymc()
    print("\nDone. Launch app with:  .venv/bin/shiny run app.py --reload")
