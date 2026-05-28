"""
Run the full causal-inference analysis suite end-to-end.

  $ .venv/bin/python run_all.py

Produces:
  - respondents.csv, conjoint_long.csv, conjoint_merged.csv  (prepared data)
  - outputs/pymc_*                                            (Bayesian conjoint)
  - outputs/econml_*                                          (heterogeneous effects)
  - outputs/dowhy_*                                           (causal DAG + refutation)
  - outputs/causalnex_*                                       (learned network)
"""
from __future__ import annotations

import time
import traceback

import analysis_causalnex
import analysis_dowhy
import analysis_econml
import analysis_pymc
import data_prep


def section(title: str):
    line = "=" * 78
    print(f"\n\n{line}\n  {title}\n{line}")


def timed(name, fn):
    section(name)
    t0 = time.time()
    try:
        fn()
    except Exception:
        traceback.print_exc()
        print(f"!! {name} FAILED")
        return
    print(f"\n[{name} finished in {time.time() - t0:.1f}s]")


def main():
    timed("1/5  Data preparation", lambda: data_prep.build_dataset())
    timed("2/5  DoWhy: causal DAG + refutation tests", analysis_dowhy.run)
    timed("3/5  EconML: heterogeneous treatment effects (CATE)", analysis_econml.run)
    timed("4/5  CausalNex: Bayesian network structure learning", analysis_causalnex.run)
    timed("5/5  PyMC: hierarchical Bayesian conjoint", analysis_pymc.run)
    print("\n\nAll analyses complete. See outputs/ for tables and plots.")


if __name__ == "__main__":
    main()
