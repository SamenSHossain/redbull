# Red Bull Conjoint — Causal Inference Dashboard

Interactive Shiny for Python dashboard for a conjoint-experiment causal analysis of Red Bull purchase preference. Five analysis tabs: DoWhy (effect identification and refutation), EconML (linear DML and causal forest CATE), PyMC (hierarchical Bayesian conjoint, cached posterior), CausalNex (NOTEARS structure learning), and Synthesis.

## Local run

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/shiny run app.py --launch-browser
```

## Deploy

Deployed on [Vercel](https://vercel.com) — connect this repo, deploy on push
to `main`.

> **Heads-up:** the Shiny runtime relies on persistent websockets and the
> ~25 MB PyMC posterior in `cache/pymc_idata.nc`, neither of which fits the
> default Vercel serverless model. The Vercel deploy is a Next.js shell
> backed by Python serverless functions over pre-computed artifacts (no live
> Shiny process). The unmodified Shiny app remains the best way to run
> everything end-to-end locally via the **Local run** instructions above,
> and as a Docker container (see `Dockerfile`).

### Pre-computed artifacts

- `cache/pymc_idata.nc` — ArviZ NetCDF posterior (~25 MB). MCMC fit takes
  ~17 s, so we precompute via `precompute.py`.
- `cache/econml_linear.pkl`, `cache/econml_forest.pkl` — fitted EconML
  `LinearDML` + `CausalForestDML` estimators.

Regenerate with:

```bash
.venv/bin/python precompute.py
```
