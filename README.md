---
title: Red Bull Conjoint
emoji: 🐂
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Red Bull Conjoint — Causal Inference Dashboard

Interactive Shiny for Python dashboard for a conjoint-experiment causal analysis of Red Bull purchase preference. Five analysis tabs: DoWhy (effect identification and refutation), EconML (linear DML and causal forest CATE), PyMC (hierarchical Bayesian conjoint, cached posterior), CausalNex (NOTEARS structure learning), and Synthesis.

Live dashboard: **https://huggingface.co/spaces/samenh/redbull**

## Local run

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/shiny run app.py --launch-browser
```

## Deploy

Primary deploy is Hugging Face Spaces — the bundled `Dockerfile` builds the
container with the cached artifacts baked in, and Spaces runs it as a
long-lived process with the websockets Shiny needs.

```bash
git push space main
```

(The `space` remote points at `huggingface.co/spaces/samenh/redbull`.)

### Pre-computed artifacts

Shipped inside the image so the container boots instantly:

- `cache/pymc_idata.nc` — ArviZ NetCDF posterior (~25 MB). MCMC fit takes
  ~17 s, so we precompute via `precompute.py`.
- `cache/econml_linear.pkl`, `cache/econml_forest.pkl` — fitted EconML
  `LinearDML` + `CausalForestDML` estimators.

Regenerate with:

```bash
.venv/bin/python precompute.py
```

### Optional: Next.js landing on Vercel

A Next.js 15 shell lives at the repo root (`app/`, `package.json`) and can be
deployed on Vercel as a static landing page that links into the Hugging Face
Space for the interactive views. Vercel's serverless model can't host the
Shiny runtime directly (no persistent websockets, no room for the 25 MB
posterior in a function bundle), so the dynamic tabs in the shell link out to
the HF Space rather than re-implementing them.
