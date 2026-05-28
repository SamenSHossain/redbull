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

Interactive Shiny for Python dashboard for a conjoint-experiment causal analysis
of Red Bull purchase preference. Five analysis tabs: DoWhy (effect identification
+ refutation), EconML (linear DML + causal forest CATE), PyMC (hierarchical
Bayesian conjoint, cached posterior), CausalNex (NOTEARS structure learning),
and Synthesis.

## Local run

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/shiny run app.py --launch-browser
```

## Deploy

This Space is built from the bundled `Dockerfile`. The PyMC posterior under
`cache/pymc_idata.nc` and the EconML pickles under `cache/*.pkl` are shipped
with the image — they take ~17 s per fit so we precompute via `precompute.py`.
