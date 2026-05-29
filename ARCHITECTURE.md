# Red Bull Conjoint — Architecture & Format

A reference for what the dashboard is, how its pieces fit together, and what
shapes the data and UI take.

---

## 1. What the app is

A Shiny-for-Python web app that takes a conjoint-survey dataset of Red Bull
purchase preferences (64 respondents × 12 product profiles = 768 ratings) and
runs the same effect through **four different causal-inference toolkits**, so
you can see where they agree and where they disagree.

### The data

A standard conjoint experiment. Each respondent rated profiles that varied on
three attributes:

- **Brand** — Red Bull / Monster / Celsius
- **Price** — $2.50 / $3.50 / $4.50
- **Engagement** — Light / Medium / High social-media engagement with the brand

Plus per-respondent covariates: `sm_engagement`, `price_importance`,
`rb_brand_image`, `rb_cans_per_week`, `purchase_intent_rb`, `rb_affordability`,
`familiar_with_rb`.

The central causal question: **does High social-media engagement actually cause
higher favorability, or is it just correlated with it?**

### The six tabs

**1. Overview** — sample summary. Sidebar has three filters (sm_engagement
bucket, price-sensitivity bucket, familiar Y/N) that the user can intersect.
The DoWhy and EconML tabs re-estimate live on whatever subgroup you pick.

**2. DoWhy** (`analysis_dowhy.py`) — classical identification. Assumes a DAG
(rating ← engagement_high, sm_engagement, price, brand, rb_brand_image,
price_importance), runs `CausalModel.identify_effect` + linear-regression
backdoor estimator, then three refutation tests:

- Add a random common cause (should leave the estimate ~unchanged)
- Permute the treatment (placebo — should drive the estimate to ~0)
- Re-estimate on an 80% bootstrap subset (should be stable)

On the full sample the ATE is **+0.328 favorability points** on a 1–7 scale,
and the refutations all pass (random-cc ≈ 0.328, placebo ≈ −0.01, subset
≈ 0.317). The point of this tab is *robustness*, not magnitude.

**3. EconML** (`analysis_econml.py`) — heterogeneity. Fits a `LinearDML` for
ATE inference and a `CausalForestDML` for CATE. The sidebar has six sliders
describing a hypothetical respondent (sm_engagement, price_importance,
rb_brand_image, cans/week, purchase intent, affordability), and the page
predicts the CATE for *that profile* with a 90% CI. There's also a
CATE-vs-modifier curve so you can see, e.g., that the engagement effect is
near zero for people who already drink ~0 cans/week and jumps to +0.7 for
moderate users.

Models are cached as pickles (`cache/econml_linear.pkl`,
`cache/econml_forest.pkl`) and only refit when the sidebar filter is narrower
than full sample.

**4. PyMC** (`analysis_pymc.py`) — hierarchical Bayesian conjoint. Each
respondent gets their own part-worth vector `beta_i` drawn from a population
`mu_beta` with a diagonal `sigma_beta`. The plate notation is the standard
MNL-on-ratings setup. Produces:

- Population part-worths with 95% HDIs (forest plot)
- A willingness-to-pay table (engagement effect ÷ |price slope|, in dollars,
  with credible intervals — engagement_high ≈ $0.28–$0.63)
- A grid of per-respondent histograms showing how individual part-worths
  spread around the population mean (the "is this effect universal?" check)

Posterior is precomputed by `precompute.py` and shipped as
`cache/pymc_idata.nc` — each MCMC fit is ~17 s so doing it live would be
unusable.

**5. CausalNex** (`analysis_causalnex.py`) — structure learning. Runs NOTEARS
to *learn* a DAG from the data instead of assuming one, with a slider to
control the edge-weight pruning threshold. Renders the resulting dependency
network and an edge table sorted by `|weight|`. The interesting finding here
is the lack of a direct edge from `sm_engagement` to `purchase_intent_rb`
once you condition on actual consumption — i.e. social engagement isn't
directly causing purchase, it's a downstream correlate.

**6. Synthesis** — a four-row table of "headline marketing claim → what the
causal analysis actually shows → how the recommendation should change", plus
an ordered list of the strongest causal levers (price >> functional
repositioning > targeted social).

### How the pieces fit

- `app.py` wires the UI and reactive server. The two key reactives are
  `filtered_respondents()` and `filtered_merged()` — every DoWhy/EconML render
  depends on them, so changing a sidebar filter triggers re-estimation
  downstream.
- The analysis modules (`analysis_dowhy.py`, `analysis_econml.py`,
  `analysis_pymc.py`, `analysis_causalnex.py`) are each independent:
  `from analysis_X import …` and the module owns the prep + fit + helper
  functions. Easy to run any one of them standalone.
- `precompute.py` runs the slow stuff (PyMC MCMC, EconML fits) once and dumps
  pickles into `cache/` so the deployed app boots instantly.
- The whole thing is packaged for Hugging Face Spaces — the `Dockerfile` is
  the deploy target, the README frontmatter is HF Space metadata, and the
  cache artifacts ship inside the image so the container doesn't have to
  refit on cold start.

### The narrative the dashboard tells

The marketing-team intuition was "social engagement drives Red Bull purchase
intent" with a recommendation to invest in shoppable social. The four causal
methods agree the *direction* is right but disagree on *who it works for* —
the effect is concentrated in already-engaged moderate users, near zero for
heavy users (saturation) and near zero for disengaged users (no leverage).
So the Synthesis tab's recommendation is: stop treating social as a
top-of-funnel conversion engine, treat it as a retention/activation tool for
the engaged tail, and put the price/bundling lever (4× larger) ahead of it.

---

## 2. Format

### The input data — three CSV files, two shapes

**`respondents.csv` — wide, one row per respondent.** The Likert/numeric
covariates and demographics.

```
respondent_id  sm_engagement  price_importance  rb_brand_image  rb_cans_per_week  purchase_intent_rb  rb_affordability  familiar_with_rb  …
1              3.5            4                 4               1                 0.40                3                 1
2              2.0            3                 2               0                 0.05                2                 0
…              (64 rows)
```

**`conjoint_long.csv` — long, one row per (respondent × profile).** This is
the actual conjoint data — each respondent saw 12 product profiles and gave
each a 1–7 favorability rating.

```
respondent_id  profile_id  brand     price  engagement  rating
1              1           Red Bull  2.50   Light       6
1              2           Monster   3.50   Medium      4
1              3           Celsius   4.50   High        2
…              (768 rows = 64 × 12)
```

**`conjoint_merged.csv` — the join.** `respondents` left-joined onto
`conjoint_long` so every row has both the rating *and* the rater's covariates.
This is what every analysis module actually consumes — DoWhy, EconML, and
PyMC all build their design matrix from `MERGED` via `data_prep.py`'s
`build_dataset()`.

The brand / price / engagement attributes get one-hot encoded in
`data_prep.py`: `redbull`, `monster` (Celsius = baseline); `price_350`,
`price_450` ($2.50 = baseline); `eng_medium`, `eng_high` (Light = baseline).
That's the `PYMC_XCOLS` list hard-coded in `app.py`.

### The dashboard layout — Shiny's `navset_underline` over `page_fluid`

Single-page app with a tab bar at the top, no routing, no SPA framework — just
Shiny's reactive server pushing HTML/SVG into named output slots over a
websocket.

```
┌─────────────────────────────────────────────────────────────────┐
│ Red Bull Conjoint  |  Causal Inference Dashboard                │  ← RB_HEADER (HTML)
├─────────────────────────────────────────────────────────────────┤
│ [Overview] DoWhy  EconML  PyMC  CausalNex  Synthesis            │  ← navset_underline
├──────────────────┬──────────────────────────────────────────────┤
│                  │                                              │
│  Sidebar         │   Main content                               │
│  (filters/       │   (cards + plots + tables)                   │
│   sliders)       │                                              │
│                  │                                              │
└──────────────────┴──────────────────────────────────────────────┘
```

**Three tabs use `layout_sidebar` (left-sidebar + main):**

- *Overview*: sidebar = 3 checkbox groups → drives `filtered_respondents()`.
- *EconML*: sidebar = 6 numeric sliders (the "hypothetical respondent") + 1 select.
- *CausalNex*: sidebar = 1 slider (NOTEARS pruning threshold).

**Three tabs are full-width** — *DoWhy*, *PyMC*, *Synthesis*. They consume the
global filter from Overview but don't add their own sidebar.

**Inside each tab, content is composed of three reusable shapes:**

1. **`ui.card(card_header(...), output_X(...))`** — every plot, table, and
   dataframe is wrapped in a card with a small navy header label (the all-caps
   "REFUTATION TESTS", "ASSUMED CAUSAL DAG").
2. **`hero_stat(eyebrow, number, unit, note)`** — the big colored callout
   (e.g. "+0.328 favorability points (1–7 scale)"). Used for ATE, CATE, and
   the EconML calculator output. Three text levels: small eyebrow label →
   giant number + unit → muted note underneath.
3. **`ui.value_box(title, value)`** in a `layout_column_wrap(width=1/5)` —
   the five small stat cards across the top of Overview ("Respondents 64",
   "Avg sm_engagement 3.03", …).

**Plots within cards use `layout_column_wrap(width=1/2)`** — two columns on
wide screens, one column on narrow. That's what produces the 2×2 histogram
grid on Overview and the side-by-side refutation-table / DAG-plot on DoWhy.

### The visual identity — `www/redbull.css` + matplotlib rcParams

Not the Red Bull corporate red/yellow despite the name — it's been re-skinned
to a navy/light-blue palette inspired by jeffreyli.me. The palette constants:

```python
RB_NAVY   = "#184A81"   # primary text, axes, labels
RB_RED    = "#184A81"   # primary accent (yes, also navy — name stuck after rename)
RB_YELLOW = "#BFD6F4"   # soft blue highlight (not yellow)
RB_NEG    = "#8A93A2"   # neutral grey for negative-direction edges in CausalNex
RB_GRID   = "#E7EBF2"   # gridline
RB_MUTED  = "#3A6286"   # secondary text
RB_BORDER = "#d1e1f6"
```

The `plt.rcParams.update({...})` block in `app.py` means every chart shares
the same look without per-plot styling: top/right spines off, soft grid in
`RB_GRID`, axis labels in `RB_NAVY`, ticks in `RB_MUTED`, "Plus Jakarta Sans"
everywhere. That's why the screenshots look consistent across tabs —
Overview histograms, DoWhy DAG, EconML CATE curve, PyMC forest plot,
CausalNex network all inherit the same defaults.

Fonts are loaded from Google Fonts at runtime — `Plus Jakarta Sans` for
body, `JetBrains Mono` for code chips like the `sm_engagement → purchase_intent`
snippet in the Synthesis table.

### The output formats

- **Charts** — server-side matplotlib `Figure` objects returned from
  `@render.plot` decorated functions; Shiny serializes them to PNG and sends
  them down the websocket.
- **Tables** — `pandas.DataFrame` wrapped in `render.DataGrid(df, height=…)`;
  rendered as an interactive scrollable table in the browser.
- **Hero / stat callouts** — `@render.ui` returns a tree of `ui.tags.div`s
  with CSS classes (`.rb-hero-stat`, `.rb-number`, `.rb-unit`, `.rb-note`)
  styled in `redbull.css`. The "+0.328" is just an HTML span with a CSS
  font-size, not an image.
- **Synthesis grid** — custom CSS grid (`.rb-synth`) defined in
  `rb_synth_grid()` in `app.py`. Three columns × N rows, with a coloured
  left border on each claim cell whose class encodes the recommendation type
  (`rb-keep` / `rb-strat` / `rb-reframe`).

### The persistence format

- **EconML models** — pickled directly: `cache/econml_linear.pkl`,
  `cache/econml_forest.pkl`. The whole fitted estimator object.
- **PyMC posterior** — NetCDF via ArviZ: `cache/pymc_idata.nc`. Standard PPL
  persistence format — multidimensional posterior arrays with
  chain/draw/variable coordinates, ~20 MB. There's a pickle fallback in
  `app.py` for older caches.
- **No DoWhy or CausalNex cache** — they're cheap enough to refit live on
  each filter change.
