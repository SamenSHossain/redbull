"""
Red Bull Conjoint — Causal Inference Dashboard (Shiny for Python).

Run:
    .venv/bin/shiny run app.py --reload --launch-browser

Tabs:
    Overview          sample summary + respondent filter
    DoWhy             causal DAG + live re-estimation under filter
    EconML            interactive CATE calculator + ATE
    PyMC              cached Bayesian posterior + WTP
    CausalNex         NOTEARS structure learning with adjustable threshold
    Synthesis         takeaways
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

warnings.filterwarnings("ignore")

from data_prep import build_dataset
import analysis_dowhy
import analysis_econml
import analysis_causalnex

# ---------------------------------------------------------------------------
# Palette (matches www/redbull.css — jeffreyli.me-inspired light blue / navy)
# ---------------------------------------------------------------------------
RB_NAVY   = "#184A81"   # dark-blue-3 — primary text / axes
RB_RED    = "#184A81"   # primary accent (now blue, not red)
RB_YELLOW = "#BFD6F4"   # soft highlight
RB_NEG    = "#8A93A2"   # neutral grey for negative-direction edges
RB_GRID   = "#E7EBF2"
RB_MUTED  = "#3A6286"   # dark-blue-1
RB_BORDER = "#d1e1f6"

plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.edgecolor":     RB_BORDER,
    "axes.labelcolor":    RB_NAVY,
    "axes.titlecolor":    RB_NAVY,
    "axes.titlesize":     11,
    "axes.titleweight":   "600",
    "axes.labelsize":     10,
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         RB_GRID,
    "grid.linewidth":     0.6,
    "xtick.color":        RB_MUTED,
    "ytick.color":        RB_MUTED,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.frameon":     False,
    "legend.fontsize":    9,
    "font.family":        ["Plus Jakarta Sans", "Inter", "system-ui", "Helvetica Neue", "Arial", "sans-serif"],
})

# ---------------------------------------------------------------------------
# Load data + cached artifacts
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
RESP, LONG, MERGED = build_dataset()
CACHE = APP_DIR / "cache"

def _load_pickle(path):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

ECONML_LINEAR_CACHED = _load_pickle(CACHE / "econml_linear.pkl")
ECONML_FOREST_CACHED = _load_pickle(CACHE / "econml_forest.pkl")

PYMC_IDATA = None
PYMC_XCOLS = ["redbull", "monster", "price_350", "price_450", "eng_medium", "eng_high"]
PYMC_LOAD_NOTE = ""  # surfaced in the UI so we can debug failures without log access

_nc_path = CACHE / "pymc_idata.nc"
PYMC_LOAD_NOTE = (
    f"exists={_nc_path.exists()} "
    f"size={_nc_path.stat().st_size if _nc_path.exists() else 'N/A'} "
    f"path={_nc_path}"
)
if _nc_path.exists():
    try:
        import arviz as az
        PYMC_IDATA = az.from_netcdf(_nc_path)
        PYMC_LOAD_NOTE += " | loaded OK"
    except Exception as e:
        PYMC_LOAD_NOTE += f" | load FAILED: {type(e).__name__}: {e}"
if PYMC_IDATA is None and (CACHE / "pymc_idata.pkl").exists():
    PYMC_IDATA, PYMC_XCOLS = _load_pickle(CACHE / "pymc_idata.pkl")
    PYMC_LOAD_NOTE += " | fell back to pkl"
print(f"PYMC cache: {PYMC_LOAD_NOTE}", flush=True)

_ARCH_PATH = APP_DIR / "ARCHITECTURE.md"
ARCHITECTURE_MD = (
    _ARCH_PATH.read_text(encoding="utf-8")
    if _ARCH_PATH.exists()
    else "# About\n\n_ARCHITECTURE.md not found — see the repo for the full architecture doc._"
)


# ---------------------------------------------------------------------------
# Theme helpers
# ---------------------------------------------------------------------------
def hero_stat(eyebrow: str, number: str, unit: str | None = None, note: str | None = None):
    """Render an .rb-hero-stat callout (big colored number)."""
    inner = [ui.tags.span(number, class_="rb-number")]
    if unit:
        inner.append(" ")
        inner.append(ui.tags.span(unit, class_="rb-unit"))
    children = [
        ui.tags.div(eyebrow, class_="rb-eyebrow"),
        ui.tags.div(*inner),
    ]
    if note:
        children.append(ui.tags.div(note, class_="rb-note"))
    return ui.tags.div(*children, class_="rb-hero-stat")


def rb_pill(label: str, value: str):
    """Small navy pill with yellow accent value."""
    return ui.tags.span(
        f"{label} ",
        ui.tags.strong(value),
        class_="rb-pill",
    )


def tab_intro(eyebrow: str, method: str, body_html: str):
    """Per-tab preamble: small uppercase eyebrow on the left, monospace
    ``method · X`` pill on the right, body paragraph below. Mirrors the
    section-intro pattern in ARCHITECTURE.md."""
    return ui.tags.div(
        ui.tags.div(
            ui.tags.span(eyebrow, class_="rb-tab-intro-eyebrow"),
            ui.tags.span(f"method · {method}", class_="rb-method-pill"),
            class_="rb-tab-intro-head",
        ),
        ui.tags.div(ui.HTML(body_html), class_="rb-tab-intro-body"),
        class_="rb-tab-intro",
    )


def rb_synth_grid(rows):
    """rows: list of (headline_claim, causal_finding, recommendation, kind) tuples
    where kind ∈ {'keep', 'strat', 'reframe'} for the left border accent."""
    children = [
        ui.tags.div("Headline claim", class_="rb-synth-head"),
        ui.tags.div("Causal finding", class_="rb-synth-head"),
        ui.tags.div("Recommendation change", class_="rb-synth-head"),
    ]
    for claim, finding, rec, kind in rows:
        children.extend([
            ui.tags.div(ui.HTML(claim), class_=f"rb-synth-cell rb-claim rb-{kind}"),
            ui.tags.div(ui.HTML(finding), class_="rb-synth-cell"),
            ui.tags.div(ui.HTML(rec), class_="rb-synth-cell"),
        ])
    return ui.tags.div(*children, class_="rb-synth")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
SM_ENG_CHOICES    = ["Low (1-3)", "Medium (3-5)", "High (5-7)"]
PRICE_IMP_CHOICES = ["Low (1-2)", "Medium (3)", "High (4-5)"]

GOOGLE_FONTS_LINK = ui.tags.link(
    rel="stylesheet",
    href=("https://fonts.googleapis.com/css2"
          "?family=Plus+Jakarta+Sans:wght@400;500;600;700;800"
          "&family=JetBrains+Mono:wght@400;500;600"
          "&display=swap"),
)

RB_HEADER = ui.HTML(
    '<div class="rb-header">'
    '  <span class="rb-wordmark">Red Bull Conjoint</span>'
    '  <span class="rb-divider"></span>'
    '  <span class="rb-subtitle">Causal Inference Dashboard</span>'
    '</div>'
)

app_ui = ui.TagList(
    ui.head_content(
        ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
        ui.tags.link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin=""),
        GOOGLE_FONTS_LINK,
        ui.tags.title("Red Bull Conjoint — Causal Inference Dashboard"),
    ),
    ui.include_css(APP_DIR / "www" / "redbull.css"),
    RB_HEADER,
    ui.page_fluid(
    ui.navset_underline(
    ui.nav_panel(
        "Overview",
        tab_intro(
            "Sample summary",
            "filter · respondent-level",
            "Three sidebar filters can be intersected. The <strong>DoWhy</strong> "
            "and <strong>EconML</strong> tabs re-estimate live on whichever "
            "subgroup you pick — value boxes and histograms below show the "
            "shape of that subgroup.",
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Respondent filter"),
                ui.tags.p(
                    "Applies live to DoWhy & EconML.",
                    style="font-size:12px;color:var(--rb-fg-subtle);margin-top:-6px;",
                ),
                ui.input_checkbox_group(
                    "sm_eng_filter", "Social-media engagement w/ Red Bull",
                    SM_ENG_CHOICES, selected=SM_ENG_CHOICES,
                ),
                ui.input_checkbox_group(
                    "price_imp_filter", "Price sensitivity",
                    PRICE_IMP_CHOICES, selected=PRICE_IMP_CHOICES,
                ),
                ui.input_checkbox_group(
                    "familiar_filter", "Familiar with Red Bull",
                    ["Yes", "No"], selected=["Yes", "No"],
                ),
                width=320,
            ),
            ui.output_ui("sample_cards"),
            ui.layout_column_wrap(
                ui.card(ui.card_header("Social-media engagement"),
                        ui.output_plot("hist_sm_engagement")),
                ui.card(ui.card_header("Price importance"),
                        ui.output_plot("hist_price_imp")),
                ui.card(ui.card_header("Weekly Red Bull cans"),
                        ui.output_plot("hist_cans")),
                ui.card(ui.card_header("Self-rated Red Bull purchase intent"),
                        ui.output_plot("hist_intent")),
                width=1 / 2,
            ),
        ),
    ),

    ui.nav_panel(
        "About",
        tab_intro(
            "Architecture reference",
            "ARCHITECTURE.md",
            "What the app is, how the pieces fit, and the format of the data, "
            "layout, visuals, and persistence. Rendered from "
            "<code>ARCHITECTURE.md</code> at the repo root.",
        ),
        ui.tags.div(
            ui.markdown(ARCHITECTURE_MD),
            class_="rb-about",
        ),
    ),

    ui.nav_panel(
        "DoWhy",
        tab_intro(
            "Classical identification",
            "dowhy",
            "Assumes a DAG (rating ← engagement_high, sm_engagement, price, brand, "
            "rb_brand_image, price_importance), runs <code>CausalModel.identify_effect</code> "
            "+ linear-regression backdoor estimator, then three refutation tests: "
            "<strong>random common cause</strong> (~unchanged), <strong>placebo</strong> "
            "(~0), <strong>80% bootstrap subset</strong> (stable). The point of this "
            "tab is <em>robustness</em>, not magnitude.",
        ),
        ui.card(
            ui.card_header("Causal effect of High engagement on favorability"),
            ui.output_ui("dowhy_hero"),
        ),
        ui.layout_column_wrap(
            ui.card(ui.card_header("Refutation tests"),
                    ui.output_data_frame("dowhy_refutation_table")),
            ui.card(ui.card_header("Assumed causal DAG"),
                    ui.output_plot("dowhy_dag_plot")),
            width=1 / 2,
        ),
    ),

    ui.nav_panel(
        "EconML",
        tab_intro(
            "Heterogeneity",
            "econml",
            "<code>LinearDML</code> for ATE inference, <code>CausalForestDML</code> "
            "for CATE. The sidebar sliders describe a hypothetical respondent and the "
            "page predicts the CATE for <em>that</em> profile with a 90% CI. The CATE "
            "curve below traces the effect against any single modifier — useful for "
            "spotting saturation or floors.",
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Hypothetical respondent"),
                ui.tags.p(
                    "Predict the engagement effect for someone with these traits.",
                    style="font-size:12px;color:var(--rb-fg-subtle);margin-top:-6px;",
                ),
                ui.input_slider("calc_sm",     "sm_engagement (1-7)",     1, 7, 4, step=0.5),
                ui.input_slider("calc_priceimp", "price_importance (1-5)", 1, 5, 4, step=1),
                ui.input_slider("calc_image",  "rb_brand_image (1-5)",    1, 5, 4, step=1),
                ui.input_slider("calc_cans",   "rb_cans_per_week",        0, 15, 1, step=1),
                ui.input_slider("calc_intent", "purchase_intent_rb (0-1)", 0, 1, 0.4, step=0.05),
                ui.input_slider("calc_afford", "rb_affordability (1-5)",  1, 5, 3, step=1),
                ui.hr(),
                ui.input_select(
                    "cate_modifier", "Plot CATE as a function of",
                    {m: m for m in analysis_econml.MODIFIERS},
                    selected="sm_engagement",
                ),
                width=340,
            ),
            ui.output_ui("econml_calculator"),
            ui.layout_column_wrap(
                ui.card(ui.card_header("CATE curve for selected modifier"),
                        ui.output_plot("econml_cate_curve")),
                ui.output_ui("econml_ate_card"),
                width=1 / 2,
            ),
        ),
    ),

    ui.nav_panel(
        "PyMC",
        tab_intro(
            "Hierarchical Bayesian conjoint",
            "pymc",
            "Each respondent gets their own part-worth vector <code>beta_i</code> "
            "drawn from a population <code>mu_beta</code> with a diagonal "
            "<code>sigma_beta</code>. The posterior is cached "
            "(<code>precompute.py</code> writes <code>cache/pymc_idata.nc</code>) "
            "because each MCMC fit takes ~17 seconds.",
        ),
        ui.layout_column_wrap(
            ui.card(ui.card_header("Population part-worths (95% HDI)"),
                    ui.output_plot("pymc_forest")),
            ui.card(ui.card_header("Willingness-to-pay — 95% credible intervals"),
                    ui.output_data_frame("pymc_wtp_table")),
            width=1 / 2,
        ),
        ui.card(
            ui.card_header("Individual-level part-worths (each bar = one respondent)"),
            ui.output_plot("pymc_individual"),
        ),
    ),

    ui.nav_panel(
        "CausalNex",
        tab_intro(
            "Structure learning",
            "causalnex · NOTEARS",
            "Instead of assuming a DAG, learn one from the data. NOTEARS returns a "
            "weighted dependency network; the sidebar slider prunes edges by "
            "<code>|weight|</code>. Key finding: after conditioning on actual "
            "consumption, there's no direct edge "
            "<code>sm_engagement → purchase_intent_rb</code> — social engagement "
            "is a downstream correlate, not a direct cause of purchase.",
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("NOTEARS structure learning"),
                ui.input_slider(
                    "notears_threshold",
                    "Edge weight threshold (|w| ≥ ...)",
                    0.05, 0.6, 0.25, step=0.05,
                ),
                ui.tags.p(
                    "Higher threshold → sparser graph. Edges below the threshold are pruned "
                    "after NOTEARS converges.",
                    style="font-size:12px;color:var(--rb-fg-subtle);line-height:1.5;",
                ),
                width=320,
            ),
            ui.card(ui.card_header("Learned dependency network"),
                    ui.output_plot("cnex_network_plot")),
            ui.card(ui.card_header("Edge table — sorted by |weight|"),
                    ui.output_data_frame("cnex_edges_table")),
        ),
    ),

    ui.nav_panel(
        "Synthesis",
        tab_intro(
            "Recommendation deltas",
            "summary",
            "Four-row comparison of the marketing-team headline claim vs. what "
            "the causal analysis actually shows, plus the ordered list of the "
            "strongest causal levers. The four methods agree on direction but "
            "disagree on <em>who</em> the effect works for — that's what the "
            "deltas below capture.",
        ),
        ui.card(
            ui.card_header("How the recommendations change after causal analysis"),
            ui.output_ui("synth_grid"),
        ),
        ui.card(
            ui.card_header("Strongest causal levers (in order)"),
            ui.tags.ol(
                ui.tags.li(
                    ui.tags.strong("Price"),
                    " — still dominates 4:1. Subscription / bundling is the biggest dial.",
                ),
                ui.tags.li(
                    ui.tags.strong("Functional repositioning"),
                    " — addresses CausalNex's finding that broader ED-category consumption is ",
                    ui.tags.em("negatively"),
                    " correlated with Red Bull intent. Reclaim share from competitors directly.",
                ),
                ui.tags.li(
                    ui.tags.strong("Shoppable social, targeted"),
                    " — only for the already-engaged tail. Wastes impressions on the disengaged.",
                ),
                style="font-size:15px;line-height:1.65;color:var(--rb-fg);padding-left:20px;",
            ),
        ),
    ),

    id="navbar",
    ),
    ),
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def server(input, output, session):

    # ----- shared reactive: filtered respondents -----------------------------
    @reactive.calc
    def filtered_respondents() -> pd.DataFrame:
        sm = input.sm_eng_filter() or []
        pi = input.price_imp_filter() or []
        fa = input.familiar_filter() or []

        df = RESP.copy()
        sm_mask = pd.Series(False, index=df.index)
        if "Low (1-3)" in sm:    sm_mask |= df["sm_engagement"] < 3
        if "Medium (3-5)" in sm: sm_mask |= (df["sm_engagement"] >= 3) & (df["sm_engagement"] < 5)
        if "High (5-7)" in sm:   sm_mask |= df["sm_engagement"] >= 5
        df = df[sm_mask]

        pi_mask = pd.Series(False, index=df.index)
        if "Low (1-2)" in pi:    pi_mask |= df["price_importance"] <= 2
        if "Medium (3)" in pi:   pi_mask |= df["price_importance"] == 3
        if "High (4-5)" in pi:   pi_mask |= df["price_importance"] >= 4
        df = df[pi_mask]

        fa_mask = pd.Series(False, index=df.index)
        if "Yes" in fa: fa_mask |= df["familiar_with_rb"] == 1
        if "No" in fa:  fa_mask |= df["familiar_with_rb"] == 0
        df = df[fa_mask]
        return df

    @reactive.calc
    def filtered_merged() -> pd.DataFrame:
        ids = filtered_respondents()["respondent_id"].tolist()
        return MERGED[MERGED["respondent_id"].isin(ids)].copy()

    # ----- Overview ----------------------------------------------------------
    @render.ui
    def sample_cards():
        df = filtered_respondents()
        cards = [
            ("Respondents",          f"{len(df)}"),
            ("Avg sm_engagement",    f"{df['sm_engagement'].mean():.2f}"),
            ("Avg price_importance", f"{df['price_importance'].mean():.2f}"),
            ("Avg RB cans/week",     f"{df['rb_cans_per_week'].mean():.1f}"),
            ("Avg purchase intent",  f"{df['purchase_intent_rb'].mean()*100:.1f}%"),
        ]
        return ui.layout_column_wrap(
            *[ui.value_box(title=t, value=v) for t, v in cards],
            width=1 / 5,
        )

    def _hist(col):
        df = filtered_respondents()
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.hist(df[col].dropna(), bins=15, color=RB_RED, alpha=0.9, edgecolor="white")
        ax.set_xlabel(col)
        ax.set_ylabel("respondents")
        fig.tight_layout()
        return fig

    @render.plot
    def hist_sm_engagement(): return _hist("sm_engagement")
    @render.plot
    def hist_price_imp():     return _hist("price_importance")
    @render.plot
    def hist_cans():          return _hist("rb_cans_per_week")
    @render.plot
    def hist_intent():        return _hist("purchase_intent_rb")

    # ----- DoWhy -------------------------------------------------------------
    @reactive.calc
    def dowhy_results():
        from dowhy import CausalModel
        merged = filtered_merged()
        if merged["respondent_id"].nunique() < 5:
            return None

        df = analysis_dowhy.build_frame(merged)
        model = CausalModel(
            data=df, treatment="engagement_high", outcome="rating",
            graph=analysis_dowhy.GML_GRAPH,
        )
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")

        refute_rcc = model.refute_estimate(identified, estimate, method_name="random_common_cause", random_seed=42)
        refute_pl  = model.refute_estimate(identified, estimate, method_name="placebo_treatment_refuter",
                                           placebo_type="permute", num_simulations=30, random_seed=42)
        refute_sub = model.refute_estimate(identified, estimate, method_name="data_subset_refuter",
                                           subset_fraction=0.8, num_simulations=30, random_seed=42)
        refutations = pd.DataFrame([
            {"test": "Original (linear regression backdoor)",
             "new_effect": float(estimate.value)},
            {"test": "Add random common cause",
             "new_effect": float(np.atleast_1d(refute_rcc.new_effect).mean())},
            {"test": "Placebo (permuted) treatment",
             "new_effect": float(np.atleast_1d(refute_pl.new_effect).mean())},
            {"test": "80% data subset bootstrap",
             "new_effect": float(np.atleast_1d(refute_sub.new_effect).mean())},
        ]).round(4)
        return estimate.value, refutations

    @render.ui
    def dowhy_hero():
        res = dowhy_results()
        n_resp = len(filtered_respondents())
        n_obs  = len(filtered_merged())
        if res is None:
            return hero_stat(
                "Causal estimate",
                "—",
                note="Too few respondents in the current filter. Loosen the selection.",
            )
        value, _ = res
        sign = "+" if value >= 0 else ""
        return hero_stat(
            "Average causal effect — High vs non-High engagement",
            f"{sign}{value:.3f}",
            unit="favorability points (1–7 scale)",
            note=f"Estimated on {n_resp} respondents ({n_obs} respondent × profile rows).",
        )

    @render.data_frame
    def dowhy_refutation_table():
        res = dowhy_results()
        if res is None:
            return pd.DataFrame({"test": ["(filter too narrow)"], "new_effect": [np.nan]})
        _, ref = res
        return render.DataGrid(ref, height="320px")

    @render.plot
    def dowhy_dag_plot():
        G = nx.parse_gml(analysis_dowhy.GML_GRAPH)
        fig, ax = plt.subplots(figsize=(8, 5))
        pos = nx.spring_layout(G, seed=7, k=1.6)
        nx.draw_networkx_nodes(G, pos, node_color="#FBE9EE",
                               edgecolors=RB_RED, linewidths=1.5, node_size=2200, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_color=RB_NAVY, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color=RB_NAVY, arrowsize=14,
                               width=1.2, connectionstyle="arc3,rad=0.08", ax=ax)
        ax.set_axis_off()
        ax.grid(False)
        return fig

    # ----- EconML ------------------------------------------------------------
    @reactive.calc
    def econml_models():
        """Use cached models when filter is unchanged from full sample; otherwise refit."""
        n_full = len(RESP)
        n_filt = len(filtered_respondents())
        if n_filt == n_full and ECONML_FOREST_CACHED is not None:
            return ECONML_LINEAR_CACHED, ECONML_FOREST_CACHED
        merged = filtered_merged()
        if merged["respondent_id"].nunique() < 8:
            return None
        T, Y, W, X, _ = analysis_econml.prepare(merged)
        linear = analysis_econml.fit_linear_dml(T, Y, W, X)
        forest = analysis_econml.fit_causal_forest(T, Y, W, X)
        return linear, forest

    @render.ui
    def econml_calculator():
        res = econml_models()
        if res is None:
            return ui.card(
                ui.card_header("Predicted engagement effect"),
                hero_stat(
                    "Predicted CATE",
                    "—",
                    note="Too few respondents in current filter for CATE estimation.",
                ),
            )
        _, forest = res
        x = np.array([[
            input.calc_sm(), input.calc_priceimp(), input.calc_image(),
            input.calc_cans(), input.calc_intent(), input.calc_afford(),
        ]])
        cate = float(forest.effect(x)[0])
        lb, ub = forest.effect_interval(x, alpha=0.1)
        sign = "+" if cate >= 0 else ""
        return ui.card(
            ui.card_header("Predicted engagement effect for this profile"),
            hero_stat(
                "Conditional Average Treatment Effect (CATE)",
                f"{sign}{cate:.3f}",
                unit="favorability points",
                note=(f"90% CI: ({float(lb[0]):+.3f}, {float(ub[0]):+.3f}). "
                      "Interpretation: a respondent matching the sliders to the left "
                      "would rate a High-engagement profile this much higher than the same "
                      "product with Light engagement."),
            ),
        )

    @render.plot
    def econml_cate_curve():
        res = econml_models()
        if res is None:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "Too few respondents in current filter",
                    ha="center", va="center", color=RB_MUTED, fontsize=11)
            ax.set_axis_off(); ax.grid(False)
            return fig
        _, forest = res
        merged = filtered_merged()
        for col in analysis_econml.MODIFIERS:
            merged[col] = merged[col].fillna(merged[col].median())
        mod = input.cate_modifier()
        col = analysis_econml.MODIFIERS.index(mod)
        x = np.linspace(merged[mod].quantile(0.05), merged[mod].quantile(0.95), 50)
        X_grid = np.tile(merged[analysis_econml.MODIFIERS].median().to_numpy(), (len(x), 1))
        X_grid[:, col] = x
        cate = forest.effect(X_grid)
        lb, ub = forest.effect_interval(X_grid, alpha=0.1)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, cate, color=RB_RED, linewidth=2.2, label="CATE")
        ax.fill_between(x, lb, ub, alpha=0.18, color=RB_RED, label="90% CI")
        ax.axhline(0, color=RB_NAVY, linestyle="--", linewidth=0.9, alpha=0.5)
        ax.set_xlabel(mod); ax.set_ylabel("CATE on favorability")
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    @render.ui
    def econml_ate_card():
        res = econml_models()
        if res is None:
            return ui.card(
                ui.card_header("ATE on the filtered subgroup"),
                hero_stat("ATE", "—", note="Too few respondents in current filter."),
            )
        linear, _ = res
        merged = filtered_merged()
        for col in analysis_econml.MODIFIERS:
            merged[col] = merged[col].fillna(merged[col].median())
        X = merged[analysis_econml.MODIFIERS].to_numpy().astype(float)
        ate_inf = linear.ate_inference(X=X)
        lo, hi = ate_inf.conf_int_mean()
        sign = "+" if ate_inf.mean_point >= 0 else ""
        return ui.card(
            ui.card_header("ATE on the filtered subgroup"),
            hero_stat(
                "Average treatment effect",
                f"{sign}{ate_inf.mean_point:.3f}",
                unit="favorability points",
                note=(f"95% CI: ({lo:+.3f}, {hi:+.3f}). "
                      f"n = {len(merged)} respondent-profile rows "
                      f"({merged['respondent_id'].nunique()} respondents)."),
            ),
        )

    # ----- PyMC --------------------------------------------------------------
    @render.plot
    def pymc_forest():
        if PYMC_IDATA is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5,
                    "PyMC posterior not loaded.\n\n" + PYMC_LOAD_NOTE,
                    ha="center", va="center", color=RB_MUTED, fontsize=9, wrap=True)
            ax.set_axis_off(); ax.grid(False)
            return fig
        import arviz as az
        # custom forest with brand palette
        post = PYMC_IDATA.posterior["mu_beta"].values.reshape(-1, len(PYMC_XCOLS))
        means = post.mean(axis=0)
        lo = np.quantile(post, 0.025, axis=0)
        hi = np.quantile(post, 0.975, axis=0)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        y_pos = np.arange(len(PYMC_XCOLS))[::-1]
        ax.hlines(y_pos, lo, hi, color=RB_NAVY, linewidth=2.2)
        ax.scatter(means, y_pos, color=RB_RED, s=70, zorder=3, edgecolor="white", linewidth=1)
        ax.axvline(0, color=RB_NAVY, linestyle="--", linewidth=0.9, alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"mu_beta[{c}]" for c in PYMC_XCOLS])
        ax.set_xlabel("part-worth utility (95% credible interval)")
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        return fig

    @render.data_frame
    def pymc_wtp_table():
        if PYMC_IDATA is None:
            return pd.DataFrame({"contrast": ["(run precompute.py)"], "mean_$": [np.nan]})
        import analysis_pymc
        return render.DataGrid(analysis_pymc.wtp_engagement(PYMC_IDATA, PYMC_XCOLS),
                               height="300px")

    @render.plot
    def pymc_individual():
        if PYMC_IDATA is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "Run precompute.py first to cache PyMC posterior.",
                    ha="center", va="center", color=RB_MUTED)
            ax.set_axis_off(); ax.grid(False)
            return fig
        beta_i = PYMC_IDATA.posterior["beta_i"].mean(("chain", "draw")).values
        fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharey=True)
        titles = ["Red Bull (vs Celsius)", "Monster (vs Celsius)",
                  "Price $3.50 (vs $2.50)",  "Price $4.50 (vs $2.50)",
                  "Medium engagement (vs Light)", "High engagement (vs Light)"]
        for j, (ax, title) in enumerate(zip(axes.flat, titles)):
            ax.hist(beta_i[:, j], bins=18, alpha=0.9, edgecolor="white", color=RB_RED)
            ax.axvline(0, color=RB_NAVY, linestyle="--", linewidth=0.9, alpha=0.6)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("posterior-mean part-worth")
        fig.tight_layout()
        return fig

    # ----- CausalNex ---------------------------------------------------------
    @reactive.calc
    def cnex_results():
        df = analysis_causalnex.build_node_frame(RESP, LONG)
        sm = analysis_causalnex.learn_structure(df, w_threshold=input.notears_threshold())
        edges = analysis_causalnex.edges_table(sm)
        return df, sm, edges

    @render.plot
    def cnex_network_plot():
        df, sm, _ = cnex_results()
        G = nx.DiGraph()
        G.add_nodes_from(df.columns)
        for u, v, w in sm.edges(data="weight"):
            G.add_edge(u, v, weight=float(w))
        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=11, k=2.2)
        weights = np.array([abs(d["weight"]) for *_, d in G.edges(data=True)])
        edge_colors = [RB_RED if d["weight"] > 0 else RB_NEG for *_, d in G.edges(data=True)]
        nx.draw_networkx_nodes(G, pos, node_color="#FBE9EE", edgecolors=RB_NAVY,
                               linewidths=1.5, node_size=2400, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_color=RB_NAVY, ax=ax)
        if len(weights):
            nx.draw_networkx_edges(
                G, pos, edge_color=edge_colors,
                width=1 + 3 * (weights / (weights.max() + 1e-9)),
                arrowsize=14, connectionstyle="arc3,rad=0.08", ax=ax,
            )
        ax.set_axis_off(); ax.grid(False)
        return fig

    @render.data_frame
    def cnex_edges_table():
        _, _, edges = cnex_results()
        if edges.empty:
            return pd.DataFrame({"source": [], "target": [], "weight": []})
        return render.DataGrid(edges.round(3), height="400px")

    # ----- Synthesis ---------------------------------------------------------
    @render.ui
    def synth_grid():
        rows = [
            (
                "High engagement adds utility",
                "DoWhy: identified, refutation-robust.<br>"
                "PyMC: WTP 95% CrI <strong>$0.28 – $0.63</strong>",
                "Keep, but <strong>report the uncertainty band</strong> alongside the point estimate.",
                "keep",
            ),
            (
                "Engagement is the 3rd most important attribute (15%)",
                "True on average. But CATE quintiles span <strong>+0.10 → +0.91</strong> — a ~9× spread.",
                "<strong>Stratify the recommendation</strong> by respondent segment.",
                "strat",
            ),
            (
                "Frictionless conversion is universal",
                "EconML: largest lift for already-engaged users. "
                "Heavy users saturated (negative moderator, t = -3.3).",
                "Reframe as a <strong>retention / activation</strong> tool, not acquisition.",
                "reframe",
            ),
            (
                "Social media drives the funnel",
                "CausalNex: no direct edge <code>sm_engagement → purchase_intent</code> "
                "after conditioning on consumption.",
                "Treat social as <strong>top-of-funnel awareness</strong>, not a conversion engine.",
                "reframe",
            ),
        ]
        return rb_synth_grid(rows)


app = App(app_ui, server)
