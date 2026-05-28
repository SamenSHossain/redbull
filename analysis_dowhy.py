"""
Formalize the conjoint as a causal model with DoWhy.

Why this matters for the deck: DoWhy makes the causal assumptions EXPLICIT.
The deck's claim is "engagement converts to favorability." That's a causal claim.
With DoWhy we:
  1. Write down the causal graph (what we assume causes what).
  2. Let the library prove the effect is identified from the data.
  3. Estimate it with linear regression.
  4. Run refutation tests (placebo, random common cause, data subset) to see if the
     estimate falls apart under perturbation. If it does, the claim is fragile.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# DoWhy spews several deprecation warnings on import; silence them for class-deck output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from dowhy import CausalModel

from data_prep import build_dataset

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


# Causal DAG in GML format. The conjoint design randomizes brand/price/engagement
# WITHIN profile, so they have no parents from respondent-side covariates.
# Respondent attitudes affect the favorability rating directly.
GML_GRAPH = """
graph [
  directed 1
  node [ id "engagement_high" label "engagement_high" ]
  node [ id "brand_redbull"   label "brand_redbull"   ]
  node [ id "brand_monster"   label "brand_monster"   ]
  node [ id "price"           label "price"           ]
  node [ id "sm_engagement"   label "sm_engagement"   ]
  node [ id "price_importance" label "price_importance" ]
  node [ id "rb_brand_image"  label "rb_brand_image"  ]
  node [ id "rating"          label "rating"          ]
  edge [ source "engagement_high"   target "rating" ]
  edge [ source "brand_redbull"     target "rating" ]
  edge [ source "brand_monster"     target "rating" ]
  edge [ source "price"             target "rating" ]
  edge [ source "sm_engagement"     target "rating" ]
  edge [ source "price_importance"  target "rating" ]
  edge [ source "rb_brand_image"    target "rating" ]
]
""".strip()


def build_frame(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()
    for col in ["sm_engagement", "price_importance", "rb_brand_image"]:
        df[col] = df[col].fillna(df[col].median())
    df["engagement_high"] = (df["engagement"] == "High").astype(int)
    df["brand_redbull"]   = df["redbull"].astype(int)
    df["brand_monster"]   = df["monster"].astype(int)
    return df[[
        "rating", "engagement_high", "brand_redbull", "brand_monster", "price",
        "sm_engagement", "price_importance", "rb_brand_image",
    ]]


def run():
    _, _, merged = build_dataset()
    df = build_frame(merged)

    model = CausalModel(
        data=df,
        treatment="engagement_high",
        outcome="rating",
        graph=GML_GRAPH,
    )

    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.linear_regression",
        test_significance=True,
    )

    print("\n=== Identified causal estimand ===")
    print(identified)
    print("\n=== Estimated effect of switching engagement to High ===")
    print(f"Estimate (avg causal effect on 1-7 favorability scale): {estimate.value:.4f}")

    # Refutation tests
    refute_rcc = model.refute_estimate(
        identified, estimate, method_name="random_common_cause", random_seed=42,
    )
    refute_placebo = model.refute_estimate(
        identified, estimate, method_name="placebo_treatment_refuter",
        placebo_type="permute", num_simulations=50, random_seed=42,
    )
    refute_subset = model.refute_estimate(
        identified, estimate, method_name="data_subset_refuter",
        subset_fraction=0.8, num_simulations=50, random_seed=42,
    )

    refutations = pd.DataFrame([
        {"test": "Add random common cause",
         "new_effect": float(refute_rcc.new_effect),
         "p_value":    getattr(refute_rcc, "refutation_result", {}).get("p_value", np.nan)
                         if isinstance(getattr(refute_rcc, "refutation_result", None), dict)
                         else np.nan},
        {"test": "Placebo (permuted) treatment",
         "new_effect": float(np.atleast_1d(refute_placebo.new_effect).mean()),
         "p_value": getattr(refute_placebo, "refutation_result", {}).get("p_value", np.nan)
                    if isinstance(getattr(refute_placebo, "refutation_result", None), dict)
                    else np.nan},
        {"test": "80% data subset bootstrap",
         "new_effect": float(np.atleast_1d(refute_subset.new_effect).mean()),
         "p_value": getattr(refute_subset, "refutation_result", {}).get("p_value", np.nan)
                    if isinstance(getattr(refute_subset, "refutation_result", None), dict)
                    else np.nan},
    ])
    refutations["original_effect"] = float(estimate.value)
    print("\n=== Refutation tests ===")
    print(refutations.to_string(index=False))

    refutations.to_csv(OUT / "dowhy_refutations.csv", index=False)

    # Save graph as PNG
    try:
        import matplotlib.pyplot as plt
        G = nx.parse_gml(GML_GRAPH)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        pos = nx.spring_layout(G, seed=7, k=1.6)
        nx.draw_networkx_nodes(G, pos, node_color="#fff3f3",
                               edgecolors="#cc0000", linewidths=1.5, node_size=2200, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="#444", arrowsize=18,
                               connectionstyle="arc3,rad=0.08", ax=ax)
        ax.set_title("Assumed causal DAG (DoWhy)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / "dowhy_causal_dag.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"(skipped DAG plot: {e})")

    return estimate, refutations


if __name__ == "__main__":
    run()
