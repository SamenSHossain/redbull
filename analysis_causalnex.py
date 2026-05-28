"""
Bayesian-network structure learning with CausalNex (NOTEARS).

This is exploratory: we let the algorithm discover which respondent attitudes
plausibly drive purchase intent for Red Bull, without imposing a graph by hand.

Variables (one row per respondent):
  - sm_engagement       (composite Likert, 1-7)   how much they engage w/ RB on social
  - rb_brand_image      (1-5)                     perceived brand image
  - rb_affordability    (1-5)                     perceived affordability
  - price_importance    (1-5)                     importance of price in choice
  - rb_cans_per_week    (count)                   actual baseline consumption
  - ed_per_week         (count)                   any-brand consumption
  - weekly_food_spend   ($/week)                  proxy for disposable income
  - purchase_intent_rb  (0-1)                     stated probability of buying RB
  - avg_rb_rating       (1-7)                     average favorability across the 4 RB profiles

NOTEARS learns a DAG over these from continuous data; we then prune weak edges and
plot the learned structure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from causalnex.structure.notears import from_pandas

from data_prep import build_dataset

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


VARS = [
    "sm_engagement",
    "rb_brand_image",
    "rb_affordability",
    "price_importance",
    "rb_cans_per_week",
    "ed_per_week",
    "weekly_food_spend",
    "purchase_intent_rb",
    "avg_rb_rating",
]


def build_node_frame(resp: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """One row per respondent with the attitudinal + outcome variables."""
    rb_rows = long[long["brand"] == "Red Bull"]
    avg_rb_rating = rb_rows.groupby("respondent_id")["rating"].mean().rename("avg_rb_rating")
    df = resp.merge(avg_rb_rating, on="respondent_id", how="left")
    df = df[VARS].copy()
    # Standardize for NOTEARS (algorithm is scale-sensitive)
    for col in VARS:
        df[col] = df[col].fillna(df[col].median())
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
    return df


def learn_structure(df: pd.DataFrame, w_threshold: float = 0.25):
    sm = from_pandas(df, w_threshold=w_threshold, max_iter=200)
    return sm


def plot_network(sm, df: pd.DataFrame):
    G = nx.DiGraph()
    G.add_nodes_from(df.columns)
    for u, v, w in sm.edges(data="weight"):
        G.add_edge(u, v, weight=float(w))

    fig, ax = plt.subplots(figsize=(11, 7.5))
    pos = nx.spring_layout(G, seed=11, k=2.2)
    weights = np.array([abs(d["weight"]) for *_, d in G.edges(data=True)])
    edge_colors = ["#cc0000" if d["weight"] > 0 else "#1f77b4" for *_, d in G.edges(data=True)]
    nx.draw_networkx_nodes(G, pos, node_color="#fff3f3", edgecolors="#444",
                           linewidths=1.5, node_size=2400, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors,
        width=1 + 3 * (weights / (weights.max() + 1e-9)),
        arrowsize=15, connectionstyle="arc3,rad=0.08", ax=ax,
    )
    ax.set_title("Learned Bayesian network (NOTEARS) — red = positive, blue = negative")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "causalnex_network.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def edges_table(sm) -> pd.DataFrame:
    rows = [{"source": u, "target": v, "weight": float(w)}
            for u, v, w in sm.edges(data="weight")]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["abs_weight"] = df["weight"].abs()
    return df.sort_values("abs_weight", ascending=False).reset_index(drop=True)


def run():
    resp, long, _ = build_dataset()
    df = build_node_frame(resp, long)

    sm = learn_structure(df, w_threshold=0.25)
    edges = edges_table(sm)
    print("\n=== Learned edges (|weight| > 0.25 in standardized units) ===")
    print(edges.to_string(index=False) if not edges.empty else "(no edges above threshold)")

    plot_network(sm, df)
    edges.to_csv(OUT / "causalnex_edges.csv", index=False)
    print(f"\nWrote network plot and edge table to {OUT}/")

    # Direct parents of key outcomes
    parents_purchase = edges[edges["target"] == "purchase_intent_rb"]
    parents_rating   = edges[edges["target"] == "avg_rb_rating"]
    print("\n=== Direct parents of purchase_intent_rb ===")
    print(parents_purchase.to_string(index=False) if not parents_purchase.empty else "(none)")
    print("\n=== Direct parents of avg_rb_rating ===")
    print(parents_rating.to_string(index=False) if not parents_rating.empty else "(none)")
    return sm, edges


if __name__ == "__main__":
    run()
