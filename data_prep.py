"""
Data prep for Red Bull conjoint causal analysis.

Reshapes the Google Forms wide export into long-format (one row per respondent x profile)
and builds respondent-level covariates from the attitudinal block.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

XLSX_PATH = Path("/Users/samenhossain/Downloads/MKTG 2120  Redbull Conjoint (Responses).xlsx")

# The 12 conjoint profiles, in the order they appear as columns 18..29 (1-indexed).
PROFILES = [
    ("Celsius",  3.50, "High"),
    ("Monster",  2.50, "High"),
    ("Red Bull", 4.50, "Light"),
    ("Monster",  4.50, "Medium"),
    ("Celsius",  2.50, "Light"),
    ("Red Bull", 3.50, "Medium"),
    ("Celsius",  2.50, "Medium"),
    ("Red Bull", 4.50, "High"),
    ("Monster",  3.50, "High"),
    ("Celsius",  4.50, "High"),
    ("Red Bull", 2.50, "High"),
    ("Monster",  3.50, "Light"),
]

# Likert text -> int (favorability scale)
FAVORABILITY_MAP = {
    "1 = Very Unfavorable": 1,
    "2 = Unfavorable": 2,
    "3 = Somewhat Unfavorable": 3,
    "4 = Neutral": 4,
    "5 = Somewhat Favorable": 5,
    "6 = Favorable": 6,
    "7 = Very Favorable": 7,
}

CONSUMPTION_TREND_MAP = {
    "Significantly decreased": -1,
    "Stayed the same": 0,
    "Significantly increased": 1,
}

RANK_MAP = {"1st": 1, "2nd": 2, "3rd": 3}


def _parse_numeric(x):
    """Pull a number out of mixed-type free-text cells like '15%', '$120', '2-4', 'a lot...'."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace(",", "").replace("$", "").replace("%", "").strip()
    # range -> midpoint
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else np.nan


def load_raw() -> pd.DataFrame:
    return pd.read_excel(XLSX_PATH, sheet_name="Form Responses 1")


def build_respondent_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per respondent with cleaned attitudinal covariates."""
    cols = list(raw.columns)
    df = pd.DataFrame({
        "respondent_id": np.arange(len(raw)),
        "follows_rb":          raw[cols[1]].astype(float),
        "likes_energy_drink":  raw[cols[2]].astype(float),
        "rb_entertaining":     raw[cols[3]].astype(float),
        "rb_feels_relevant":   raw[cols[4]].astype(float),
        "rb_cans_per_week":    raw[cols[5]].apply(_parse_numeric),
        "consumption_trend":   raw[cols[6]].map(CONSUMPTION_TREND_MAP),
        "purchase_intent_rb":  raw[cols[7]].apply(_parse_numeric),  # 0-100
        "familiar_with_rb":    (raw[cols[8]].astype(str).str.strip().str.lower() == "yes").astype(int),
        "ed_per_week":         raw[cols[9]].apply(_parse_numeric),
        "price_importance":    raw[cols[10]].astype(float),
        "weekly_food_spend":   raw[cols[11]].apply(_parse_numeric),
        "rank_rb":             raw[cols[12]].map(RANK_MAP),
        "rank_monster":        raw[cols[13]].map(RANK_MAP),
        "rank_celsius":        raw[cols[14]].map(RANK_MAP),
        "rb_brand_image":      raw[cols[15]].astype(float),
        "rb_affordability":    raw[cols[16]].astype(float),
    })
    # Composite: social-media engagement w/ Red Bull (mean of 4 Likert items, 1-7)
    df["sm_engagement"] = df[["follows_rb", "likes_energy_drink",
                              "rb_entertaining", "rb_feels_relevant"]].mean(axis=1)
    # Normalize purchase intent to [0, 1]
    df["purchase_intent_rb"] = df["purchase_intent_rb"].clip(lower=0, upper=100) / 100
    return df


def build_long_conjoint(raw: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (respondent, profile). 12 profiles x N respondents."""
    cols = list(raw.columns)
    profile_cols = cols[17:29]  # columns 18..29 (0-indexed 17..28)
    assert len(profile_cols) == 12, f"expected 12 profile cols, got {len(profile_cols)}"

    rows = []
    for resp_idx, (_, row) in enumerate(raw.iterrows()):
        for profile_idx, (brand, price, engagement) in enumerate(PROFILES):
            rating_raw = row[profile_cols[profile_idx]]
            rating = FAVORABILITY_MAP.get(rating_raw, np.nan) if isinstance(rating_raw, str) else rating_raw
            rows.append({
                "respondent_id": resp_idx,
                "profile_id":    profile_idx,
                "brand":         brand,
                "price":         price,
                "engagement":    engagement,
                "rating":        rating,
            })
    long = pd.DataFrame(rows)

    # Drop respondents with any missing ratings (so the panel is balanced)
    bad = long.groupby("respondent_id")["rating"].apply(lambda s: s.isna().any())
    keep_ids = bad.index[~bad].tolist()
    long = long[long["respondent_id"].isin(keep_ids)].reset_index(drop=True)

    # Dummy encoding (Celsius / $2.50 / Light = baseline, matching the original analysis)
    long["redbull"]    = (long["brand"] == "Red Bull").astype(int)
    long["monster"]    = (long["brand"] == "Monster").astype(int)
    long["price_350"]  = (long["price"] == 3.50).astype(int)
    long["price_450"]  = (long["price"] == 4.50).astype(int)
    long["eng_medium"] = (long["engagement"] == "Medium").astype(int)
    long["eng_high"]   = (long["engagement"] == "High").astype(int)

    # Numeric engagement score (Light=0, Medium=1, High=2) for CATE
    long["eng_score"] = long["engagement"].map({"Light": 0, "Medium": 1, "High": 2})
    return long


def _build_from_xlsx():
    raw = load_raw()
    resp = build_respondent_table(raw)
    long = build_long_conjoint(raw)
    # Keep only respondents that survived the balance filter
    resp = resp[resp["respondent_id"].isin(long["respondent_id"].unique())].reset_index(drop=True)
    # Merge respondent covariates onto the long panel (handy for CATE)
    merged = long.merge(resp, on="respondent_id", how="left")
    return resp, long, merged


def build_dataset():
    out_dir = Path(__file__).parent
    csv_resp   = out_dir / "respondents.csv"
    csv_long   = out_dir / "conjoint_long.csv"
    csv_merged = out_dir / "conjoint_merged.csv"
    if csv_resp.exists() and csv_long.exists() and csv_merged.exists():
        return pd.read_csv(csv_resp), pd.read_csv(csv_long), pd.read_csv(csv_merged)
    return _build_from_xlsx()


if __name__ == "__main__":
    # Regenerate from the source XLSX. Delete the existing CSVs first if you
    # want this to take effect — build_dataset() prefers the cached CSVs.
    resp, long, merged = _build_from_xlsx()
    print(f"Respondents kept: {len(resp)}")
    print(f"Long-format rows: {len(long)} ({len(long) // 12} x 12)")
    print("\nRespondent covariate summary:")
    print(resp.describe().round(2))
    print("\nLong-format head:")
    print(long.head(8))
    out_dir = Path(__file__).parent
    resp.to_csv(out_dir / "respondents.csv", index=False)
    long.to_csv(out_dir / "conjoint_long.csv", index=False)
    merged.to_csv(out_dir / "conjoint_merged.csv", index=False)
    print(f"\nWrote: respondents.csv, conjoint_long.csv, conjoint_merged.csv")
