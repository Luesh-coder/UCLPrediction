"""Model-ready tables written to data/model/: no missing values in features or targets, no
duplicate rows, and only the columns a model needs (identifiers kept for splitting/inspection).

Missing values here are structural, not random (e.g. no first leg outside knockout ties, no UCL
history for most league clubs, no domestic data for leagues outside the 12 covered). They are
filled with neutral values and paired with *_missing flags, so a model can still tell a filled
value from a real one. Neutral values are long-run averages from the pre-2010 seasons.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config

NEUTRAL_PPG = 1.37  # points per game of an average team
NEUTRAL_GOALS_PG = 1.34  # goals scored (and conceded) per game of an average team
NEUTRAL_REST_DAYS = 7.0
NEUTRAL_RANK_PCT = 0.5  # middle of the table
WEAK_ASSOC_PPG = 0.75  # association without group/league-phase matches in 5 seasons (~10th percentile)

# Categorical text columns are one-hot encoded with fixed category lists, so train and predict
# files always have the same columns. Domestic matches have is_ucl = 0 and every phase_* = 0;
# UCL matches have every comp_* = 0.
UCL_PHASES = ["qualifying", "group", "league", "knockout"]
PHASE_COLUMNS = [f"phase_{p}" for p in UCL_PHASES]
COMPETITION_COLUMNS = {lg: "comp_" + lg.replace(".", "_") for lg in config.DOMESTIC_LEAGUES}
RESULT_CODES = {"A": 0, "D": 1, "H": 2}  # result_90: 0 = away win, 1 = draw, 2 = home win

MATCH_IDS = ["match_id", "date", "season_start", "competition", "home_team", "away_team"]
MATCH_TARGETS = ["home_goals", "away_goals", "result_90"]
MATCH_CONTEXT = ["is_ucl"] + list(COMPETITION_COLUMNS.values()) + PHASE_COLUMNS + \
    ["leg", "neutral", "covid_no_fans", "same_association", "elo_diff", "first_leg_gd"]
SIDE = ["elo", "elo_n", "dom_ppg_l5", "dom_ppg_l10", "dom_gf_pg_l10", "dom_ga_pg_l10", "dom_days_since_last",
        "dom_prev_ppg", "dom_prev_gd_pg", "dom_prev_rank_pct", "ucl_prev1_stage", "ucl_best_stage_prev5",
        "ucl_main_apps_prev5", "ucl_titles_prev10", "ucl_main_ppg_prev3", "ucl_main_gd_pg_prev3",
        "ucl_history_seasons", "assoc_ppg_prev5", "assoc_main_teams", "ucl_ss_played", "ucl_ss_ppg", "ucl_ss_gd_pg",
        "dom_form_missing", "dom_prev_missing", "ucl_prev3_missing", "assoc_missing"]

TEAM_SEASON_IDS = ["season_start", "team_id", "team"]
TEAM_SEASON_FEATURES = ["elo", "elo_n", "elo_rank_in_season", "elo_minus_season_mean", "n_main_stage_teams",
                        "dom_ppg_l5", "dom_ppg_l10", "dom_gf_pg_l10", "dom_ga_pg_l10", "dom_prev_ppg",
                        "dom_prev_gd_pg", "dom_prev_rank_pct", "ucl_prev1_stage", "ucl_best_stage_prev5",
                        "ucl_main_apps_prev5", "ucl_titles_prev10", "ucl_main_ppg_prev3", "ucl_main_gd_pg_prev3",
                        "ucl_history_seasons", "assoc_ppg_prev5", "assoc_main_teams",
                        "dom_form_missing", "dom_prev_missing", "ucl_prev3_missing", "assoc_missing"]
TEAM_SEASON_TARGETS = ["stage_score", "reached_last16", "reached_qf", "reached_sf", "reached_final", "is_winner"]


def complete_side(df: pd.DataFrame, p: str = "") -> pd.DataFrame:
    """Fill one side's structurally missing features (column prefix `p`) and add *_missing flags."""
    df = df.copy()
    df[f"{p}dom_prev_rank_pct"] = (df[f"{p}dom_prev_rank"] - 1) / (df[f"{p}dom_prev_n_teams"] - 1)
    df[f"{p}dom_form_missing"] = df[f"{p}dom_ppg_l10"].isna().astype(int)
    df[f"{p}dom_prev_missing"] = df[f"{p}dom_prev_ppg"].isna().astype(int)
    df[f"{p}ucl_prev3_missing"] = df[f"{p}ucl_main_ppg_prev3"].isna().astype(int)
    df[f"{p}assoc_missing"] = df[f"{p}assoc_ppg_prev5"].isna().astype(int)
    fill = {
        "dom_ppg_l5": NEUTRAL_PPG, "dom_ppg_l10": NEUTRAL_PPG, "dom_gf_pg_l10": NEUTRAL_GOALS_PG,
        "dom_ga_pg_l10": NEUTRAL_GOALS_PG, "dom_days_since_last": NEUTRAL_REST_DAYS, "dom_prev_ppg": NEUTRAL_PPG,
        "dom_prev_gd_pg": 0.0, "dom_prev_rank_pct": NEUTRAL_RANK_PCT, "ucl_main_ppg_prev3": NEUTRAL_PPG,
        "ucl_main_gd_pg_prev3": 0.0, "assoc_ppg_prev5": WEAK_ASSOC_PPG, "ucl_ss_ppg": NEUTRAL_PPG,
        "ucl_ss_gd_pg": 0.0, "ucl_ss_played": 0.0,
    }
    return df.fillna({f"{p}{c}": v for c, v in fill.items() if f"{p}{c}" in df})


def drop_duplicate_matches(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicates, repeated match ids, and the same fixture listed twice on one date
    (in either home/away orientation)."""
    n = len(df)
    h, a = df["home_id"], df["away_id"]
    key = pd.DataFrame({"c": df["competition"], "d": df["date"], "lo": np.where(h < a, h, a), "hi": np.where(h < a, a, h)},
                       index=df.index)
    df = df[~df.duplicated() & ~df["match_id"].duplicated() & ~key.duplicated()]
    return df, n - len(df)


def match_tables(mf: pd.DataFrame, today: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """(train, predict, duplicates removed). Train = played matches; predict = upcoming fixtures.
    Unplayed fixtures dated in the past (results the source has not published yet), postponed,
    cancelled and awarded matches go to neither."""
    df, dups = drop_duplicate_matches(mf)
    df = complete_side(complete_side(df, "home_"), "away_")
    df["leg"] = df["leg"].fillna(0).astype(int)  # 0 = not part of a two-legged tie
    df["first_leg_gd"] = df["first_leg_gd"].fillna(0.0)
    for c in ["is_ucl", "neutral", "covid_no_fans", "same_association"]:
        df[c] = df[c].astype(bool).astype(int)
    df = encode_categories(df)
    feats = match_feature_columns()
    train = df[(df["status"] == "played") & df["home_goals"].notna() & df["away_goals"].notna()]
    predict = df[(df["status"] == "scheduled") & (df["date"] >= today)]
    train = train[MATCH_IDS + MATCH_TARGETS + feats].copy()
    train[["home_goals", "away_goals"]] = train[["home_goals", "away_goals"]].astype(int)
    train["result_90"] = train["result_90"].map(RESULT_CODES).astype(int)
    predict = predict[MATCH_IDS + feats]
    return train.reset_index(drop=True), predict.reset_index(drop=True), dups


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the match phase and the domestic competition."""
    unknown = set(df["phase"]) - set(UCL_PHASES) - {"domestic"}
    unknown |= set(df.loc[df["competition"] != "UCL", "competition"]) - set(COMPETITION_COLUMNS)
    if unknown:
        raise ValueError(f"categories without an encoding column: {sorted(unknown)}")
    df = df.copy()
    for p, col in zip(UCL_PHASES, PHASE_COLUMNS):
        df[col] = (df["phase"] == p).astype(int)
    for lg, col in COMPETITION_COLUMNS.items():
        df[col] = (df["competition"] == lg).astype(int)
    return df


def match_feature_columns() -> list[str]:
    return MATCH_CONTEXT + [f"{s}_{f}" for s in ("home", "away") for f in SIDE]


def write_column_guide(path) -> None:
    """data/model/columns.json: which columns are identifiers, targets and (numeric) features."""
    guide = {
        "match": {"identifiers": MATCH_IDS, "targets": MATCH_TARGETS, "features": match_feature_columns(),
                  "result_90_codes": {"0": "away win", "1": "draw", "2": "home win"}},
        "team_season": {"identifiers": TEAM_SEASON_IDS, "targets": TEAM_SEASON_TARGETS,
                        "features": TEAM_SEASON_FEATURES},
    }
    path.write_text(json.dumps(guide, indent=2))


def team_season_tables(tsf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """(train, predict, duplicates removed). Train = finished seasons; predict = seasons in progress."""
    n = len(tsf)
    df = tsf.drop_duplicates(["season_start", "team_id"])
    dups = n - len(df)
    df = complete_side(df)
    done = df["season_complete"] == True  # noqa: E712 (object column)
    train = df.loc[done, TEAM_SEASON_IDS + TEAM_SEASON_FEATURES + TEAM_SEASON_TARGETS].copy()
    train[TEAM_SEASON_TARGETS] = train[TEAM_SEASON_TARGETS].astype(int)
    predict = df.loc[~done, TEAM_SEASON_IDS + TEAM_SEASON_FEATURES]
    return train.reset_index(drop=True), predict.reset_index(drop=True), dups


def check_complete(name: str, df: pd.DataFrame, id_cols: list[str]) -> list[str]:
    issues = []
    na = df.isna().sum()
    if na.any():
        issues.append(f"{name}: missing values in {na[na > 0].to_dict()}")
    text = [c for c in df.columns if c not in id_cols and not pd.api.types.is_numeric_dtype(df[c]) and len(df)]
    if text:
        issues.append(f"{name}: non-numeric feature/target columns {text}")
    if df.duplicated(id_cols[:1] if id_cols[0] == "match_id" else id_cols[:2]).any():
        issues.append(f"{name}: duplicate ids")
    return issues
