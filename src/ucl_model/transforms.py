"""Feature transforms and engineered candidate features for the match model.

engineer() is row-wise and uses fixed constants only (nothing is estimated from the data), so it cannot leak
across a train/validation boundary. Scaling does estimate statistics, so it sits inside the sklearn pipeline
from make_pipeline() and is fitted on each fold's training rows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ucl_data.features import ELO_HOME_ADV
from ucl_data.model_data import COMPETITION_COLUMNS, NEUTRAL_PPG

SIDES = ("home", "away")
ELO_N_CAP = 200  # a rating has settled after this many matches; larger counts only measure how old the data is
REST_DAYS_CAP = 30  # longer gaps are season breaks
SS_SHRINK = 3  # this season's UCL record gets weight played / (played + 3); the rest is a neutral record
# ucl_history_seasons = min(10, seasons since 1992): 10 for every season after 2002, so it only marks the era
DROPPED = [f"{s}_ucl_history_seasons" for s in SIDES]
REPLACED_PHASES = ["phase_group", "phase_league"]  # league phase (2024-) has only two seasons of matches
DIFF = ["dom_ppg_l10", "dom_gf_pg_l10", "dom_ga_pg_l10", "dom_prev_ppg", "assoc_ppg_prev5", "ucl_main_ppg_prev3",
        "ucl_best_stage_prev5"]
CANDIDATES = ["home_exp"] + [f"diff_{f}" for f in DIFF]  # kept only if rolling CV shows they help

BINARY = {"is_ucl", "neutral", "covid_no_fans", "same_association"}
TREE_PARAMS = dict(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                   l2_regularization=1.0, early_stopping=False, random_state=0)
LINEAR_C = 0.1


def _side(*names: str) -> list[str]:
    return [f"{s}_{n}" for n in names for s in SIDES]


# Selection keeps or drops whole groups: a home_ column without its away_ twin (or the diff) makes no sense.
FEATURE_GROUPS = {
    "elo": ["elo_diff", "home_elo", "away_elo", "home_exp"],
    "elo_n": _side("elo_n"),
    "dom_form": _side("dom_ppg_l5", "dom_ppg_l10", "dom_gf_pg_l10", "dom_ga_pg_l10", "dom_days_since_last")
    + ["diff_dom_ppg_l10", "diff_dom_gf_pg_l10", "diff_dom_ga_pg_l10"],
    "dom_prev": _side("dom_prev_ppg", "dom_prev_gd_pg", "dom_prev_rank_pct") + ["diff_dom_prev_ppg"],
    "ucl_pedigree": _side("ucl_prev1_stage", "ucl_best_stage_prev5", "ucl_main_apps_prev5", "ucl_titles_prev10",
                          "ucl_main_ppg_prev3", "ucl_main_gd_pg_prev3")
    + ["diff_ucl_main_ppg_prev3", "diff_ucl_best_stage_prev5"],
    "assoc": _side("assoc_ppg_prev5", "assoc_main_teams") + ["diff_assoc_ppg_prev5"],
    "ucl_season_so_far": _side("ucl_ss_played", "ucl_ss_ppg", "ucl_ss_gd_pg"),
    "match_context": ["leg", "neutral", "covid_no_fans", "same_association", "first_leg_gd"],
    "competition": ["is_ucl"] + list(COMPETITION_COLUMNS.values()),
    "phase": ["phase_qualifying", "phase_main_group", "phase_knockout"],
    "missing_flags": _side("dom_form_missing", "dom_prev_missing", "ucl_prev3_missing", "assoc_missing"),
}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Transformed copy of a match table (match_train or match_predict) with the candidate features added."""
    out = df.drop(columns=DROPPED)
    for s in SIDES:
        out[f"{s}_elo_n"] = np.log1p(out[f"{s}_elo_n"].clip(upper=ELO_N_CAP))
        out[f"{s}_dom_days_since_last"] = np.log1p(out[f"{s}_dom_days_since_last"].clip(upper=REST_DAYS_CAP))
        w = out[f"{s}_ucl_ss_played"] / (out[f"{s}_ucl_ss_played"] + SS_SHRINK)
        out[f"{s}_ucl_ss_ppg"] = w * out[f"{s}_ucl_ss_ppg"] + (1 - w) * NEUTRAL_PPG
        out[f"{s}_ucl_ss_gd_pg"] = w * out[f"{s}_ucl_ss_gd_pg"]
    out["phase_main_group"] = out[REPLACED_PHASES].sum(axis=1)
    out = out.drop(columns=REPLACED_PHASES)
    # Elo expected score of the home side, the same formula the ratings are updated with
    out["home_exp"] = 1 / (1 + 10 ** (-(out["elo_diff"] + ELO_HOME_ADV * (1 - out["neutral"])) / 400))
    for f in DIFF:
        out[f"diff_{f}"] = out[f"home_{f}"] - out[f"away_{f}"]
    return out


def feature_columns(groups=None, candidates: bool = True) -> list[str]:
    """Columns of the given groups (default: all), with or without the engineered candidates."""
    cols = [c for g in (FEATURE_GROUPS if groups is None else groups) for c in FEATURE_GROUPS[g]]
    return cols if candidates else [c for c in cols if c not in CANDIDATES]


def uncovered(raw_features: list[str]) -> list[str]:
    """Raw features (columns.json) that engineer() neither transforms into a group nor drops on purpose."""
    known = set(feature_columns()) | set(DROPPED) | set(REPLACED_PHASES)
    return [c for c in raw_features if c not in known]


def is_binary(col: str) -> bool:
    return col in BINARY or col.startswith(("comp_", "phase_")) or col.endswith("_missing")


def make_pipeline(kind: str, features: list[str]) -> Pipeline:
    """Unfitted model for `features`: "prior" (class frequencies), "linear" (scaled multinomial logistic
    regression) or "tree" (gradient boosting, no scaling needed)."""
    if kind == "prior":
        return Pipeline([("model", DummyClassifier(strategy="prior"))])
    if kind == "linear":
        scale = [c for c in features if not is_binary(c)]
        pre = ColumnTransformer([("scale", StandardScaler(), scale)], remainder="passthrough")
        return Pipeline([("pre", pre), ("model", LogisticRegression(C=LINEAR_C, max_iter=3000))])
    if kind == "tree":
        return Pipeline([("model", HistGradientBoostingClassifier(**TREE_PARAMS))])
    raise ValueError(f"unknown model kind {kind!r}")
