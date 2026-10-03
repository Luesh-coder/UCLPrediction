"""Season-based splits. Rows are never split at random: outcomes drift over time (the UCL draw rate fell
from about 31 % in 1992-95 to 18 % in the 2020s), so models train on past seasons and are scored on later ones.

- Test: TEST_SEASONS, locked. Scored once, after every feature and model choice is made.
- Validation: rolling origin. For each season in CV_SEASONS, train on every earlier season and validate
  on that season's UCL main-stage matches.
- Final model: refit on everything played so far before predicting match_predict.csv.

Whole seasons go to one side, domestic matches included, so a fold never trains on matches played after
the ones it validates on. Features already use only pre-match information (ucl_data/features.py).
"""
from __future__ import annotations

import json
from collections.abc import Iterator

import numpy as np
import pandas as pd

from ucl_data import config

TEST_SEASONS = (2023, 2024, 2025)  # 503 UCL main-stage matches, incl. both league-phase seasons
CV_SEASONS = tuple(range(2012, 2023))  # 11 validation folds, ~1,370 UCL main-stage matches


def load_match(kind: str = "train") -> tuple[pd.DataFrame, dict]:
    """data/model/match_{kind}.csv and its entry in columns.json (identifiers, targets, features)."""
    guide = json.loads((config.MODEL / "columns.json").read_text())["match"]
    return pd.read_csv(config.MODEL / f"match_{kind}.csv", parse_dates=["date"]), guide


def ucl_main_mask(df: pd.DataFrame) -> pd.Series:
    """UCL group/league phase and knockout matches. Qualifiers are missing for 2011-12 to 2023-24."""
    return (df["is_ucl"] == 1) & (df["phase_qualifying"] == 0)


def holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(development, test): every season before the test seasons, and the test seasons."""
    return df[df["season_start"] < min(TEST_SEASONS)], df[df["season_start"].isin(TEST_SEASONS)]


def rolling_folds(df: pd.DataFrame, seasons=CV_SEASONS, eval_mask: pd.Series | None = None
                  ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """(season, train positions, validation positions) per season. Train = all rows of earlier seasons;
    validation = `eval_mask` rows of the season (default: UCL main stage, or every row of a team-season table)."""
    if set(seasons) & set(TEST_SEASONS):
        raise ValueError(f"test seasons {TEST_SEASONS} cannot be validation folds")
    if eval_mask is None:
        eval_mask = ucl_main_mask(df) if "is_ucl" in df else pd.Series(True, index=df.index)
    s, ok = df["season_start"].to_numpy(), eval_mask.to_numpy()
    for season in seasons:
        train, val = np.flatnonzero(s < season), np.flatnonzero((s == season) & ok)
        assert s[train].max() < season and not np.isin(s[train], TEST_SEASONS).any()
        yield season, train, val
