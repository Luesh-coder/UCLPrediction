"""Match model on result_90 (0 = away win, 1 = draw, 2 = home win).

Three predictors are averaged into the final ensemble:
- hybrid: XGBoost on every feature in columns.json, including the Dixon-Coles rating probabilities;
- plain: the same XGBoost without the Dixon-Coles columns;
- Dixon-Coles: the rating model's own probabilities (dc_p_away, dc_p_draw, dc_p_home).
UCL matches count UCL_WEIGHT times in training: the model learns from every league but is judged on the UCL.
These choices were tested on the season split (train <= 2021-22, tune on 2022-23, test 2023-24 onwards).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from ucl_data.ratings import DC_COLUMNS

PARAMS = dict(objective="multi:softprob", num_class=3, eta=0.03, max_depth=4, min_child_weight=50, subsample=0.8,
              colsample_bytree=0.8, eval_metric="mlogloss", tree_method="hist", seed=0)
UCL_WEIGHT = 3.0
MAX_ROUNDS, EARLY_STOPPING = 5000, 200
DC_PROBS = ["dc_p_away", "dc_p_draw", "dc_p_home"]


def weights(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["is_ucl"] == 1, UCL_WEIGHT, 1.0)


def log_loss_probs(y: np.ndarray, p: np.ndarray) -> float:
    """Mean cross-entropy of class probabilities p (n x 3) for labels y."""
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def _matrix(df: pd.DataFrame, cols: list[str], label: bool = True) -> xgb.DMatrix:
    return xgb.DMatrix(df[cols], df["result_90"] if label else None, weight=weights(df) if label else None)


@dataclass
class MatchModel:
    features: list[str]
    hybrid: xgb.Booster
    plain: xgb.Booster

    @property
    def plain_features(self) -> list[str]:
        return [c for c in self.features if c not in DC_COLUMNS]

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Class probabilities (n x 3) from each predictor and the ensemble."""
        n = len(df)  # xgboost returns shape (0, 0) for no rows, hence the reshape
        out = {"hybrid": self.hybrid.predict(_matrix(df, self.features, False)).reshape(n, 3),
               "plain": self.plain.predict(_matrix(df, self.plain_features, False)).reshape(n, 3),
               "dixon_coles": df[DC_PROBS].to_numpy()}
        out["ensemble"] = (out["hybrid"] + out["plain"] + out["dixon_coles"]) / 3
        return out

    @property
    def rounds(self) -> tuple[int, int]:
        return self.hybrid.num_boosted_rounds(), self.plain.num_boosted_rounds()


def _boost(train: pd.DataFrame, cols: list[str], val: pd.DataFrame | None, rounds: int | None) -> xgb.Booster:
    """Early-stopped on val when given (trees beyond the best iteration are dropped), else `rounds` trees."""
    if val is None:
        return xgb.train(PARAMS, _matrix(train, cols), rounds)
    b = xgb.train(PARAMS, _matrix(train, cols), MAX_ROUNDS, evals=[(_matrix(val, cols), "val")],
                  early_stopping_rounds=EARLY_STOPPING, verbose_eval=False)
    return b[: b.best_iteration + 1]


def fit(train: pd.DataFrame, features: list[str], val: pd.DataFrame | None = None,
        rounds: tuple[int, int] | None = None) -> MatchModel:
    """Fit hybrid and plain boosters: early-stopped on `val`, or with fixed (hybrid, plain) `rounds`."""
    plain = [c for c in features if c not in DC_COLUMNS]
    rh, rp = rounds if rounds else (None, None)
    return MatchModel(features, _boost(train, features, val, rh), _boost(train, plain, val, rp))
