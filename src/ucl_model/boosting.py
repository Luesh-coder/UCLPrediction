"""Building blocks for a gradient-boosted match model on result_90 (0 = away win, 1 = draw,
2 = home win), trained with the multiclass softmax cross-entropy loss.

The model keeps one raw score per class for every match, F (n x K), and turns it into
probabilities with softmax. Each boosting round fits trees to the first and second derivatives
of the loss with respect to F (gradients g and Hessians h), then adds the trees' output to F.
Round 0 starts every match at the same scores F0, the log of each class's training frequency.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ucl_data import config

N_CLASSES = 3
CLASS_NAMES = ["away", "draw", "home"]  # index = result_90 code
TEST_FROM_SEASON = 2023  # train on 1992-93 -> 2022-23, test on 2023-24 onwards


def load_match_train() -> tuple[pd.DataFrame, list[str]]:
    """data/model/match_train.csv and its feature columns (from columns.json)."""
    df = pd.read_csv(config.MODEL / "match_train.csv")
    guide = json.loads((config.MODEL / "columns.json").read_text())
    return df, guide["match"]["features"]


def split_by_season(df: pd.DataFrame, test_from: int = TEST_FROM_SEASON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test): seasons before `test_from` train, the rest test, so no later match leaks
    into training."""
    past = df["season_start"] < test_from
    return df[past].reset_index(drop=True), df[~past].reset_index(drop=True)


def softmax(F: np.ndarray) -> np.ndarray:
    """Row-wise class probabilities from raw scores F (n x K)."""
    z = np.exp(F - F.max(axis=1, keepdims=True))  # subtracting the row max avoids overflow
    return z / z.sum(axis=1, keepdims=True)


def initial_scores(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """F0 (K,): the constant scores with the lowest training loss, log of each class's
    frequency, so softmax(F0) gives back the class frequencies."""
    prior = np.bincount(y, minlength=n_classes) / len(y)
    return np.log(prior)


def gradients(y: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(g, h), each n x K: first and second derivative of the loss L = -log p[y] with respect to
    each raw score. g = p - onehot(y); h = p * (1 - p), the diagonal of the Hessian (the part
    tree boosting uses, as in XGBoost and LightGBM)."""
    p = softmax(F)
    onehot = np.eye(F.shape[1])[y]
    return p - onehot, p * (1 - p)


def log_loss(y: np.ndarray, F: np.ndarray) -> float:
    """Mean softmax cross-entropy, the loss the gradients come from."""
    return float(-np.log(softmax(F)[np.arange(len(y)), y]).mean())
