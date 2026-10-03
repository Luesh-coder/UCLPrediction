"""Cross-validated scoring and feature-group selection for the match model.

Comparisons are paired: two set-ups are scored on the same validation matches and the per-match difference
is bootstrapped. Season-to-season noise (log loss 0.87-1.03 per UCL season for the same model) is larger
than most feature effects, so a single holdout cannot tell them apart.

Choices follow the one-standard-error rule: the simpler set-up (fewer features, no sample weights) is kept
unless the richer one is better by more than one bootstrap standard error.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .transforms import FEATURE_GROUPS, feature_columns, make_pipeline

TARGET = "result_90"
CLASSES = [0, 1, 2]  # away win, draw, home win: ordered, as the ranked probability score needs


@dataclass(frozen=True)
class Setup:
    name: str
    kind: str  # "prior", "linear" or "tree" (transforms.make_pipeline)
    features: tuple[str, ...]
    ucl_weight: float = 1.0  # weight of UCL rows relative to domestic rows
    half_life: float | None = None  # recency weight: a season this many seasons back counts half

    def with_groups(self, groups, candidates: bool, name: str | None = None) -> Setup:
        return replace(self, name=name or self.name, features=tuple(feature_columns(groups, candidates)))


@dataclass
class Scores:
    """Out-of-fold predictions of one set-up."""
    name: str
    season: np.ndarray
    y: np.ndarray
    proba: np.ndarray
    log_loss: np.ndarray = field(init=False)
    rps: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.log_loss = log_loss_per_match(self.proba, self.y)
        self.rps = rps_per_match(self.proba, self.y)

    def summary(self) -> dict:
        return {"log_loss": round(float(self.log_loss.mean()), 4), "rps": round(float(self.rps.mean()), 4),
                "accuracy": round(float((self.proba.argmax(1) == self.y).mean()), 4), "matches": len(self.y)}


def log_loss_per_match(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(proba[np.arange(len(y)), y], 1e-15, 1))


def rps_per_match(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ranked probability score: squared error of the cumulative distribution over away < draw < home."""
    k = proba.shape[1]
    observed = (np.arange(k - 1)[None, :] >= y[:, None]).astype(float)
    return ((proba.cumsum(1)[:, :-1] - observed) ** 2).sum(1) / (k - 1)


def sample_weights(df: pd.DataFrame, ref_season: int, setup: Setup) -> np.ndarray | None:
    if setup.ucl_weight == 1 and not setup.half_life:
        return None  # unweighted (gradient boosting fits about twice as fast without weights)
    w = np.where(df["is_ucl"].to_numpy() == 1, setup.ucl_weight, 1.0)
    if setup.half_life:
        w = w * 0.5 ** ((ref_season - df["season_start"].to_numpy()) / setup.half_life)
    return w / w.mean()


def fit(setup: Setup, train: pd.DataFrame, ref_season: int):
    """Fitted pipeline; `ref_season` is the first season after the training data (for recency weights)."""
    model = make_pipeline(setup.kind, list(setup.features))
    model.fit(train[list(setup.features)], train[TARGET],
              model__sample_weight=sample_weights(train, ref_season, setup))
    return model


def predict(model, df: pd.DataFrame, features) -> np.ndarray:
    assert list(model.classes_) == CLASSES
    return model.predict_proba(df[list(features)])


def cross_validate(setup: Setup, df: pd.DataFrame, folds) -> Scores:
    season, y, proba = [], [], []
    for s, train, val in folds:
        v = df.iloc[val]
        proba.append(predict(fit(setup, df.iloc[train], s), v, setup.features))
        season.append(v["season_start"].to_numpy())
        y.append(v[TARGET].to_numpy())
    return Scores(setup.name, np.concatenate(season), np.concatenate(y), np.vstack(proba))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    """Mean of a - b per match with its bootstrap standard error and 95 % interval."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = np.concatenate([d[rng.integers(0, len(d), (k, len(d)))].mean(1)  # in chunks to bound memory
                            for k in np.diff(np.r_[0:n:200, n])])
    return {"diff": float(d.mean()), "se": float(means.std()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5))}


def better_beyond_se(richer: Scores, simpler: Scores) -> tuple[bool, dict]:
    """One-standard-error rule: is the richer set-up's log loss lower by more than one standard error?"""
    c = paired_bootstrap(richer.log_loss, simpler.log_loss)
    return c["diff"] < -c["se"], c


def calibration_table(proba: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted probability vs observed frequency, pooled over the three outcomes."""
    p = proba.ravel()
    hit = (y[:, None] == np.array(CLASSES)[None, :]).ravel()
    b = np.minimum((p * bins).astype(int), bins - 1)
    t = pd.DataFrame({"bin": b, "predicted": p, "observed": hit}).groupby("bin")
    return t.agg(predicted=("predicted", "mean"), observed=("observed", "mean"), n=("observed", "size")).round(3)


def grouped_permutation_importance(setup: Setup, df: pd.DataFrame, folds, groups, seed: int = 0) -> pd.DataFrame:
    """Increase in validation log loss when one group's columns are shuffled together (the same permutation
    for every column, so the group's internal structure is kept), pooled over folds."""
    rng = np.random.default_rng(seed)
    deltas = {g: [] for g in groups}
    for s, train, val in folds:
        v = df.iloc[val]
        model = fit(setup, df.iloc[train], s)
        y = v[TARGET].to_numpy()
        base = log_loss_per_match(predict(model, v, setup.features), y)
        for g in groups:
            cols = [c for c in FEATURE_GROUPS[g] if c in setup.features]
            shuffled = v.copy()
            shuffled[cols] = v[cols].to_numpy()[rng.permutation(len(v))]
            deltas[g].append(log_loss_per_match(predict(model, shuffled, setup.features), y) - base)
    rows = []
    for g, d in deltas.items():
        d = np.concatenate(d)
        c = paired_bootstrap(d, np.zeros_like(d))
        rows.append({"group": g, "importance": c["diff"], "se": c["se"], "lo": c["lo"], "hi": c["hi"]})
    return pd.DataFrame(rows).sort_values("importance").reset_index(drop=True)


def backward_elimination(setup: Setup, groups: list[str], candidates: bool, df: pd.DataFrame, folds,
                         reference: Scores, order: list[str], log=print) -> tuple[list[str], list[dict]]:
    """Drop groups one at a time in `order` (least important first). A drop is kept when the reduced set's
    log loss is within one standard error of `reference` (the full set), so dropped groups cannot add up
    to a real loss."""
    kept, history = list(groups), []
    for g in order:
        trial = [k for k in kept if k != g]
        scores = cross_validate(setup.with_groups(trial, candidates, f"without {g}"), df, folds)
        c = paired_bootstrap(scores.log_loss, reference.log_loss)
        dropped = c["diff"] <= c["se"]
        history.append({"group": g, "dropped": dropped, **{k: round(v, 4) for k, v in c.items()}})
        log(f"  drop {g:18s} vs full set {c['diff']:+.4f} (se {c['se']:.4f}) -> {'dropped' if dropped else 'kept'}")
        if dropped:
            kept = trial
    return kept, history
