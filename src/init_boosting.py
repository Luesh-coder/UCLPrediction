"""Round 0 of the gradient-boosted match model: initial predictions, gradients and Hessians on
data/model/match_train.csv (target result_90), and the log loss of that start on every split.

Usage:  python src/init_boosting.py
"""
import numpy as np
import pandas as pd

from ucl_model.boosting import (CLASS_NAMES, SPLITS, gradients, initial_scores, load_match, log_loss, match_features,
                                softmax, ucl_main_stage)

if __name__ == "__main__":
    splits = {name: load_match(name) for name in SPLITS}
    train = splits["train"]
    y = train["result_90"].to_numpy()
    for name, df in splits.items():
        print(f"{name:5s}: {len(df)} matches, seasons {df['season_start'].min()}-{df['season_start'].max()}")
    print(f"{len(match_features())} features\n")

    F0 = initial_scores(y)
    F = np.tile(F0, (len(y), 1))  # every training match starts at the same scores
    g, h = gradients(y, F)
    cols = [f"{s}_{c}" for s in ("F0", "p0", "h") for c in CLASS_NAMES]
    print("initial scores F0, probabilities p0 = softmax(F0) and Hessian h (same for every match):")
    print(pd.Series(np.concatenate([F0, softmax(F0[None])[0], h[0]]), index=cols).round(6).to_string(), "\n")

    print("gradient g = p0 - onehot(result_90), one row per actual result:")
    by_result = pd.DataFrame(g, columns=[f"g_{c}" for c in CLASS_NAMES]).groupby(y).first()
    by_result.index = [f"{CLASS_NAMES[k]} win" if k != 1 else "draw" for k in by_result.index]
    by_result["matches"] = np.bincount(y)
    print(by_result.round(6).to_string(), "\n")

    print("first five training matches:")
    head = train[["date", "home_team", "away_team", "result_90"]].head().copy()
    head[[f"g_{c}" for c in CLASS_NAMES]] = g[:5]
    print(head.round(6).to_string(index=False), "\n")

    print("sum of g per class (0 at the optimal constant):", g.sum(axis=0).round(6) + 0.0)  # + 0.0 drops -0.0
    print("sum of h per class (root of the first tree):  ", h.sum(axis=0).round(3), "\n")
    print("log loss of F0 (the baseline a model has to beat):")
    rows = {}
    for name, df in splits.items():
        yy = df["result_90"].to_numpy()
        ucl = ucl_main_stage(df)
        rows[name] = {"all matches": log_loss(yy, np.tile(F0, (len(yy), 1))),
                      "UCL main stage": log_loss(yy[ucl], np.tile(F0, (ucl.sum(), 1)))}
    print(pd.DataFrame(rows).T.round(4).to_string())
