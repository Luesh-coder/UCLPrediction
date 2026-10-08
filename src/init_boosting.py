"""Round 0 of the gradient-boosted match model: initial predictions, gradients and Hessians on
the training split of data/model/match_train.csv (target result_90).

Usage:  python src/init_boosting.py [--test-from 2023]
"""
import argparse

import numpy as np
import pandas as pd

from ucl_model.boosting import (CLASS_NAMES, TEST_FROM_SEASON, gradients, initial_scores, load_match_train,
                                log_loss, softmax, split_by_season)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-from", type=int, default=TEST_FROM_SEASON, help="first test season (start year)")
    args = ap.parse_args()

    df, features = load_match_train()
    train, test = split_by_season(df, args.test_from)
    y = train["result_90"].to_numpy()
    print(f"train: {len(train)} matches, seasons {train['season_start'].min()}-{train['season_start'].max()}")
    print(f"test:  {len(test)} matches, seasons {test['season_start'].min()}-{test['season_start'].max()}")
    print(f"{len(features)} features\n")

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
    print("sum of h per class (root of the first tree):  ", h.sum(axis=0).round(3))
    y_test = test["result_90"].to_numpy()
    print(f"log loss of F0: train {log_loss(y, F):.6f}, test {log_loss(y_test, np.tile(F0, (len(y_test), 1))):.6f}"
          " (the baseline a model has to beat)")
