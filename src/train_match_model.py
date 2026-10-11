"""Train and evaluate the match model, then predict the upcoming fixtures.

1. Evaluation: train on seasons before --val-season, stop adding trees when the validation season stops
   improving, and report log loss and accuracy on every later season (all matches and the UCL main stage).
2. Final model: retrain on every played season with the same number of trees and write
   data/model/match_predictions.csv (ensemble probabilities for data/model/match_predict.csv).

Usage:  python src/train_match_model.py [--val-season 2022]
"""
import argparse

import numpy as np
import pandas as pd

from ucl_data import config
from ucl_model.boosting import load_match_train
from ucl_model.match_model import MatchModel, fit, log_loss_probs


def evaluate(model: MatchModel, train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print log loss and accuracy on the test seasons for each predictor and the base rates."""
    prior = np.bincount(train["result_90"], minlength=3) / len(train)
    probs = {"base rates": np.tile(prior, (len(test), 1)), **model.predict(test)}
    ucl_main = ((test["is_ucl"] == 1) & (test["phase_qualifying"] == 0)).to_numpy()
    y = test["result_90"].to_numpy()
    rows = []
    for name, p in probs.items():
        row = {"model": name}
        for subset, mask in [("all", np.ones(len(test), bool)), ("UCL main stage", ucl_main)]:
            row[f"log loss, {subset}"] = log_loss_probs(y[mask], p[mask])
            row[f"accuracy, {subset}"] = float((p[mask].argmax(1) == y[mask]).mean())
        rows.append(row)
    print(f"test seasons ({ucl_main.sum()} UCL main-stage matches):")
    print(pd.DataFrame(rows).set_index("model").round(4).to_string(), "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-season", type=int, default=2022, help="validation season (start year); later seasons are the test")
    args = ap.parse_args()

    df, features = load_match_train()
    s = df["season_start"]
    train, val, test = df[s < args.val_season], df[s == args.val_season], df[s > args.val_season]
    if val.empty:
        raise SystemExit(f"no matches in validation season {args.val_season}")
    print(f"train {len(train)} matches ({s.min()}-{args.val_season - 1}), validation {len(val)} ({args.val_season}), "
          f"test {len(test)}" + (f" ({args.val_season + 1}-{s.max()})" if len(test) else ""))

    model = fit(train, features, val)
    print(f"trees: hybrid {model.rounds[0]}, plain {model.rounds[1]}\n")
    if test.empty:
        print("no seasons after the validation season: evaluation skipped\n")
    else:
        evaluate(model, train, test)

    final = fit(df, features, rounds=model.rounds)
    upcoming = pd.read_csv(config.MODEL / "match_predict.csv")
    out = upcoming[["match_id", "date", "competition", "home_team", "away_team", "dc_home_xg", "dc_away_xg"]].copy()
    p = final.predict(upcoming)["ensemble"] if len(upcoming) else np.empty((0, 3))
    out["p_away"], out["p_draw"], out["p_home"] = p[:, 0], p[:, 1], p[:, 2]
    out.round(4).to_csv(config.MODEL / "match_predictions.csv", index=False)  # rewritten even when empty
    print(f"wrote {len(out)} predictions to {config.MODEL / 'match_predictions.csv'}")
    ucl = out[out["competition"] == "UCL"]
    if len(ucl):
        print(ucl.head(10).round(3).to_string(index=False))
