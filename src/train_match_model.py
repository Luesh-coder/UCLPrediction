"""Train and evaluate the match model, then predict the upcoming fixtures.

Uses the chronological split written by build_dataset.py (data/model/match_{train,valid,test,all}.csv):
1. Fit on train; trees are added until log loss on valid stops improving (valid picks the tree count).
2. Report log loss and accuracy on valid and test (all matches and the UCL main stage). Look at test once.
3. Refit on match_all.csv with the same number of trees and write data/model/match_predictions.csv
   (ensemble probabilities for data/model/match_predict.csv).

Usage:  python src/train_match_model.py
"""
import numpy as np
import pandas as pd

from ucl_data import config
from ucl_model.boosting import load_match, match_features, ucl_main_stage
from ucl_model.match_model import MatchModel, fit, log_loss_probs


def evaluate(model: MatchModel, train: pd.DataFrame, df: pd.DataFrame, name: str) -> None:
    """Print log loss and accuracy on df for each predictor and the training base rates."""
    prior = np.bincount(train["result_90"], minlength=3) / len(train)
    probs = {"base rates": np.tile(prior, (len(df), 1)), **model.predict(df)}
    ucl = ucl_main_stage(df)
    y = df["result_90"].to_numpy()
    rows = []
    for model_name, p in probs.items():
        row = {"model": model_name}
        for subset, mask in [("all", np.ones(len(df), bool)), ("UCL main stage", ucl)]:
            row[f"log loss, {subset}"] = log_loss_probs(y[mask], p[mask])
            row[f"accuracy, {subset}"] = float((p[mask].argmax(1) == y[mask]).mean())
        rows.append(row)
    print(f"{name} ({len(df)} matches, {ucl.sum()} UCL main stage):")
    print(pd.DataFrame(rows).set_index("model").round(4).to_string(), "\n")


if __name__ == "__main__":
    features = match_features()
    train, valid, test = load_match("train"), load_match("valid"), load_match("test")
    if valid.empty:
        raise SystemExit("match_valid.csv is empty: rebuild with build_dataset.py")
    print(f"train {len(train)} matches, valid {len(valid)}, test {len(test)}")

    model = fit(train, features, valid)
    print(f"trees: hybrid {model.rounds[0]}, plain {model.rounds[1]}\n")
    evaluate(model, train, valid, "valid (also chose the tree counts)")
    if len(test):
        evaluate(model, train, test, "test")

    final = fit(load_match("all"), features, rounds=model.rounds)
    upcoming = load_match("predict")
    out = upcoming[["match_id", "date", "competition", "home_team", "away_team", "dc_home_xg", "dc_away_xg"]].copy()
    p = final.predict(upcoming)["ensemble"] if len(upcoming) else np.empty((0, 3))
    out["p_away"], out["p_draw"], out["p_home"] = p[:, 0], p[:, 1], p[:, 2]
    out.round(4).to_csv(config.MODEL / "match_predictions.csv", index=False)  # rewritten even when empty
    print(f"wrote {len(out)} predictions to {config.MODEL / 'match_predictions.csv'}")
    ucl = out[out["competition"] == "UCL"]
    if len(ucl):
        print(ucl.head(10).round(3).to_string(index=False))
