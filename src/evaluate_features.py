"""Choose the match model's features, model kind and sample weights with rolling-origin CV
(ucl_model/splits.py), then, with --test, score the locked test seasons once.

Usage:  python src/evaluate_features.py [--test]
Writes data/model/selected_features.json. Run --test only after the choices are final: the test seasons
must not influence them.
"""
import argparse
import json
from dataclasses import replace

import pandas as pd

from ucl_data import config
from ucl_model import selection as S, splits, transforms as T

pd.set_option("display.width", 160)
ELO_ONLY = S.Setup("Elo only, linear", "linear", ("elo_diff", "neutral", "is_ucl"))


def compare(scores: list[S.Scores], base: S.Scores) -> pd.DataFrame:
    rows = []
    for s in scores:
        c = S.paired_bootstrap(s.log_loss, base.log_loss)
        rows.append({"setup": s.name, **s.summary(), "vs_elo": c["diff"], "lo": c["lo"], "hi": c["hi"]})
    return pd.DataFrame(rows).round(4)


def main(score_test: bool) -> dict:
    raw, guide = splits.load_match()
    if missing := T.uncovered(guide["features"]):
        raise SystemExit(f"features without a group in ucl_model/transforms.py: {missing}")
    df = T.engineer(raw)  # same rows in the same order, so the folds index both tables
    folds = list(splits.rolling_folds(df))
    print(f"rolling-origin CV over seasons {splits.CV_SEASONS[0]}-{splits.CV_SEASONS[-1]}: "
          f"{sum(len(v) for *_, v in folds)} UCL main-stage matches; test seasons {splits.TEST_SEASONS} untouched\n")

    # 1. Baselines and the transformed / engineered feature sets
    plain, full = tuple(T.feature_columns(candidates=False)), tuple(T.feature_columns())
    ladder = [(S.Setup("class prior", "prior", ("elo_diff",)), df), (ELO_ONLY, df),
              (S.Setup("raw features, tree", "tree", tuple(guide["features"])), raw),
              (S.Setup("transformed, tree", "tree", plain), df),
              (S.Setup("transformed + candidates, tree", "tree", full), df),
              (S.Setup("transformed, linear", "linear", plain), df),
              (S.Setup("transformed + candidates, linear", "linear", full), df)]
    scores = {}
    for setup, data in ladder:
        scores[setup.name] = S.cross_validate(setup, data, folds)
        print(f"  {setup.name:34s} log loss {scores[setup.name].log_loss.mean():.4f}")
    ladder_table = compare(list(scores.values()), scores[ELO_ONLY.name])
    print("\n" + ladder_table.to_string(index=False) + "\n")

    # 2. Model kind (lower log loss), then the engineered candidates (only if they help beyond one SE)
    kind = min(["tree", "linear"], key=lambda k: scores[f"transformed + candidates, {k}"].log_loss.mean())
    candidates, c = S.better_beyond_se(scores[f"transformed + candidates, {kind}"], scores[f"transformed, {kind}"])
    print(f"model kind: {kind}; candidates vs none {c['diff']:+.4f} (se {c['se']:.4f}) -> "
          f"{'kept' if candidates else 'not kept'}")
    setup = S.Setup("selected", kind, full if candidates else plain)
    best = scores[f"transformed{' + candidates' if candidates else ''}, {kind}"]

    # 3. Sample weights, one option at a time; kept only if better than unweighted beyond one SE
    weights, winners = [], []
    for option in [replace(setup, ucl_weight=w) for w in (3.0, 5.0)] + [replace(setup, half_life=h) for h in (10.0, 20.0)]:
        s = S.cross_validate(option, df, folds)
        better, c = S.better_beyond_se(s, best)
        weights.append({"ucl_weight": option.ucl_weight, "half_life": option.half_life,
                        **{k: round(v, 4) for k, v in c.items()}, "better": better})
        print(f"  ucl_weight {option.ucl_weight:g}, half_life {option.half_life}: {c['diff']:+.4f} (se {c['se']:.4f})")
        if better:
            winners.append((s.log_loss.mean(), option, s))
    if winners:
        _, setup, best = min(winners, key=lambda w: w[0])
    print(f"sample weights: ucl_weight {setup.ucl_weight:g}, half_life {setup.half_life}\n")

    # 4. Grouped permutation importance, then 5. backward elimination of groups that are not clearly useful
    groups = list(T.FEATURE_GROUPS)
    importance = S.grouped_permutation_importance(setup, df, folds, groups)
    print(importance.round(4).to_string(index=False) + "\n")
    order = importance.loc[importance["lo"] <= 0, "group"].tolist()  # least important first
    kept, elimination = S.backward_elimination(setup, groups, candidates, df, folds, best, order)
    final = setup.with_groups(kept, candidates, "selected")
    final_scores = S.Scores("selected", best.season, best.y, best.proba) if kept == groups \
        else S.cross_validate(final, df, folds)
    print(f"\nkept groups: {kept}\n" + compare([final_scores], scores[ELO_ONLY.name]).to_string(index=False))

    result = {
        "cv_seasons": list(splits.CV_SEASONS), "test_seasons": list(splits.TEST_SEASONS),
        "model": kind, "model_params": T.TREE_PARAMS if kind == "tree" else {"C": T.LINEAR_C},
        "candidates": candidates, "ucl_weight": final.ucl_weight, "half_life": final.half_life,
        "groups": kept, "features": list(final.features),
        "cv": compare([*scores.values(), final_scores], scores[ELO_ONLY.name]).to_dict("records"),
        "sample_weights": weights, "importance": importance.round(4).to_dict("records"), "elimination": elimination,
    }
    if score_test:
        result["test"] = score_locked_test(df, final)
    path = config.MODEL / "selected_features.json"
    path.write_text(json.dumps(result, indent=1, default=lambda o: o.item()))
    print(f"\nwrote {path.relative_to(config.ROOT)}")
    return result


def score_locked_test(df: pd.DataFrame, final: S.Setup) -> dict:
    """Train on every season before the test seasons, score the test seasons' UCL main-stage matches
    (and their domestic matches as a side check)."""
    dev, test = splits.holdout(df)
    models = [(setup, S.fit(setup, dev, min(splits.TEST_SEASONS))) for setup in (ELO_ONLY, final)]
    out = {}
    for name, rows in [("ucl_main", test[splits.ucl_main_mask(test)]), ("domestic", test[test["is_ucl"] == 0])]:
        scored = [S.Scores(setup.name, rows["season_start"].to_numpy(), rows[S.TARGET].to_numpy(),
                           S.predict(model, rows, setup.features)) for setup, model in models]
        table = compare(scored, scored[0])
        print(f"\nLOCKED TEST, {name} ({len(rows)} matches, seasons {splits.TEST_SEASONS}):\n"
              + table.to_string(index=False))
        out[name] = table.to_dict("records")
        if name == "ucl_main":
            cal = S.calibration_table(scored[1].proba, scored[1].y)
            print("calibration of the selected model (all three outcomes pooled):\n" + cal.to_string())
            out["calibration"] = cal.reset_index().to_dict("records")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="also score the locked test seasons (do this once)")
    main(ap.parse_args().test)
