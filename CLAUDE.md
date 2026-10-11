# Project
Building a Machine Learning Full Stack Project where, using Python and its libraries (NumPy, Pandas, PyTorch, TensorFlow, Scikit-learn) to create a Model that predicts UCL Matches as they come along as well as predict the Overall winner of the UCL Season as more match data from current matches are inputted. I plan to implement this model by building an XGBoost Decision Tree.

Target season: UCL 2026-27.

# Current state
- **Done**: the data pipeline (`src/ucl_data/`) — downloads open data, harmonises team names, builds
  leakage-safe features, and writes clean train/valid/test/predict splits with validation checks.
- **Not started**: the model. `src/training.ipynb`, `src/validation.ipynb` and `src/testing.ipynb`
  exist but are empty. `xgboost` (and any other ML library) is not yet in `requirements.txt`, which
  only lists `pandas`, `numpy`, `requests`.
- No web app / front end yet.

`README.md` is the detailed reference for the dataset (sources, licences, every column, cleaning rules,
known gaps, and the round-0 boosting baseline). Keep it in sync when the pipeline changes.

# Commands
Use Python 3.12 on this machine (`py -3.12`): the 3.13 install has a broken numpy build that crashes on `import pandas`.

```bash
py -3.12 -m pip install -r requirements.txt
py -3.12 src/download_data.py [--refresh]   # sync raw files into data/raw (only changed files unless --refresh)
py -3.12 src/build_dataset.py               # build data/processed + data/model; exits 1 if validation fails
```

Scripts import the package as `ucl_data`, so run them from the repo root as `src/<script>.py`
(or have `src` on the path, e.g. notebook kernel started in `src/`).

# Layout
```
src/
  download_data.py, build_dataset.py   CLI entry points
  ucl_data/
    config.py       paths, source repos, league list, split boundaries (VALID_FIRST_SEASON=2023, TEST_FIRST_SEASON=2025)
    download.py     GitHub sync by blob SHA
    sources.py      load raw files into uniform match tables
    openfootball.py parser for openfootball Football.TXT files
    teams.py        team-name -> stable team_id (aliases in data/manual/team_aliases.csv)
    tables.py       UCL matches, knockout ties, team-season outcomes
    features.py     Elo, domestic form, UCL history, association strength (all pre-match only)
    model_data.py   fill structural missing values (+ *_missing flags), dedupe, one-hot encode, split
    build.py        orchestrates everything, validate(), build_all()
  training.ipynb, validation.ipynb, testing.ipynb   (empty, for the model work)
data/                (gitignored — regenerate with the two scripts)
  raw/        downloaded source files
  manual/     team_aliases.csv, ucl_participants.csv (hand-maintained)
  processed/  reference tables (ucl_matches, ucl_ties, ucl_team_seasons, domestic_*, teams)
  model/      model-ready CSVs + columns.json
```

# Modelling data (data/model/)
- **Match model**: `match_{train,valid,test,all,predict}.csv`, targets `result_90` (0 away, 1 draw, 2 home)
  or `home_goals`/`away_goals`. ~122k matches: UCL 1992-93 onward + 12 domestic top flights.
- **Winner model**: `team_season_{train,valid,test,all,predict}.csv`, targets `is_winner`, `reached_*`, `stage_score`.
  Alternatively simulate the bracket with the match model (Monte Carlo).
- `columns.json` lists identifiers / targets / features per file: `X = df[guide["match"]["features"]]`.
  Identifiers (`match_id`, `date`, team names, `competition`, `team_id`) are never features.
- Splits are chronological by season: train 1992-93 → 2022-23, valid 2023-24 → 2024-25, test 2025-26 onward.
  Never shuffle across splits. Tune on valid, look at test once, refit on `*_all.csv` before predicting.
- Evaluate UCL main stage with `is_ucl == 1 & phase_qualifying == 0`. Baseline to beat (class-frequency
  log loss on UCL main stage): valid 1.0352, test 1.0376.

# Conventions
- Features must only use information from before the match (team-season rows: before the season's first
  group/league-phase match). Don't introduce leakage.
- Don't fill targets; don't invent values in `data/processed/` (blanks there mean the source lacks the data).
- Football-Data.co.uk must not be used (its terms exclude automated/AI collection).
- 2026-27 UCL isn't published by openfootball yet: re-run both scripts when it is, or fill
  `data/manual/ucl_participants.csv` to get prediction rows.
