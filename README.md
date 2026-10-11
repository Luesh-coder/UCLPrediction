# UCL Prediction

## Summary
I am building a Machine Learning Model Application that will predict the UEFA Champions League (UCL) Upcoming Matches and the Overall UCL 26/27 Season. I plan to train the model on past season, pooling in from all top European league and plan to put in data along the way as well. 

## Dataset

Reproducible data pipeline for predicting UEFA Champions League (UCL) **match results** and the
**season winner**. It downloads openly licensed data, harmonises team names across sources, and
writes clean reference tables to `data/processed/` and leakage-safe, model-ready feature tables to `data/model/`.

### Quickstart

```bash
py -3.12 -m pip install -r requirements.txt   # pandas, numpy, requests, scipy, xgboost
py -3.12 src/download_data.py                 # fetch/refresh raw files (only changed files are re-downloaded)
py -3.12 src/build_dataset.py                 # build data/processed + data/model and run validation checks (~10 min)
py -3.12 src/train_match_model.py             # evaluate the match model and predict upcoming fixtures
```

> On this machine use Python 3.12: the Python 3.13 install has an experimental MinGW build of
> numpy that crashes when pandas is imported.

### Sources and licences

| Source | Used for | Licence |
|---|---|---|
| [openfootball/champions-league](https://github.com/openfootball/champions-league) | UCL 2011-12 → 2025-26 (group/league phase + knockouts), qualifying rounds 2024-25 → 2025-26; Europa League 2020-21 → 2024-25 and Conference League 2021-22 → 2024-25, plus both competitions' qualifiers 2024-25 → 2025-26 (team ratings only) | CC0 (public domain) |
| [jalapic/engsoccerdata](https://github.com/jalapic/engsoccerdata) | UCL 1992-93 → 2010-11 incl. qualifying (`champs.csv`); European Cup 1955-92 and domestic top-flight history up to 2024-25 (Elo warm-up and training data); English and German second tiers up to 2024-25 (team ratings only) | GPL (≥ 2) — fine for private use; check the licence before redistributing derived data |
| [openfootball/football.json](https://github.com/openfootball/football.json) | Domestic top flights 2025-26 and the current 2026-27 season; Austria from 2010-11; any older season engsoccerdata lacks; second tiers (team ratings only) | CC0 |

Each domestic league-season comes from one source: engsoccerdata when it is complete, otherwise
football.json (`source` column in `domestic_matches.csv`).

| League | Code | Seasons |
|---|---|---|
| Premier League / First Division | `en.1` | 1955-56 → 2026-27 |
| La Liga | `es.1` | 1955-56 → 2026-27 |
| Serie A | `it.1` | 1955-56 → 2026-27 |
| Ligue 1 | `fr.1` | 1955-56 → 2026-27 |
| Eredivisie | `nl.1` | 1956-57 → 2026-27 |
| Bundesliga | `de.1` | 1963-64 → 2026-27 |
| Primeira Liga | `pt.1` | 1994-95 → 2026-27 |
| Belgian Pro League, Greek Super League, Süper Lig, Scottish Premiership | `be.1`, `gr.1`, `tr.1`, `sco.1` | 1994-95 (Belgium 1995-96) → 2025-26 (partial) |
| Austrian Bundesliga | `at.1` | 2010-11 → 2025-26 (partial) |
| Second tiers (team ratings only): Championship, 2. Bundesliga, Segunda División, Serie B, Ligue 2 | `en.2`, `de.2`, `es.2`, `it.2`, `fr.2` | England 1955-56 → 2026-27; Germany 1974-75, Spain 2012-13, Italy 2013-14, France 2014-15 → 2025-26 |

Europa League, Conference League and second-tier matches only feed the Elo and Dixon-Coles ratings (more
matches between leagues, and promoted clubs arrive with a rating); they are not training rows. Set
`RATING_EXTRA_SOURCES = False` in `src/ucl_data/config.py` to leave them out.

Candidate extra sources (xG, betting odds, squad values, lineups) and what their terms allow are
researched in [docs/data_sources.md](docs/data_sources.md); none of them is used yet.

Football-Data.co.uk is **not** used: its terms restrict use to private league-match prediction and
exclude automated/AI collection, and it has no Champions League data.

### Model-ready files (`data/model/`) — use these for training

No missing values, no duplicate rows, only model columns (plus a few identifiers for splitting and
inspection). Every target and feature column is numeric; the only text columns are identifiers
(`match_id`, `date`, `competition`, team names, `team_id`). `columns.json` lists the identifiers,
targets and features of each file, so `X = df[guide["match"]["features"]]` gives the model input.

| File | Rows | What it is for |
|---|---|---|
| `match_all.csv` | every played match since 1992-93: UCL + 12 domestic top flights (~122k) | **Match model** data: targets `home_goals`, `away_goals`, `result_90` (0 = away win, 1 = draw, 2 = home win) + pre-match features |
| `match_train/valid/test.csv` | `match_all.csv` split by season (see below) | Fit / tune / final evaluation |
| `match_predict.csv` | upcoming fixtures (today onwards) | Same features, no targets — predict these |
| `team_season_all.csv` | group/league-phase club × finished season (984) | **Winner model**: features at the start of the main stage + targets `stage_score`, `reached_last16/qf/sf/final`, `is_winner` |
| `team_season_train/valid/test.csv` | `team_season_all.csv` split by season | Fit / tune / final evaluation |
| `team_season_predict.csv` | clubs in a season still in progress | Same features, no targets (fills once 2026-27 is published or `ucl_participants.csv` is filled) |

**Split** (chronological on `season_start`, boundaries in `config.VALID_FIRST_SEASON` / `TEST_FIRST_SEASON`;
the build checks that the parts add up to the `_all` file and do not overlap in seasons):

| Split | Seasons | Matches (UCL main stage) | Team-seasons (winners) |
|---|---|---|---|
| train | 1992-93 → 2022-23 | ~110.9k (3,527) | 880 (31) |
| valid | 2023-24, 2024-25 | ~7.8k (314) | 68 (2) |
| test | 2025-26 onwards | ~3.4k (189) | 36 (1) |

Test also collects matches of the season in progress once they are played (domestic 2026-27 so far).
Tune on valid, look at test once, then refit on `*_all.csv` before predicting.

How the files are cleaned (`src/ucl_data/model_data.py`):

- **Missing values** are structural (no first leg outside knockout ties, no UCL history for most league
  clubs, no domestic data for leagues outside the 12 covered). They are filled with neutral values and
  a `*_missing` flag per side records where that happened, so a model can tell filled values from real ones:

  | Feature | Filled with | Flag |
  |---|---|---|
  | `dom_ppg_l5/l10`, `dom_prev_ppg`, `ucl_main_ppg_prev3`, `ucl_ss_ppg` | 1.37 (average points per game) | `dom_form_missing`, `dom_prev_missing`, `ucl_prev3_missing` (`ucl_ss_played = 0` for `ucl_ss_*`) |
  | `dom_gf_pg_l10`, `dom_ga_pg_l10` | 1.34 (average goals per game) | `dom_form_missing` |
  | `dom_prev_gd_pg`, `ucl_main_gd_pg_prev3`, `ucl_ss_gd_pg`, `first_leg_gd` | 0 | as above; `leg = 0` for non-tie matches |
  | `dom_days_since_last` | 7 days | `dom_form_missing` |
  | `dom_prev_rank_pct` (0 = champion, 1 = bottom) | 0.5 | `dom_prev_missing` |
  | `assoc_ppg_prev5` | 0.75 (weak association: no group/league-phase matches in 5 seasons) | `assoc_missing` |

  Targets are never filled: rows without a result are either in the predict files or dropped
  (fixtures dated in the past whose results the source has not published yet, postponed, cancelled
  and awarded matches).
- **Duplicates**: exact duplicate rows, repeated ids, and the same fixture listed twice on one date
  (either home/away orientation) are removed. The build also checks that no club plays twice on one day.
- **Categories as numbers**: `phase` is one-hot encoded as `phase_qualifying`, `phase_group`,
  `phase_league`, `phase_knockout`, and the league as `comp_en_1`, `comp_es_1`, … (one per league).
  UCL matches have `is_ucl = 1` and all `comp_*` = 0; domestic matches have `is_ucl = 0` and all
  `phase_*` = 0. The category lists are fixed, so train and predict files always share columns.
  Team names stay identifiers: Elo and the other features already describe team strength, and
  numbering teams would imply an order that does not exist.
- **Dropped columns**: raw ids and country codes, text labels duplicated by other columns
  (`season`, `round_code`, `is_main_stage`, `tie_id`), constant or post-match columns
  (`status`, `went_to_extra_time`, `home_advanced`), `entered_via_qualifying` (unknown for 2011-12 → 2023-24),
  and `dom_prev_rank`/`dom_prev_n_teams` (replaced by `dom_prev_rank_pct`).

### Reference tables (`data/processed/`)

| File | Grain | What it is for |
|---|---|---|
| `ucl_matches.csv` | one row per UCL match (1992-93 →) | Clean results: round, leg, 90-min score, HT, extra time, penalties, neutral flag |
| `ucl_ties.csv` | one row per knockout tie | Aggregate score, away goals, **winner**, how it was decided |
| `ucl_team_seasons.csv` | team × season | Entry round, furthest stage, group/league-phase record, winner flag |
| `domestic_matches.csv`, `domestic_tables.csv` | match / team × league-season (1955 →) | Domestic results incl. current-season fixtures, and season summaries |
| `teams.csv` | raw spelling × source | How every raw team name maps to a `team_id` |

Blanks in the reference tables are facts the sources do not have (e.g. half-time scores for most
engsoccerdata seasons); they are left empty rather than invented.

`team_id` is stable across sources and seasons (e.g. `ITA_inter` covers "Internazionale", "Inter" and
"FC Internazionale Milano"). Manual merges live in `data/manual/team_aliases.csv`;
`src/ucl_data/known_aliases.csv` ships merges for spellings that differ between the bundled sources (checked
club by club). The manual file wins: a shipped alias is ignored when the manual file mentions that name.

#### Key definitions

- `round_code`: `QPR`/`Q1`/`Q2`/`Q3`/`QPO` qualifying, `R1`/`R2` pre-group knockout rounds (1992-94),
  `GS` group stage, `GS2` second group stage (1999-2003), `LP` league phase (2024-), `KPO` knockout
  play-off (2024-), `R16`, `QF`, `SF`, `F`. `phase` groups these into qualifying / group / league / knockout.
- `result_90` (H/D/A) and `home_goals`/`away_goals` are **90-minute** scores; extra time and penalties are separate columns.
- `stage_score` (comparable across formats): 0 qualifying only, 1 group/league phase, 2 last 16,
  3 quarter-final, 4 semi-final, 5 runner-up, 6 winner. `teams_remaining` is the number of clubs still
  alive at the furthest round reached (1 = champion). Targets are empty for seasons not yet finished.
- Features never use information from the match day itself or later (team-season features use only
  data from before the season's first group/league-phase match).

#### Features

| Prefix (`home_`/`away_` in match features) | Meaning |
|---|---|
| `elo`, `elo_n` | Club Elo before the match and number of rated matches. One rating over domestic top flights and second tiers (K=20) and European Cup/UCL/Europa/Conference League matches (K=30) since 1955, home advantage 65, goal-difference multiplier: league games rank clubs within a league, European games calibrate leagues against each other. A club new to a league starts at the rating of the clubs relegated from it (clubs promoted from a rated second tier keep their own rating) |
| `dom_ppg_l5`, `dom_ppg_l10`, `dom_gf_pg_l10`, `dom_ga_pg_l10`, `dom_days_since_last` | Domestic league form over the last 5/10 league games (missing if stale > 150 days) |
| `dom_prev_ppg`, `dom_prev_gd_pg`, `dom_prev_rank_pct` | Previous domestic season (rank ordered by points per game, then goal difference, scaled 0 = champion … 1 = bottom; approximate for split leagues) |
| `ucl_prev1_stage`, `ucl_best_stage_prev5`, `ucl_main_apps_prev5`, `ucl_titles_prev10`, `ucl_main_ppg_prev3` | UCL pedigree from previous seasons |
| `assoc_ppg_prev5`, `assoc_main_teams` | Association (country) strength: its clubs' points per UCL match over the last 5 seasons; clubs in this season's main stage |
| `ucl_ss_played`, `ucl_ss_ppg`, `ucl_ss_gd_pg` | This season's main-stage record before the match |
| `elo_diff`, `first_leg_gd`, `same_association`, `neutral`, `covid_no_fans` | Match context (`first_leg_gd` is set on second legs, from the home side's view) |
| `dc_home_xg`, `dc_away_xg`, `dc_p_away`, `dc_p_draw`, `dc_p_home` (no side prefix) | Dixon-Coles ratings (`src/ucl_data/ratings.py`): each club's attack and defence strength fitted to goals in all rating matches, older matches weighted down (half-life 240 days, 4-year window), refitted every week on matches before that week. Gives expected goals for each side and win/draw/loss probabilities |

### Known gaps and caveats

- **Qualifying rounds are missing for 2011-12 → 2023-24** (not in the source). History features use
  main-stage data only, so they are consistent across eras; to evaluate on the main stage use
  `is_ucl == 1 & phase_qualifying == 0`.
- **Domestic coverage**: `dom_*` features exist for 73–87 % of UCL main-stage matches from 1995-96 on
  (about half in 1992-95). Clubs from leagues outside the 12 above (e.g. Norway, Czechia, Croatia,
  Ukraine) have empty `dom_*` features; their Elo comes from European matches only.
- **Austria, Belgium, Greece, Scotland and Turkey** are complete only through 2024-25: open sources stop
  in early November for 2025-26 and have no 2026-27 yet, so these clubs' current form is missing.
- engsoccerdata lacks the 2022-23 Premier League (football.json is used for it). Corrected there:
  Gazélec Ajaccio's 2015-16 Ligue 1 season labelled as AC Ajaccio, and the second half of the 2006-07
  Süper Lig dated in 2006 instead of 2007. One Serie A clash remains: Lecce has two games dated
  22 Sep 2001 (which one is misdated is unknown), accepted by the validation as a known source issue.
- **2026-27 is not published yet** by openfootball. Re-run both scripts once it appears, or list the
  league-phase clubs in `data/manual/ucl_participants.csv` to get prediction rows now.
- `gs_position_approx` ranks by points, goal difference, goals scored (UEFA head-to-head tie-breakers
  are not applied). `neutral` covers finals, the 2019-20 Lisbon final tournament and replays only.
  `covid_no_fans` is an approximate window (10 Mar 2020 – 29 May 2021).
- Corrections applied to engsoccerdata: the 2009-10 Inter–Barcelona semi-final date (year typo); two
  1995-96 group A matches dated 23 Sep instead of 27 Sep;
  a wrong tie winner (1999-00 SF: Real Madrid, not Bayern); the leg labels (derived from dates instead).
  The 1993-94 Dinamo Tbilisi and 1995-96 Dynamo Kyiv expulsions are recorded as `decided_by = expulsion`.
- Elo ratings warm up from 1955, but leagues that enter the data in 1994 (Portugal, Belgium, Greece,
  Turkey, Scotland) and Austria (2010) need a few seasons to settle.

### Using it

- **Match model**: train on `data/model/match_train.csv` (`match_all.csv` holds about 116.5k domestic + 5.5k UCL
  matches) with the `is_ucl`, `comp_*` and `phase_*` flags as features, then evaluate on the UCL rows; targets `result_90` or
  `home_goals`/`away_goals`. Split by `season_start` (train on the past, test on later seasons) to
  avoid leakage. `match_id`, `date`, team names and `competition` are identifiers, not features (the league is encoded in `comp_*`).
  Predict `match_predict.csv`.
  `python src/train_match_model.py` does this (`src/ucl_model/match_model.py`): XGBoost on all features
  (UCL matches weighted ×3), the same without the Dixon-Coles columns, and the Dixon-Coles probabilities,
  averaged into an ensemble. It fits on `match_train.csv`, adds trees until log loss on `match_valid.csv`
  stops improving, prints log loss and accuracy on valid and test, then refits on `match_all.csv` with the
  same number of trees and writes `data/model/match_predictions.csv` (`p_away`, `p_draw`, `p_home` and the
  Dixon-Coles expected goals) for `match_predict.csv`. Results are in the hybrid model section below.
  `python src/init_boosting.py` prints round 0 of a gradient-boosted `result_90` model (softmax loss) on
  `match_train.csv` (initial scores, gradients and Hessians, from `src/ucl_model/boosting.py`) and the
  baseline log loss on each split (the round-0 section below).
- **Winner model**: either train on `team_season_train.csv` (targets `is_winner`, `reached_*`,
  `stage_score`), or simulate the bracket many times with a match model (Monte Carlo) using
  `data/processed/ucl_ties.csv` for knockout rules. One test season (36 clubs, 1 winner) is too small
  to score reliably: also evaluate walk-forward on `team_season_all.csv` (train on seasons < s, test on s,
  for each recent s).
- **Validation**: `build_dataset.py` checks that the model files have no missing values or duplicate ids,
  that no club plays twice on one day, that every knockout winner appears in the next round and
  every loser does not, that each finished season has exactly one champion, that there are no duplicate or unparsed fixtures,
  that every UCL club from a covered country appears in its own league that season, and that no league
  season loses most of its clubs from one season to the next (the sign of a team-name mismatch).

### Match model: hybrid XGBoost + Dixon-Coles

`src/ucl_model/match_model.py`, run with `py -3.12 src/train_match_model.py` (about 30 s once the data is built).
Three predictors of `result_90`, averaged with equal weights into the ensemble:

| Predictor | What it is |
|---|---|
| `hybrid` | XGBoost on every feature in `columns.json`, including the Dixon-Coles columns (`dc_*`) |
| `plain` | the same XGBoost without the `dc_*` columns |
| `dixon_coles` | the Dixon-Coles probabilities themselves (`dc_p_away`, `dc_p_draw`, `dc_p_home`) |

Both boosters use depth-4 trees, learning rate 0.03 and `min_child_weight` 50. Every UCL match (qualifying
included) counts 3× in training, and the number of trees is chosen by early stopping on valid
(200 rounds without improvement).

Log loss (lower is better), from the last run on 2026-10-11:

| Predictor | valid, all (7,796) | valid, UCL main stage (314) | test, all (3,417) | test, UCL main stage (189) |
|---|---|---|---|---|
| base rates (round 0) | 1.0754 | 1.0352 | 1.0769 | 1.0376 |
| hybrid (280 trees) | 0.9670 | 0.8964 | 0.9782 | 0.9114 |
| plain (337 trees) | 0.9682 | 0.9006 | 0.9817 | 0.9209 |
| Dixon-Coles | 0.9718 | 0.8947 | 0.9837 | 0.9158 |
| **ensemble** | **0.9670** | **0.8933** | **0.9791** | **0.9123** |

The ensemble's accuracy is 52 % on all test matches and 60 % on the UCL main stage. With a few hundred UCL
matches per split, differences of about 0.01 between predictors are within noise; every predictor beats the
base rates clearly. Test has been looked at, so don't tune on it. Retraining on new data can change the
numbers slightly.

### Match model: boosting starting point (round 0)

Gradient boosting on `result_90` with the softmax cross-entropy (log) loss. Before any tree is fitted, every
match gets the same scores F0 = log(class frequency in `match_train.csv`), so the starting probabilities
p0 = softmax(F0) are just the training frequencies. Values below are computed from `match_train.csv`
(1992-93 → 2022-23, 110,859 matches).

| | Away win (0) | Draw (1) | Home win (2) |
|---|---|---|---|
| Training matches | 30,840 | 28,235 | 51,784 |
| Starting score F0 = log p0 | −1.2794 | −1.3677 | −0.7612 |
| Starting probability p0 | 0.2782 | 0.2547 | 0.4671 |
| Hessian h = p0 (1 − p0), same for every match | 0.2008 | 0.1898 | 0.2489 |

Gradient g = p0 − onehot(result_90). At round 0 it depends only on the actual result:

| Actual result | g_away | g_draw | g_home |
|---|---|---|---|
| Away win | −0.7218 | +0.2547 | +0.4671 |
| Draw | +0.2782 | −0.7453 | +0.4671 |
| Home win | +0.2782 | +0.2547 | −0.5329 |

Log loss of the starting prediction, the baseline a model has to beat:

| Split | All matches | UCL main stage |
|---|---|---|
| train | 1.0598 | 1.0487 |
| valid | 1.0754 | 1.0352 |
| test | 1.0769 | 1.0376 |

Notes:
- Only the gaps between the three scores matter (adding a constant to all three leaves p0 unchanged).
- Moving the train/valid boundary anywhere from 2019 to 2025 changes F0 by at most 0.026.
- The start pools domestic and UCL matches; UCL alone has fewer draws (22.9 % vs 25.6 %) and more home
  wins (49.3 % vs 46.6 %), which the `is_ucl` feature lets the trees pick up.
- h is the diagonal of the 3 × 3 Hessian, the part tree boosting uses. Libraries rescale it (XGBoost uses
  2 p (1 − p), LightGBM K/(K − 1) p (1 − p)), so their internal values differ by that factor.
