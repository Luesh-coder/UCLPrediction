# UCL Prediction

## Summary
I am building a Machine Learning Model Application that will predict the UEFA Champions League (UCL) Upcoming Matches and the Overall UCL 26/27 Season. I plan to train the model on past season, pooling in from all top European league and plan to put in data along the way as well. 

## Dataset

Reproducible data pipeline for predicting UEFA Champions League (UCL) **match results** and the
**season winner**. It downloads openly licensed data, harmonises team names across sources, and
writes clean tables plus leakage-safe, model-ready feature tables to `data/processed/`.

### Quickstart

```bash
py -3.12 -m pip install -r requirements.txt   # pandas, numpy, requests
py -3.12 src/download_data.py                 # fetch/refresh raw files (only changed files are re-downloaded)
py -3.12 src/build_dataset.py                 # build data/processed + data/model and run validation checks
```

> On this machine use Python 3.12: the Python 3.13 install has an experimental MinGW build of
> numpy that crashes when pandas is imported.

### Sources and licences

| Source | Used for | Licence |
|---|---|---|
| [openfootball/champions-league](https://github.com/openfootball/champions-league) | UCL 2011-12 → 2025-26 (group/league phase + knockouts), qualifying rounds 2024-25 → 2025-26 | CC0 (public domain) |
| [jalapic/engsoccerdata](https://github.com/jalapic/engsoccerdata) | UCL 1992-93 → 2010-11 incl. qualifying (`champs.csv`); European Cup 1955-92 and domestic top-flight history up to 2024-25 (Elo warm-up and training data) | GPL (≥ 2) — fine for private use; check the licence before redistributing derived data |
| [openfootball/football.json](https://github.com/openfootball/football.json) | Domestic top flights 2025-26 and the current 2026-27 season; Austria from 2010-11; any older season engsoccerdata lacks | CC0 |

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

Football-Data.co.uk is **not** used: its terms restrict use to private league-match prediction and
exclude automated/AI collection, and it has no Champions League data.

### Model-ready files (`data/model/`) — use these for training

No missing values, no duplicate rows, only model columns (plus a few identifiers for splitting and
inspection). Every target and feature column is numeric; the only text columns are identifiers
(`match_id`, `date`, `competition`, team names, `team_id`). `columns.json` lists the identifiers,
targets and features of each file, so `X = df[guide["match"]["features"]]` gives the model input.

| File | Rows | What it is for |
|---|---|---|
| `match_train.csv` | every played match since 1992-93: UCL + 12 domestic top flights (~122k) | **Match model** training data: targets `home_goals`, `away_goals`, `result_90` (0 = away win, 1 = draw, 2 = home win) + pre-match features |
| `match_predict.csv` | upcoming fixtures (today onwards) | Same features, no targets — predict these |
| `team_season_train.csv` | group/league-phase club × finished season (984) | **Winner model**: features at the start of the main stage + targets `stage_score`, `reached_last16/qf/sf/final`, `is_winner` |
| `team_season_predict.csv` | clubs in a season still in progress | Same features, no targets (fills once 2026-27 is published or `ucl_participants.csv` is filled) |

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
"FC Internazionale Milano"). Manual merges live in `data/manual/team_aliases.csv`.

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
| `elo`, `elo_n` | Club Elo before the match and number of rated matches. One rating over domestic top flights (K=20) and European Cup/UCL matches (K=30) since 1955, home advantage 65, goal-difference multiplier: league games rank clubs within a league, European games calibrate leagues against each other. Promoted clubs start at the rating of the clubs they replace |
| `dom_ppg_l5`, `dom_ppg_l10`, `dom_gf_pg_l10`, `dom_ga_pg_l10`, `dom_days_since_last` | Domestic league form over the last 5/10 league games (missing if stale > 150 days) |
| `dom_prev_ppg`, `dom_prev_gd_pg`, `dom_prev_rank_pct` | Previous domestic season (rank ordered by points per game, then goal difference, scaled 0 = champion … 1 = bottom; approximate for split leagues) |
| `ucl_prev1_stage`, `ucl_best_stage_prev5`, `ucl_main_apps_prev5`, `ucl_titles_prev10`, `ucl_main_ppg_prev3` | UCL pedigree from previous seasons |
| `assoc_ppg_prev5`, `assoc_main_teams` | Association (country) strength: its clubs' points per UCL match over the last 5 seasons; clubs in this season's main stage |
| `ucl_ss_played`, `ucl_ss_ppg`, `ucl_ss_gd_pg` | This season's main-stage record before the match |
| `elo_diff`, `first_leg_gd`, `same_association`, `neutral`, `covid_no_fans` | Match context (`first_leg_gd` is set on second legs, from the home side's view) |

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

- **Match model**: train on `data/model/match_train.csv` (about 116.5k domestic + 5.5k UCL matches) with
  the `is_ucl`, `comp_*` and `phase_*` flags as features, then evaluate on the UCL rows; targets `result_90` or
  `home_goals`/`away_goals`. Split by `season_start` (train on the past, test on later seasons) to
  avoid leakage. `match_id`, `date`, team names and `competition` are identifiers, not features (the league is encoded in `comp_*`).
  Predict `match_predict.csv`.
  `python src/init_boosting.py` prints round 0 of a gradient-boosted `result_90` model (softmax loss) on a
  season split (train 1992-93 → 2022-23 by default, `--test-from` to change): initial scores, gradients and
  Hessians, from `src/ucl_model/boosting.py`.
- **Winner model**: either train on `team_season_train.csv` (targets `is_winner`, `reached_*`,
  `stage_score`), or simulate the bracket many times with a match model (Monte Carlo) using
  `data/processed/ucl_ties.csv` for knockout rules.
- **Validation**: `build_dataset.py` checks that the model files have no missing values or duplicate ids,
  that no club plays twice on one day, that every knockout winner appears in the next round and
  every loser does not, that each finished season has exactly one champion, that there are no duplicate or unparsed fixtures,
  that every UCL club from a covered country appears in its own league that season, and that no league
  season loses most of its clubs from one season to the next (the sign of a team-name mismatch).
