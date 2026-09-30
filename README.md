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
py -3.12 src/build_dataset.py                 # build data/processed/*.csv and run validation checks
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

### Output files (`data/processed/`)

| File | Grain | What it is for |
|---|---|---|
| `ucl_matches.csv` | one row per UCL match (1992-93 →) | Clean results: round, leg, 90-min score, HT, extra time, penalties, neutral flag |
| `ucl_ties.csv` | one row per knockout tie | Aggregate score, away goals, **winner**, how it was decided |
| `ucl_team_seasons.csv` | team × season | Entry round, furthest stage, group/league-phase record, winner flag |
| `match_features.csv` | one row per match: UCL + domestic top flights, 1992-93 → | **Pooled training data** for the match model: targets + pre-match features for home and away side; `competition` is `UCL` or a league code |
| `ucl_match_features.csv` | one row per UCL match | The UCL rows of `match_features.csv` (evaluate and predict on these) |
| `ucl_team_season_features.csv` | main-stage participant × season | **Winner model**: features frozen at the start of the group/league phase + stage targets |
| `domestic_matches.csv`, `domestic_tables.csv` | match / team × league-season (1955 →) | Domestic results incl. current-season fixtures, and season summaries |
| `teams.csv` | raw spelling × source | How every raw team name maps to a `team_id` |

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
| `dom_prev_ppg`, `dom_prev_gd_pg`, `dom_prev_rank` | Previous domestic season (rank ordered by points per game, then goal difference; approximate for split leagues) |
| `ucl_prev1_stage`, `ucl_best_stage_prev5`, `ucl_main_apps_prev5`, `ucl_titles_prev10`, `ucl_main_ppg_prev3` | UCL pedigree from previous seasons |
| `assoc_ppg_prev5`, `assoc_main_teams` | Association (country) strength: its clubs' points per UCL match over the last 5 seasons; clubs in this season's main stage |
| `ucl_ss_played`, `ucl_ss_ppg`, `ucl_ss_gd_pg` | This season's main-stage record before the match |
| `elo_diff`, `first_leg_gd`, `same_association`, `neutral`, `covid_no_fans` | Match context (`first_leg_gd` is set on second legs, from the home side's view) |

### Known gaps and caveats

- **Qualifying rounds are missing for 2011-12 → 2023-24** (not in the source). History features use
  main-stage data only, so they are consistent across eras; filter on `is_main_stage` for training.
- **Domestic coverage**: `dom_*` features exist for 73–87 % of UCL main-stage matches from 1995-96 on
  (about half in 1992-95). Clubs from leagues outside the 12 above (e.g. Norway, Czechia, Croatia,
  Ukraine) have empty `dom_*` features; their Elo comes from European matches only.
- **Austria, Belgium, Greece, Scotland and Turkey** are complete only through 2024-25: open sources stop
  in early November for 2025-26 and have no 2026-27 yet, so these clubs' current form is missing.
- engsoccerdata lacks the 2022-23 Premier League (football.json is used for it) and labels
  Gazélec Ajaccio's 2015-16 Ligue 1 season as AC Ajaccio (corrected).
- **2026-27 is not published yet** by openfootball. Re-run both scripts once it appears, or list the
  league-phase clubs in `data/manual/ucl_participants.csv` to get prediction rows now.
- `gs_position_approx` ranks by points, goal difference, goals scored (UEFA head-to-head tie-breakers
  are not applied). `neutral` covers finals, the 2019-20 Lisbon final tournament and replays only.
  `covid_no_fans` is an approximate window (10 Mar 2020 – 29 May 2021).
- Corrections applied to engsoccerdata: the 2009-10 Inter–Barcelona semi-final date (year typo);
  a wrong tie winner (1999-00 SF: Real Madrid, not Bayern); the leg labels (derived from dates instead).
  The 1993-94 Dinamo Tbilisi and 1995-96 Dynamo Kyiv expulsions are recorded as `decided_by = expulsion`.
- Elo ratings warm up from 1955, but leagues that enter the data in 1994 (Portugal, Belgium, Greece,
  Turkey, Scotland) and Austria (2010) need a few seasons to settle.

### Using it

- **Match model**: train on `match_features.csv` (about 119k domestic + 5.5k UCL played matches) with
  `is_ucl` / `competition` as features, then evaluate on the UCL rows (`is_ucl & is_main_stage`);
  targets `result_90` or `home_goals`/`away_goals`. Split by season (train on the past, test on later
  seasons) to avoid leakage. Rows with `status == "scheduled"` are upcoming fixtures to predict.
- **Winner model**: either train on `ucl_team_season_features.csv` (targets `is_winner`,
  `reached_*`, `stage_score`), or simulate the bracket many times with a match model (Monte Carlo) using `ucl_ties.csv` for knockout rules.
- **Validation**: `build_dataset.py` checks that every knockout winner appears in the next round and
  every loser does not, that each finished season has exactly one champion, that there are no duplicate or unparsed fixtures,
  that every UCL club from a covered country appears in its own league that season, and that no league
  season loses most of its clubs from one season to the next (the sign of a team-name mismatch).
