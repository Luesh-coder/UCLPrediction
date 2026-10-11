# New data sources for UCLPrediction: what you can use

*Researched and checked on 11 Oct 2026. The criteria were: automated download, personal non-commercial ML use, and data kept out of git.*

**Status labels.** **Allowed** means the licence clearly permits this use. **Allowed with conditions** means it does, with limits you must follow. **Unclear** means nobody could read the deciding clause, or the clause does not cover this use. **Forbidden** means the terms ban it. The researchers could not open most terms pages directly. Each quote is marked by where it came from: *direct*, *copy* (a third party's word-for-word copy) or *extract* (a search-engine snippet). Re-read the live page before you sign up or pay.

One note on your current features: `dc_home_xg` / `dc_away_xg` are expected goals from your Dixon-Coles model, which is fitted on goals. Shot-based xG would be new information for the model.

---

## 1. Expected goals (xG) and shot data

| Source | Coverage | Access | Cost | Licence status and deciding clause |
|---|---|---|---|---|
| [FiveThirtyEight SPI matches](https://github.com/fivethirtyeight/data/tree/master/soccer-spi) | Team xG, non-shot xG, adjusted goals and pre-match SPI probabilities. xG is confirmed for the UCL and big 5 in a 2016/17 sample. NL, PT, BE, TR, GR, SCO and AT are **missing** from the 2016/17 rows, and later seasons are unverified. Covers 2016/17 to 2022/23 and has been frozen since 13 Jun 2023. | One CSV. The official URL is gone, so use the [Wayback snapshot](https://web.archive.org/web/20250306125411id_/https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv). | Free | **Allowed (credit required).** "Unless otherwise noted, our data sets are available under the Creative Commons Attribution 4.0 International License" (*direct*) |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Event data with shot xG. The only full seasons are EPL, La Liga, Serie A and Ligue 1 in 2015/16. Bundesliga has Leverkusen matches only. The UCL has 1 match per season. Nothing for the 7 smaller leagues. | GitHub JSON or statsbombpy | Free | **Allowed with conditions.** For "analysis, research"; you may not "distribute, reproduce, sell … to any external or third party" or "commercially exploit"; published analysis must carry the StatsBomb logo (*direct*) |
| [API-Football](https://www.api-football.com/) | Shots per fixture (total, on target, inside the box). `expected_goals` was seen only in a third-party test on the Brazilian league, so it is unconfirmed for your leagues. Says it covers all 12 leagues and the UCL (unverified). | REST API with a key | The free plan only reaches seasons 2022–2024. Pro costs $19/month. | **Unclear** (*extract*). "the resale of our data without permission is not permitted". It grants no licence to publish the data and says nothing about ML. |
| [Sportmonks xG add-on](https://www.sportmonks.com/football-api/xg-data/) | Team and player xG and xGoT. The coverage list includes the UCL, big 5, NL, PT, TR, BE, SCO and AT. Greece is unconfirmed. **xG starts in 2024/25.** | REST API with a token | Paid. Roughly €99/month for the base plan plus €69/month for the xG add-on (prices unverified). | **Allowed with conditions.** "distribution, transfer, and storage of data provided by our services is allowed, but reselling the product is forbidden without our consent" (*copy*, 31 Aug 2026) |
| [Understat](https://understat.com/) | Big 5 and Russia only. No UCL. 2014/15 onwards. | Scraping only | Free | **Unclear.** No terms found. |
| [FBref](https://fbref.com/) | Advanced xG was reportedly removed in Jan 2026 (unverified). | HTML behind a CAPTCHA | Free | **Unclear, do not use.** The terms were not read. Extracts report a ban on using its stats to "support machine learning methods that predict". |
| [FotMob](https://www.fotmob.com/) | xG and shots from Opta (coverage unverified) | Undocumented API | n/a | **Forbidden.** "The use of automatic services (robots, spiders, indexing, etc.) … is expressly forbidden" (two copies). The soccerdata Python package removed its FotMob scraper at FotMob's request. |
| [Sofascore](https://www.sofascore.com/) | xG and shots (unverified) | Undocumented API that blocks scrapers | n/a | **Unclear, do not use.** An unverified extract bans "robots or scripts, scraping, crawling". |

**Recommendation**
- **Usable now, for free:** FiveThirtyEight SPI, but only as history. Credit FiveThirtyEight (CC BY 4.0) in the README. After downloading, check which leagues actually have xG by counting non-empty `xg1` values per league.
- **Current-season xG:** only Sportmonks has clear terms, and it is paid. API-Football's terms are unclear and its xG is unconfirmed for your leagues.
- **Ruled out:** FotMob (forbidden). FBref and Sofascore (terms unclear, reported bans, and both block bots). Understat (no terms, no UCL, only 5 of your 12 leagues). StatsBomb is allowed but has too little coverage for match features.

**Value and effort**
- **Value:** shot-based xG measures chance quality with less noise than goals, and your model does not have it yet. The size of the gain for your model is unknown. Measure it before paying for anything.
- **Catch with FiveThirtyEight:** the data ends in 2022-23, but your test seasons are 2023-24 onwards and your predictions are for 2026-27. The feature would be missing exactly where you evaluate and predict. Use it as an **offline experiment** instead: test on 2020-21 to 2022-23, on UCL and big-5 rows, with and without rolling xG features. Sportmonks xG only starts in 2024/25, which is short. Mixing two providers' xG is also uncertain, because their xG models may be scaled differently.
- **Effort: medium.**
  - Map FiveThirtyEight team names to your `team_id` through `data/manual/team_aliases.csv`. UCL rows carry no country, so match them against that season's clubs in `ucl_team_seasons.csv`.
  - Join on (date, home_id, away_id).
  - Build **rolling pre-match** averages (xG for and against over the last N matches), the same way `dom_gf_pg_l10` works. Never use a match's own xG as a feature: it is only known after the match.
  - Fill missing values with a neutral number and add an `xg_missing` flag, the same pattern you already use for other features.

---

## 2. Betting odds

| Source | Coverage | Access | Cost | Licence status and deciding clause |
|---|---|---|---|---|
| [The Odds API](https://the-odds-api.com/historical-odds-data/) | Home/draw/away odds from each bookmaker, plus totals. Has a key for all 12 leagues and the UCL. History starts 6 Jun 2020, with snapshots every 10 minutes (every 5 minutes from Sep 2022). Whether the smaller leagues have history that far back is unverified. | REST API with a key | Free Starter: 500 credits/month, upcoming odds only. History is paid: $30 for 20K, $59 for 100K or $119 for 5M credits per month. The checker estimates a 6-season backfill at about 120–130k credits, so about $118–119. | **Allowed with conditions.** Permitted uses include "Storing our data and retaining it indefinitely" and "Using our data to train statistical and machine learning models". "Do not resell, repackage, or redistribute our data as a standalone data product" (*copy*, terms dated 31 Aug 2026) |
| [Betfair Exchange Historical Data](https://developer.betfair.com/historical-data-services-api/) | Exchange prices for the Match Odds market (home/draw/away). The free BASIC tier gives the last traded price every minute. Football runs from Apr/May 2015. Your 12 leagues and the UCL are likely covered but unverified, because the site needs a login. Trading is thin in Austria and Greece. | Betfair account, then "buy" the free BASIC plan and use the Historical Data API. Parse the files with `betfairlightweight`. | BASIC is free | **Allowed with conditions.** Use the data "only as strictly required for your permitted personal, non-exclusive, non-sublicensable, non-commercial purpose"; the developer site says "Free for personal use" (*extracts*). Only for Betfair.com customers. |
| [API-Football odds](https://www.api-football.com/) | Pre-match odds, kept for only about 7 days. No back history. | REST API with a key | The free plan does not work for this (seasons 2022–2024 only). Pro costs $19/month. | **Unclear** (*extract*). No resale and no publishing licence. |
| [BeatTheBookie](https://github.com/Lisandro79/BeatTheBookie) | Closing odds (highest, average and count of bookmakers) for 880,494 matches, 2000-01-01 to 2015-09-06. UCL is likely included but unverified. | Google Drive, Dropbox or a Kaggle mirror | Free | **Unclear.** The repo is GPL-3.0 and the Kaggle copy is labelled CC-BY-SA-4.0, but the authors say the data was "assembled over months of scraping online sport portals". |
| [OddsPortal](https://www.oddsportal.com/) | The deepest archive | Scraping only | n/a | **Forbidden.** "not permitted to use our content … by embedding, aggregating, scraping or recreating it without our express consent" |
| Republished Football-Data.co.uk files ([datasets/football-datasets](https://github.com/datasets/football-datasets), xgabora, Kaggle European Soccer DB) | Domestic odds only. No UCL. | GitHub or Kaggle | Free | **Forbidden.** The original source says "NOT commerical or data training products using automated bots/scrapers/AI" and "league match prediction only". A republisher's open licence cannot grant rights it never had. |

**Recommendation**
- **First choice: The Odds API.** It has the clearest licence of any source here: it explicitly allows ML training and keeping the data forever. Get a free Starter key now and start collecting upcoming odds. It is unverified whether 500 credits a month is enough for 13 competitions, so check the cost per call. The paid backfill from June 2020 is optional. Before paying, make one test history call per league, because the smaller leagues may start later than 2020.
- **Second choice: Betfair BASIC.** It is free and takes the odds history back to 2015. Conditions: personal and non-commercial use only, no redistribution, and you need a Betfair.com account in a country Betfair serves. Reading the clause as covering a private prediction model is reasonable, but it is not an explicit ML permission.
- **Ruled out:**
  - OddsPortal and the Football-Data.co.uk republishers are forbidden.
  - API-Football odds have no history, and its free plan is unusable for current seasons.
  - BeatTheBookie has no clean chain of rights. Leave it out, consistent with how cautious your README already is.

**Value and effort**
- **Value: probably the highest of everything here.** Betting prices combine information your model cannot see, such as injuries, lineups, transfers and motivation. Closing odds are the usual benchmark that models are compared against. There is no known figure for how much they would help your model, so compare log loss on your season split.
- **Coverage against your split.** You train on seasons before 2022-23, validate on 2022-23 and test on 2023-24 onwards. The Odds API alone gives only 2 training seasons (2020-21 and 2021-22). Adding Betfair makes it about 7. Most of your ~122k rows (1992–2015) will have no odds, so either use a missing flag or train a separate model on the odds era only.
- **Timing.** Your rule is that features never use information from match day or later. Take the **last snapshot before match day (00:00 UTC)**, and use the same rule at prediction time. A side benefit is that one history call per competition per day captures all of the next day's matches. That may cost fewer credits than the checker's per-kickoff estimate, but this is unverified.
- **Conversion.** Turn decimal odds into probabilities (1/odds) and divide each by their sum to remove the bookmaker's margin. Store them as `odds_p_home`, `odds_p_draw` and `odds_p_away`, plus an `odds_missing` flag.
- **Mixing sources.** Betfair exchange prices carry almost no margin, while bookmaker prices do. If you train on Betfair (2015–2020) and predict with Odds API bookmaker prices, the inputs will differ somewhat. Normalising both and adding a source flag reduces the gap, but treat it as a risk.
- **Effort: medium to high.** Each source names clubs its own way (`home_team`/`away_team` in The Odds API, runner names on Betfair). You will need alias rows in `team_aliases.csv`, and UCL events need the participant-list lookup described in section 1. Join on (date ±1 day, home_id, away_id), because kickoff times are in UTC.

---

## 3. Squad market values and squad strength

*No independent checker reviewed this section. Everything below is the researcher's claim, apart from the Transfermarkt clause, which the lineups checker confirmed.*

| Source | Coverage | Access | Cost | Licence status and deciding clause |
|---|---|---|---|---|
| [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | Player valuations, lineups and transfers. All 12 leagues and the UCL, 2012-13 to 2025-26. **Frozen since 6 Jul 2026**, so there is no 2026-27. | Public download bucket or Kaggle | Free | **Unclear.** The repo is CC0, but CC0 "disclaims responsibility for clearing rights of other persons". Transfermarkt itself bans bots and AI training (last row). |
| [EA FC 24 / FIFA dataset (Kaggle)](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset) | Team overall, attack, midfield and defence ratings and player values, FIFA 15 to FC 24 (2014-15 to 2023-24). Greece is partial. | Kaggle API (free account) | Free | **Unclear.** The dataset says "License: CC0", but it was scraped from sofifa.com, and EA's terms reportedly ban using EA content to train ML (not read word for word). |
| [SoFIFA API](https://sofifa.com/document) | The same ratings, up to the current FC27 rosters | REST JSON, 60 requests/minute | Free | **Allowed with conditions (unverified, extracts only).** "Your project must be a NON-COMMERCIAL project"; "we do not support building applications entirely based on our API". It reportedly also requires a public website showing SoFIFA's logo and link, and that you already have your own FIFA database. The same EA question applies. |
| [ClubElo](http://clubelo.com/API) | Daily Elo for all European top flights | The API now returns errors and appears to have moved behind a login, with registration not yet open | Was free | **Unclear.** No terms found. |
| [CIES Football Observatory](https://football-observatory.com/-Value-) | Occasional big-5 snapshots. Full data is paid. | HTML/PDF | Free or paid | **Unclear.** No terms found. |
| [Transfermarkt](https://www.transfermarkt.com) | Everything | HTML behind anti-bot protection | Paid licence | **Forbidden.** §11.1: "not permitted to access or copy the Digital Content using bots, spiders, screen scraping … also prohibited from using the digital content for the training or development of artificial intelligence (AI)" (two independent copies) |

**Recommendation.** No source here has clean terms.
- SoFIFA is the only operator that explicitly allows non-commercial programmatic use. Its conditions are unverified and the EA question is open, so read sofifa.com/document yourself before building anything.
- Transfermarkt and anything derived from it are ruled out by its AI-training ban, and that data is frozen anyway.
- ClubElo adds little, because your Elo already uses the same results, and its API is currently down.

**Value and effort.** Squad value and game ratings describe squad quality looking forward, for example after summer transfers. They are likely most useful early in a season and for the winner model (`team_season_*`). This is not measured. **Effort: medium.** You need an as-of join that takes the latest roster before each match date, plus name mapping. Greece and the years before 2014 are coverage gaps.

---

## 4. Lineups, injuries and suspensions

| Source | Coverage | Access | Cost | Licence status and deciding clause |
|---|---|---|---|---|
| [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | Starting XI and bench for each game, all 12 leagues and the UCL, 2012-13 to 2025-26. No injury table. Frozen. | Download bucket or Kaggle | Free | **Unclear.** CC0 applies only "to the extent that he or she is an owner", and Transfermarkt bans AI training. |
| [API-Football](https://www.api-football.com/) | Lineups, plus `/injuries` and `/sidelined` endpoints. Coverage per league is unverified. | REST API with a key | Free plan: seasons 2022–2024 only. Pro costs $19/month. | **Unclear** (*extract*). "prohibited to resell this data to third parties"; any licence to publish "must be requested … from the competent authorities". |
| [Sportmonks](https://www.sportmonks.com/football-api/) | Lineups, injured and suspended players, and predicted lineups. Paid coverage and how far back the data goes are unverified. | REST API with a token | The free plan covers only Denmark and Scotland. Paid plan prices are unverified. | **Allowed with conditions.** Storing the data is allowed; reselling is not. "Your Sportmonks account is your own personal account". Data is "exclusively available per domain" (*copy*) |
| [football-data.org](https://www.football-data.org/) | **No injuries.** Lineups only with the paid Deep Data add-on. The free tier has the UCL and 7 of your 12 leagues. | REST API with a token | Free tier. Paid prices unverified. | **Allowed with conditions.** One key per application (§2.3). Keys must not be stored in open-source repos (§6.1). You must show the credit "Football data provided by the Football-Data.org API" (§7.1). After cancelling, you may not reference the data on your own site or service (§9.1) (*copy*) |
| Transfermarkt website | Injury histories | Scraping | n/a | **Forbidden** (§11.1, see section 3) |
| FotMob / Sofascore | Predicted lineups and missing players | Undocumented APIs | n/a | FotMob: **Forbidden.** Sofascore: **Unclear, do not use.** |

**Recommendation.** There is no free source with clean terms. If you are willing to pay, Sportmonks is the only option with clear terms and absence data, but confirm its paid league coverage first. football-data.org has no injuries. **Leave this for later.**

**Value and effort.** Absences matter for individual matches, for example when a star player is out. Odds taken shortly before kickoff already reflect most of that, so once you have odds the extra gain is probably small (an expectation, not measured). **Effort: high.** The data is per player, so you would need weights for how important each player is (for example their share of minutes) to turn it into team features. It also needs one API call per fixture, and confirmed lineups only appear shortly before kickoff.

---

## What to add first, in order

1. **The Odds API.** Start collecting upcoming odds on the free plan now. Do the paid backfill from 2020 (about $118–119, an estimate) if your budget allows. It has the clearest ML permission of any source.
2. **Betfair BASIC history.** Free, from 2015, personal and non-commercial use only. It fills in odds across your training seasons.
3. **FiveThirtyEight SPI xG.** Free under CC BY. Use it in an offline experiment (2016-17 to 2022-23, UCL and big 5) to find out whether xG helps before you pay for any xG source.
4. **Sportmonks (paid).** Only if step 3 shows a clear gain. It gives current xG (from 2024/25) plus injuries and lineups. Confirm the prices and the per-league coverage, especially Greece.
5. **SoFIFA ratings,** with the EA FC Kaggle file for history. Only after you have read SoFIFA's conditions yourself and are comfortable with the EA question.

**Do not use:** Transfermarkt or anything derived from it, FotMob, Sofascore, OddsPortal, republished Football-Data.co.uk files, or FBref. Understat, ClubElo, CIES and BeatTheBookie stay unclear.

## Risks

- **Terms were read second-hand.** Most quotes are copies or search extracts. Re-read each live page before signing up, and keep a dated copy of the terms in your notes (not the data).
- **Terms and access change.** FotMob's ban, the FiveThirtyEight shutdown, ClubElo moving behind a login, API-Football limiting its free plan and transfermarkt-datasets pausing all happened recently. Make each new source optional with an on/off setting, like `RATING_EXTRA_SOURCES`, plus `*_missing` flags.
- **Coverage gaps.**
  - Odds history starts in 2015 (Betfair) and 2020 (The Odds API).
  - FiveThirtyEight xG may not cover the smaller leagues.
  - Greece is unconfirmed in Sportmonks xG and partial in the EA ratings.
  - Austria and Greece have thin trading on Betfair.
  - Understat and Football-Data.co.uk have no UCL at all.
- **Training data that differs from prediction data.** A feature can be present in training but missing when you predict (FiveThirtyEight), or come from a different source (Betfair vs bookmakers). Compare each new feature's distribution in `match_predict.csv` against `match_train.csv`.
- **Team-name matching.** Every source spells clubs differently, and UCL events carry no country. Add rows to `team_aliases.csv`, print any names that fail to match, and add a check on the share of matches joined per league-season, next to your existing team-name mismatch check.
- **Dates.** Odds sources give kickoff times in UTC, while your dates are local. Join with a ±1 day tolerance.
- **Secrets.** Keep API keys out of git (for example in a `.env` file listed in `.gitignore`). football-data.org requires this explicitly.
- **Credits you must give.** FiveThirtyEight (CC BY) in the README. The StatsBomb logo on any analysis you publish with its data. The football-data.org credit line if you use it.