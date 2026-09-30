"""Leakage-safe features. Every value uses only matches played strictly before the match
(or, for team-season rows, before the season's first group/league-phase match)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import teams

# Elo over domestic top flights and European matches together (in the style of ClubElo): league
# results rank clubs within a league, European results calibrate the leagues against each other.
ELO_K_DOMESTIC = 20
ELO_K_EUROPE = 30
ELO_HOME_ADV = 65
ELO_INIT = 1500
ELO_INIT_QUAL = 1400  # club first seen in a European qualifier, from an association with no rated clubs
ELO_NEWCOMER_GAP = 100  # new club: association mean minus this (unless relegated clubs give a reference)
DOMESTIC_STALE_DAYS = 150  # older domestic form is treated as missing


def _gd_mult(gd: float) -> float:
    gd = abs(gd)
    return 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8


class Elo:
    """Sequential Elo over all matches, recording each club's rating before every match.

    A club new to a league starts at the mean rating of the clubs that left that league after
    the previous season (i.e. the relegated clubs it replaces); otherwise at its association's
    mean minus ELO_NEWCOMER_GAP.
    """

    def __init__(self, league_teams: dict[tuple[str, int], set[str]]) -> None:
        self.league_teams = league_teams  # (league, season_start) -> club ids
        self.rating: dict[str, float] = {}
        self.n: dict[str, int] = {}
        self.by_country: dict[str, list[str]] = {}
        self._newcomer: dict[tuple[str, int], float] = {}

    def _assoc_mean(self, country: str) -> float | None:
        clubs = self.by_country.get(country, [])
        return float(np.mean([self.rating[t] for t in clubs])) if len(clubs) >= 3 else None

    def _init(self, country: str, league, season: int, qualifying: bool) -> float:
        am = self._assoc_mean(country)
        if isinstance(league, str):
            key = (league, season)
            if key not in self._newcomer:
                gone = self.league_teams.get((league, season - 1), set()) - self.league_teams.get(key, set())
                vals = [self.rating[t] for t in gone if t in self.rating]
                self._newcomer[key] = float(np.mean(vals)) if vals else (
                    am - ELO_NEWCOMER_GAP if am is not None else ELO_INIT)
            return self._newcomer[key]
        if am is not None:
            return am - ELO_NEWCOMER_GAP
        return ELO_INIT_QUAL if qualifying else ELO_INIT

    def get(self, team: str, country: str, league=None, season: int = 0, qualifying: bool = False) -> float:
        if team not in self.rating:
            self.rating[team] = self._init(country, league, season, qualifying)
            self.n[team] = 0
            self.by_country.setdefault(country, []).append(team)
        return self.rating[team]

    def run(self, games: pd.DataFrame) -> pd.DataFrame:
        """games: match_id, date, season_start, league (NaN for European matches), home/away _id and
        _country, home/away_goals, neutral, qualifying, status."""
        out = []
        g = games.sort_values(["date", "match_id"])
        played = g["status"] == "played"
        for _, day in g[played].groupby("date", sort=True):
            pre = []
            for r in day.itertuples(index=False):
                rh = self.get(r.home_id, r.home_country, r.league, r.season_start, r.qualifying)
                ra = self.get(r.away_id, r.away_country, r.league, r.season_start, r.qualifying)
                out.append((r.match_id, rh, ra, self.n[r.home_id], self.n[r.away_id]))
                pre.append((r, rh, ra))
            for r, rh, ra in pre:  # a club plays at most once per day, so update after recording
                e = 1 / (1 + 10 ** ((ra - rh - (0 if r.neutral else ELO_HOME_ADV)) / 400))
                s = 1.0 if r.home_goals > r.away_goals else 0.5 if r.home_goals == r.away_goals else 0.0
                k = ELO_K_DOMESTIC if isinstance(r.league, str) else ELO_K_EUROPE
                d = k * _gd_mult(r.home_goals - r.away_goals) * (s - e)
                self.rating[r.home_id] += d
                self.rating[r.away_id] -= d
                self.n[r.home_id] += 1
                self.n[r.away_id] += 1
        for r in g[~played].itertuples(index=False):  # fixtures not yet played: current rating
            out.append((r.match_id, self.get(r.home_id, r.home_country, r.league, r.season_start, r.qualifying),
                        self.get(r.away_id, r.away_country, r.league, r.season_start, r.qualifying),
                        self.n[r.home_id], self.n[r.away_id]))
        return pd.DataFrame(out, columns=["match_id", "home_elo", "away_elo", "home_elo_n", "away_elo_n"])


# ---------------------------------------------------------------- domestic league features

def domestic_team_games(dom: pd.DataFrame) -> pd.DataFrame:
    """Team-perspective domestic results with rolling form over the last 5 and 10 league games."""
    p = dom[dom["home_goals"].notna() & ~dom["status"].isin(["cancelled", "canceled", "postponed"])].copy()
    p["home_id"] = [teams.team_id(c, r) for c, r in zip(p["country"], p["home_raw"])]
    p["away_id"] = [teams.team_id(c, r) for c, r in zip(p["country"], p["away_raw"])]
    cols = ["league", "season_start", "date"]
    t = pd.concat([
        p[cols + ["home_id", "home_goals", "away_goals"]].set_axis(cols + ["team_id", "gf", "ga"], axis=1),
        p[cols + ["away_id", "away_goals", "home_goals"]].set_axis(cols + ["team_id", "gf", "ga"], axis=1),
    ], ignore_index=True)
    t["pts"] = np.select([t["gf"] > t["ga"], t["gf"] == t["ga"]], [3, 1], 0)
    t = t.sort_values(["team_id", "date"]).reset_index(drop=True)
    g = t.groupby("team_id")
    for w in (5, 10):
        roll = g[["pts", "gf", "ga"]].rolling(w, min_periods=1)
        sums = roll.sum().reset_index(level=0, drop=True)
        n = g["pts"].rolling(w, min_periods=1).count().reset_index(level=0, drop=True)
        t[f"dom_ppg_l{w}"] = sums["pts"] / n
        if w == 10:
            t["dom_gf_pg_l10"], t["dom_ga_pg_l10"], t["dom_n_l10"] = sums["gf"] / n, sums["ga"] / n, n
    return t


def domestic_season_tables(games: pd.DataFrame) -> pd.DataFrame:
    """Full-season domestic summary per team (all league games incl. split rounds)."""
    s = games.groupby(["league", "season_start", "team_id"]).agg(
        played=("pts", "size"), pts=("pts", "sum"), gf=("gf", "sum"), ga=("ga", "sum")).reset_index()
    s["ppg"] = s["pts"] / s["played"]
    s["gd_pg"] = (s["gf"] - s["ga"]) / s["played"]
    s = s.sort_values(["league", "season_start", "ppg", "gd_pg"], ascending=[True, True, False, False])
    s["rank"] = s.groupby(["league", "season_start"]).cumcount() + 1
    s["n_teams"] = s.groupby(["league", "season_start"])["team_id"].transform("size")
    return s


def attach_domestic_form(df: pd.DataFrame, games: pd.DataFrame, team_col: str, prefix: str) -> pd.DataFrame:
    """Latest domestic form strictly before df['date'] for df[team_col]."""
    left = df[["_row", "date", team_col]].rename(columns={team_col: "team_id"}).sort_values("date")
    right = games[["team_id", "date", "dom_ppg_l5", "dom_ppg_l10", "dom_gf_pg_l10", "dom_ga_pg_l10", "dom_n_l10"]]
    right = right.rename(columns={"date": "dom_last_date"}).sort_values("dom_last_date")
    j = pd.merge_asof(left, right, left_on="date", right_on="dom_last_date", by="team_id",
                      allow_exact_matches=False, direction="backward")
    j["dom_days_since_last"] = (j["date"] - j["dom_last_date"]).dt.days
    stale = j["dom_days_since_last"] > DOMESTIC_STALE_DAYS
    feats = ["dom_ppg_l5", "dom_ppg_l10", "dom_gf_pg_l10", "dom_ga_pg_l10", "dom_n_l10", "dom_days_since_last"]
    j.loc[stale, feats] = np.nan
    j = j[["_row"] + feats].rename(columns={f: f"{prefix}{f}" for f in feats})
    return df.merge(j, on="_row", how="left")


def attach_domestic_prev_season(df: pd.DataFrame, tables: pd.DataFrame, team_col: str, prefix: str) -> pd.DataFrame:
    prev = tables[["team_id", "season_start", "ppg", "gd_pg", "rank", "n_teams"]].copy()
    prev["season_start"] += 1
    prev = prev.drop_duplicates(["team_id", "season_start"])
    prev.columns = ["team_id", "season_start"] + [f"{prefix}dom_prev_{c}" for c in ["ppg", "gd_pg", "rank", "n_teams"]]
    return df.merge(prev.rename(columns={"team_id": team_col}), on=[team_col, "season_start"], how="left")


# ---------------------------------------------------------------- UCL history / association strength

def ucl_history(ts: pd.DataFrame, tp: pd.DataFrame, pairs: pd.DataFrame, first_season: int) -> pd.DataFrame:
    """Per requested (team_id, season_start) pair: record in previous UCL seasons. Uses main-stage
    data only, whose coverage is complete for every season (qualifiers are missing 2011-12..2023-24)."""
    done = ts[ts["season_complete"]]
    stage = {(t, s): v for t, s, v in zip(done["team_id"], done["season_start"], done["stage_score"])}
    main = tp[tp["is_main_stage"]].assign(gd=lambda x: x["gf"] - x["ga"])
    per = main.groupby(["team_id", "season_start"]).agg(n=("pts", "size"), pts=("pts", "sum"), gd=("gd", "sum"))
    per = per.to_dict("index")
    rows = []
    for team, s in pairs[["team_id", "season_start"]].drop_duplicates().itertuples(index=False):
        prev = [stage.get((team, s - k), 0) for k in range(1, 11)]
        last3 = [per.get((team, s - k)) for k in range(1, 4)]
        n3 = sum(x["n"] for x in last3 if x)
        rows.append(dict(
            team_id=team, season_start=s,
            ucl_prev1_stage=prev[0], ucl_best_stage_prev5=max(prev[:5]),
            ucl_main_apps_prev5=sum(v >= 1 for v in prev[:5]), ucl_titles_prev10=sum(v == 6 for v in prev),
            ucl_main_ppg_prev3=sum(x["pts"] for x in last3 if x) / n3 if n3 else np.nan,
            ucl_main_gd_pg_prev3=sum(x["gd"] for x in last3 if x) / n3 if n3 else np.nan,
            ucl_history_seasons=min(10, s - first_season),
        ))
    return pd.DataFrame(rows)


def association_strength(tp: pd.DataFrame, ts: pd.DataFrame, countries, seasons, window: int = 5) -> pd.DataFrame:
    """Association (country) points per main-stage match over the previous `window` seasons,
    plus the number of its clubs in this season's main stage (known at the draw)."""
    main = tp[tp["is_main_stage"]]
    per = main.groupby(["country", "season_start"]).agg(n=("pts", "size"), pts=("pts", "sum"))
    rows = []
    for c in countries:
        for s in seasons:
            idx = [(c, s - k) for k in range(1, window + 1) if (c, s - k) in per.index]
            n = per.loc[idx, "n"].sum() if idx else 0
            rows.append(dict(country=c, season_start=s, assoc_ppg_prev5=per.loc[idx, "pts"].sum() / n if n else np.nan,
                             assoc_matches_prev5=n))
    a = pd.DataFrame(rows)
    cnt = ts[ts["reached_main_stage"]].groupby(["country", "season_start"]).size().rename("assoc_main_teams")
    return a.merge(cnt.reset_index(), on=["country", "season_start"], how="left").fillna({"assoc_main_teams": 0})

