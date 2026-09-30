"""Load raw source files into uniform match tables (raw team names, no canonicalisation)."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import config
from .openfootball import parse_file

UCL_COLUMNS = [
    "season", "season_start", "date", "time", "round_raw", "round_code", "group", "matchday", "leg",
    "home_raw", "home_country", "away_raw", "away_country",
    "home_goals", "away_goals", "ht_home_goals", "ht_away_goals",
    "aet_home_goals", "aet_away_goals", "pen_home", "pen_away",
    "status", "tie_winner_raw", "source", "source_ref",
]


def season_label(start: int) -> str:
    return f"{start}-{(start + 1) % 100:02d}"


# ---------------------------------------------------------------- openfootball UCL (2011-12 onwards)

def _of_round(raw: str, file_stem: str) -> tuple[str, str | None, float]:
    """(round_code, group, matchday) for an openfootball round header."""
    r = raw.strip()
    md = re.search(r"Matchday\s+(\d+)", r)
    matchday = float(md[1]) if md else np.nan
    if file_stem == "clq":
        if re.search(r"play-?offs?", r, re.I):
            return "QPO", None, matchday
        n = re.search(r"(\d)", r)
        return f"Q{n[1]}", None, matchday
    if g := re.match(r"(?:Group|Gruppe)\s+([A-H])$", r):
        return "GS", g[1], matchday
    if r.startswith("Group"):
        return "GS", None, matchday
    if r.startswith("League"):
        return "LP", None, matchday
    if r.startswith("Playoffs"):
        return "KPO", None, matchday
    for pat, code in [(r"Round of 16", "R16"), (r"Quarter", "QF"), (r"Semi", "SF"), (r"Final$", "F")]:
        if re.search(pat, r):
            return code, None, matchday
    raise ValueError(f"unknown round header {raw!r}")


def load_openfootball_ucl(min_season: int = config.OPENFOOTBALL_FIRST_SEASON) -> pd.DataFrame:
    rows = []
    for f in sorted(config.OPENFOOTBALL_UCL["dest"].glob("*/*.txt")):
        start = int(f.parent.name[:4])
        if start < min_season:
            continue
        for m in parse_file(f, start):
            code, group, md = _of_round(m["round_raw"], f.stem)
            rows.append(dict(m, season=season_label(start), season_start=start, round_code=code, group=group,
                             matchday=md, leg=np.nan, tie_winner_raw=None, source="openfootball",
                             source_ref=m["source_line"]))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df[UCL_COLUMNS]


# ---------------------------------------------------------------- engsoccerdata (1992-93 to 2010-11)

# Known errors in engsoccerdata champs.csv: (Date, home, visitor) -> corrected values.
ENGSOCCERDATA_FIXES = {
    ("2009-04-20", "Internazionale", "Barcelona"): {"Date": "2010-04-20"},  # 2009-10 SF 1st leg, year typo
}

_ES_ROUNDS = {"PrelimF": "QPR", "prelim": "QPR", "Round1": "R1", "Round2": "R2", "Q-1": "Q1", "Q-2": "Q2",
              "Q-3": "Q3", "Q-PO": "QPO", "R16": "R16", "QF": "QF", "SF": "SF", "final": "F"}


def _es_round(raw: str) -> tuple[str, str | None]:
    if g := re.match(r"Group([A-H])(-prelim|-inter)?$", raw):
        return ("GS2" if g[2] == "-inter" else "GS"), g[1]
    return _ES_ROUNDS[raw], None


def load_european_cup_history(first: int = config.HISTORY_FIRST_SEASON, last: int = config.FIRST_SEASON - 1) -> pd.DataFrame:
    """European Cup matches before the Champions League era; only used to warm up Elo ratings."""
    es = pd.read_csv(config.ENGSOCCERDATA["dest"] / "data-raw" / "champs.csv")
    es = es[(es["Season"] >= first) & (es["Season"] <= last)].reset_index()
    return pd.DataFrame(dict(
        match_id="EC_" + es["index"].astype(str), season=es["Season"].map(season_label), season_start=es["Season"],
        date=pd.to_datetime(es["Date"]), home_raw=es["home"], home_country=es["hcountry"],
        away_raw=es["visitor"], away_country=es["vcountry"],
        home_goals=es["hgoal"].astype(float), away_goals=es["vgoal"].astype(float),
        neutral=(es["round"] == "final") | (es["leg"] == "replay"), status="played",
    ))


def _split_score(s) -> tuple[float, float]:
    if isinstance(s, str) and (m := re.match(r"^\s*(\d+)-(\d+)\s*$", s)):
        return float(m[1]), float(m[2])
    return np.nan, np.nan


def load_engsoccerdata_ucl(first: int = config.FIRST_SEASON, last: int = config.OPENFOOTBALL_FIRST_SEASON - 1) -> pd.DataFrame:
    path = config.ENGSOCCERDATA["dest"] / "data-raw" / "champs.csv"
    es = pd.read_csv(path)
    for (d, h, v), fix in ENGSOCCERDATA_FIXES.items():
        hit = (es["Date"] == d) & (es["home"] == h) & (es["visitor"] == v)
        for col, val in fix.items():
            es.loc[hit, col] = val
    es = es[(es["Season"] >= first) & (es["Season"] <= last)].reset_index()
    rounds = es["round"].map(_es_round)
    ht = es["HT"].map(_split_score)
    aet = es["aet"].map(_split_score)
    pens = es["pens"].map(_split_score)
    df = pd.DataFrame(dict(
        season=es["Season"].map(season_label), season_start=es["Season"], date=pd.to_datetime(es["Date"]),
        time=None, round_raw=es["round"], round_code=[r[0] for r in rounds], group=[r[1] for r in rounds],
        matchday=np.nan, leg=np.nan,  # the source's leg labels are unreliable; legs are derived from dates
        home_raw=es["home"], home_country=es["hcountry"], away_raw=es["visitor"], away_country=es["vcountry"],
        home_goals=es["hgoal"].astype(float), away_goals=es["vgoal"].astype(float),
        ht_home_goals=[h[0] for h in ht], ht_away_goals=[h[1] for h in ht],
        aet_home_goals=[a[0] for a in aet], aet_away_goals=[a[1] for a in aet],
        pen_home=[p[0] for p in pens], pen_away=[p[1] for p in pens],
        status="played", tie_winner_raw=es["tiewinner"], source="engsoccerdata",
        source_ref="champs.csv:" + (es["index"] + 2).astype(str),
    ))
    return df[UCL_COLUMNS]


# ---------------------------------------------------------------- domestic leagues

# Team names engsoccerdata gets wrong for a given season: (file, season, name) -> correct name.
ENGSOCCERDATA_DOMESTIC_NAME_FIXES = {
    ("france", 2015, "AC Ajaccio"): "Gazélec FC Ajaccio",  # Gazélec, not AC Ajaccio, played Ligue 1 in 2015-16
}


def load_engsoccerdata_domestic(first: int = config.HISTORY_FIRST_SEASON) -> pd.DataFrame:
    """Top-flight results from engsoccerdata (complete seasons up to 2024-25)."""
    parts = []
    for name, league in config.ENGSOCCERDATA_LEAGUES.items():
        country, league_name = config.DOMESTIC_LEAGUES[league]
        d = pd.read_csv(config.ENGSOCCERDATA["dest"] / "data-raw" / f"{name}.csv", low_memory=False)
        d = d[d["tier"].astype(str).isin(["1", "1.0"]) & (d["Season"] >= first)].dropna(subset=["hgoal", "vgoal"])
        for (file, season, wrong), right in ENGSOCCERDATA_DOMESTIC_NAME_FIXES.items():
            if file == name:
                for col in ("home", "visitor"):
                    d.loc[(d["Season"] == season) & (d[col] == wrong), col] = right
        ht = d["HT"].map(_split_score) if "HT" in d else pd.Series([(np.nan, np.nan)] * len(d), index=d.index)
        parts.append(pd.DataFrame(dict(
            season=d["Season"].map(season_label), season_start=d["Season"], league=league, league_name=league_name,
            country=country, date=pd.to_datetime(d["Date"]), time=None, round=None, stage=None,
            home_raw=d["home"], away_raw=d["visitor"],
            home_goals=d["hgoal"].astype(float), away_goals=d["vgoal"].astype(float),
            ht_home_goals=[h[0] for h in ht], ht_away_goals=[h[1] for h in ht],
            aet_home_goals=np.nan, aet_away_goals=np.nan, status="played", source="engsoccerdata",
        )))
    return pd.concat(parts, ignore_index=True)


def load_domestic(first: int = config.HISTORY_FIRST_SEASON) -> pd.DataFrame:
    """One source per league-season: engsoccerdata when it is (near) complete, else football.json."""
    es, fj = load_engsoccerdata_domestic(first), load_footballjson_domestic()
    fj = fj[fj["season_start"] >= first]
    played = pd.concat([df[df["home_goals"].notna()].groupby(["league", "season_start"]).size() for df in (es, fj)],
                       axis=1, keys=["es", "fj"]).fillna(0)
    use_es = set(played.index[played["es"] >= 0.95 * played.max(axis=1)])
    pick_es = [k in use_es for k in zip(es["league"], es["season_start"])]
    pick_fj = [k not in use_es for k in zip(fj["league"], fj["season_start"])]
    out = pd.concat([es[pick_es], fj[pick_fj]], ignore_index=True)
    return out.sort_values(["date", "league", "home_raw"], kind="stable").reset_index(drop=True)


def load_footballjson_domestic() -> pd.DataFrame:
    """Top-flight results from openfootball/football.json (2010-11 onwards, incl. the current season)."""
    rows = []
    for f in sorted(config.OPENFOOTBALL_DOMESTIC["dest"].glob("*/*.json")):
        league = f.stem
        country, league_name = config.DOMESTIC_LEAGUES[league]
        start = int(f.parent.name[:4])
        for m in json.loads(f.read_text(encoding="utf-8"))["matches"]:
            score = m.get("score")
            ft = ht = et = None
            if isinstance(score, list) and len(score) == 2:
                ft = score
            elif isinstance(score, dict):
                ft, ht, et = score.get("ft"), score.get("ht"), score.get("et")
            status = m.get("status") or ("played" if ft else "scheduled")
            rows.append(dict(
                season=season_label(start), season_start=start, league=league, league_name=league_name,
                country=country, date=m.get("date"), time=m.get("time"), round=m.get("round"),
                stage=m.get("stage"), home_raw=m["team1"], away_raw=m["team2"],
                home_goals=ft[0] if ft else np.nan, away_goals=ft[1] if ft else np.nan,
                ht_home_goals=ht[0] if ht else np.nan, ht_away_goals=ht[1] if ht else np.nan,
                aet_home_goals=et[0] if et else np.nan, aet_away_goals=et[1] if et else np.nan,
                status=status, source="openfootball",
            ))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df
