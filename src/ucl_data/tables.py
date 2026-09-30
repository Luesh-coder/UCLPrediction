"""Core UCL tables: matches, knockout ties and team-season outcomes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import sources, teams

QUALIFYING = {"QPR", "Q1", "Q2", "Q3", "QPO", "R1", "R2"}
KNOCKOUT = QUALIFYING | {"KPO", "R16", "QF", "SF", "F"}
PHASE = {**{c: "qualifying" for c in QUALIFYING}, "GS": "group", "GS2": "group", "LP": "league",
         "KPO": "knockout", "R16": "knockout", "QF": "knockout", "SF": "knockout", "F": "knockout"}
AWAY_GOALS_LAST_SEASON = 2020  # away-goals rule abolished from 2021-22
COVID_NO_FANS = (pd.Timestamp("2020-03-10"), pd.Timestamp("2021-05-29"))

# Ties decided off the pitch: tie_id -> (team that advanced, note).
TIE_OVERRIDES = {
    "1993-94_QPR_GEO_dinamo-tbilisi|NIR_linfield":
        ("NIR_linfield", "Dinamo Tbilisi expelled for attempted bribery; Linfield advanced"),
    "1995-96_QPR_DEN_aalborg|UKR_dinamo-kiev":
        ("DEN_aalborg", "Dynamo Kyiv won the tie but were expelled from the group stage and replaced by Aalborg"),
}
# engsoccerdata tiewinner values contradicted by the scores and the next round.
KNOWN_SOURCE_WINNER_ERRORS = {
    "1999-00_SF_ESP_real-madrid|GER_bayern-munchen",  # Real Madrid won 3-2 on aggregate and played the final
}


def _goals_incl_et(df: pd.DataFrame, side: str) -> pd.Series:
    return df[f"aet_{side}_goals"].fillna(df[f"{side}_goals"])


def build_matches() -> pd.DataFrame:
    df = pd.concat([sources.load_engsoccerdata_ucl(), sources.load_openfootball_ucl()], ignore_index=True)
    for side in ("home", "away"):
        df[f"{side}_country"] = df[f"{side}_country"].map(teams.norm_country)
        df[f"{side}_id"] = [teams.team_id(c, r) for c, r in zip(df[f"{side}_country"], df[f"{side}_raw"])]
    df["phase"] = df["round_code"].map(PHASE)
    df["is_main_stage"] = df["phase"] != "qualifying"
    df = df.sort_values(["season_start", "date", "time", "home_id"], na_position="first", kind="stable")
    df = df.reset_index(drop=True)
    df["match_id"] = df["season"] + "_" + (df.groupby("season").cumcount() + 1).map("{:04d}".format)

    # Knockout ties: one id per pairing per round; legs ordered by date.
    ko = df["round_code"].isin(KNOCKOUT)
    h_first = df["home_id"] < df["away_id"]
    lo, hi = df["home_id"].where(h_first, df["away_id"]), df["away_id"].where(h_first, df["home_id"])
    df["tie_id"] = (df["season"] + "_" + df["round_code"] + "_" + lo + "|" + hi).where(ko)
    k = df[ko]
    size = k.groupby("tie_id")["match_id"].transform("size")
    df.loc[ko, "leg"] = (k.groupby("tie_id").cumcount() + 1).where(size > 1)  # leg 3 = replay

    played = df["status"] == "played"
    df["result_90"] = np.select(
        [df["home_goals"] > df["away_goals"], df["home_goals"] < df["away_goals"]], ["H", "A"], "D")
    df.loc[~played, "result_90"] = None
    df["went_to_extra_time"] = df["aet_home_goals"].notna()
    df["neutral"] = ((df["round_code"] == "F") | (df["leg"] == 3)
                     | ((df["season"] == "2019-20") & df["round_code"].isin(["QF", "SF"])))
    df["covid_no_fans"] = df["date"].between(*COVID_NO_FANS)

    cols = ["match_id", "season", "season_start", "date", "time", "phase", "round_code", "round_raw", "group",
            "matchday", "tie_id", "leg", "is_main_stage", "neutral", "covid_no_fans",
            "home_id", "away_id", "home_raw", "away_raw", "home_country", "away_country",
            "status", "home_goals", "away_goals", "result_90", "ht_home_goals", "ht_away_goals",
            "went_to_extra_time", "aet_home_goals", "aet_away_goals", "pen_home", "pen_away",
            "tie_winner_raw", "source", "source_ref"]
    return df[cols]


def build_ties(m: pd.DataFrame) -> pd.DataFrame:
    """One row per knockout tie with aggregate score and the team that advanced."""
    ko = m[m["tie_id"].notna()].sort_values(["tie_id", "date"])
    rows = []
    for tie_id, g in ko.groupby("tie_id", sort=False):
        first, last = g.iloc[0], g.iloc[-1]
        a, b = first["home_id"], first["away_id"]
        complete = (g["status"] == "played").all()
        goals, away = {a: 0.0, b: 0.0}, {a: 0.0, b: 0.0}
        hg, ag = _goals_incl_et(g, "home"), _goals_incl_et(g, "away")
        for h, aw, x, y in zip(g["home_id"], g["away_id"], hg, ag):
            goals[h] += x
            goals[aw] += y
            away[aw] += y
        winner = decided_by = note = None
        last_hg, last_ag = hg.iloc[-1], ag.iloc[-1]
        if not complete:
            pass
        elif len(g) == 3:  # the replay alone decides the tie
            if last_hg != last_ag:
                winner = last["home_id"] if last_hg > last_ag else last["away_id"]
            elif pd.notna(last["pen_home"]):
                winner = last["home_id"] if last["pen_home"] > last["pen_away"] else last["away_id"]
            decided_by = "replay"
        elif goals[a] != goals[b]:
            winner = a if goals[a] > goals[b] else b
            decided_by = "extra_time" if last["went_to_extra_time"] else "regular"
        elif len(g) > 1 and first["season_start"] <= AWAY_GOALS_LAST_SEASON and away[a] != away[b]:
            winner, decided_by = (a if away[a] > away[b] else b), "away_goals"
        elif pd.notna(last["pen_home"]):
            winner = last["home_id"] if last["pen_home"] > last["pen_away"] else last["away_id"]
            decided_by = "penalties"
        src_winner = None  # engsoccerdata states a tiewinner; used as a cross-check and fallback
        if isinstance(first["tie_winner_raw"], str):
            raw_to_id = dict(zip(g["home_raw"], g["home_id"])) | dict(zip(g["away_raw"], g["away_id"]))
            src_winner = raw_to_id.get(first["tie_winner_raw"])
        if tie_id in TIE_OVERRIDES:
            winner, note = TIE_OVERRIDES[tie_id]
            decided_by = "expulsion"
        rows.append(dict(
            tie_id=tie_id, season=first["season"], season_start=first["season_start"],
            phase=first["phase"], round_code=first["round_code"], n_legs=len(g),
            first_date=first["date"], last_date=last["date"],
            team_a_id=a, team_b_id=b, team_a_goals=goals[a], team_b_goals=goals[b],
            team_a_away_goals=away[a], team_b_away_goals=away[b], complete=complete,
            winner_id=winner or src_winner, decided_by=decided_by if winner else ("source" if src_winner else None),
            source_winner_disagrees=bool(src_winner and winner and src_winner != winner
                                         and tie_id not in TIE_OVERRIDES and tie_id not in KNOWN_SOURCE_WINNER_ERRORS),
            note=note or ("engsoccerdata tiewinner is wrong here" if tie_id in KNOWN_SOURCE_WINNER_ERRORS else None),
        ))
    t = pd.DataFrame(rows)
    t["loser_id"] = np.where(t["winner_id"] == t["team_a_id"], t["team_b_id"],
                             np.where(t["winner_id"] == t["team_b_id"], t["team_a_id"], None))
    return t


def _stage_score(teams_remaining: pd.Series, main: pd.Series) -> pd.Series:
    """0 qualifying only, 1 group/league phase, 2 last 16, 3 QF, 4 SF, 5 runner-up, 6 winner."""
    s = np.select(
        [teams_remaining == 1, teams_remaining <= 2, teams_remaining <= 4, teams_remaining <= 8,
         teams_remaining <= 16], [6, 5, 4, 3, 2], 1)
    return pd.Series(np.where(main, s, 0), index=teams_remaining.index)


def build_team_seasons(m: pd.DataFrame, ties: pd.DataFrame) -> pd.DataFrame:
    """One row per team per season: entry round, furthest stage and group/league-phase record."""
    long = pd.concat([
        m[["season", "season_start", "date", "round_code", "home_id", "home_country"]]
        .set_axis(["season", "season_start", "date", "round_code", "team_id", "country"], axis=1),
        m[["season", "season_start", "date", "round_code", "away_id", "away_country"]]
        .set_axis(["season", "season_start", "date", "round_code", "team_id", "country"], axis=1),
    ])
    # Round order within a season, by first match date; teams still alive = teams in this or any later round.
    order = long.groupby(["season", "round_code"])["date"].min().rename("first").reset_index()
    order["round_order"] = order.groupby("season")["first"].rank(method="dense")
    long = long.merge(order[["season", "round_code", "round_order"]], on=["season", "round_code"])
    last_order = long.groupby(["season", "team_id"])["round_order"].max().rename("last_order").reset_index()
    alive = order[["season", "round_code", "round_order"]].copy()
    alive["teams_alive"] = [
        (last_order[(last_order["season"] == s)]["last_order"] >= o).sum()
        for s, o in zip(alive["season"], alive["round_order"])
    ]

    first_round = long.sort_values("round_order").drop_duplicates(["season", "team_id"])
    last_round = long.sort_values("round_order").drop_duplicates(["season", "team_id"], keep="last")
    ts = (last_round[["season", "season_start", "team_id", "country", "round_code", "round_order"]]
          .merge(alive[["season", "round_order", "teams_alive"]], on=["season", "round_order"])
          .rename(columns={"round_code": "stage_reached", "teams_alive": "teams_remaining"}))
    ts = ts.merge(first_round[["season", "team_id", "round_code"]].rename(columns={"round_code": "entry_round"}),
                  on=["season", "team_id"])

    finals = ties[(ties["round_code"] == "F") & ties["complete"]]
    champion = dict(zip(finals["season"], finals["winner_id"]))
    is_w = ts["team_id"] == ts["season"].map(champion)
    ts.loc[is_w, ["stage_reached", "teams_remaining"]] = ["W", 1]
    ts["reached_main_stage"] = ~ts["stage_reached"].isin(QUALIFYING)
    ts["stage_score"] = _stage_score(ts["teams_remaining"], ts["reached_main_stage"])
    ts["season_complete"] = ts["season"].isin(champion.keys())
    for col, thr in [("reached_last16", 2), ("reached_qf", 3), ("reached_sf", 4), ("reached_final", 5)]:
        ts[col] = ts["stage_score"] >= thr
    ts["is_winner"] = is_w
    flags = ["reached_last16", "reached_qf", "reached_sf", "reached_final", "is_winner", "stage_score",
             "stage_reached", "teams_remaining"]
    ts[flags] = ts[flags].astype(object)
    ts.loc[~ts["season_complete"], flags] = None  # unknown until the season ends

    ts = ts.merge(_group_records(m), on=["season", "team_id"], how="left")
    main_size = (ts[ts["reached_main_stage"]].groupby("season").size().rename("n_main_stage_teams"))
    ts = ts.merge(main_size, on="season", how="left")
    ts = ts.drop(columns="round_order").sort_values(["season_start", "stage_score", "team_id"],
                                                    ascending=[True, False, True], na_position="last")
    return ts.reset_index(drop=True)


def team_perspective(m: pd.DataFrame) -> pd.DataFrame:
    """Two rows per played match (one per team) with goals for/against and 90-minute points."""
    p = m[m["status"] == "played"]
    base = ["match_id", "season", "season_start", "date", "round_code", "phase", "group", "is_main_stage"]
    h = p[base + ["home_id", "away_id", "home_country", "home_goals", "away_goals"]].set_axis(
        base + ["team_id", "opp_id", "country", "gf", "ga"], axis=1).assign(is_home=True)
    a = p[base + ["away_id", "home_id", "away_country", "away_goals", "home_goals"]].set_axis(
        base + ["team_id", "opp_id", "country", "gf", "ga"], axis=1).assign(is_home=False)
    t = pd.concat([h, a], ignore_index=True)
    t["pts"] = np.select([t["gf"] > t["ga"], t["gf"] == t["ga"]], [3, 1], 0)
    return t.sort_values(["date", "match_id", "is_home"]).reset_index(drop=True)


def _group_records(m: pd.DataFrame) -> pd.DataFrame:
    """Record in the first group stage / league phase, with an approximate position (pts, GD, GF)."""
    t = team_perspective(m)
    g = t[t["round_code"].isin(["GS", "LP"])].copy()
    # Seasons listed by matchday (2023-24) carry no group letter: recover groups as connected components.
    g["group"] = g["group"].fillna(g.groupby("season", group_keys=False).apply(_component_groups, include_groups=False))
    rec = g.groupby(["season", "team_id"]).agg(
        group=("group", "first"), gs_played=("pts", "size"), gs_wins=("pts", lambda s: (s == 3).sum()),
        gs_draws=("pts", lambda s: (s == 1).sum()), gs_losses=("pts", lambda s: (s == 0).sum()),
        gs_gf=("gf", "sum"), gs_ga=("ga", "sum"), gs_pts=("pts", "sum")).reset_index()
    rec["gs_gd"] = rec["gs_gf"] - rec["gs_ga"]
    rec = rec.sort_values(["season", "group", "gs_pts", "gs_gd", "gs_gf"], ascending=[True, True, False, False, False])
    rec["gs_position_approx"] = rec.groupby(["season", "group"]).cumcount() + 1
    return rec


def _component_groups(g: pd.DataFrame) -> pd.Series:
    if g["group"].notna().all() or (g["round_code"] == "LP").all():
        return g["group"] if g["group"].notna().all() else pd.Series("LP", index=g.index)
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for a, b in zip(g["team_id"], g["opp_id"]):
        parent[find(a)] = find(b)
    roots = {r: f"G{i + 1}" for i, r in enumerate(dict.fromkeys(find(t) for t in g["team_id"]))}
    return g["team_id"].map(lambda t: roots[find(t)])
