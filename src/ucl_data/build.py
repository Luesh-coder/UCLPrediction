"""Assemble processed tables and model-ready feature sets, validate them and write CSVs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features as F, sources, tables, teams

BASE_COLUMNS = ["match_id", "competition", "season", "season_start", "date", "phase", "round_code", "leg", "tie_id",
                "is_main_stage", "neutral", "covid_no_fans", "home_id", "away_id", "home_country", "away_country",
                "status", "home_goals", "away_goals", "result_90", "went_to_extra_time"]
SIDE_FEATURES = ["elo", "elo_n", "dom_ppg_l5", "dom_ppg_l10", "dom_gf_pg_l10", "dom_ga_pg_l10", "dom_n_l10",
                 "dom_days_since_last", "dom_prev_ppg", "dom_prev_gd_pg", "dom_prev_rank", "dom_prev_n_teams",
                 "ucl_prev1_stage", "ucl_best_stage_prev5", "ucl_main_apps_prev5", "ucl_titles_prev10",
                 "ucl_main_ppg_prev3", "ucl_main_gd_pg_prev3", "ucl_history_seasons",
                 "assoc_ppg_prev5", "assoc_main_teams", "ucl_ss_played", "ucl_ss_ppg", "ucl_ss_gd_pg"]
TEAM_SEASON_TARGETS = ["season_complete", "stage_reached", "stage_score", "teams_remaining", "reached_last16",
                       "reached_qf", "reached_sf", "reached_final", "is_winner"]


def manual_participants(ts: pd.DataFrame) -> pd.DataFrame:
    """Rows from data/manual/ucl_participants.csv for seasons that have no match data yet."""
    path = config.MANUAL / "ucl_participants.csv"
    if not path.exists():
        return ts.iloc[0:0]
    p = pd.read_csv(path, dtype=str, comment="#").dropna(how="all")
    p = p[~p["season"].isin(ts["season"])]
    if p.empty:
        return ts.iloc[0:0]
    p["country"] = p["country"].map(teams.norm_country)
    p["team_id"] = [teams.team_id(c, r) for c, r in zip(p["country"], p["team"])]
    p["season_start"] = p["season"].str[:4].astype(int)
    p["n_main_stage_teams"] = p.groupby("season")["team_id"].transform("size")
    return p.assign(reached_main_stage=True, season_complete=False, entry_round=None)[
        ["season", "season_start", "team_id", "country", "reached_main_stage", "season_complete", "entry_round",
         "n_main_stage_teams"]]


def load_domestic() -> pd.DataFrame:
    """Domestic top-flight matches (from 1955) with team ids and a match_id."""
    d = sources.load_domestic()
    d["home_id"] = [teams.team_id(c, r) for c, r in zip(d["country"], d["home_raw"])]
    d["away_id"] = [teams.team_id(c, r) for c, r in zip(d["country"], d["away_raw"])]
    d["match_id"] = d["league"] + "_" + d["season"] + "_" + \
        (d.groupby(["league", "season"]).cumcount() + 1).map("{:04d}".format)
    return d


def ucl_base(m: pd.DataFrame) -> pd.DataFrame:
    return m.assign(competition="UCL")[BASE_COLUMNS]


def domestic_base(dom: pd.DataFrame) -> pd.DataFrame:
    """Domestic matches in the same layout as UCL matches, from the Champions League era on."""
    d = dom[dom["season_start"] >= config.FIRST_SEASON]
    res = np.select([d["home_goals"] > d["away_goals"], d["home_goals"] < d["away_goals"]], ["H", "A"], "D")
    return pd.DataFrame(dict(
        match_id=d["match_id"], competition=d["league"], season=d["season"], season_start=d["season_start"],
        date=d["date"], phase="domestic", round_code=None, leg=np.nan, tie_id=None, is_main_stage=False,
        neutral=False, covid_no_fans=d["date"].between(*tables.COVID_NO_FANS),
        home_id=d["home_id"], away_id=d["away_id"], home_country=d["country"], away_country=d["country"],
        status=d["status"], home_goals=d["home_goals"], away_goals=d["away_goals"],
        result_90=np.where(d["home_goals"].notna(), res, None), went_to_extra_time=d["aet_home_goals"].notna(),
    ))[BASE_COLUMNS]


def elo_games(m: pd.DataFrame, dom: pd.DataFrame) -> pd.DataFrame:
    """Every match that feeds the Elo: European Cup 1955-1992, UCL, and domestic top flights."""
    cols = ["match_id", "date", "season_start", "home_id", "away_id", "home_country", "away_country",
            "home_goals", "away_goals", "neutral", "status"]
    ec = sources.load_european_cup_history()
    for side in ("home", "away"):
        ec[f"{side}_country"] = ec[f"{side}_country"].map(teams.norm_country)
        ec[f"{side}_id"] = [teams.team_id(c, r) for c, r in zip(ec[f"{side}_country"], ec[f"{side}_raw"])]
    d = dom.assign(home_country=dom["country"], away_country=dom["country"], neutral=False)
    return pd.concat([
        ec[cols].assign(league=np.nan, qualifying=False),
        m[cols].assign(league=np.nan, qualifying=m["phase"] == "qualifying"),
        d[cols].assign(league=d["league"], qualifying=False),
    ], ignore_index=True)


def season_so_far(base: pd.DataFrame, tp: pd.DataFrame) -> pd.DataFrame:
    """This season's UCL main-stage record before each fixture (any competition, played or not)."""
    played = tp[tp["is_main_stage"]].sort_values("date").copy()
    played["gd"] = played["gf"] - played["ga"]
    g = played.groupby(["season", "team_id"])
    played["cum_n"], played["cum_pts"], played["cum_gd"] = g.cumcount() + 1, g["pts"].cumsum(), g["gd"].cumsum()
    left = pd.concat([base[["match_id", "season", "date", "home_id"]].rename(columns={"home_id": "team_id"}),
                      base[["match_id", "season", "date", "away_id"]].rename(columns={"away_id": "team_id"})])
    right = played[["season", "team_id", "date", "cum_n", "cum_pts", "cum_gd"]].rename(columns={"date": "d"})
    j = pd.merge_asof(left.sort_values("date"), right.sort_values("d"), left_on="date", right_on="d",
                      by=["season", "team_id"], allow_exact_matches=False)
    j["ucl_ss_played"] = j["cum_n"].fillna(0)
    j["ucl_ss_ppg"] = j["cum_pts"] / j["cum_n"]
    j["ucl_ss_gd_pg"] = j["cum_gd"] / j["cum_n"]
    return j[["match_id", "team_id", "ucl_ss_played", "ucl_ss_ppg", "ucl_ss_gd_pg"]]


def _merge_prefixed(df, right, keys: dict[str, str], prefix: str) -> pd.DataFrame:
    """Merge `right` into df, prefixing its value columns. keys maps right column -> df column."""
    right = right.rename(columns={c: f"{prefix}{c}" for c in right.columns if c not in keys}).rename(columns=keys)
    return df.merge(right, on=list(keys.values()), how="left")


def attach_side(df, side, ctx) -> pd.DataFrame:
    tid, cc, p = f"{side}_id", f"{side}_country", f"{side}_"
    df = F.attach_domestic_form(df, ctx["games"], tid, p)
    df = F.attach_domestic_prev_season(df, ctx["dtab"], tid, p)
    df = _merge_prefixed(df, ctx["hist"], {"team_id": tid, "season_start": "season_start"}, p)
    df = _merge_prefixed(df, ctx["assoc"], {"country": cc, "season_start": "season_start"}, p)
    if "match_id" in df:
        df = _merge_prefixed(df, ctx["sofar"], {"match_id": "match_id", "team_id": tid}, p)
    return df


def match_features(base, m, ties, ctx) -> pd.DataFrame:
    df = base.merge(ctx["elo"], on="match_id", how="left").reset_index(drop=True)
    df["_row"] = np.arange(len(df))
    for side in ("home", "away"):
        df = attach_side(df, side, ctx)
    df["is_ucl"] = df["competition"] == "UCL"
    df["same_association"] = df["home_country"] == df["away_country"]
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # Second legs: goal difference carried over from the first leg, from this match's home-team view.
    leg1 = m.loc[m["leg"] == 1, ["tie_id", "home_goals", "away_goals"]].set_axis(["tie_id", "l1_h", "l1_a"], axis=1)
    df = df.merge(leg1, on="tie_id", how="left")
    df["first_leg_gd"] = np.where(df["leg"] == 2, df["l1_a"] - df["l1_h"], np.nan)

    winner = df["tie_id"].map(ties.set_index("tie_id")["winner_id"])
    df["home_advanced"] = (winner == df["home_id"]).astype(object).where(winner.notna(), None)
    cols = BASE_COLUMNS[:2] + ["is_ucl"] + BASE_COLUMNS[2:] + \
        ["home_advanced", "same_association", "elo_diff", "first_leg_gd"] + \
        [f"{s}_{f}" for s in ("home", "away") for f in SIDE_FEATURES]
    return df[cols].sort_values(["date", "match_id"]).reset_index(drop=True)


def team_season_features(m, ts, ctx) -> pd.DataFrame:
    """One row per group/league-phase participant with features frozen at the start of the main stage."""
    rows = ts.loc[ts["reached_main_stage"] == True].copy()  # noqa: E712 (object column)
    main = m[m["is_main_stage"]]
    rows["date"] = rows["season"].map(main.groupby("season")["date"].min()).fillna(pd.Timestamp.today().normalize())

    # Elo before the team's first main-stage match; current rating when a season has no fixtures yet.
    first = pd.concat([
        main[["season", "date", "match_id", "home_id"]].set_axis(["season", "date", "match_id", "team_id"], axis=1).assign(h=True),
        main[["season", "date", "match_id", "away_id"]].set_axis(["season", "date", "match_id", "team_id"], axis=1).assign(h=False),
    ]).sort_values(["date", "match_id"]).drop_duplicates(["season", "team_id"]).merge(ctx["elo"], on="match_id")
    first["elo"] = np.where(first["h"], first["home_elo"], first["away_elo"])
    first["elo_n"] = np.where(first["h"], first["home_elo_n"], first["away_elo_n"])
    rows = rows.merge(first[["season", "team_id", "elo", "elo_n"]], on=["season", "team_id"], how="left")
    todo = rows["elo"].isna()
    if todo.any():
        model = ctx["elo_model"]
        rows.loc[todo, "elo"] = [model.get(t, c) for t, c in rows.loc[todo, ["team_id", "country"]].values]
        rows.loc[todo, "elo_n"] = [float(model.n[t]) for t in rows.loc[todo, "team_id"]]

    rows = rows.rename(columns={"team_id": "x_id", "country": "x_country"})
    rows["_row"] = np.arange(len(rows))
    rows = attach_side(rows, "x", ctx).rename(columns=lambda c: c[2:] if c.startswith("x_") else c)
    rows = rows.rename(columns={"id": "team_id"})
    g = rows.groupby("season")["elo"]
    rows["elo_rank_in_season"] = g.rank(ascending=False, method="min")
    rows["elo_minus_season_mean"] = rows["elo"] - g.transform("mean")
    rows["entered_via_qualifying"] = rows["entry_round"].isin(tables.QUALIFYING).astype(object)
    no_quali_data = rows["season_start"].between(config.OPENFOOTBALL_FIRST_SEASON, 2023) | rows["entry_round"].isna()
    rows.loc[no_quali_data, "entered_via_qualifying"] = None
    rows["team"] = rows["team_id"].map(ctx["names"])
    feats = ["elo", "elo_n", "elo_rank_in_season", "elo_minus_season_mean", "entered_via_qualifying",
             "n_main_stage_teams"] + [f for f in SIDE_FEATURES if f not in ("elo", "elo_n") and not f.startswith("ucl_ss")]
    out = rows.rename(columns={"date": "main_stage_start"})
    return out[["season", "season_start", "main_stage_start", "team_id", "team", "country"] + feats + TEAM_SEASON_TARGETS] \
        .sort_values(["season_start", "elo"], ascending=[True, False]).reset_index(drop=True)


def team_table(m: pd.DataFrame, dom: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for side in ("home", "away"):
        parts.append(pd.DataFrame({"country": m[f"{side}_country"], "raw": m[f"{side}_raw"],
                                   "source": np.where(m["source"] == "openfootball", "openfootball_ucl", "engsoccerdata_ucl"),
                                   "season_start": m["season_start"]}))
        parts.append(pd.DataFrame({"country": dom["country"], "raw": dom[f"{side}_raw"],
                                   "source": dom["source"] + "_domestic", "season_start": dom["season_start"]}))
    return teams.build_team_table(pd.concat(parts, ignore_index=True))


def add_names(df: pd.DataFrame, names: dict[str, str], cols: dict[str, str]) -> pd.DataFrame:
    """Insert a display-name column right after each id column."""
    df = df.copy()
    for id_col, name_col in cols.items():
        df.insert(df.columns.get_loc(id_col) + 1, name_col, df[id_col].map(names))
    return df


def validate(m, ties, ts, dom) -> list[str]:
    issues = []
    if (n := (m["status"] == "unparsed").sum()):
        issues.append(f"{n} matches with unparsed scores")
    bad = ties[ties["complete"] & ties["winner_id"].isna()]
    if len(bad):
        issues.append(f"{len(bad)} completed ties without a winner: {bad['tie_id'].tolist()[:5]}")
    dis = ties[ties["source_winner_disagrees"]]
    if len(dis):
        issues.append(f"{len(dis)} ties where the computed winner != engsoccerdata tiewinner: {dis['tie_id'].tolist()[:5]}")
    # Every knockout winner must appear in the season's next round (when that round is in the data).
    order = m.groupby(["season", "round_code"])["date"].min().reset_index().sort_values(["season", "date"])
    order["next_round"] = order.groupby("season")["round_code"].shift(-1)
    seen = set(zip(m["season"], m["round_code"], m["home_id"])) | set(zip(m["season"], m["round_code"], m["away_id"]))
    t = ties[ties["complete"]].merge(order[["season", "round_code", "next_round"]], on=["season", "round_code"])
    miss = [r.tie_id for r in t.itertuples()
            if isinstance(r.next_round, str) and (r.season, r.next_round, r.winner_id) not in seen]
    if miss:
        issues.append(f"{len(miss)} knockout winners missing from the next round: {miss[:5]}")
    losers_on = [r.tie_id for r in t.itertuples()
                 if isinstance(r.next_round, str) and (r.season, r.next_round, r.loser_id) in seen]
    if losers_on:
        issues.append(f"{len(losers_on)} knockout losers appearing in the next round: {losers_on[:5]}")
    champs = ts[ts["is_winner"] == True].groupby("season").size()  # noqa: E712
    wrong = [s for s in ts.loc[ts["season_complete"], "season"].unique() if champs.get(s, 0) != 1]
    if wrong:
        issues.append(f"seasons without exactly one winner: {wrong}")
    if (dup := m.duplicated(["date", "home_id", "away_id"]).sum()):
        issues.append(f"{dup} duplicate fixtures")

    # Domestic: every UCL club from a covered association must be in its own league that season,
    # and consecutive league-seasons must share most clubs (a sudden drop means a naming mismatch).
    league_of = {v[0]: k for k, v in config.DOMESTIC_LEAGUES.items()}
    sets = pd.concat([dom[["league", "season_start", "home_id"]].set_axis(["lg", "s", "t"], axis=1),
                      dom[["league", "season_start", "away_id"]].set_axis(["lg", "s", "t"], axis=1)]) \
        .groupby(["lg", "s"])["t"].agg(set)
    ucl = pd.concat([m[["season_start", "home_id", "home_country"]].set_axis(["s", "t", "cc"], axis=1),
                     m[["season_start", "away_id", "away_country"]].set_axis(["s", "t", "cc"], axis=1)]).drop_duplicates()
    absent = [(s, t) for s, t, cc in ucl.itertuples(index=False)
              if (league_of.get(cc), s) in sets.index and t not in sets[(league_of[cc], s)]]
    if absent:
        issues.append(f"{len(absent)} UCL club-seasons missing from their own domestic league: {absent[:5]}")
    low = [(lg, s) for (lg, s), clubs in sets.items() if s > config.FIRST_SEASON and (lg, s - 1) in sets.index
           and len(clubs & sets[(lg, s - 1)]) < 0.6 * len(clubs)]
    if low:
        issues.append(f"{len(low)} league-seasons with low club carry-over (possible name mismatch): {low[:5]}")
    return issues


def build_all(out_dir=config.PROCESSED) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = tables.build_matches()
    ties = tables.build_ties(m)
    ts = tables.build_team_seasons(m, ties)
    ts_all = pd.concat([ts, manual_participants(ts)], ignore_index=True)
    dom = load_domestic()
    team_rows = team_table(m, dom)
    names = team_rows.drop_duplicates("team_id").set_index("team_id")["team"].to_dict()

    base = pd.concat([ucl_base(m), domestic_base(dom)], ignore_index=True)
    tp = tables.team_perspective(m)
    league_teams = pd.concat([dom[["league", "season_start", "home_id"]].set_axis(["lg", "s", "t"], axis=1),
                              dom[["league", "season_start", "away_id"]].set_axis(["lg", "s", "t"], axis=1)]) \
        .groupby(["lg", "s"])["t"].agg(set).to_dict()
    elo_model = F.Elo(league_teams)
    games = F.domestic_team_games(dom)
    pairs = pd.concat([base[["home_id", "season_start"]].set_axis(["team_id", "season_start"], axis=1),
                       base[["away_id", "season_start"]].set_axis(["team_id", "season_start"], axis=1),
                       ts_all[["team_id", "season_start"]]]).drop_duplicates()
    countries = set(base["home_country"]) | set(base["away_country"]) | set(ts_all["country"])
    seasons = sorted(set(base["season_start"]) | set(ts_all["season_start"]))
    ctx = dict(names=names, elo_model=elo_model, elo=elo_model.run(elo_games(m, dom)), games=games,
               dtab=F.domestic_season_tables(games), hist=F.ucl_history(ts_all, tp, pairs, config.FIRST_SEASON),
               assoc=F.association_strength(tp, ts_all, countries, seasons), sofar=season_so_far(base, tp))
    mf = match_features(base, m, ties, ctx)
    tsf = team_season_features(m, ts_all, ctx)

    pair = {"home_id": "home_team", "away_id": "away_team"}
    mf = add_names(mf, names, pair)
    result = {
        "ucl_matches": add_names(m, names, pair),
        "ucl_ties": add_names(ties, names, {"team_a_id": "team_a", "team_b_id": "team_b", "winner_id": "winner"}),
        "ucl_team_seasons": add_names(ts, names, {"team_id": "team"}),
        "match_features": mf,
        "ucl_match_features": mf[mf["is_ucl"]],
        "ucl_team_season_features": tsf,
        "domestic_matches": add_names(dom, names, pair),
        "domestic_tables": add_names(ctx["dtab"], names, {"team_id": "team"}),
        "teams": team_rows,
    }
    for name, df in result.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    return result | {"_issues": validate(m, ties, ts, dom)}
