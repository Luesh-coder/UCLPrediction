"""Dixon-Coles team ratings (Dixon & Coles, 1997): every club gets an attack and a defence strength,
fitted to the goals of past matches with older matches weighted down exponentially.

Goals are Poisson: log(expected home goals) = mu + home advantage + attack[home] - defence[away], and the
same for the away side without the home advantage. Each strength is shrunk towards its club's current
league level, so a club with few matches sits near its league and leagues are calibrated against each
other through European matches. The model is refitted every week on matches strictly before that week,
so a match's ratings never use its own result or later ones (like the Elo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

HALF_LIFE_DAYS = 240  # a match 240 days old counts half as much as one played today (tuned on 2022-23)
WINDOW_DAYS = 1460  # older matches weigh under 2% and are left out
TEAM_SHRINK = 3.0  # ridge penalty pulling a club towards its league level (tuned on 2022-23)
LEAGUE_SHRINK = 1.0
RHO = -0.01  # Dixon-Coles low-score correction (tuned on 2022-23)
MAX_GOALS = 10
STEP_DAYS = 7
NO_LEAGUE = "europe"  # clubs seen only in European matches (association outside the covered leagues)
DC_COLUMNS = ["dc_home_xg", "dc_away_xg", "dc_p_away", "dc_p_draw", "dc_p_home"]


def outcome_probs(lh: np.ndarray, la: np.ndarray, rho: float = RHO) -> np.ndarray:
    """(n, 3) probabilities [away win, draw, home win] from expected goals, with the low-score correction."""
    k = np.arange(MAX_GOALS + 1)
    grid = poisson.pmf(k[None], lh[:, None])[:, :, None] * poisson.pmf(k[None], la[:, None])[:, None, :]
    grid[:, 0, 0] *= 1 - lh * la * rho
    grid[:, 0, 1] *= 1 + lh * rho
    grid[:, 1, 0] *= 1 + la * rho
    grid[:, 1, 1] *= 1 - rho
    grid /= grid.sum(axis=(1, 2), keepdims=True)
    home = np.tril(np.ones((MAX_GOALS + 1,) * 2), -1)  # [home goals, away goals]: home goals > away goals
    return np.stack([(grid * home.T).sum((1, 2)), np.trace(grid, axis1=1, axis2=2), (grid * home).sum((1, 2))], 1)


class DixonColes:
    """Rolling Dixon-Coles over `games`: match_id, date, home_id, away_id, home_goals, away_goals, neutral,
    status and league (NaN for European matches)."""

    def __init__(self, games: pd.DataFrame) -> None:
        g = games.sort_values(["date", "match_id"]).reset_index(drop=True)
        self.teams = pd.Index(pd.unique(pd.concat([g["home_id"], g["away_id"]])))
        self.leagues = pd.Index(sorted(set(g["league"].dropna())) + [NO_LEAGUE])
        self.g = g
        self.h, self.a = self.teams.get_indexer(g["home_id"]), self.teams.get_indexer(g["away_id"])
        self.lg = self.leagues.get_indexer(g["league"].fillna(NO_LEAGUE))
        self.euro = g["league"].isna().to_numpy()
        self.home_on = 1.0 - g["neutral"].astype(bool).to_numpy()
        self.days = (g["date"] - g["date"].min()).dt.days.to_numpy()
        self.played = ((g["status"] == "played") & g["home_goals"].notna() & g["away_goals"].notna()).to_numpy()
        self.yh, self.ya = g["home_goals"].to_numpy(float), g["away_goals"].to_numpy(float)
        T, L = len(self.teams), len(self.leagues)
        self.p = np.zeros(3 + 2 * T + 2 * L)  # mu, home, home_europe, attack[T], defence[T], league att/def[L]
        self.team_league = np.full(T, L - 1)
        self.in_window = np.zeros(T, bool)  # clubs rated by the latest fit; others count as league average

    def _split(self, p):
        T, L = len(self.teams), len(self.leagues)
        return p[0], p[1], p[2], p[3:3 + T], p[3 + T:3 + 2 * T], p[3 + 2 * T:3 + 2 * T + L], p[3 + 2 * T + L:]

    def fit(self, day: int) -> None:
        """Fit on played matches in [day - WINDOW_DAYS, day), warm-started from the previous fit."""
        w_idx = np.flatnonzero(self.played & (self.days < day) & (self.days >= day - WINDOW_DAYS))
        if len(w_idx) == 0:
            return
        T, L = len(self.teams), len(self.leagues)
        # Current league of each club: its latest domestic match in the window.
        dom = w_idx[~self.euro[w_idx]]
        t_all = np.concatenate([self.h[dom], self.a[dom]])
        l_all = np.concatenate([self.lg[dom], self.lg[dom]])
        order = np.argsort(np.concatenate([self.days[dom], self.days[dom]]), kind="stable")[::-1]
        uniq, first = np.unique(t_all[order], return_index=True)
        self.team_league = np.full(T, L - 1)
        self.team_league[uniq] = l_all[order][first]

        # Local indices: only clubs in the window get parameters in this fit.
        clubs, inv = np.unique(np.concatenate([self.h[w_idx], self.a[w_idx]]), return_inverse=True)
        self.in_window = np.zeros(T, bool)
        self.in_window[clubs] = True
        n, C = len(w_idx), len(clubs)
        h, a = inv[:n], inv[n:]
        lh, la = self.team_league[clubs][h], self.team_league[clubs][a]
        w = 0.5 ** ((day - self.days[w_idx]) / HALF_LIFE_DAYS)
        home_on, euro = self.home_on[w_idx], self.euro[w_idx].astype(float)
        yh, ya = self.yh[w_idx], self.ya[w_idx]
        o = np.cumsum([0, 3, C, C, L, L])

        def f(x):
            at, dt, al, dl = x[o[1]:o[2]], x[o[2]:o[3]], x[o[3]:o[4]], x[o[4]:o[5]]
            eh = x[0] + (x[1] + x[2] * euro) * home_on + at[h] + al[lh] - dt[a] - dl[la]
            ea = x[0] + at[a] + al[la] - dt[h] - dl[lh]
            mh, ma = np.exp(eh), np.exp(ea)
            nll = np.sum(w * (mh - yh * eh + ma - ya * ea)) \
                + 0.5 * TEAM_SHRINK * (at @ at + dt @ dt) + 0.5 * LEAGUE_SHRINK * (al @ al + dl @ dl)
            rh, ra = w * (mh - yh), w * (ma - ya)  # d nll / d log-rate
            g = np.empty_like(x)
            g[0] = rh.sum() + ra.sum()
            g[1] = (rh * home_on).sum()
            g[2] = (rh * home_on * euro).sum()
            g[o[1]:o[2]] = np.bincount(h, rh, C) + np.bincount(a, ra, C) + TEAM_SHRINK * at
            g[o[2]:o[3]] = -np.bincount(a, rh, C) - np.bincount(h, ra, C) + TEAM_SHRINK * dt
            g[o[3]:o[4]] = np.bincount(lh, rh, L) + np.bincount(la, ra, L) + LEAGUE_SHRINK * al
            g[o[4]:o[5]] = -np.bincount(la, rh, L) - np.bincount(lh, ra, L) + LEAGUE_SHRINK * dl
            return nll, g

        mu, hm, he, at, dt, al, dl = self._split(self.p)
        x0 = np.concatenate([[mu if mu else np.log(max(ya.mean(), 0.1)), hm, he], at[clubs], dt[clubs], al, dl])
        x = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
        self.p[:3] = x[:3]
        self.p[3:3 + T][clubs] = x[o[1]:o[2]]
        self.p[3 + T:3 + 2 * T][clubs] = x[o[2]:o[3]]
        self.p[3 + 2 * T:] = x[o[3]:]
        self.objective = f  # kept for gradient checks

    def expected_goals(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Expected (home, away) goals for games rows idx under the current fit. A club outside the fit
        window takes the league of the domestic match it plays, if any."""
        mu, hm, he, at, dt, al, dl = self._split(self.p)
        at, dt = at * self.in_window, dt * self.in_window
        h, a = self.h[idx], self.a[idx]
        own = np.where(self.euro[idx], len(self.leagues) - 1, self.lg[idx])
        lh = np.where(self.in_window[h], self.team_league[h], own)
        la = np.where(self.in_window[a], self.team_league[a], own)
        euro, home_on = self.euro[idx].astype(float), self.home_on[idx]
        eh = mu + (hm + he * euro) * home_on + at[h] + al[lh] - dt[a] - dl[la]
        ea = mu + at[a] + al[la] - dt[h] - dl[lh]
        return np.exp(eh), np.exp(ea)


def rolling_ratings(games: pd.DataFrame, target_ids, today: pd.Timestamp, verbose: bool = True) -> pd.DataFrame:
    """Dixon-Coles features for the games whose match_id is in target_ids. Refits every STEP_DAYS on
    matches before each step; fixtures from today's step onwards use the latest fit."""
    dc = DixonColes(games)
    target = dc.g["match_id"].isin(set(target_ids)).to_numpy()
    start = dc.g.loc[target, "date"].min().normalize()
    origin = dc.g["date"].min()
    steps = pd.date_range(start, max(today, start), freq=f"{STEP_DAYS}D")
    rows, fits = [], 0
    for i, d in enumerate(steps):
        last = i == len(steps) - 1
        lo = (d - origin).days
        sel = target & (dc.days >= lo) & ((dc.days < lo + STEP_DAYS) | last)
        if not sel.any():
            continue
        dc.fit(lo)
        fits += 1
        idx = np.flatnonzero(sel)
        lh, la = dc.expected_goals(idx)
        rows.append(pd.DataFrame({"match_id": dc.g["match_id"].to_numpy()[idx], "dc_home_xg": lh, "dc_away_xg": la}))
        if verbose and fits % 250 == 0:
            print(f"  Dixon-Coles: {fits} weekly fits, up to {d.date()}", flush=True)
    out = pd.concat(rows, ignore_index=True)
    probs = outcome_probs(out["dc_home_xg"].to_numpy(), out["dc_away_xg"].to_numpy())
    out["dc_p_away"], out["dc_p_draw"], out["dc_p_home"] = probs[:, 0], probs[:, 1], probs[:, 2]
    return out[["match_id"] + DC_COLUMNS]
