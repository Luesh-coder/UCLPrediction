"""Parser for openfootball Football.TXT competition files (e.g. champions-league/2025-26/cl.txt).

Handles round headers (`▪ Group A`, `▪ League, Matchday 3`, `▪ Finals, Round of 16`),
date lines with an optional year, optional kick-off times, and the score notations
`2-1 (1-0)`, `0-0`, `3-2 a.e.t. (3-0, 1-0)` and `4-3 pen. 1-1 a.e.t. (1-1, 0-1)`.
"""
from __future__ import annotations

import re
import warnings
from datetime import date
from pathlib import Path

MONTHS = {m: i for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}
WEEKDAYS = {d: i for i, d in enumerate("Mon Tue Wed Thu Fri Sat Sun".split())}

HEADER_RE = re.compile(r"^▪\s*(?P<round>.+?)\s*$")
DATE_RE = re.compile(
    r"^\s*(?P<wd>Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})(?:\s+(?P<year>\d{4}))?\s*$"
)
MATCH_RE = re.compile(
    r"^\s*(?:(?P<time>\d{1,2}[:.]\d{2})\s+)?"
    r"(?P<home>.+?)\s+\((?P<hc>[A-Z]{3})\)\s+v\s+(?P<away>.+?)\s+\((?P<ac>[A-Z]{3})\)"
    r"\s*(?P<rest>.*?)\s*$"
)
_S = r"(\d+)\s*[-–]\s*(\d+)"
PEN_RE = re.compile(rf"^{_S}\s+pen\.\s+{_S}\s+a\.e\.t\.\s*(?:\(\s*{_S}(?:\s*,\s*{_S})?\s*\))?$")
AET_RE = re.compile(rf"^{_S}\s+a\.e\.t\.\s*(?:\(\s*{_S}(?:\s*,\s*{_S})?\s*\))?$")
FT_RE = re.compile(rf"^{_S}\s*(?:\(\s*{_S}\s*\))?$")


def _pair(groups: tuple, i: int) -> tuple[int | None, int | None]:
    a, b = groups[i], groups[i + 1]
    return (int(a), int(b)) if a is not None else (None, None)


def parse_score(rest: str) -> dict:
    """Split a score string into 90-minute, half-time, extra-time and penalty scores."""
    rest = re.split(r"\s+@\s+", rest)[0].strip()
    out = dict(ft=(None, None), ht=(None, None), aet=(None, None), pen=(None, None), status="played")
    if not rest:
        out["status"] = "scheduled"
    elif m := PEN_RE.match(rest):
        g = m.groups()
        out.update(pen=_pair(g, 0), aet=_pair(g, 2), ft=_pair(g, 4), ht=_pair(g, 6))
    elif m := AET_RE.match(rest):
        g = m.groups()
        out.update(aet=_pair(g, 0), ft=_pair(g, 2), ht=_pair(g, 4))
    elif m := FT_RE.match(rest):
        g = m.groups()
        out.update(ft=_pair(g, 0), ht=_pair(g, 2))
    else:
        out["status"] = "unparsed"
    if out["status"] == "played" and out["ft"] == (None, None):
        out["status"] = "unparsed"  # e.g. 'a.e.t.' without the 90-minute score
    if out["ft"] == (0, 0) and out["ht"] == (None, None):
        out["ht"] = (0, 0)  # the format omits the half-time score for goalless games
    return out


def _resolve_date(wd: str, mon: str, day: int, year: int | None, season_start: int, last_year: int | None) -> date:
    if year is not None:
        return date(year, MONTHS[mon], day)
    # No explicit year: the weekday pins it down between the season's two calendar years.
    candidates = [y for y in (season_start, season_start + 1) if date(y, MONTHS[mon], day).weekday() == WEEKDAYS[wd]]
    if len(candidates) == 1:
        return date(candidates[0], MONTHS[mon], day)
    fallback = last_year if last_year is not None else (season_start if MONTHS[mon] >= 6 else season_start + 1)
    return date(fallback, MONTHS[mon], day)


def parse_file(path: Path, season_start: int) -> list[dict]:
    """Parse one competition file into match dicts (one per fixture line)."""
    matches: list[dict] = []
    round_name = None
    cur_date: date | None = None
    last_year: int | None = None
    cur_time = None
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith(("=", "#")) or line.lstrip().startswith(("(", "[")):
            continue
        if m := HEADER_RE.match(line):
            round_name = m["round"]
            cur_time = None
            continue
        if m := DATE_RE.match(line):
            year = int(m["year"]) if m["year"] else None
            cur_date = _resolve_date(m["wd"], m["mon"], int(m["day"]), year, season_start, last_year)
            last_year = cur_date.year
            cur_time = None
            continue
        if m := MATCH_RE.match(line):
            if m["time"]:
                cur_time = m["time"].replace(".", ":")
            s = parse_score(m["rest"])
            if s["status"] == "unparsed":
                warnings.warn(f"{path}:{lineno}: could not parse score {m['rest']!r}")
            matches.append(
                dict(
                    date=cur_date, time=cur_time, round_raw=round_name,
                    home_raw=m["home"].strip(), home_country=m["hc"],
                    away_raw=m["away"].strip(), away_country=m["ac"],
                    home_goals=s["ft"][0], away_goals=s["ft"][1],
                    ht_home_goals=s["ht"][0], ht_away_goals=s["ht"][1],
                    aet_home_goals=s["aet"][0], aet_away_goals=s["aet"][1],
                    pen_home=s["pen"][0], pen_away=s["pen"][1],
                    status=s["status"], source_line=f"{path.parent.name}/{path.name}:{lineno}",
                )
            )
            continue
        if re.match(r"^\s*(?:Group|Gruppe)\s+[A-H]\s*\|", line):
            continue  # group roster line
        warnings.warn(f"{path}:{lineno}: unrecognised line {line!r}")
    return matches
