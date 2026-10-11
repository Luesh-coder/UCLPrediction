"""Team-name canonicalisation across sources.

Each (country, raw name) pair is mapped to a stable team_id in two steps:
1. `data/manual/team_aliases.csv` and the bundled `known_aliases.csv` rewrite known variants
   ("Internazionale" -> "Inter").
2. The (possibly rewritten) name is normalised (accents, punctuation and generic
   affixes such as FC/CF/SK/04 removed) and grouped by that key within the country.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import config

# Clubs that play in another association's league or changed code between sources.
COUNTRY_FIXES = {"MCO": "FRA"}
# Spellings that differ between the bundled sources (e.g. engsoccerdata vs openfootball), checked club by club.
KNOWN_ALIASES = Path(__file__).with_name("known_aliases.csv")

_TRANSLIT = str.maketrans({"ø": "o", "Ø": "O", "ß": "ss", "æ": "ae", "Æ": "AE", "ð": "d", "þ": "th",
                           "ı": "i", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ə": "a", "Ə": "A"})
_GENERIC = {
    "fc", "cf", "afc", "sc", "ac", "ssc", "ss", "sk", "fk", "kv", "bc", "cd", "ud", "rc", "rcd", "sv", "bv",
    "tsg", "vfl", "vfb", "club", "clube", "de", "da", "do", "del", "della", "the", "of", "and", "e", "nk",
    "hnk", "gnk", "pfc", "pfk", "kf", "ks", "mfk", "ofk", "bk", "fck", "as", "sd", "ae", "acf", "ogc", "osc",
    "hsc", "rsc", "krc", "royal", "royale", "calcio", "football", "futbol", "cp", "sl", "pae", "sfp", "osfp",
    "jk", "cs", "csm", "sad", "if", "ff", "ifk", "us", "ca", "cfr", "kaa",
}


def norm_country(cc: str) -> str:
    return COUNTRY_FIXES.get(cc, cc)


def norm_key(name: str) -> str:
    s = unicodedata.normalize("NFKD", name.translate(_TRANSLIT))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    tokens = re.sub(r"[^a-z0-9]+", " ", s).split()
    kept = [t for t in tokens if t not in _GENERIC and not t.isdigit()]
    return " ".join(kept) if kept else " ".join(tokens)


def _read_aliases(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False, comment="#")
    return {(norm_country(r.country), r.alias): r.canonical for r in df.itertuples()}


@lru_cache(maxsize=1)
def load_aliases() -> dict[tuple[str, str], str]:
    """Aliases shipped with the code (KNOWN_ALIASES), overridden by data/manual/team_aliases.csv. A shipped
    alias is skipped when the manual file mentions that name at all, so the two files cannot form a loop."""
    manual = _read_aliases(config.MANUAL / "team_aliases.csv")
    named = set(manual) | {(cc, canonical) for (cc, _), canonical in manual.items()}
    return {k: v for k, v in _read_aliases(KNOWN_ALIASES).items() if k not in named} | manual


def team_key(country: str, raw: str) -> tuple[str, str]:
    """(association, normalised key) for a raw team name."""
    cc = norm_country(country)
    aliases, seen = load_aliases(), set()
    while (cc, raw) in aliases and raw not in seen:  # follow chains such as AEK -> AEK Athen -> AEK Athens
        seen.add(raw)
        raw = aliases[(cc, raw)]
    return cc, norm_key(raw)


@lru_cache(maxsize=None)
def team_id(country: str, raw: str) -> str:
    cc, key = team_key(country, raw)
    return f"{cc}_{key.replace(' ', '-')}"


def build_team_table(names: pd.DataFrame) -> pd.DataFrame:
    """names: columns country, raw, source, season_start. Returns one row per (country, raw).

    The display name of a team is its most recent openfootball spelling (UCL first, then other UEFA
    competitions, then domestic), falling back to engsoccerdata (UCL, then domestic).
    """
    df = names.copy()
    df["country"] = df["country"].map(norm_country)
    df["team_id"] = [team_id(c, r) for c, r in zip(df["country"], df["raw"])]
    pref = {"openfootball_ucl": 0, "openfootball_uefa": 1, "openfootball_domestic": 2, "engsoccerdata_ucl": 3,
            "engsoccerdata_domestic": 4}
    df["_pref"] = df["source"].map(pref)
    display = (df.sort_values(["_pref", "season_start"], ascending=[True, False])
                 .drop_duplicates("team_id")
                 .set_index("team_id")["raw"])
    agg = (df.groupby(["country", "raw", "source"], as_index=False)
             .agg(team_id=("team_id", "first"), first_season=("season_start", "min"),
                  last_season=("season_start", "max")))
    agg["team"] = agg["team_id"].map(display)
    return agg[["team_id", "team", "country", "raw", "source", "first_season", "last_season"]]
