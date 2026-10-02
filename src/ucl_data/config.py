"""Paths and source settings shared by the download and build steps."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MODEL = DATA / "model"
MANUAL = DATA / "manual"

USER_AGENT = "UCLPrediction-data/0.1 (personal, non-commercial research)"

# openfootball/champions-league (CC0): UCL main stage from 2011-12, qualifiers from 2024-25.
OPENFOOTBALL_UCL = {
    "owner": "openfootball",
    "repo": "champions-league",
    "branch": "master",
    "pattern": r"^\d{4}-\d{2}/(cl|clq)\.txt$",
    "dest": RAW / "openfootball" / "champions-league",
}

# Top-flight domestic leagues (league code -> association, name). History comes from
# engsoccerdata, recent seasons from openfootball/football.json (CC0).
DOMESTIC_LEAGUES = {
    "en.1": ("ENG", "Premier League"),
    "es.1": ("ESP", "La Liga"),
    "de.1": ("GER", "Bundesliga"),
    "it.1": ("ITA", "Serie A"),
    "fr.1": ("FRA", "Ligue 1"),
    "nl.1": ("NED", "Eredivisie"),
    "pt.1": ("POR", "Primeira Liga"),
    "be.1": ("BEL", "Pro League"),
    "tr.1": ("TUR", "Super Lig"),
    "gr.1": ("GRE", "Super League"),
    "sco.1": ("SCO", "Premiership"),
    "at.1": ("AUT", "Bundesliga"),
}
OPENFOOTBALL_DOMESTIC = {
    "owner": "openfootball",
    "repo": "football.json",
    "branch": "master",
    "pattern": r"^\d{4}-\d{2}/(" + "|".join(k.replace(".", r"\.") for k in DOMESTIC_LEAGUES) + r")\.json$",
    "dest": RAW / "openfootball" / "football.json",
}

# engsoccerdata (GPL >= 2): European Cup / UCL 1955-56 to 2017-18 (champs.csv; used for 1992-93 to
# 2010-11, earlier seasons only warm up the Elo) and domestic league history up to 2024-25.
ENGSOCCERDATA_LEAGUES = {
    "england": "en.1", "spain": "es.1", "italy": "it.1", "germany": "de.1", "france": "fr.1",
    "holland": "nl.1", "portugal": "pt.1", "belgium": "be.1", "greece": "gr.1", "turkey": "tr.1",
    "scotland": "sco.1",
}
ENGSOCCERDATA = {
    "owner": "jalapic",
    "repo": "engsoccerdata",
    "branch": "master",
    "pattern": r"^data-raw/(champs|" + "|".join(ENGSOCCERDATA_LEAGUES) + r")\.csv$",
    "dest": RAW / "engsoccerdata",
}

FIRST_SEASON = 1992  # Champions League era starts in 1992-93
OPENFOOTBALL_FIRST_SEASON = 2011  # UCL seasons before this come from engsoccerdata
HISTORY_FIRST_SEASON = 1955  # European Cup and domestic history from here warm up the Elo ratings
