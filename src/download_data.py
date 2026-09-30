"""Download / refresh the raw source files into data/raw.

Usage:  python src/download_data.py [--refresh]
Only files whose content changed upstream are re-downloaded unless --refresh is given.
"""
import argparse

from ucl_data import config
from ucl_data.download import sync_github

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-download every file")
    args = ap.parse_args()
    for source in (config.OPENFOOTBALL_UCL, config.ENGSOCCERDATA, config.OPENFOOTBALL_DOMESTIC):
        sync_github(source, refresh=args.refresh)
