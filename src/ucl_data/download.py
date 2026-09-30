"""Download raw data files from openly licensed sources.

GitHub-hosted sources are synced by git blob SHA, so re-running only fetches files
that changed upstream (e.g. the in-progress season).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

from . import config

_session = requests.Session()
_session.headers["User-Agent"] = config.USER_AGENT


def _get(url: str, retries: int = 4, timeout: int = 60) -> requests.Response:
    """GET with exponential backoff on rate limits, server errors and timeouts."""
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            r = _session.get(url, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            status = getattr(e.response, "status_code", None) if isinstance(e, requests.HTTPError) else None
            if attempt == retries or (status is not None and 400 <= status < 500 and status != 429):
                raise
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def sync_github(source: dict, refresh: bool = False) -> list[Path]:
    """Mirror files matching source['pattern'] from a GitHub repo into source['dest']."""
    owner, repo, branch = source["owner"], source["repo"], source["branch"]
    dest: Path = source["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    tree = _get(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1").json()
    if tree.get("truncated"):
        raise RuntimeError(f"{owner}/{repo}: git tree listing truncated")
    pattern = re.compile(source["pattern"])
    blobs = [b for b in tree["tree"] if b["type"] == "blob" and pattern.match(b["path"])]

    changed = []
    for blob in blobs:
        out = dest / blob["path"]
        if not refresh and out.exists() and manifest.get(blob["path"]) == blob["sha"]:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        r = _get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{blob['path']}")
        out.write_bytes(r.content)
        manifest[blob["path"]] = blob["sha"]
        changed.append(out)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"{owner}/{repo}: {len(blobs)} files tracked, {len(changed)} downloaded/updated")
    return changed

