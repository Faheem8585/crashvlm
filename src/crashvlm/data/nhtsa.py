"""Thin, cached client for the public NHTSA Vehicle Crash Test Database API.

Every JSON response is cached on disk, so re-running the catalog build is free and the exact
API state a run used can be archived next to the results.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_ROOT = "https://nrd.api.nhtsa.dot.gov/nhtsa/vehicle/api/v1"
TESTS = "vehicle-database-test-results"


def _cache_name(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path).strip("_") + ".json"


class NHTSAClient:
    def __init__(self, cache_dir: str | Path, timeout: float = 60.0, retries: int = 5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        retry = Retry(total=retries, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
        self.session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))

    def get_json(self, path: str, use_cache: bool = True) -> dict[str, Any]:
        """GET ``API_ROOT/path`` and cache the decoded JSON."""
        cache = self.cache_dir / _cache_name(path)
        if use_cache and cache.exists():
            return json.loads(cache.read_text())
        resp = self.session.get(f"{API_ROOT}/{path}", timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        cache.write_text(json.dumps(data))
        return data

    def get_all(self, path: str) -> list[dict[str, Any]]:
        """Follow the API's page-number pagination and concatenate ``results``."""
        out: list[dict[str, Any]] = []
        page = 0
        while True:
            sep = "&" if "?" in path else "?"
            data = self.get_json(f"{path}{sep}pageNumber={page}")
            results = data.get("results") or []
            out.extend(results)
            total = ((data.get("meta") or {}).get("pagination") or {}).get("total") or 0
            if not results or len(out) >= total:
                return out
            page += 1

    # --- endpoints ---------------------------------------------------------------------------

    def tests_page(self, page: int) -> dict[str, Any]:
        return self.get_json(f"{TESTS}?pageNumber={page}")

    def all_tests(self, workers: int = 8) -> list[dict[str, Any]]:
        """All test summaries. The API ignores filters and pageSize (20 rows/page), so we crawl."""
        first = self.tests_page(0)
        total = first["meta"]["pagination"]["total"]
        per_page = max(1, first["meta"]["pagination"]["count"])
        n_pages = -(-total // per_page)
        with ThreadPoolExecutor(workers) as ex:
            pages = list(ex.map(self.tests_page, range(n_pages)))
        return [r for p in pages for r in (p.get("results") or [])]

    def multimedia(self, test_no: int) -> dict[str, Any]:
        res = self.get_json(f"{TESTS}/get-multimedia-files/{test_no}").get("results") or [{}]
        return res[0] or {}

    def vehicles(self, test_no: int) -> list[dict[str, Any]]:
        return self.get_all(f"{TESTS}/get-vehicle-info/{test_no}")

    def occupants(self, test_no: int) -> list[dict[str, Any]]:
        return self.get_all(f"{TESTS}/get-occupant-info/{test_no}")

    def map(self, fn, items: Iterable, workers: int = 8) -> list:
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(fn, items))
