"""Pluggable fetchers that resolve spec/datasheet text for a BOM part.

Two concrete fetchers are provided:

- LocalFileFetcher: reads pre-existing text files from a directory
  (<part_number>.txt). Reliable, offline, the recommended path if you
  already have datasheets saved locally.
- WebSearchFetcher: best-effort, no-API-key web search + page/PDF text
  extraction. Search markup and site structure change often and some
  hosts may be blocked by network policy, so treat failures as expected.

CompositeFetcher chains fetchers in priority order, caches the first
successful result to disk, and falls back to the BOM description text so
the pipeline never breaks on a missing datasheet.

To use a real parts-data provider (Octopart, Digi-Key, Mouser, ...),
implement a class with the same `fetch(part_number, description) -> str |
None` method and pass it into CompositeFetcher ahead of WebSearchFetcher.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Protocol

import requests

logger = logging.getLogger(__name__)


class DatasheetFetcher(Protocol):
    def fetch(self, part_number: str, description: str = "") -> Optional[str]: ...


def _normalize(part_number: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", part_number).upper()


@dataclasses.dataclass
class LocalFileFetcher:
    """Reads pre-existing datasheet/spec text from a local directory.

    Matches <directory>/<part_number>.txt, tolerant of case and
    punctuation differences in the filename (e.g. "RC0805FR-071KL.txt"
    matches part number "RC0805FR071KL").
    """

    directory: Path

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)

    def fetch(self, part_number: str, description: str = "") -> Optional[str]:
        if not self.directory.is_dir():
            return None
        target = _normalize(part_number)
        for path in self.directory.glob("*.txt"):
            if _normalize(path.stem) == target:
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    return text
        return None


@dataclasses.dataclass
class WebSearchFetcher:
    """Best-effort fetch of datasheet/spec text via web search + extraction.

    No API key required: searches "<part_number> datasheet", fetches the
    top result(s), and extracts text from HTML or PDF. This is inherently
    fragile -- always pair it with a fallback (see CompositeFetcher).
    """

    search_url: str = "https://html.duckduckgo.com/html/"
    user_agent: str = "Mozilla/5.0 (compatible; bom-alt-ranker/1.0)"
    timeout: float = 15.0
    request_delay: float = 1.0
    max_results: int = 5
    max_chars: int = 20000

    def _search(self, query: str) -> List[str]:
        try:
            resp = requests.post(
                self.search_url,
                data={"q": query},
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("web search failed for %r: %s", query, exc)
            return []

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(resp.text, "html.parser")
        urls = [a.get("href") for a in soup.select("a.result__a") if a.get("href")]
        return urls[: self.max_results]

    def _extract_pdf_text(self, content: bytes) -> Optional[str]:
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            pages = [page.extract_text() or "" for page in reader.pages[:10]]
            text = " ".join(" ".join(pages).split())
            return text or None
        except Exception as exc:
            logger.warning("PDF text extraction failed: %s", exc)
            return None

    def _extract_text(self, url: str) -> Optional[str]:
        try:
            resp = requests.get(
                url, headers={"User-Agent": self.user_agent}, timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("fetch failed for %s: %s", url, exc)
            return None

        content_type = resp.headers.get("Content-Type", "")
        if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
            return self._extract_pdf_text(resp.content)

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text or None

    def fetch(self, part_number: str, description: str = "") -> Optional[str]:
        for url in self._search(f"{part_number} datasheet"):
            time.sleep(self.request_delay)
            text = self._extract_text(url)
            if text:
                return text[: self.max_chars]
        return None


@dataclasses.dataclass
class CompositeFetcher:
    """Tries fetchers in order, caches the first hit to disk, and falls
    back to the BOM description text if every fetcher comes up empty."""

    fetchers: List[DatasheetFetcher]
    cache_dir: Path

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, part_number: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", part_number)
        return self.cache_dir / f"{safe}.txt"

    def fetch(self, part_number: str, description: str = "", refresh: bool = False) -> str:
        cache_path = self._cache_path(part_number)
        if not refresh and cache_path.exists():
            cached = cache_path.read_text(encoding="utf-8", errors="ignore").strip()
            if cached:
                return cached

        for fetcher in self.fetchers:
            try:
                text = fetcher.fetch(part_number, description)
            except Exception as exc:
                logger.warning("%s failed for %s: %s", type(fetcher).__name__, part_number, exc)
                text = None
            if text:
                cache_path.write_text(text, encoding="utf-8")
                return text

        logger.warning("no datasheet text found for %s; falling back to BOM description", part_number)
        fallback = description.strip()
        if fallback:
            cache_path.write_text(fallback, encoding="utf-8")
        return fallback
