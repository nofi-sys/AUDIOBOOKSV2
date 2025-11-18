from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib import error, parse, request

DEFAULT_LANGUAGES: Sequence[str] = ("es", "en")
USER_AGENT = "txt2md-mvp/1.0 (+https://example.com)"


def _request_json(url: str, *, timeout: int = 10) -> Optional[Dict[str, Any]]:
    req = request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            data = resp.read()
    except (error.HTTPError, error.URLError, TimeoutError):
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except json.JSONDecodeError:
        return None


def _build_summary(entry: Dict[str, Any], lang: str, original: str, source_title: Optional[str] = None) -> Dict[str, str]:
    content_urls = entry.get("content_urls") or {}
    desktop = content_urls.get("desktop") or {}
    return {
        "title": entry.get("title") or source_title or original,
        "lang": lang,
        "extract": entry.get("extract") or "",
        "description": entry.get("description") or "",
        "url": desktop.get("page") or "",
        "source_url": entry.get("source") or "",
        "source_title": source_title or entry.get("title") or "",
        "search_term": original,
    }


def _fetch_summary_for_lang(term: str, lang: str, *, timeout: int = 10) -> Optional[Dict[str, str]]:
    if not term:
        return None

    params = {
        'action': 'query',
        'format': 'json',
        'titles': term,
        'prop': 'extracts',
        'exintro': True,
        'explaintext': True,
    }
    url = f"https://{lang}.wikipedia.org/w/api.php?{parse.urlencode(params)}"

    data = _request_json(url, timeout=timeout)
    if not data:
        return None

    page = next(iter(data['query']['pages'].values()))
    extract = page.get('extract', '').strip()

    if not extract:
        return None

    # Simplified build_summary logic since we don't have all the fields from the summary API
    return {
        "title": page.get("title") or term,
        "lang": lang,
        "extract": extract,
        "description": "",  # Not available in this API
        "url": f"https://{lang}.wikipedia.org/wiki/{parse.quote(term.replace(' ', '_'))}",
        "source_url": url,
        "source_title": term,
        "search_term": term,
    }


def search_wikipedia(term: str, *, lang: str = "es", limit: int = 3, timeout: int = 10) -> List[str]:
    params = {
        "action": "opensearch",
        "format": "json",
        "limit": str(limit),
        "search": term,
    }
    url = f"https://{lang}.wikipedia.org/w/api.php?{parse.urlencode(params)}"
    data = _request_json(url, timeout=timeout)
    if not isinstance(data, list) or len(data) < 2:
        return []
    titles = data[1]
    if not isinstance(titles, list):
        return []
    return [str(item) for item in titles if isinstance(item, str)]


def fetch_wikipedia_summary(
    title: str,
    *,
    languages: Optional[Sequence[str]] = None,
    allow_search: bool = True,
    timeout: int = 10,
) -> Optional[Dict[str, str]]:
    if not title:
        return None
    langs: Iterable[str] = languages or DEFAULT_LANGUAGES
    tried_terms: List[str] = []
    for lang in langs:
        summary = _fetch_summary_for_lang(title, lang, timeout=timeout)
        if summary:
            return summary
        if allow_search:
            suggestions = search_wikipedia(title, lang=lang, limit=3, timeout=timeout)
            for candidate in suggestions:
                if candidate in tried_terms:
                    continue
                tried_terms.append(candidate)
                summary = _fetch_summary_for_lang(candidate, lang, timeout=timeout)
                if summary:
                    summary["source_title"] = candidate
                    summary["search_term"] = title
                    return summary
    return None
