import json
import os
import re
from html import unescape
from urllib.parse import quote, urlsplit

import requests
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException, RatelimitException

from agent.results import failure, skill, success

MWMBL_URL = "https://api.mwmbl.org/api/v1/search/"
WIKTIONARY_URL = "https://en.wiktionary.org"
USER_AGENT = "PersonalAI-Assistant/1.0 (https://github.com/yosef2222/Ai-Agent)"
_TAGS = re.compile(r"<[^>]+>")
_WORDS = re.compile(r"[^\W_]+")
_GENERIC_WORDS = {
    "a", "about", "an", "and", "as", "at", "by", "current", "for", "from", "in", "information", "is", "latest",
    "news", "of", "official", "on", "or", "results", "search", "site", "the", "to", "today", "web", "website", "with",
}
_TERM = r"[a-zA-Z][a-zA-Z0-9]*(?:[-'][a-zA-Z0-9]+)*"
_QUOTED_TERM = rf"[\"'‘“]?(?P<term>{_TERM})[\"'’”]?"
_LABEL = r"(?:(?:the\s+)?(?:word|term|abbreviation|acronym|initialism)\s+)?"
_END = r"(?:\s+(?:in|on)\s+.+)?[.?!]*$"
_DEFINITION_QUERIES = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    rf"\b(?:meaning|definition)\s+of\s+{_LABEL}{_QUOTED_TERM}{_END}",
    rf"\bdefine\s+{_LABEL}{_QUOTED_TERM}{_END}",
    rf"\bwhat\s+(?:does\s+)?{_LABEL}{_QUOTED_TERM}\s+means?\b",
    rf"^{_QUOTED_TERM}\s+(?:(?:abbreviation|acronym|slang)\s+)?(?:meaning|definition)\b",
))
_CASE_ALIAS = re.compile(rf"Alternative letter-case form of ({_TERM})\.", re.IGNORECASE)


def _plain(text):
    return " ".join(_TAGS.sub("", unescape(text or "")).split())


def _clean_results(hits, terms):
    results = []
    seen = set()
    for hit in hits:
        url = hit.get("href") or ""
        try:
            parsed = urlsplit(url)
        except ValueError:
            continue
        title = _plain(hit.get("title"))
        if title and parsed.scheme in {"http", "https"} and parsed.hostname and url not in seen:
            snippet = _plain(hit.get("body"))
            if terms and terms.isdisjoint(_WORDS.findall(f"{title} {url} {snippet}".casefold())):
                continue
            results.append({"title": title[:250], "url": url, "snippet": snippet[:600]})
            seen.add(url)
        if len(results) == 3:
            break
    return results


def _get_json(url, params=None):
    with requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=(5, 12),
                      allow_redirects=False, stream=True) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("Unexpected search response status.")
        data = bytearray()
        for chunk in response.iter_content(chunk_size=16384):
            data.extend(chunk)
            if len(data) > 512 * 1024:
                raise ValueError("Search response exceeded the size limit.")
    return json.loads(data)


def _segments(data):
    if not isinstance(data, list):
        return ""
    return "".join(segment["value"] for segment in data
                   if isinstance(segment, dict) and isinstance(segment.get("value"), str))


def _mwmbl(query):
    hits = _get_json(MWMBL_URL, {"s": query})
    if not isinstance(hits, list):
        raise ValueError("Invalid Mwmbl search response.")
    return [{"title": _segments(hit.get("title")), "href": hit["url"], "body": _segments(hit.get("extract"))}
            for hit in hits if isinstance(hit, dict) and isinstance(hit.get("url"), str)]


def _english_definitions(term):
    data = _get_json(f"{WIKTIONARY_URL}/api/rest_v1/page/definition/{quote(term, safe='')}")
    if not isinstance(data, dict) or not isinstance(data.get("en", []), list):
        raise ValueError("Invalid Wiktionary response.")
    return [definition for entry in data.get("en", []) if entry.get("language") == "English"
            for sense in entry.get("definitions", []) if (definition := _plain(sense.get("definition")))]


def _lookup_definition(term):
    definitions = _english_definitions(term)
    alias = _CASE_ALIAS.fullmatch(definitions[0]) if definitions else None
    if alias and alias[1] != term and alias[1].casefold() == term.casefold():
        term = alias[1]
        definitions = _english_definitions(term)
    if not definitions:
        return []
    return [{"title": f"{term} — Wiktionary", "url": f"{WIKTIONARY_URL}/wiki/{quote(term, safe='')}",
             "snippet": " ".join(dict.fromkeys(definitions[:3]))[:600]}]


@skill
def web_search(query: str) -> dict:
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
        return failure("Provide a search query between 1 and 500 characters.")
    query = query.strip()
    terms = set(_WORDS.findall(query.casefold())) - _GENERIC_WORDS
    fallbacks = os.environ.get("WEB_SEARCH_FALLBACK", os.environ.get("BING_SEARCH_FALLBACK", "1")) != "0"
    if fallbacks:
        for pattern in _DEFINITION_QUERIES:
            match = pattern.search(query)
            if match and len(match["term"]) <= 64:
                try:
                    results = _lookup_definition(match["term"])
                    if results:
                        return success(results)
                except (requests.RequestException, ValueError):
                    pass
                break
    duckduckgo_error = "DuckDuckGo returned no usable results."
    try:
        with DDGS(proxy=os.environ.get("HTTPS_PROXY"), timeout=8) as search:
            results = _clean_results(search.text(query, backend="api", max_results=3), terms)
        if results:
            return success(results)
    except RatelimitException:
        duckduckgo_error = "DuckDuckGo rate-limited this search."
    except DuckDuckGoSearchException:
        duckduckgo_error = "DuckDuckGo search is temporarily unavailable."
    if not fallbacks:
        return failure(duckduckgo_error, {"unavailable": True})
    try:
        results = _clean_results(_mwmbl(query), terms)
    except (requests.RequestException, ValueError):
        return failure(f"{duckduckgo_error} Mwmbl search could not load results either.", {"unavailable": True})
    if not results:
        return failure("No relevant links found. Try a more specific query that names the subject and its context.")
    return success(results)
