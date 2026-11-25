import os
import re
from html import unescape
from urllib.parse import urlsplit

import requests
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException, RatelimitException
from lxml import etree

from agent.results import failure, skill, success

BING_RSS_URL = "https://www.bing.com/search"
_TAGS = re.compile(r"<[^>]+>")
_WORDS = re.compile(r"[^\W_]+")
_GENERIC_WORDS = {
    "a", "about", "an", "and", "as", "at", "by", "current", "for", "from", "in", "information", "is", "latest",
    "news", "of", "official", "on", "or", "results", "search", "site", "the", "to", "today", "web", "website", "with",
}


def _clean_results(hits, terms):
    results = []
    seen = set()
    for hit in hits:
        url = hit.get("href") or ""
        try:
            parsed = urlsplit(url)
        except ValueError:
            continue
        title = _TAGS.sub("", unescape(hit.get("title") or "")).strip()
        if title and parsed.scheme in {"http", "https"} and parsed.hostname and url not in seen:
            snippet = _TAGS.sub("", unescape(hit.get("body") or "")).strip()
            if terms and terms.isdisjoint(_WORDS.findall(f"{title} {url} {snippet}".casefold())):
                continue
            results.append({"title": title, "url": url, "snippet": snippet})
            seen.add(url)
        if len(results) == 3:
            break
    return results


def _bing_rss(query):
    with requests.get(BING_RSS_URL, params={"format": "rss", "q": query}, timeout=(5, 12),
                      allow_redirects=False, stream=True) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("Bing search redirected the request.")
        data = bytearray()
        for chunk in response.iter_content(chunk_size=16384):
            data.extend(chunk)
            if len(data) > 512 * 1024:
                raise ValueError("Bing search response exceeded the size limit.")
    root = etree.fromstring(bytes(data), parser=etree.XMLParser(resolve_entities=False, no_network=True))
    if root.tag != "rss" or root.find("channel") is None:
        raise ValueError("Bing did not return an RSS search feed.")
    return [{"title": item.findtext("title"), "href": item.findtext("link"), "body": item.findtext("description")}
            for item in root.findall("channel/item")]


@skill
def web_search(query: str) -> dict:
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
        return failure("Provide a search query between 1 and 500 characters.")
    query = query.strip()
    terms = set(_WORDS.findall(query.casefold())) - _GENERIC_WORDS
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
    if os.environ.get("BING_SEARCH_FALLBACK", "1") == "0":
        return failure(duckduckgo_error, {"unavailable": True})
    try:
        results = _clean_results(_bing_rss(query), terms)
    except (requests.RequestException, etree.XMLSyntaxError, ValueError):
        return failure(f"{duckduckgo_error} Bing search could not load results either.", {"unavailable": True})
    if not results:
        return failure("No relevant links found. Try a more specific query that names the subject and its context.")
    return success(results)
