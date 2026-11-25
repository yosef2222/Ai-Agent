import os
from urllib.parse import urlsplit

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException, RatelimitException

from agent.results import failure, skill, success


@skill
def web_search(query: str) -> dict:
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
        return failure("Provide a search query between 1 and 500 characters.")
    try:
        with DDGS(proxy=os.environ.get("HTTPS_PROXY"), timeout=30) as search:
            try:
                hits = search.text(query.strip(), backend="api", max_results=3)
            except DuckDuckGoSearchException:
                hits = search.text(query.strip(), backend="html", max_results=3)
        results = []
        for hit in hits:
            url = hit.get("href", "")
            parsed = urlsplit(url)
            if parsed.scheme in {"http", "https"} and parsed.hostname:
                results.append({"title": hit["title"], "url": url, "snippet": hit.get("body", "")})
    except RatelimitException:
        return failure("DuckDuckGo rate-limited this search. Try again later.")
    except DuckDuckGoSearchException:
        return failure("DuckDuckGo search is temporarily unavailable.")
    return success(results[:3])
