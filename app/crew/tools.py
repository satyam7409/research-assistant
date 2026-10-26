from crewai.tools import tool
from tavily import TavilyClient

from app.config import settings


@tool("web_search")
def web_search(query: str) -> str:
    """Search the web. Returns the title, URL and a text excerpt for the top results."""
    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        res = client.search(query=query, max_results=5)
    except Exception as exc:  # noqa: BLE001 - a failed search must not crash the crew
        return f"Search failed: {exc}. Try a different query."

    return "\n\n".join(
        f"TITLE: {r['title']}\nURL: {r['url']}\nEXCERPT: {r['content']}"
        for r in res.get("results", [])
    )
