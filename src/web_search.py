"""Web search integration using Tavily API"""

from tavily import TavilyClient
from src.schemas import RetrievedDoc


class WebSearcher:
    """Tavily-powered web search with result formatting."""
    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 3) -> list[RetrievedDoc]:
        """Searches the web and returns results as RetrievedDoc instances."""
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=False,
                include_raw_content=False,
                days=180,
                exclude_domains=["youtube.com", "youtu.be"],
            )

            results = []
            for result in response.get("results", []):
                results.append(RetrievedDoc(
                    content=result.get("content", ""),
                    url=result.get("url", "")
                ))

            return results

        except Exception as e:
            print(f"Warning: Web search failed: {e}")
            print("Continuing with local documentation only...")
            return []
