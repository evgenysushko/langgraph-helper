"""Simple agent with direct orchestration"""

from src.config import Config, MAX_WEB_RESULTS
from src.retrieval import BaseRetriever, RetrievedDoc
from src.web_search import WebSearcher
from src.llm_utils import create_model, generate_answer
from src.agents.base import BaseAgent


class SimpleAgent(BaseAgent):
    """Orchestrates retrieval, web search, and LLM-based answer generation."""

    def __init__(self, config: Config, retriever: BaseRetriever):
        self.config = config
        self.retriever = retriever

        self.searcher = None
        if config.web_search_enabled:
            self.searcher = WebSearcher(api_key=config.web_search_api_key)

        self.model = create_model(config.gemini_api_key)

    def answer(self, query: str) -> str:
        """Main entry point: retrieves docs, searches web if enabled, generates and displays answer."""
        print()
        print("=" * 60)
        print("Processing Query")
        print("=" * 60)
        print()

        docs = self.retriever.retrieve(query)
        print()

        web_results = []
        if self.config.web_search_enabled and self.searcher:
            print("Searching the web for additional context...")
            web_results = self.searcher.search(query, max_results=MAX_WEB_RESULTS)
            if web_results:
                print(f"  Found {len(web_results)} web results:")
                for i, result in enumerate(web_results, 1):
                    print(f"    [{i}] {result.url}")
            else:
                print("  No web results found (continuing with docs only)")
            print()

        print("Generating answer...")
        result = generate_answer(
            self.model,
            query,
            docs,
            web_results if web_results else None
        )

        self._display_results(result, docs, web_results)

        return result.answer