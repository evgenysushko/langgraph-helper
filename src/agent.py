"""Agent for orchestrating document retrieval, web search, and answer generation"""

from src.config import Config
from src.retrieval import BaseRetriever
from src.schemas import Answer, RetrievedDoc
from src.web_search import WebSearcher
from src.llm_utils import create_model, generate_answer


class Agent:
    """Orchestrates retrieval, web search, and LLM-based answer generation."""

    def __init__(self, config: Config, retriever: BaseRetriever):
        """Initialize agent with configuration and retriever."""
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
        if self.searcher:
            print("Searching the web for additional context...")
            web_results = self.searcher.search(query, max_results=Config.MAX_WEB_RESULTS)
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

    def _display_results(
        self,
        result: Answer,
        docs: list[RetrievedDoc],
        web_results: list[RetrievedDoc]
    ) -> None:
        """Display answer and sources with proper formatting."""
        print()
        print("=" * 60)
        print("Answer")
        print("=" * 60)
        print()
        print(result.answer)
        print()

        print("=" * 60)
        print("Sources Used")
        print("=" * 60)

        if result.sources_used:
            unique_sources = sorted(set(result.sources_used))

            local_sources = [n for n in unique_sources if 1 <= n <= len(docs)]
            web_sources = [
                n for n in unique_sources
                if n > len(docs) and n <= len(docs) + len(web_results)
            ]

            if local_sources:
                print("\nOfficial Documentation:")

                # Separate sources with URLs from MCP results
                url_sources = [
                    (display_num, doc_num)
                    for display_num, doc_num in enumerate(local_sources, 1)
                    if docs[doc_num - 1].url
                ]
                mcp_sources = [
                    (display_num, doc_num)
                    for display_num, doc_num in enumerate(local_sources, 1)
                    if not docs[doc_num - 1].url
                ]

                # Display URL-based sources
                for display_num, doc_num in url_sources:
                    doc = docs[doc_num - 1]
                    print(f"  [{display_num}] {doc.url}")

                # Display MCP sources as summary
                if mcp_sources:
                    print(f"  {len(mcp_sources)} document{'' if len(mcp_sources) == 1 else 's'} from MCP search")

            if web_sources:
                print("\nWeb Sources (may include outdated information):")
                for display_num, doc_num in enumerate(web_sources, 1):
                    web_idx = doc_num - len(docs) - 1
                    result_item = web_results[web_idx]
                    print(f"  [{display_num}] {result_item.url}")

                print("\n  Note: Web results may reference deprecated features.")
                print("  Always verify against official documentation.")

            print()
        else:
            print("(No specific sources cited)")
            print()
