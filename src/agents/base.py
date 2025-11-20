"""Abstract base class for agent implementations"""

from abc import ABC, abstractmethod
from src.schemas import Answer, RetrievedDoc


class BaseAgent(ABC):
    """Abstract interface for agent implementations."""

    @abstractmethod
    def answer(self, query: str) -> str:
        """Process query and return answer."""
        ...

    def _display_results(
        self,
        result: Answer,
        docs: list[RetrievedDoc],
        web_results: list[RetrievedDoc] | None = None
    ) -> None:
        """Shared display logic for formatting and printing answer with sources."""
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
                if n > len(docs) and n <= len(docs) + len(web_results or [])
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
                    result_item = (web_results or [])[web_idx]
                    print(f"  [{display_num}] {result_item.url}")

                print("\n  Note: Web results may reference deprecated features.")
                print("  Always verify against official documentation.")

            print()
        else:
            print("(No specific sources cited)")
            print()
