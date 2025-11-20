"""Main orchestration logic for question answering"""

import google.generativeai as genai

from src.config import Config
from src.retrieval import BaseRetriever, RetrievedDoc
from src.web_search import WebSearcher
from src.schemas import Answer


class Agent:
    """Orchestrates retrieval, web search, and LLM-based answer generation."""
    MAX_WEB_RESULTS = 3

    ANSWER_GENERATION_PROMPT = """You are a helpful assistant that answers questions about LangGraph and LangChain based on official documentation.

User Question: {query}

{doc_range_info}

Documentation Context:
{context}

Instructions:
- Answer the user's question using ONLY the provided documentation
- Be specific and provide code examples if present in the docs
- If the documentation doesn't fully answer the question, say so
- Format your answer clearly with markdown
- Include relevant code snippets if available
- In sources_used, list ONLY the document numbers you actually referenced in your answer
- Do not include documents you didn't use, even if they were provided
- If you reference deprecated features found in web results, explicitly note they are deprecated

Return your response as JSON with:
- "answer": your complete answer in markdown
- "sources_used": array of document numbers that you actually used
"""

    def __init__(self, config: Config, retriever: BaseRetriever):
        self.config = config
        self.retriever = retriever

        self.searcher = None
        if config.web_search_enabled:
            self.searcher = WebSearcher(api_key=config.web_search_api_key)

        genai.configure(api_key=config.gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def _format_context(self, docs: list[RetrievedDoc], web_results: list[RetrievedDoc] | None = None) -> str:
        """Combines local docs and web results into numbered, formatted context for LLM."""
        all_docs = docs + (web_results if web_results else [])
        context_parts = []

        for i, doc in enumerate(all_docs, 1):
            # Mark web results
            if web_results and i > len(docs):
                context_parts.append(f"[Document {i}: Web Search Result]")
            else:
                context_parts.append(f"[Document {i}]")

            if doc.url:
                context_parts.append(f"URL: {doc.url}")

            context_parts.append(f"\n{doc.content}\n")
            context_parts.append("=" * 80)

        return "\n\n".join(context_parts)

    def _generate_answer(self, query: str, context: str, num_local_docs: int, num_web_results: int) -> Answer:
        """Generates structured answer using LLM with context-aware prompting based on source types."""
        if num_web_results > 0:
            doc_range_info = """
Document Types:
- Local documentation (official, current, authoritative)
- Web search results (may include outdated content)

IMPORTANT: Prioritize local documentation over web search results.
If web results contradict local documentation, trust the local docs and note the discrepancy.
If web results mention deprecated features, explicitly identify them as deprecated.

When referencing sources in your answer:
- Use generic terms like "the official documentation" or "local documentation" rather than document number ranges
- Only use specific document numbers in the sources_used field, not in your answer text
"""
        else:
            doc_range_info = "All documents provided are official local documentation."

        prompt = self.ANSWER_GENERATION_PROMPT.format(
            query=query,
            doc_range_info=doc_range_info,
            context=context
        )

        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=Answer.model_json_schema()
            )
        )

        return Answer.model_validate_json(response.text)

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
            try:
                web_results = self.searcher.search(query, max_results=self.MAX_WEB_RESULTS)
                if web_results:
                    print(f"  Found {len(web_results)} web results:")
                    for i, result in enumerate(web_results, 1):
                        print(f"    [{i}] {result.url}")
                else:
                    print("  No web results found (continuing with docs only)")
            except Exception as e:
                print(f"  Warning: Web search failed: {e}")
                print("  Continuing with local documentation only...")
            print()

        print("Generating answer...")
        context = self._format_context(docs, web_results if web_results else None)

        result = self._generate_answer(query, context, len(docs), len(web_results))

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
            web_sources = [n for n in unique_sources if n > len(docs) and n <= len(docs) + len(web_results)]

            if local_sources:
                print("\nOfficial Documentation:")

                # Separate sources with URLs from MCP results
                url_sources = [(display_num, doc_num) for display_num, doc_num in enumerate(local_sources, 1) if docs[doc_num - 1].url]
                mcp_sources = [(display_num, doc_num) for display_num, doc_num in enumerate(local_sources, 1) if not docs[doc_num - 1].url]

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

        return result.answer