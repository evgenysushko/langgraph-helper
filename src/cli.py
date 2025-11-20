"""Command-line interface and argument parsing"""

import argparse
import sys
from src.config import Config, Mode, RetrievalMethod

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangGraph Helper Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run main.py "How do I add persistence?"
    → Uses defaults: offline, map retrieval, simple agent, no web search

  uv run main.py --mode online "How do I add persistence?"
    → Online mode with map retrieval (fetches live docs)

  uv run main.py --mode online --retrieval mcp "How do I handle errors?"
    → Online mode with MCP server (live docs from LangChain)

  uv run main.py --mode online --web-search "How do I add persistence?"
    → Online mode with map retrieval + web search enhancement

  uv run main.py --agent-type langgraph "How do I add persistence?"
    → Use LangGraph-based agent implementation

Defaults:
  --mode offline       Local docs only
  --retrieval map      Map-based retrieval using llms.txt
  --agent-type simple  Direct orchestration
  No --web-search      Web search disabled

Environment Variables:
  GEMINI_API_KEY       - Google Gemini API key (required)
  WEB_SEARCH_API_KEY   - Web search API key (only if using --web-search)
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="offline",
        choices=["offline", "online"],
        help="Operating mode: offline (local docs) or online (live docs). Default: offline"
    )

    parser.add_argument(
        "--retrieval",
        type=str,
        default="map",
        choices=["map", "mcp"],
        help="Retrieval method: map (llms.txt) or mcp (MCP server). Default: map"
    )

    parser.add_argument(
        "--web-search",
        action="store_true",
        help="Enable web search enhancement (requires WEB_SEARCH_API_KEY)"
    )

    parser.add_argument(
        "--agent-type",
        type=str,
        default="simple",
        choices=["simple", "langgraph"],
        help="Agent implementation: simple (direct) or langgraph (graph-based). Default: simple"
    )

    parser.add_argument(
        "query",
        type=str,
        help="Your question about LangGraph or LangChain"
    )

    return parser.parse_args()


def main() -> None:
    """Orchestrates CLI argument parsing, config validation, and agent execution."""
    try:
        args = parse_args()
        config = Config()

        config.mode = Mode[args.mode.upper()]
        config.retrieval_method = RetrievalMethod[args.retrieval.upper()]
        config.web_search_enabled = args.web_search

        config.validate()

        print(f"Mode: {config.mode.value}")
        print(f"Retrieval: {config.retrieval_method.value}")
        if config.web_search_enabled:
            print(f"Web Search: enabled")
        print(f"Agent: {args.agent_type}")
        print(f"Query: {args.query}")

        if config.retrieval_method == RetrievalMethod.MAP:
            from src.retrieval import MapRetriever
            fetch_live = config.mode == Mode.ONLINE
            retriever = MapRetriever(
                llms_txt_path=config.llms_txt_path,
                docs_dir=config.docs_dir,
                api_key=config.gemini_api_key,
                fetch_live=fetch_live
            )
        elif config.retrieval_method == RetrievalMethod.MCP:
            from src.retrieval import MCPRetriever
            retriever = MCPRetriever()

        if args.agent_type == "langgraph":
            from src.agents import LangGraphAgent
            agent = LangGraphAgent(config, retriever)
        else:
            from src.agents import SimpleAgent
            agent = SimpleAgent(config, retriever)

        agent.answer(args.query)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
