"""LangGraph based agent implementation"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from src.config import Config, MAX_WEB_RESULTS
from src.retrieval import BaseRetriever, RetrievedDoc
from src.web_search import WebSearcher
from src.llm_utils import create_model, generate_answer
from src.agents.base import BaseAgent
from src.schemas import Answer


class AgentState(TypedDict):
    """Shared state across graph execution"""
    query: str
    docs: list[RetrievedDoc]
    web_results: list[RetrievedDoc]
    result: Answer | None


class LangGraphAgent(BaseAgent):
    """LangGraph-based agent using StateGraph for orchestration"""

    def __init__(self, config: Config, retriever: BaseRetriever):
        self.config = config
        self.retriever = retriever

        self.searcher = None
        if config.web_search_enabled:
            self.searcher = WebSearcher(api_key=config.web_search_api_key)

        self.model = create_model(config.gemini_api_key)
        self.graph = self._build_graph()

    def _retrieve_docs_node(self, state: AgentState) -> dict:
        """Node: Retrieve relevant documentation"""
        docs = self.retriever.retrieve(state["query"])
        print()
        return {"docs": docs}

    def _web_search_node(self, state: AgentState) -> dict:
        """Node: Search web for additional context"""
        print("Searching the web for additional context...")
        web_results = self.searcher.search(state["query"], max_results=MAX_WEB_RESULTS)
        if web_results:
            print(f"  Found {len(web_results)} web results:")
            for i, result in enumerate(web_results, 1):
                print(f"    [{i}] {result.url}")
        else:
            print("  No web results found (continuing with docs only)")
        print()
        return {"web_results": web_results}

    def _generate_answer_node(self, state: AgentState) -> dict:
        """Node: Generate answer using LLM"""
        print("Generating answer...")
        result = generate_answer(
            self.model,
            state["query"],
            state["docs"],
            state.get("web_results")
        )
        return {"result": result}

    def _should_search_web(self, state: AgentState) -> str:
        """Conditional edge: Decide whether to search web"""
        if self.config.web_search_enabled and self.searcher:
            return "web_search"
        return "generate_answer"

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve_docs", self._retrieve_docs_node)
        workflow.add_node("web_search", self._web_search_node)
        workflow.add_node("generate_answer", self._generate_answer_node)

        workflow.add_edge(START, "retrieve_docs")
        workflow.add_conditional_edges(
            "retrieve_docs",
            self._should_search_web,
            {
                "web_search": "web_search",
                "generate_answer": "generate_answer"
            }
        )
        workflow.add_edge("web_search", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def answer(self, query: str) -> str:
        """Main entry point: runs the graph and displays results"""
        print()
        print("=" * 60)
        print("Processing Query")
        print("=" * 60)
        print()

        initial_state = {
            "query": query,
            "docs": [],
            "web_results": [],
            "result": None
        }

        final_state = self.graph.invoke(initial_state)

        result = final_state.get("result")
        docs = final_state["docs"]
        web_results = final_state.get("web_results", [])

        if result:
            self._display_results(result, docs, web_results)
            return result.answer

        return ""
