"""MCP server-based document retrieval"""

import asyncio
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.retrieval.base import BaseRetriever
from src.schemas import RetrievedDoc


class MCPRetriever(BaseRetriever):
    """Retrieves documentation from LangChain MCP server via HTTP streaming."""
    MCP_SERVER_URL = "https://docs.langchain.com/mcp"

    async def _retrieve_async(self, query: str) -> list[RetrievedDoc]:
        """Connects to MCP server, finds search tool, and executes query asynchronously."""
        print("Retrieving documentation from MCP server...")

        try:
            async with streamablehttp_client(url=self.MCP_SERVER_URL) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    tools_result = await session.list_tools()
                    print(f"  Connected to MCP server")
                    print(f"  Available tools: {len(tools_result.tools)}")

                    search_tool = None
                    for tool in tools_result.tools:
                        if "search" in tool.name.lower() or "query" in tool.name.lower():
                            search_tool = tool
                            break

                    if not search_tool:
                        print(f"  Warning: No search tool found, available tools:")
                        for tool in tools_result.tools:
                            print(f"    - {tool.name}: {tool.description}")

                        if tools_result.tools:
                            search_tool = tools_result.tools[0]
                            print(f"  Using first available tool: {search_tool.name}")
                        else:
                            raise Exception("No tools available from MCP server")

                    print(f"  Using tool: {search_tool.name}")

                    result = await session.call_tool(
                        search_tool.name,
                        arguments={"query": query}
                    )

                    retrieved_docs = self._parse_mcp_results(result)

                    print(f"  Retrieved {len(retrieved_docs)} documents")

                    return retrieved_docs

        except Exception as e:
            print(f"  ✗ MCP retrieval failed: {e}")
            raise Exception(f"Failed to retrieve from MCP server: {e}")

    def _parse_mcp_results(self, result: Any) -> list[RetrievedDoc]:
        """Parses MCP server response into RetrievedDoc objects with fallback handling."""
        docs = []

        if hasattr(result, 'content') and result.content:
            for idx, content_item in enumerate(result.content, 1):
                if hasattr(content_item, 'text'):
                    text_content = content_item.text
                elif hasattr(content_item, 'content'):
                    text_content = content_item.content
                else:
                    text_content = str(content_item)

                doc = RetrievedDoc(content=text_content)
                docs.append(doc)
        else:
            doc = RetrievedDoc(content=str(result))
            docs.append(doc)

        if not docs:
            raise Exception("MCP server returned no results")

        return docs

    def retrieve(self, query: str) -> list[RetrievedDoc]:
        """Synchronous wrapper around async MCP retrieval."""
        return asyncio.run(self._retrieve_async(query))