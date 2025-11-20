# LangGraph Helper Agent

AI agent that helps developers work with LangGraph and LangChain documentation. 

## Setup Instructions

### 1. Prerequisites:

- Python 3.12 or higher
- uv package manager: `pip install uv`

### 2. API Keys:

Google Gemini (required):
- Get free API key: https://aistudio.google.com/app/apikey
- Rate limits: https://ai.google.dev/gemini-api/docs/rate-limits

Tavily (optional, for --web-search in online mode):
- Get free API key: https://tavily.com
- Free tier: 1,000 searches/month

### 3. Installation:

```bash
# Clone repository
git clone https://github.com/evgenysushko/langgraph-helper.git
cd langgraph-helper

# Install dependencies
uv sync

# Copy example env file and add your API keys
cp .env.example .env
# Then edit .env to add your actual keys
```

### 4. Download Documentation:

For offline mode, pre-download documentation:

```bash
uv run download_docs.py
```

This downloads llms.txt and all referenced markdown files to `data/docs/`. Creates `data/download_log.txt` for troubleshooting. Not required if using online mode - it auto-caches on first run.

### Example run:

```bash
uv run main.py "How do I add persistence to a LangGraph agent?"
```

### Expected output:

```
Processing Query
=============================================================

Retrieving relevant documentation...
  Selected 3 documents
Reasoning: [LLM explains why these docs were chosen]

Generating answer...

=============================================================
Answer
=============================================================

[Detailed markdown answer with code examples]

=============================================================
Sources Used
=============================================================

Official Documentation:
  [1] https://docs.langchain.com/...
  [2] https://docs.langchain.com/...
```

## Architecture Overview

### Design approach:

Single-turn Q&A architecture with straightforward orchestration. Each query goes through retrieval, optional web search, and answer generation.

### State management:

Each query is processed independently. The agent retrieves fresh documentation context for every question. No conversation history, no persistent state between queries. 

### Orchestration flow:

```
Query → Retrieval (Map/MCP) → [Optional Web Search] → LLM Answer Generation → Display
```

1. *Retrieval* - Two implementations: MapRetriever (LLM selects docs from `llms.txt`) and MCPRetriever (queries LangChain's MCP server)
2. *Web search* (optional) - Tavily API enhancement when `--web-search` flag is enabled
3. *Answer generation* - Gemini LLM with JSON schemas for structured output and source attribution
4. *Display* - Formats answer with source separation (local docs vs web results)

## Operating Modes

### Offline Mode (default):

- Uses locally cached documentation from `data/docs/`.
- The LLM analyzes the local llms.txt map, selects 2-3 relevant URLs, and fetches markdown files from cache.
- No internet required except for Gemini API calls.

Local data used:
- `llms.txt` - Documentation map
- Markdown files in `data/docs/` - Full documentation content

Retrieval options:
- *Map-based* (default) - LLM picks documents from `llms.txt`, fetches from cache

### Online Mode:

- Fetches live documentation from `docs.langchain.com` and auto-caches to `data/docs/` for future offline use.
- Falls back to cache if live fetch fails.
- This mode ensures you have the latest documentation.

Retrieval options:
- *Map-based* (default) - LLM picks documents from `llms.txt`, fetches from live URLs
- *MCP* - Connects to LangChain's MCP server at https://docs.langchain.com/mcp, queries directly

## Retrieval Strategy

Two Implementations:

### 1. Map-based URL Selection (default)

Offline Mode:
1. Loads llms.txt from local file
2. LLM receives query + full `llms.txt` content
3. LLM uses structured output to return 2-3 most relevant URLs
4. Fetches markdown files from local `data/docs/` directory
5. Adds to context for answer generation

Online Mode:
1. Fetches `llms.txt` from https://docs.langchain.com/llms.txt
2. Caches `llms.txt` locally for future offline use
3. LLM receives query + `llms.txt` content
4. LLM uses structured output to return 2-3 most relevant URLs
5. Fetches markdown files from their URLs
6. Caches fetched .md files to `data/docs/` for future offline use
7. Graceful fallback: If online fetch fails, uses local files if available
8. Adds to context for answer generation

### 2. MCP-based Retrieval (online mode only)

1. Connects to LangChain's official MCP server at `https://docs.langchain.com/mcp`
2. Uses `langchain-mcp-adapters` library with streamable HTTP transport
3. Loads MCP tools that search live documentation
4. Queries MCP server with user's question
5. Receives relevant documentation directly from MCP server

MCP provides direct access to LangChain's documentation infrastructure. No `llms.txt` parsing needed - just query the server and get relevant docs. Only works in online mode since it requires network access.

## Web Search (online mode only):

- Web search supplements official documentation with community content via Tavily API.
- Only works in online mode (requires `--mode online --web-search`).
- The agent retrieves official docs first, then searches the web, then combines results with clear source separation. The LLM is instructed to prioritize official docs over web sources and flag deprecated features.
- Why Tavily: Reliable free tier (1,000 searches/month) and 6-month recency filtering to reduce outdated content.

## Data Freshness Strategy

Download approach:
- Pre-download all docs upfront during setup
- Refresh local files regularly: re-run download script

What is being downloaded:
- `llms.txt` (resource map) from https://docs.langchain.com/llms.txt
- `llms-full.txt` (as reference, not used in current implementation)
- All individual markdown files listed in `llms.txt`

Download logging:
- Script automatically creates `data/download_log.txt` with timestamp
- Logs summary (successful/failed downloads)
- Lists any failed URLs for troubleshooting
- Helps diagnose issues when docs are missing or outdated

Online Mode:
- MAP retrieval method fetches live documentation from `docs.langchain.com` and auto-caches to `data/docs/`
- MCP retrieval method provides direct access to LangChain's documentation infrastructure

## Usage Examples

### Command-line flags:

* `--mode {offline|online}` - Operating mode (default: offline)
* `--retrieval {map|mcp}` - Retrieval method (default: map)
* `--web-search` - Enable web search (requires online mode)

### Examples:

Simplest usage (offline, map-based):

```bash
uv run main.py "How do I add persistence?"
```

Online mode with live docs:

```bash
uv run main.py --mode online "How do I add persistence?"
```

Online with web search:

```bash
uv run main.py --mode online --web-search "How do I add persistence?"
```

MCP retrieval:

```bash
uv run main.py --mode online --retrieval mcp "How do I add persistence?"
```

MCP retrieval with web search:

```bash
uv run main.py --mode online --retrieval mcp --web-search "What's new in LangGraph?"
```

### Example questions:

* "How do I add persistence to a LangGraph agent?"
* "What's the difference between StateGraph and MessageGraph?"
* "Show me how to implement human-in-the-loop with LangGraph"
* "How do I handle errors and retries in LangGraph nodes?"
* "What are best practices for state management in LangGraph?"
