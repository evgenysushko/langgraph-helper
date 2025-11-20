"""Map-based retrieval using llms.txt"""

import time
from pathlib import Path
from urllib.parse import urlparse

import google.generativeai as genai
import requests

from src.retrieval.base import BaseRetriever
from src.schemas import DocumentSelection, RetrievedDoc


class MapRetriever(BaseRetriever):
    """Retrieves docs using LLM-based selection from llms.txt documentation map."""
    LLMS_TXT_URL = "https://docs.langchain.com/llms.txt"
    REQUEST_TIMEOUT = 10
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_INITIAL_DELAY = 2.0

    DOC_SELECTION_PROMPT = """You are a helpful assistant that selects the most relevant documentation for answering user queries about LangGraph and LangChain.

Given the following resource map (llms.txt) and a user query, select 2-3 most relevant documentation URLs that would help answer the query.

Resource Map:
{llms_txt_content}

User Query: {query}

Return your response as a JSON object with exactly this structure:
{{
  "urls": ["url1", "url2", "url3"],
  "reasoning": "brief explanation of why these documents were selected"
}}

IMPORTANT: Return ONLY the JSON object, no other text. Use exactly "urls" (not "selected_urls" or anything else) and "reasoning" as keys.
"""

    def __init__(self, llms_txt_path: Path, docs_dir: Path, api_key: str, fetch_live: bool = False):
        self.llms_txt_path = llms_txt_path
        self.docs_dir = docs_dir
        self.fetch_live = fetch_live

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

        self.llms_txt_content = self._load_llms_txt()

    def _load_llms_txt(self) -> str:
        """Loads llms.txt from URL if fetch_live=True, otherwise from local cache."""
        if self.fetch_live:
            try:
                print(f"  Fetching llms.txt from {self.LLMS_TXT_URL}...")
                response = requests.get(self.LLMS_TXT_URL, timeout=self.REQUEST_TIMEOUT)
                response.raise_for_status()
                content = response.text

                self.llms_txt_path.parent.mkdir(parents=True, exist_ok=True)
                self.llms_txt_path.write_text(content, encoding="utf-8")
                print(f"  ✓ Cached llms.txt locally")

                return content

            except Exception as e:
                print(f"  ⚠ Failed to fetch llms.txt online: {e}")
                if self.llms_txt_path.exists():
                    print(f"  ↪ Using local llms.txt as fallback")
                    return self.llms_txt_path.read_text(encoding="utf-8")
                else:
                    raise FileNotFoundError(
                        f"Cannot fetch llms.txt online and no local file at {self.llms_txt_path}"
                    )
        else:
            if not self.llms_txt_path.exists():
                raise FileNotFoundError(f"llms.txt not found at {self.llms_txt_path}")
            return self.llms_txt_path.read_text(encoding="utf-8")

    def _call_llm_with_retry(
        self,
        prompt: str,
        max_retries: int = None,
        initial_delay: float = None
    ) -> DocumentSelection:
        """Calls LLM with exponential backoff retry logic for document selection."""
        if max_retries is None:
            max_retries = self.DEFAULT_MAX_RETRIES
        if initial_delay is None:
            initial_delay = self.DEFAULT_INITIAL_DELAY

        delay = initial_delay

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema={
                            "type": "object",
                            "properties": {
                                "urls": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of 2-3 most relevant documentation URLs"
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Why these documents were selected"
                                }
                            },
                            "required": ["urls", "reasoning"]
                        },
                        temperature=0
                    )
                )

                return DocumentSelection.model_validate_json(response.text)

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"  Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(f"  ✗ LLM call failed after {max_retries} attempts")
                    raise

    def _url_to_relative_path(self, url: str) -> str:
        """Extracts relative path from URL (e.g., 'https://.../oss/python/intro.md' -> 'oss/python/intro.md')."""
        parsed = urlparse(url)
        return parsed.path.lstrip('/')

    def _fetch_doc(self, url: str) -> RetrievedDoc | None:
        """Fetches document from URL in online mode or reads from cache in offline mode."""
        relative_path = self._url_to_relative_path(url)
        file_path = self.docs_dir / relative_path

        if self.fetch_live:
            try:
                if not url.startswith("http"):
                    full_url = f"https://docs.langchain.com/{url}"
                else:
                    full_url = url

                print(f"  Fetching {relative_path} from {full_url}...")
                response = requests.get(full_url, timeout=self.REQUEST_TIMEOUT)
                response.raise_for_status()
                content = response.text

                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")
                print(f"  ✓ Cached {relative_path} locally")

                return RetrievedDoc(content=content, url=url)

            except Exception as e:
                print(f"  ⚠ Failed to fetch {relative_path} online: {e}")
                if file_path.exists():
                    print(f"  ↪ Using local {relative_path} as fallback")
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        return RetrievedDoc(content=content, url=url)
                    except Exception as read_error:
                        print(f"  ⚠ Error reading local {relative_path}: {read_error}")
                        return None
                else:
                    print(f"  ✗ No local fallback available for {relative_path}")
                    return None
        else:
            if not file_path.exists():
                print(f"  ⚠ Document not found locally: {relative_path}")
                return None

            try:
                content = file_path.read_text(encoding="utf-8")
                return RetrievedDoc(content=content, url=url)
            except Exception as e:
                print(f"  ⚠ Error reading {relative_path}: {e}")
                return None

    def retrieve(self, query: str) -> list[RetrievedDoc]:
        """Retrieves documents by using LLM to select URLs from llms.txt, then fetching content."""
        print("Retrieving relevant documentation...")

        prompt = self.DOC_SELECTION_PROMPT.format(
            llms_txt_content=self.llms_txt_content,
            query=query
        )

        selection = self._call_llm_with_retry(prompt)

        print(f"  Selected {len(selection.urls)} documents")
        print(f"Reasoning: {selection.reasoning}")
        print()

        retrieved_docs = []
        for url in selection.urls:
            doc = self._fetch_doc(url)
            if doc:
                retrieved_docs.append(doc)
                if not self.fetch_live:
                    relative_path = self._url_to_relative_path(url)
                    print(f"  ✓ Retrieved: {relative_path}")

        if not retrieved_docs:
            raise Exception("No documents could be retrieved")

        return retrieved_docs
