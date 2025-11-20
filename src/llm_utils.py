"""Shared LLM utilities"""

import google.generativeai as genai

from src.config import GEMINI_MODEL
from src.schemas import Answer, RetrievedDoc


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


def create_model(api_key: str) -> genai.GenerativeModel:
    """Initialize and configure Gemini model"""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


def format_context(docs: list[RetrievedDoc], web_results: list[RetrievedDoc] | None = None) -> str:
    """Formats retrieved docs and web results into numbered LLM context"""
    all_docs = docs + (web_results if web_results else [])
    context_parts = []

    for i, doc in enumerate(all_docs, 1):
        if web_results and i > len(docs):
            context_parts.append(f"[Document {i}: Web Search Result]")
        else:
            context_parts.append(f"[Document {i}]")

        if doc.url:
            context_parts.append(f"URL: {doc.url}")

        context_parts.append(f"\n{doc.content}\n")
        context_parts.append("=" * 80)

    return "\n\n".join(context_parts)


def generate_answer(
    model: genai.GenerativeModel,
    query: str,
    docs: list[RetrievedDoc],
    web_results: list[RetrievedDoc] | None = None
) -> Answer:
    """Generates structured answer using LLM with retrieved context"""
    num_web_results = len(web_results) if web_results else 0

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

    context = format_context(docs, web_results)
    prompt = ANSWER_GENERATION_PROMPT.format(
        query=query,
        doc_range_info=doc_range_info,
        context=context
    )

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Comprehensive answer in markdown format"
                    },
                    "sources_used": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Document numbers (1-indexed) actually used"
                    }
                },
                "required": ["answer", "sources_used"]
            }
        )
    )

    return Answer.model_validate_json(response.text)
