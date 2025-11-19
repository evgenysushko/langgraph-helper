"""Pydantic schemas for structured LLM outputs"""

from pydantic import BaseModel, Field


class DocumentSelection(BaseModel):
    """LLM output schema for selecting relevant documentation URLs."""
    urls: list[str] = Field(description="List of 2-3 most relevant documentation URLs")
    reasoning: str = Field(description="Why these documents were selected")


class Answer(BaseModel):
    """LLM output schema for final answer with source attribution."""
    answer: str = Field(description="Comprehensive answer in markdown format")
    sources_used: list[int] = Field(description="Document numbers (1-indexed) actually used")