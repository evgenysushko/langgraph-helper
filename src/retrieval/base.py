"""Abstract base class for document retrievers"""

from abc import ABC, abstractmethod
from typing import NamedTuple


class RetrievedDoc(NamedTuple):
    """Retrieved documentation with content and optional source URL."""
    content: str
    url: str | None = None


class BaseRetriever(ABC):
    """Abstract interface for document retrieval implementations."""
    @abstractmethod
    def retrieve(self, query: str) -> list[RetrievedDoc]:
        """Retrieves relevant documents for a query."""
        ...