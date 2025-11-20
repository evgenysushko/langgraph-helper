"""Abstract base class for document retrievers"""

from abc import ABC, abstractmethod
from src.schemas import RetrievedDoc


class BaseRetriever(ABC):
    """Abstract interface for document retrieval implementations."""
    @abstractmethod
    def retrieve(self, query: str) -> list[RetrievedDoc]:
        """Retrieves relevant documents for a query."""
        ...