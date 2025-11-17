"""Base class for search tools."""

from abc import ABC, abstractmethod
from typing import Any


class SearchTool(ABC):
    """Base class for search tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Execute the search and return results."""
        pass

    def __repr__(self):
        return f"{self.name}: {self.description}"

