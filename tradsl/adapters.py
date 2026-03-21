from abc import ABC, abstractmethod
from typing import Optional


class Adapter(ABC):
    @abstractmethod
    def tick(self) -> Optional[list]:
        """Return a list of values (one per column), or None if no new data."""
        pass
