from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class Function(ABC):
    @abstractmethod
    def apply(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply function to DataFrame. Handle missing data internally."""
        pass
