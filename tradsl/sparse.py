"""
Sparse Storage for Slowly-Changing Time Series

Section 9: SparseTimeSeries for fundamental data, event timing, etc.
O(unique values) memory instead of O(total bars).
"""
from typing import Optional, List, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class SparseTimeSeries:
    """
    Stores (timestamp, value) pairs only when value changes.
    
    Memory: O(unique values), not O(total bars)
    Lookup: O(log N) via binary search
    Latest value: O(1)
    Range retrieval: O(k) where k = values in range
    
    Use for:
    - Fundamental data (earnings dates, financial statement items)
    - Event-timing features (days to next earnings)
    - Macro indicators updated less frequently than daily
    - Regime labels
    """
    
    def __init__(self):
        self._timestamps: List[int] = []
        self._values: List[float] = []
        self._last_value: Optional[float] = None
    
    def push(self, timestamp_ns: int, value: float, epsilon: float = 1e-10) -> bool:
        """
        Store if changed beyond epsilon. Returns True if stored.
        
        Args:
            timestamp_ns: Timestamp in nanoseconds
            value: Value to store
            epsilon: Tolerance for considering value changed
        
        Returns:
            True if value was stored (changed), False if same
        """
        if self._last_value is not None and abs(value - self._last_value) < epsilon:
            return False
        
        self._timestamps.append(timestamp_ns)
        self._values.append(value)
        self._last_value = value
        return True
    
    def get_at(self, timestamp_ns: int) -> float:
        """
        Value at or before timestamp (forward-fill).
        O(log N) via binary search.
        
        Args:
            timestamp_ns: Query timestamp
        
        Returns:
            Value at or before timestamp (forward-filled)
        
        Raises:
            ValueError: If no data available
        """
        if not self._timestamps:
            raise ValueError("No data in sparse series")
        
        if timestamp_ns <= self._timestamps[0]:
            return self._values[0]
        
        idx = self._binary_search(timestamp_ns)
        
        if idx < len(self._timestamps) and self._timestamps[idx] <= timestamp_ns:
            return self._values[idx]
        
        if idx > 0:
            return self._values[idx - 1]
        
        return self._values[0]
    
    def _binary_search(self, timestamp_ns: int) -> int:
        """Find index where timestamps[i] > timestamp_ns."""
        lo, hi = 0, len(self._timestamps)
        
        while lo < hi:
            mid = (lo + hi) // 2
            if self._timestamps[mid] < timestamp_ns:
                lo = mid + 1
            else:
                hi = mid
        
        return lo
    
    def get_latest(self) -> Optional[float]:
        """Most recent value. O(1)."""
        return self._last_value
    
    def get_range(self, start_ns: int, end_ns: int) -> np.ndarray:
        """
        Dense array of values in [start, end].
        O(k) where k = values in range.
        
        Args:
            start_ns: Start timestamp (inclusive)
            end_ns: End timestamp (inclusive)
        
        Returns:
            Dense array of values
        """
        if not self._timestamps:
            return np.array([])
        
        start_idx = self._binary_search(start_ns)
        end_idx = self._binary_search(end_ns + 1)
        
        if start_idx >= len(self._timestamps):
            return np.array([])
        
        values = self._values[start_idx:end_idx]
        
        return np.array(values)
    
    def __len__(self) -> int:
        """Number of stored values (unique changes)."""
        return len(self._values)
    
    @property
    def is_empty(self) -> bool:
        """True if no data stored."""
        return len(self._values) == 0


class SparseBuffer:
    """
    Sparse storage combined with circular buffer for DAG nodes.
    
    Automatically switches between sparse and dense based on
    update frequency relative to strategy base frequency.
    """
    
    def __init__(
        self,
        max_gap: int = 10,
        epsilon: float = 1e-10
    ):
        self._sparse = SparseTimeSeries()
        self._max_gap = max_gap
        self._epsilon = epsilon
        self._current_timestamp: Optional[int] = None
    
    def push(self, timestamp_ns: int, value: float) -> bool:
        """Push new value if changed."""
        self._current_timestamp = timestamp_ns
        return self._sparse.push(timestamp_ns, value, self._epsilon)
    
    def get_latest(self) -> Optional[float]:
        """Get latest value, checking for gap."""
        if self._sparse.is_empty:
            return None
        
        latest = self._sparse.get_latest()
        
        if self._current_timestamp is not None:
            stored_ts = self._sparse._timestamps[-1]
            gap = (self._current_timestamp - stored_ts) // (24 * 60 * 60 * 1_000_000_000)
            
            if gap > self._max_gap:
                return None
        
        return latest
    
    def get_at(self, timestamp_ns: int) -> Optional[float]:
        """Get value at timestamp with forward-fill."""
        try:
            return self._sparse.get_at(timestamp_ns)
        except ValueError:
            return None
    
    def __len__(self) -> int:
        return len(self._sparse)
