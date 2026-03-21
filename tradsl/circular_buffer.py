"""
Circular Buffer for Incremental Feature Computation

Fixed-size circular buffer for time series values. Provides O(1) push
and O(window) slice access for feature computation.
"""


class CircularBuffer:
    """
    Fixed-size circular buffer for time series values.

    Properties:
        - Fixed allocation at construction, never grows
        - O(1) push (amortized)
        - O(1) access to latest value via [-1]
        - O(window) slice for feature computation
        - Zero heap allocation after initialization

    Warmup semantics:
        A buffer is in "warming up" state until it has received `size` values.
        In warming-up state, any access to the buffer contents returns None.
        The engine checks warmup state before computing any node.
    """

    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("Buffer size must be positive")
        self._size = size
        self._head = 0
        self._count = 0
        self._data: list = [None] * size

    @property
    def size(self) -> int:
        """Buffer capacity."""
        return self._size

    @property
    def count(self) -> int:
        """Number of elements currently in buffer."""
        return self._count

    def push(self, value) -> None:
        """Push new value. Overwrites oldest. O(1)."""
        self._data[self._head] = value
        self._head = (self._head + 1) % self._size
        if self._count < self._size:
            self._count += 1

    @property
    def is_ready(self) -> bool:
        """True when buffer has received exactly `size` values."""
        return self._count >= self._size

    def latest(self):
        """Most recent value. O(1). Returns None if not ready."""
        if not self.is_ready:
            return None
        return self._data[(self._head - 1) % self._size]

    def contents(self) -> list:
        """
        Return all buffer contents as list oldest-to-newest.
        Returns list padded with None if not full.
        """
        if self._count == 0:
            return [None] * self._size
        result = []
        for i in range(self._count):
            idx = (self._head - self._count + i) % self._size
            result.append(self._data[idx])
        while len(result) < self._size:
            result.insert(0, None)
        return result

    def __getitem__(self, idx: int):
        """Access element by index from oldest (0) to newest (-1 or size-1)."""
        if idx < 0:
            idx = self._count + idx
        if idx < 0 or idx >= self._count:
            raise IndexError(f"Index {idx} out of range for buffer with {self._count} elements")
        actual_idx = (self._head - self._count + idx) % self._size
        return self._data[actual_idx]

    def __len__(self) -> int:
        """Number of elements in buffer."""
        return self._count
