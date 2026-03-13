"""Tests for CircularBuffer."""
import numpy as np
import pytest
from tradsl.circular_buffer import CircularBuffer


class TestCircularBuffer:
    def test_basic_push_and_access(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)

        assert buf.count == 3
        assert buf.is_ready is True
        assert buf.latest() == 3.0

    def test_overwrite_behavior(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)
        buf.push(4.0)

        assert buf.count == 3
        assert buf.latest() == 4.0

    def test_to_array(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)

        arr = buf.to_array()
        assert arr is not None
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_not_ready_returns_none(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)

        assert buf.is_ready is False
        assert buf.latest() is None
        assert buf.to_array() is None

    def test_index_access(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)

        assert buf[0] == 1.0
        assert buf[1] == 2.0
        assert buf[2] == 3.0
        assert buf[-1] == 3.0
        assert buf[-2] == 2.0

    def test_index_out_of_range(self):
        buf = CircularBuffer(3)
        buf.push(1.0)
        buf.push(2.0)

        with pytest.raises(IndexError):
            _ = buf[2]

    def test_len(self):
        buf = CircularBuffer(5)
        assert len(buf) == 0

        buf.push(1.0)
        buf.push(2.0)
        assert len(buf) == 2

    def test_float_dtype(self):
        buf = CircularBuffer(3, dtype=np.float32)
        buf.push(1.5)
        buf.push(2.5)
        buf.push(3.5)

        arr = buf.to_array()
        assert arr is not None
        assert arr.dtype == np.float32

    def test_negative_size_raises(self):
        with pytest.raises(ValueError):
            CircularBuffer(0)

        with pytest.raises(ValueError):
            CircularBuffer(-1)
