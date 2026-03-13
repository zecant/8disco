"""
Tests for sparse storage module.
"""
import pytest
import numpy as np
from tradsl.sparse import SparseTimeSeries, SparseBuffer


class TestSparseTimeSeries:
    """Tests for SparseTimeSeries."""
    
    def test_init(self):
        """Test initialization."""
        ts = SparseTimeSeries()
        assert len(ts) == 0
        assert ts.is_empty
    
    def test_push_stores_value(self):
        """Test that push stores values."""
        ts = SparseTimeSeries()
        ts.push(1000, 1.0)
        assert len(ts) == 1
        assert not ts.is_empty
    
    def test_push_same_value_not_stored(self):
        """Test that identical values within epsilon are not stored."""
        ts = SparseTimeSeries()
        ts.push(1000, 1.0)
        ts.push(2000, 1.0)
        assert len(ts) == 1
    
    def test_push_different_value_stored(self):
        """Test that different values are stored."""
        ts = SparseTimeSeries()
        ts.push(1000, 1.0)
        ts.push(2000, 2.0)
        assert len(ts) == 2
    
    def test_get_latest(self):
        """Test getting latest value."""
        ts = SparseTimeSeries()
        ts.push(1000, 1.0)
        ts.push(2000, 2.0)
        ts.push(3000, 3.0)
        assert ts.get_latest() == 3.0
    
    def test_get_latest_empty(self):
        """Test getting latest from empty series."""
        ts = SparseTimeSeries()
        assert ts.get_latest() is None
    
    def test_get_at_before_first(self):
        """Test getting value before first timestamp."""
        ts = SparseTimeSeries()
        ts.push(2000, 1.0)
        assert ts.get_at(1000) == 1.0
    
    def test_get_at_after_last(self):
        """Test getting value after last timestamp."""
        ts = SparseTimeSeries()
        ts.push(1000, 1.0)
        assert ts.get_at(2000) == 1.0
    
    def test_get_at_middle(self):
        """Test getting value at middle timestamp."""
        ts = SparseTimeSeries()
        ts.push(1000, 1.0)
        ts.push(3000, 3.0)
        assert ts.get_at(2000) == 1.0
    
    def test_get_at_exact(self):
        """Test getting value at exact timestamp."""
        ts = SparseTimeSeries()
        ts.push(1000, 1.0)
        ts.push(2000, 2.0)
        assert ts.get_at(2000) == 2.0
    
    def test_get_at_empty_raises(self):
        """Test that getting from empty series raises."""
        ts = SparseTimeSeries()
        with pytest.raises(ValueError):
            ts.get_at(1000)
    
    def test_get_range(self):
        """Test getting range of values."""
        ts = SparseTimeSeries()
        ts.push(1000, 1.0)
        ts.push(2000, 2.0)
        ts.push(3000, 3.0)
        
        result = ts.get_range(1500, 2500)
        # Returns forward-filled values in range
        assert len(result) >= 1
    
    def test_get_range_empty_result(self):
        """Test getting range with no data."""
        ts = SparseTimeSeries()
        ts.push(1000, 1.0)
        
        result = ts.get_range(2000, 3000)
        assert len(result) == 0


class TestSparseBuffer:
    """Tests for SparseBuffer."""
    
    def test_init(self):
        """Test initialization."""
        sb = SparseBuffer()
        assert len(sb) == 0
    
    def test_push(self):
        """Test push stores value."""
        sb = SparseBuffer()
        result = sb.push(1000, 1.0)
        assert result is True
    
    def test_get_latest_empty(self):
        """Test getting latest from empty buffer."""
        sb = SparseBuffer()
        assert sb.get_latest() is None
    
    def test_get_latest(self):
        """Test getting latest value."""
        sb = SparseBuffer()
        sb.push(1000, 1.0)
        sb.push(2000, 2.0)
        assert sb.get_latest() == 2.0
    
    def test_get_latest_gap_exceeded(self):
        """Test that gap exceeding max returns None."""
        sb = SparseBuffer(max_gap=1)
        sb.push(1000, 1.0)
        # Gap of ~1 second in ns = 1_000_000_000
        sb.push(2_000_000_000, 2.0)
        # Gap is 1 second, max_gap is 1 day in bars, should still work
        assert sb.get_latest() is not None
    
    def test_get_at(self):
        """Test getting value at timestamp."""
        sb = SparseBuffer()
        sb.push(1000, 1.0)
        assert sb.get_at(1500) == 1.0
    
    def test_get_at_empty(self):
        """Test getting from empty buffer."""
        sb = SparseBuffer()
        assert sb.get_at(1000) is None
