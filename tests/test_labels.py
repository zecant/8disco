"""Tests for label functions."""
import pytest
import numpy as np
from tradsl.labels import (
    forward_return_sign, forward_return, triple_barrier,
    LABEL_REGISTRY, get_label_function, compute_labels
)


class TestForwardReturnSign:
    def test_positive_return(self):
        prices = np.array([100.0, 110.0])
        features = np.array([[1.0], [2.0]])
        result = forward_return_sign(features, prices, 1, period=1)
        
        assert result[0] == 1.0
        assert np.isnan(result[1])
    
    def test_negative_return(self):
        prices = np.array([110.0, 100.0])
        features = np.array([[1.0], [2.0]])
        result = forward_return_sign(features, prices, 1, period=1)
        
        assert result[0] == -1.0
        assert np.isnan(result[1])
    
    def test_zero_return(self):
        prices = np.array([100.0, 100.0])
        features = np.array([[1.0], [2.0]])
        result = forward_return_sign(features, prices, 1, period=1)
        
        assert result[0] == 0.0
        assert np.isnan(result[1])
    
    def test_multi_period(self):
        prices = np.array([100.0, 105.0, 110.0, 115.0])
        features = np.array([[i] for i in range(4)])
        result = forward_return_sign(features, prices, 3, period=2)
        
        assert result[0] == 1.0
        assert result[1] == 1.0
        assert np.isnan(result[2])
        assert np.isnan(result[3])
    
    def test_nan_for_insufficient_future_data(self):
        prices = np.array([100.0, 110.0])
        features = np.array([[1.0], [2.0]])
        result = forward_return_sign(features, prices, 1, period=5)
        
        assert np.all(np.isnan(result))


class TestForwardReturn:
    def test_basic_return(self):
        prices = np.array([100.0, 110.0])
        features = np.array([[1.0], [2.0]])
        result = forward_return(features, prices, 1, period=1, log=False)
        
        expected = (110.0 - 100.0) / 100.0
        assert abs(result[0] - expected) < 1e-6
        assert np.isnan(result[1])
    
    def test_log_return(self):
        prices = np.array([100.0, 110.0])
        features = np.array([[1.0], [2.0]])
        result = forward_return(features, prices, 1, period=1, log=True)
        
        expected = np.log(110.0 / 100.0)
        assert abs(result[0] - expected) < 1e-6
    
    def test_negative_prices(self):
        prices = np.array([100.0, -10.0])
        features = np.array([[1.0], [2.0]])
        result = forward_return(features, prices, 1, period=1)
        
        assert np.isnan(result[0])


class TestTripleBarrier:
    def test_upper_barrier_hit(self):
        prices = np.array([100.0, 103.0])
        features = np.array([[1.0], [2.0]])
        result = triple_barrier(
            features, prices, 1,
            upper_barrier=0.02,
            lower_barrier=0.01,
            time_barrier=10
        )
        
        assert result[0] == 1.0
    
    def test_lower_barrier_hit(self):
        prices = np.array([100.0, 98.0])
        features = np.array([[1.0], [2.0]])
        result = triple_barrier(
            features, prices, 1,
            upper_barrier=0.02,
            lower_barrier=0.01,
            time_barrier=10
        )
        
        assert result[0] == -1.0
    
    def test_time_barrier(self):
        prices = np.array([100.0, 101.0, 101.0, 101.0, 101.0])
        features = np.array([[i] for i in range(5)])
        result = triple_barrier(
            features, prices, 4,
            upper_barrier=0.02,
            lower_barrier=0.01,
            time_barrier=3
        )
        
        assert result[0] == 0.0


class TestLabelRegistry:
    def test_get_label_function(self):
        spec = get_label_function('forward_return_sign')
        assert spec is not None
    
    def test_compute_labels(self):
        prices = np.array([100.0, 110.0])
        features = np.array([[1.0], [2.0]])
        result = compute_labels('forward_return_sign', features, prices, 1, period=1)
        
        assert result[0] == 1.0
    
    def test_unknown_label(self):
        prices = np.array([100.0, 110.0])
        features = np.array([[1.0], [2.0]])
        
        with pytest.raises(ValueError):
            compute_labels('nonexistent', features, prices, 1)
    
    def test_registry_has_expected_labels(self):
        expected = ['forward_return_sign', 'forward_return', 'triple_barrier']
        for name in expected:
            assert name in LABEL_REGISTRY
