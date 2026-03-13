"""Tests for feature functions."""
import pytest
import numpy as np
from tradsl.functions import (
    log_returns, sma, ema, rsi, rolling_std, volatility, roc, zscore,
    FUNCTION_REGISTRY, get_function, compute_function, FeatureCategory
)


class TestLogReturns:
    def test_basic_log_return(self):
        arr = np.array([100.0, 110.0])
        result = log_returns(arr, period=1)
        assert result is not None
        assert abs(result - np.log(110.0 / 100.0)) < 1e-6
    
    def test_multi_period(self):
        arr = np.array([100.0, 105.0, 110.0])
        result = log_returns(arr, period=2)
        assert result is not None
        assert abs(result - np.log(110.0 / 100.0)) < 1e-6
    
    def test_insufficient_data(self):
        arr = np.array([100.0])
        result = log_returns(arr, period=1)
        assert result is None
    
    def test_nan_handling(self):
        arr = np.array([100.0, np.nan, 110.0])
        result = log_returns(arr, period=1)
        assert result is None
    
    def test_zero_price(self):
        arr = np.array([0.0, 110.0])
        result = log_returns(arr, period=1)
        assert result is None


class TestSMA:
    def test_basic_sma(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(arr, period=5)
        assert result == 3.0
    
    def test_partial_window(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = sma(arr, period=5)
        assert result is None
    
    def test_nan_handling(self):
        arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = sma(arr, period=5)
        assert result is None


class TestEMA:
    def test_basic_ema(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ema(arr, period=5)
        assert result is not None
        assert 2.0 < result < 5.0
    
    def test_alpha_calculation(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        period = 5
        alpha = 2.0 / (period + 1)
        
        ema_val = arr[0]
        for i in range(1, len(arr)):
            ema_val = alpha * arr[i] + (1 - alpha) * ema_val
        
        result = ema(arr, period=period)
        assert abs(result - ema_val) < 1e-6
    
    def test_insufficient_data(self):
        arr = np.array([1.0, 2.0])
        result = ema(arr, period=5)
        assert result is None


class TestRSI:
    def test_rsi_100(self):
        arr = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                        110.0, 111.0, 112.0, 113.0, 114.0, 115.0])
        result = rsi(arr, period=14)
        assert result == 100.0
    
    def test_rsi_0(self):
        arr = np.array([115.0, 114.0, 113.0, 112.0, 111.0, 110.0, 109.0, 108.0, 107.0, 106.0,
                        105.0, 104.0, 103.0, 102.0, 101.0, 100.0])
        result = rsi(arr, period=14)
        assert result == 0.0
    
    def test_insufficient_data(self):
        arr = np.array([100.0, 101.0, 102.0])
        result = rsi(arr, period=14)
        assert result is None


class TestRollingStd:
    def test_basic_std(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_std(arr, period=5)
        expected = np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=0)
        assert abs(result - expected) < 1e-6


class TestVolatility:
    def test_volatility_calculation(self):
        arr = np.array([0.01, -0.01, 0.01, -0.01, 0.01]) * 10
        result = volatility(arr, period=5)
        assert result is not None
        std = np.std(arr, ddof=0)
        expected = std * np.sqrt(252)
        assert abs(result - expected) < 1e-6


class TestROC:
    def test_positive_roc(self):
        arr = np.array([100.0, 110.0])
        result = roc(arr, period=1)
        assert result == 10.0
    
    def test_negative_roc(self):
        arr = np.array([110.0, 100.0])
        result = roc(arr, period=1)
        assert result == pytest.approx(-9.0909, rel=1e-3)
    
    def test_zero_roc(self):
        arr = np.array([100.0, 100.0])
        result = roc(arr, period=1)
        assert result == 0.0


class TestZScore:
    def test_basic_zscore(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = zscore(arr, period=5)
        current = 5.0
        mean = 3.0
        std = np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=0)
        expected = (current - mean) / std
        assert abs(result - expected) < 1e-6
    
    def test_zero_std(self):
        arr = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = zscore(arr, period=5)
        assert result == 0.0


class TestFunctionRegistry:
    def test_get_function(self):
        spec = get_function('ema')
        assert spec is not None
        assert spec.category == FeatureCategory.SAFE
    
    def test_compute_function(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_function('sma', arr, period=5)
        assert result == 3.0
    
    def test_unknown_function(self):
        arr = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            compute_function('nonexistent', arr)
    
    def test_registry_has_expected_functions(self):
        expected = ['log_returns', 'sma', 'ema', 'rsi', 'rolling_std', 'volatility', 'roc', 'zscore']
        for name in expected:
            assert name in FUNCTION_REGISTRY
