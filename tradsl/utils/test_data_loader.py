import pytest
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from tradsl.utils.data_loader import load_timeseries, DataLoaderError


class MockAdapter:
    def __init__(self, data=None):
        self.data = data or {}
    
    def load_historical(self, symbol, start, end, frequency):
        if symbol in self.data:
            df = self.data[symbol].copy()
            df = df[(df.index >= start) & (df.index <= end)]
            return df
        dates = pd.date_range(start, end, freq=frequency)
        df = pd.DataFrame({
            'open': np.random.randn(len(dates)) + 100,
            'high': np.random.randn(len(dates)) + 101,
            'low': np.random.randn(len(dates)) + 99,
            'close': np.random.randn(len(dates)) + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        return df


class TestDataLoader:
    def test_load_timeseries_basic(self):
        adapter = MockAdapter()
        
        config = {
            '_adapters': {'yfinance': adapter},
            'nvda': {
                'type': 'timeseries',
                'adapter': 'yfinance',
                'parameters': ['nvda']
            }
        }
        
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 10)
        
        result = load_timeseries(config, start, end)
        
        assert result is not None
        assert not result.empty
        assert 'nvda_close' in result.columns
        assert 'nvda_volume' in result.columns
    
    def test_load_multiple_timeseries(self):
        adapter = MockAdapter()
        
        config = {
            '_adapters': {'yfinance': adapter},
            'nvda': {
                'type': 'timeseries',
                'adapter': 'yfinance',
                'parameters': ['nvda']
            },
            'vix': {
                'type': 'timeseries',
                'adapter': 'yfinance',
                'parameters': ['^VIX']
            }
        }
        
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 10)
        
        result = load_timeseries(config, start, end)
        
        assert 'nvda_close' in result.columns
        assert '^VIX_close' in result.columns
    
    def test_missing_adapter_raises_error(self):
        config = {
            '_adapters': {},
            'nvda': {
                'type': 'timeseries',
                'adapter': 'nonexistent',
                'parameters': ['nvda']
            }
        }
        
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 10)
        
        with pytest.raises(DataLoaderError) as exc:
            load_timeseries(config, start, end)
        
        assert "not defined" in str(exc.value)
    
    def test_timeseries_without_adapter_skipped(self):
        adapter = MockAdapter()
        
        config = {
            '_adapters': {'yfinance': adapter},
            'nvda': {
                'type': 'timeseries',
                'adapter': 'yfinance',
                'parameters': ['nvda']
            },
            'vix': {
                'type': 'timeseries',
                'parameters': []
            }
        }
        
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 10)
        
        result = load_timeseries(config, start, end)
        
        assert 'nvda_close' in result.columns
    
    def test_flatten_columns(self):
        from tradsl.utils.data_loader import _flatten_columns
        
        df = pd.DataFrame({
            'close': [100, 101, 102],
            'high': [101, 102, 103]
        })
        
        result = _flatten_columns(df, 'nvda')
        
        assert 'nvda_close' in result.columns
        assert 'nvda_high' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
