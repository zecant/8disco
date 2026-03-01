import pytest
import sys
sys.path.insert(0, '.')


import numpy as np
import pandas as pd

from tradsl.utils.feature_engine import compute_features, compute_features_incremental, FeatureEngineError


def rolling_mean_30(series, window=30):
    return series.rolling(window=window).mean()


class SimpleModel:
    def predict(self, X, **params):
        if isinstance(X, pd.DataFrame):
            return {'allocation': (X.iloc[:, 0] * 0.5).tolist()}
        return {'allocation': [float(X) * 0.5]}


class MultiOutputModel:
    def predict(self, X, **params):
        if isinstance(X, pd.DataFrame):
            return {
                'long': (X.iloc[:, 0] * 0.6).tolist(),
                'short': (X.iloc[:, 0] * -0.4).tolist()
            }
        return {
            'long': [float(X) * 0.6],
            'short': [float(X) * -0.4]
        }


class TestFeatureEngine:
    def test_compute_timeseries_function(self):
        config = {
            '_execution_order': ['nvda', 'nvda_ma30'],
            '_graph': {'deps': {'nvda_ma30': ['nvda']}},
            'nvda': {
                'type': 'timeseries',
                'function': None
            },
            'nvda_ma30': {
                'type': 'timeseries',
                'function': rolling_mean_30,
                'inputs': ['nvda_close'],
                'params': {'window': 5}
            }
        }
        
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'nvda_close': np.arange(10) + 100
        }, index=dates)
        
        result = compute_features(config, data)
        
        assert 'nvda_ma30' in result.columns
        assert result['nvda_ma30'].iloc[-1] == 107.0
    
    def test_compute_model(self):
        config = {
            '_execution_order': ['nvda', 'signal_model'],
            '_graph': {'deps': {'signal_model': ['nvda']}},
            'nvda': {
                'type': 'timeseries',
                'function': None
            },
            'signal_model': {
                'type': 'model',
                'class': SimpleModel,
                'inputs': ['nvda_close']
            }
        }
        
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'nvda_close': np.arange(10) + 100
        }, index=dates)
        
        result = compute_features(config, data)
        
        assert 'signal_model_allocation' in result.columns
    
    def test_compute_features_incremental(self):
        config = {
            '_execution_order': ['nvda', 'nvda_ma30'],
            '_graph': {'deps': {'nvda_ma30': ['nvda']}},
            'nvda': {
                'type': 'timeseries',
                'function': None
            },
            'nvda_ma30': {
                'type': 'timeseries',
                'function': rolling_mean_30,
                'inputs': ['nvda_close'],
                'params': {'window': 3}
            }
        }
        
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'nvda_close': [100, 101, 102, 103, 104]
        }, index=dates)
        
        new_row = pd.Series({'nvda_close': 105})
        
        result = compute_features_incremental(config, data, new_row)
        
        assert len(result) == 6
        assert 'nvda_ma30' in result.columns
    
    def test_model_returns_dict(self):
        config = {
            '_execution_order': ['nvda', 'signal_model'],
            '_graph': {'deps': {'signal_model': ['nvda']}},
            'nvda': {'type': 'timeseries'},
            'signal_model': {
                'type': 'model',
                'class': MultiOutputModel,
                'inputs': ['nvda_close']
            }
        }
        
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({'nvda_close': np.arange(10) + 100}, index=dates)
        
        result = compute_features(config, data)
        
        assert 'signal_model_long' in result.columns
        assert 'signal_model_short' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
