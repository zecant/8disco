import pytest
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime

from tradsl.utils.feature_engine import compute_features, compute_features_incremental, FeatureEngineError


def rolling_mean(series, window=30):
    return series.rolling(window=window).mean()


def exponential_mean(series, span=30):
    return series.ewm(span=span).mean()


class ScalarModel:
    def predict(self, X, **params):
        return {'signal': 0.5}


class MultiInputModel:
    def predict(self, X, **params):
        if isinstance(X, pd.DataFrame):
            return {'combined': (X.sum(axis=1) * 0.1).tolist()}
        return {'combined': [float(X.sum()) * 0.1]}


class ChainedModel:
    def predict(self, X, **params):
        if isinstance(X, pd.DataFrame):
            result = X.iloc[:, 0] * 0.5
            return {
                'primary': result.tolist(),
                'secondary': (result * 2).tolist()
            }
        return {'primary': [0.5], 'secondary': [1.0]}


class TestFeatureEngineEdgeCases:
    def test_empty_dataframe(self):
        config = {
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        data = pd.DataFrame()
        
        result = compute_features(config, data)
        
        assert result.empty
    
    def test_missing_input_columns(self):
        config = {
            '_execution_order': ['nvda_ma'],
            '_graph': {'deps': {'nvda_ma': ['nvda']}},
            'nvda_ma': {
                'type': 'timeseries',
                'function': rolling_mean,
                'inputs': ['nvda_close'],
                'params': {'window': 5}
            }
        }
        
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'other_close': np.arange(10) + 100
        }, index=dates)
        
        result = compute_features(config, data)
        
        assert 'nvda_ma' not in result.columns
    
    def test_timeseries_function_returns_series(self):
        def double_values(series):
            return series * 2
        
        config = {
            '_execution_order': ['nvda_double'],
            '_graph': {'deps': {}},
            'nvda_double': {
                'type': 'timeseries',
                'function': double_values,
                'inputs': ['nvda_close']
            }
        }
        
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'nvda_close': np.arange(10) + 100
        }, index=dates)
        
        result = compute_features(config, data)
        
        assert 'nvda_double' in result.columns
        assert result['nvda_double'].iloc[0] == 200
    
    def test_multiple_inputs_to_model(self):
        config = {
            '_execution_order': ['nvda', 'vix', 'ensemble'],
            '_graph': {'deps': {'ensemble': ['nvda', 'vix']}},
            'nvda': {'type': 'timeseries'},
            'vix': {'type': 'timeseries'},
            'ensemble': {
                'type': 'model',
                'class': MultiInputModel,
                'inputs': ['nvda_close', 'vix_close']
            }
        }
        
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'nvda_close': np.arange(10) + 100,
            'vix_close': np.arange(10) + 20
        }, index=dates)
        
        result = compute_features(config, data)
        
        assert 'ensemble_combined' in result.columns
    
    def test_chained_models(self):
        config = {
            '_execution_order': ['nvda', 'model1', 'model2'],
            '_graph': {'deps': {'model1': ['nvda'], 'model2': ['model1']}},
            'nvda': {'type': 'timeseries'},
            'model1': {
                'type': 'model',
                'class': ChainedModel,
                'inputs': ['nvda_close']
            },
            'model2': {
                'type': 'model',
                'class': MultiInputModel,
                'inputs': ['model1_primary']
            }
        }
        
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'nvda_close': np.arange(10) + 100
        }, index=dates)
        
        result = compute_features(config, data)
        
        assert 'model1_primary' in result.columns
        assert 'model1_secondary' in result.columns
        assert 'model2_combined' in result.columns
    
    def test_timeseries_with_params(self):
        config = {
            '_execution_order': ['nvda', 'nvda_ema'],
            '_graph': {'deps': {'nvda_ema': ['nvda']}},
            'nvda': {'type': 'timeseries'},
            'nvda_ema': {
                'type': 'timeseries',
                'function': exponential_mean,
                'inputs': ['nvda_close'],
                'params': {'span': 5}
            }
        }
        
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        data = pd.DataFrame({
            'nvda_close': np.arange(20) + 100
        }, index=dates)
        
        result = compute_features(config, data)
        
        assert 'nvda_ema' in result.columns
    
    def test_model_with_inputs_gets_called(self):
        config = {
            '_execution_order': ['nvda', 'const_model'],
            '_graph': {'deps': {'const_model': ['nvda']}},
            'nvda': {'type': 'timeseries'},
            'const_model': {
                'type': 'model',
                'class': ScalarModel,
                'inputs': ['nvda_close']
            }
        }
        
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'nvda_close': np.arange(10) + 100
        }, index=dates)
        
        result = compute_features(config, data)
        
        assert 'const_model_signal' in result.columns
    
    def test_incremental_with_new_columns(self):
        config = {
            '_execution_order': ['nvda', 'nvda_ma'],
            '_graph': {'deps': {'nvda_ma': ['nvda']}},
            'nvda': {'type': 'timeseries'},
            'nvda_ma': {
                'type': 'timeseries',
                'function': rolling_mean,
                'inputs': ['nvda_close'],
                'params': {'window': 3}
            }
        }
        
        dates = pd.date_range('2020-01-01', periods=3, freq='D')
        data = pd.DataFrame({
            'nvda_close': [100, 101, 102]
        }, index=dates)
        
        new_row = pd.Series({'nvda_close': 103})
        
        result = compute_features_incremental(config, data, new_row)
        
        assert len(result) == 4
        assert 'nvda_ma' in result.columns
    
    def test_skip_non_model_and_timeseries_nodes(self):
        config = {
            '_execution_order': ['agent'],
            '_graph': {'deps': {}},
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': []
            }
        }
        
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'nvda_close': np.arange(10) + 100
        }, index=dates)
        
        result = compute_features(config, data)
        
        assert 'nvda_close' in result.columns


class TestFeatureEngineErrors:
    def test_invalid_function_raises_error(self):
        def bad_func(x):
            raise ValueError("Computation failed")
        
        config = {
            '_execution_order': ['nvda_bad'],
            '_graph': {'deps': {'nvda_bad': ['nvda']}},
            'nvda': {'type': 'timeseries'},
            'nvda_bad': {
                'type': 'timeseries',
                'function': bad_func,
                'inputs': ['nvda_close']
            }
        }
        
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'nvda_close': np.arange(10) + 100
        }, index=dates)
        
        with pytest.raises(FeatureEngineError):
            compute_features(config, data)
    
    def test_model_with_inputs_but_no_predict_raises(self):
        class BadModel:
            pass
        
        config = {
            '_execution_order': ['nvda', 'bad_model'],
            '_graph': {'deps': {'bad_model': ['nvda']}},
            'nvda': {'type': 'timeseries'},
            'bad_model': {
                'type': 'model',
                'class': BadModel,
                'inputs': ['nvda_close']
            }
        }
        
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'nvda_close': np.arange(10) + 100
        }, index=dates)
        
        with pytest.raises(FeatureEngineError):
            compute_features(config, data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
