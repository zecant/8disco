import pytest
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime

from tradsl.utils.interpreter import TradslInterpreter, create_interpreter, InterpreterError
from tradsl.utils.feature_engine import FeatureEngineError


class FailingSizer:
    def __call__(self, signals, tradable):
        raise ValueError("Sizer computation failed")


class WrongKeysSizer:
    def __call__(self, signals, tradable):
        return {'wrong_key': 1.0}


class NegativeSizer:
    def __call__(self, signals, tradable):
        return {'nvda': -0.5, 'aapl': 1.5}


class NaNSizer:
    def __call__(self, signals, tradable):
        return {'nvda': np.nan, 'aapl': 0.5}


class PartialSizer:
    def __call__(self, signals, tradable):
        return {'nvda': 0.5}


class TestPositionSizerEdgeCases:
    def test_sizer_raises_exception(self):
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': ['nvda', 'aapl'],
                'sizer': FailingSizer()
            },
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        interpreter = create_interpreter(config)
        
        predictions = {'signal': 0.5}
        
        with pytest.raises(ValueError, match="Sizer computation failed"):
            interpreter.get_allocation(predictions)
    
    def test_sizer_returns_wrong_keys(self):
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': ['nvda', 'aapl'],
                'sizer': WrongKeysSizer()
            },
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        interpreter = create_interpreter(config)
        
        predictions = {'signal': 0.5}
        
        with pytest.raises(ValueError, match="must return all tradable symbols"):
            interpreter.get_allocation(predictions)
    
    def test_sizer_returns_negative_values(self):
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': ['nvda', 'aapl'],
                'sizer': NegativeSizer()
            },
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        interpreter = create_interpreter(config)
        
        predictions = {'signal': 0.5}
        
        with pytest.raises(ValueError, match="negative"):
            interpreter.get_allocation(predictions)
    
    def test_sizer_returns_nan(self):
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': ['nvda', 'aapl'],
                'sizer': NaNSizer()
            },
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        interpreter = create_interpreter(config)
        
        predictions = {'signal': 0.5}
        
        with pytest.raises(ValueError, match="NaN"):
            interpreter.get_allocation(predictions)
    
    def test_sizer_returns_partial_keys(self):
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': ['nvda', 'aapl'],
                'sizer': PartialSizer()
            },
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        interpreter = create_interpreter(config)
        
        predictions = {'signal': 0.5}
        
        with pytest.raises(ValueError, match="must return all tradable"):
            interpreter.get_allocation(predictions)


class TestModelPredictionFailures:
    def test_model_predict_throws(self):
        class FailingModel:
            def predict(self, X, **params):
                raise RuntimeError("Prediction failed")
        
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': ['nvda'],
                'sizer': lambda s, t: {x: 1.0/len(t) for x in t}
            },
            'signal_model': {
                'type': 'model',
                'class': FailingModel,
                'inputs': []
            },
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        interpreter = create_interpreter(config)
        
        with pytest.raises(FeatureEngineError, match="prediction failed"):
            interpreter.predict(pd.Series({'nvda_close': 100}))
    
    def test_model_returns_wrong_shape(self):
        class WrongShapeModel:
            def predict(self, X, **params):
                return "not a dict"
        
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': ['nvda'],
                'sizer': lambda s, t: {x: 1.0/len(t) for x in t}
            },
            'signal_model': {
                'type': 'model',
                'class': WrongShapeModel,
                'inputs': []
            },
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        interpreter = create_interpreter(config)
        
        with pytest.raises(FeatureEngineError, match="must return a dict"):
            interpreter.predict(pd.Series({'nvda_close': 100}))
    
    def test_model_returns_nan(self):
        class NaNModel:
            def predict(self, X, **params):
                return {'signal': np.nan}
        
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': ['nvda'],
                'sizer': lambda s, t: {x: 1.0/len(t) for x in t}
            },
            'signal_model': {
                'type': 'model',
                'class': NaNModel,
                'inputs': []
            },
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        interpreter = create_interpreter(config)
        
        predictions = interpreter.predict(pd.Series({'nvda_close': 100}))
        
        assert 'signal_model_signal' in predictions


class TestDataAlignment:
    def test_timeseries_different_date_ranges(self):
        from tradsl.utils.data_loader import load_timeseries
        
        class MockAdapter1:
            def load_historical(self, symbol, start, end, frequency):
                dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
                return pd.DataFrame({
                    'open': [100] * len(dates),
                    'close': [100] * len(dates)
                }, index=dates)
        
        class MockAdapter2:
            def load_historical(self, symbol, start, end, frequency):
                dates = pd.date_range('2020-01-05', '2020-01-15', freq='D')
                return pd.DataFrame({
                    'open': [200] * len(dates),
                    'close': [200] * len(dates)
                }, index=dates)
        
        config = {
            '_adapters': {'a1': MockAdapter1(), 'a2': MockAdapter2()},
            'nvda': {'type': 'timeseries', 'adapter': 'a1', 'parameters': ['nvda']},
            'aapl': {'type': 'timeseries', 'adapter': 'a2', 'parameters': ['aapl']},
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        result = load_timeseries(config, datetime(2020, 1, 1), datetime(2020, 1, 15))
        
        assert 'nvda_close' in result.columns
        assert 'aapl_close' in result.columns
        assert result.index[0] == pd.Timestamp('2020-01-01')
        assert result.index[-1] == pd.Timestamp('2020-01-15')
    
    def test_forward_fill_behavior(self):
        from tradsl.utils.data_loader import load_timeseries
        
        class MockAdapter:
            def load_historical(self, symbol, start, end, frequency):
                dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
                df = pd.DataFrame({
                    'close': [100, np.nan, 102, np.nan, 104, 105, np.nan, 107, 108, 109]
                }, index=dates)
                return df
        
        config = {
            '_adapters': {'adapter': MockAdapter()},
            'nvda': {'type': 'timeseries', 'adapter': 'adapter', 'parameters': ['nvda']},
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        result = load_timeseries(config, datetime(2020, 1, 1), datetime(2020, 1, 10))
        
        assert result['nvda_close'].isna().sum() == 0
        assert result['nvda_close'].iloc[1] == 100
        assert result['nvda_close'].iloc[3] == 102


class TestMultipleModelsSchedules:
    def test_different_retrain_schedules(self):
        from tradsl.utils.training import TrainingScheduler
        
        scheduler_weekly = TrainingScheduler(schedule='weekly', n_bars=10)
        scheduler_daily = TrainingScheduler(schedule='daily', n_bars=10)
        
        interpreter = create_interpreter({
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': [],
                'sizer': lambda s, t: {}
            },
            '_execution_order': [],
            '_graph': {'deps': {}}
        })
        
        interpreter.schedulers = {
            'model_weekly': scheduler_weekly,
            'model_daily': scheduler_daily
        }
        
        day1 = datetime(2024, 1, 1)
        day2 = datetime(2024, 1, 8)
        
        scheduler_weekly.update_last_train(0, day1)
        
        should_retrain_weekly = interpreter.should_retrain(50, day2)
        
        assert 'model_weekly' in should_retrain_weekly
    
    def test_no_retrain_when_not_needed(self):
        from tradsl.utils.training import TrainingScheduler
        
        scheduler = TrainingScheduler(schedule='every_n_bars', n_bars=100)
        scheduler.last_train_bar = 50
        
        interpreter = create_interpreter({
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': [],
                'sizer': lambda s, t: {}
            },
            '_execution_order': [],
            '_graph': {'deps': {}}
        })
        
        interpreter.schedulers = {'model1': scheduler}
        
        should_retrain = interpreter.should_retrain(60, datetime.now())
        
        assert 'model1' not in should_retrain


class TestAdapterFailures:
    def test_adapter_raises_exception(self):
        from tradsl.utils.data_loader import load_timeseries, DataLoaderError
        
        class FailingAdapter:
            def load_historical(self, symbol, start, end, frequency):
                raise RuntimeError("Adapter failed")
        
        config = {
            '_adapters': {'fail': FailingAdapter()},
            'nvda': {'type': 'timeseries', 'adapter': 'fail', 'parameters': ['nvda']},
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        with pytest.raises(DataLoaderError):
            load_timeseries(config, datetime(2020, 1, 1), datetime(2020, 1, 10))
    
    def test_adapter_returns_empty(self):
        from tradsl.utils.data_loader import load_timeseries
        
        class EmptyAdapter:
            def load_historical(self, symbol, start, end, frequency):
                return pd.DataFrame()
        
        config = {
            '_adapters': {'empty': EmptyAdapter()},
            'nvda': {'type': 'timeseries', 'adapter': 'empty', 'parameters': ['nvda']},
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        result = load_timeseries(config, datetime(2020, 1, 1), datetime(2020, 1, 10))
        
        assert result.empty


class TestFeatureNaNHandling:
    def test_nan_in_function_output(self):
        from tradsl.utils.feature_engine import compute_features
        
        def returns_nan(x):
            return pd.Series([np.nan] * len(x))
        
        config = {
            '_execution_order': ['nvda', 'bad_feature'],
            '_graph': {'deps': {'bad_feature': ['nvda']}},
            'nvda': {'type': 'timeseries'},
            'bad_feature': {
                'type': 'timeseries',
                'function': returns_nan,
                'inputs': ['nvda_close']
            }
        }
        
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({'nvda_close': np.arange(10) + 100}, index=dates)
        
        result = compute_features(config, data)
        
        assert 'bad_feature' in result.columns
        assert pd.isna(result['bad_feature'].iloc[-1])
    
    def test_inf_handling(self):
        from tradsl.utils.feature_engine import compute_features
        
        def div_by_value(x):
            return x / 0
        
        config = {
            '_execution_order': ['nvda', 'inf_feature'],
            '_graph': {'deps': {'inf_feature': ['nvda']}},
            'nvda': {'type': 'timeseries'},
            'inf_feature': {
                'type': 'timeseries',
                'function': div_by_value,
                'inputs': ['nvda_close']
            }
        }
        
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({'nvda_close': np.arange(10) + 100}, index=dates)
        
        result = compute_features(config, data)
        
        assert 'inf_feature' in result.columns
        assert np.isinf(result['inf_feature'].iloc[-1])


class TestBacktestResults:
    def test_backtest_tracks_positions(self):
        from tradsl.utils.interpreter import TradslInterpreter
        
        class SimpleModel:
            def __init__(self):
                self.trained = True
            
            def predict(self, X, **params):
                return {'allocation': [0.8]}
        
        class SimpleAdapter:
            def load_historical(self, symbol, start, end, frequency):
                dates = pd.date_range(start, end, freq=frequency)
                return pd.DataFrame({
                    'close': np.linspace(100, 110, len(dates))
                }, index=dates)
        
        config = {
            '_adapters': {'sim': SimpleAdapter()},
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': ['nvda'],
                'sizer': lambda s, t: {'nvda': 1.0}
            },
            'nvda': {
                'type': 'timeseries',
                'adapter': 'sim',
                'parameters': ['nvda']
            },
            '_execution_order': ['nvda', 'agent'],
            '_graph': {'deps': {}}
        }
        
        interpreter = create_interpreter(config)
        
        results = interpreter.run_backtest(
            datetime(2020, 1, 1),
            datetime(2020, 1, 10),
            frequency='D'
        )
        
        assert len(results) > 0
        assert 'timestamp' in results.columns
        assert 'nvda' in results.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
