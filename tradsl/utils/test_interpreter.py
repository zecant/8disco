import pytest
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime

from tradsl.utils.interpreter import TradslInterpreter, create_interpreter, InterpreterError


class MockAdapter:
    def __init__(self, data=None):
        self.data = data or {}
    
    def load_historical(self, symbol, start, end, frequency):
        dates = pd.date_range(start, end, freq=frequency)
        return pd.DataFrame({
            'open': np.random.randn(len(dates)) + 100,
            'high': np.random.randn(len(dates)) + 101,
            'low': np.random.randn(len(dates)) + 99,
            'close': np.random.randn(len(dates)) + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)


class MockModel:
    def __init__(self, **params):
        self.params = params
        self.trained = False
    
    def train(self, X, y, **params):
        self.trained = True
    
    def predict(self, X, **params):
        if isinstance(X, pd.DataFrame):
            return {'allocation': (X.iloc[:, 0] * 0.01).tolist()}
        return {'allocation': [0.5]}


def kelly_sizer(signals, tradable):
    return {sym: 1.0 / len(tradable) for sym in tradable}


class TestTradslInterpreter:
    def test_create_interpreter(self):
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': ['nvda'],
                'sizer': kelly_sizer
            },
            '_execution_order': ['agent'],
            '_graph': {'deps': {'agent': []}}
        }
        
        interpreter = create_interpreter(config)
        
        assert interpreter is not None
        assert interpreter.position_sizer is kelly_sizer
    
    def test_create_interpreter_no_agent_raises_error(self):
        config = {
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        with pytest.raises(InterpreterError) as exc:
            create_interpreter(config)
        
        assert "No agent" in str(exc.value)
    
    def test_initialize_with_model(self):
        config = {
            'agent': {
                'type': 'agent',
                'inputs': ['signal_model'],
                'tradable': ['nvda'],
                'sizer': kelly_sizer
            },
            'signal_model': {
                'type': 'model',
                'class': MockModel,
                'inputs': ['nvda'],
                'params': {'lr': 0.01}
            },
            '_execution_order': ['signal_model', 'agent'],
            '_graph': {'deps': {'signal_model': ['nvda'], 'agent': ['signal_model']}}
        }
        
        interpreter = create_interpreter(config)
        
        assert 'signal_model' in interpreter.models
    
    def test_load_data(self):
        adapter = MockAdapter()
        
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': ['nvda'],
                'sizer': kelly_sizer
            },
            'nvda': {
                'type': 'timeseries',
                'adapter': 'yfinance',
                'parameters': ['nvda']
            },
            '_adapters': {'yfinance': adapter},
            '_execution_order': ['nvda', 'agent'],
            '_graph': {'deps': {'nvda': [], 'agent': []}}
        }
        
        interpreter = create_interpreter(config)
        
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 10)
        
        interpreter.load_data(start, end)
        
        assert interpreter.data is not None
        assert not interpreter.data.empty
    
    def test_load_data_no_adapter_raises_error(self):
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': [],
                'sizer': kelly_sizer
            },
            '_adapters': {},
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        interpreter = create_interpreter(config)
        
        with pytest.raises(Exception):
            interpreter.load_data(datetime(2020, 1, 1), datetime(2020, 1, 10))
    
    def test_compute_initial_features(self):
        adapter = MockAdapter()
        
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': ['nvda'],
                'sizer': kelly_sizer
            },
            'nvda': {
                'type': 'timeseries',
                'adapter': 'yfinance',
                'parameters': ['nvda']
            },
            '_adapters': {'yfinance': adapter},
            '_execution_order': ['nvda', 'agent'],
            '_graph': {'deps': {'nvda': [], 'agent': []}}
        }
        
        interpreter = create_interpreter(config)
        
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 10)
        
        interpreter.load_data(start, end)
        interpreter.compute_initial_features()
        
        assert interpreter.feature_df is not None
    
    def test_compute_initial_features_no_data_raises_error(self):
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': [],
                'sizer': kelly_sizer
            },
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        interpreter = create_interpreter(config)
        
        with pytest.raises(InterpreterError):
            interpreter.compute_initial_features()
    
    def test_predict(self):
        adapter = MockAdapter()
        
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': ['nvda'],
                'sizer': kelly_sizer
            },
            'nvda': {
                'type': 'timeseries',
                'adapter': 'yfinance',
                'parameters': ['nvda']
            },
            '_adapters': {'yfinance': adapter},
            '_execution_order': ['nvda', 'agent'],
            '_graph': {'deps': {'nvda': [], 'agent': []}}
        }
        
        interpreter = create_interpreter(config)
        
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 10)
        
        interpreter.load_data(start, end)
        interpreter.compute_initial_features()
        
        new_row = pd.Series({'nvda_close': 105.0})
        
        predictions = interpreter.predict(new_row)
        
        assert predictions is not None
    
    def test_get_allocation(self):
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': ['nvda', 'vix'],
                'sizer': kelly_sizer
            },
            '_execution_order': ['agent'],
            '_graph': {'deps': {'agent': []}}
        }
        
        interpreter = create_interpreter(config)
        
        predictions = {'signal': 0.5}
        
        allocation = interpreter.get_allocation(predictions)
        
        assert allocation['nvda'] == 0.5
        assert allocation['vix'] == 0.5
    
    def test_get_allocation_no_sizer(self):
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': ['nvda', 'vix'],
                'sizer': None
            },
            '_execution_order': ['agent'],
            '_graph': {'deps': {'agent': []}}
        }
        
        interpreter = create_interpreter(config)
        
        predictions = {}
        
        allocation = interpreter.get_allocation(predictions)
        
        assert allocation['nvda'] == 0.5
        assert allocation['vix'] == 0.5
    
    def test_should_retrain(self):
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': [],
                'sizer': kelly_sizer
            },
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        from tradsl.utils.training import TrainingScheduler
        scheduler = TrainingScheduler(schedule='every_n_bars', n_bars=10)
        
        interpreter = create_interpreter(config)
        interpreter.schedulers = {'model1': scheduler}
        
        current_time = datetime.now()
        
        result = interpreter.should_retrain(0, current_time)
        
        assert 'model1' in result
    
    def test_retrain_model(self):
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': [],
                'sizer': kelly_sizer
            },
            'signal_model': {
                'type': 'model',
                'class': MockModel,
                'inputs': []
            },
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        from tradsl.utils.training import TrainingScheduler
        scheduler = TrainingScheduler(schedule='every_n_bars', n_bars=10)
        scheduler.last_train_bar = 10
        
        interpreter = create_interpreter(config)
        interpreter.schedulers = {'signal_model': scheduler}
        
        interpreter.retrain_model('signal_model')
        
        assert scheduler.last_train_bar >= 10
    
    def test_retrain_model_no_scheduler(self):
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': [],
                'sizer': kelly_sizer
            },
            'signal_model': {
                'type': 'model',
                'class': MockModel,
                'inputs': []
            },
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        interpreter = create_interpreter(config)
        interpreter.models = {'model1': MockModel()}
        
        interpreter.retrain_model('model1')
        
        assert interpreter.models['model1'].trained is False
    
    def test_get_model_inputs(self):
        config = {
            'agent': {
                'type': 'agent',
                'inputs': [],
                'tradable': [],
                'sizer': kelly_sizer
            },
            'signal_model': {
                'type': 'model',
                'class': MockModel,
                'inputs': ['nvda_close']
            },
            '_execution_order': [],
            '_graph': {'deps': {}}
        }
        
        interpreter = create_interpreter(config)
        interpreter.feature_df = pd.DataFrame({
            'nvda_close': [100, 101, 102],
            'vix_close': [20, 21, 22]
        })
        
        inputs = interpreter._get_model_inputs('signal_model')
        
        assert 'nvda_close' in inputs


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
