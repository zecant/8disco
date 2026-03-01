import pytest
import sys
sys.path.insert(0, '.')

from tradsl.parser import parse_config


class TestParser:
    def test_basic_block(self):
        source = """:agent
type=agent
inputs=[spy, vix]
tradable=[spy]
"""
        result = parse_config(source)
        assert result['agent'] == {
            'type': 'agent',
            'inputs': ['spy', 'vix'],
            'tradable': ['spy']
        }
    
    def test_param_block(self):
        source = """mlparams:
lr=0.001
epochs=100

:agent
type=agent
"""
        result = parse_config(source)
        assert result['_params']['mlparams'] == {'lr': 0.001, 'epochs': 100}
        assert result['agent'] == {'type': 'agent'}
    
    def test_multiple_param_blocks(self):
        source = """mlparams:
lr=0.001

windowparams:
window=30
min_periods=10

:spy
type=timeseries
"""
        result = parse_config(source)
        assert result['_params']['mlparams'] == {'lr': 0.001}
        assert result['_params']['windowparams'] == {'window': 30, 'min_periods': 10}
    
    def test_adapter_block(self):
        source = """:yfinance
type=adapter
class=adapters.YFAdapter
"""
        result = parse_config(source)
        assert result['_adapters']['yfinance'] == {
            'type': 'adapter',
            'class': 'adapters.YFAdapter'
        }
    
    def test_backtest_block(self):
        source = """:backtest
type=backtest
start=2020-01-01
end=2024-01-01
capital=100000
"""
        result = parse_config(source)
        assert result['_backtest'] == {
            'type': 'backtest',
            'start': '2020-01-01',
            'end': '2024-01-01',
            'capital': 100000
        }
    
    def test_timeseries_with_adapter_and_params(self):
        source = """:nvda
type=timeseries
adapter=yfinance
parameters=["nvda"]
params=windowparams
"""
        result = parse_config(source)
        assert result['nvda'] == {
            'type': 'timeseries',
            'adapter': 'yfinance',
            'parameters': ['nvda'],
            'params': 'windowparams'
        }
    
    def test_model_with_param_ref(self):
        source = """:signal_model
type=model
class=RandomForest
inputs=[nvda, vix]
params=mlparams
"""
        result = parse_config(source)
        assert result['signal_model'] == {
            'type': 'model',
            'class': 'RandomForest',
            'inputs': ['nvda', 'vix'],
            'params': 'mlparams'
        }
    
    def test_full_config(self):
        source = """mlparams:
lr=0.001
epochs=100

:yfinance
type=adapter
class=adapters.YFAdapter

:nvda
type=timeseries
adapter=yfinance
parameters=["nvda"]

:agent
type=agent
inputs=[nvda]
tradable=[nvda]
sizer=kelly
"""
        result = parse_config(source)
        
        assert '_params' in result
        assert 'mlparams' in result['_params']
        assert result['_params']['mlparams']['lr'] == 0.001
        
        assert '_adapters' in result
        assert 'yfinance' in result['_adapters']
        
        assert 'nvda' in result
        assert result['nvda']['adapter'] == 'yfinance'
        
        assert 'agent' in result
        assert result['agent']['sizer'] == 'kelly'
    
    def test_empty_list(self):
        source = """:empty
type=timeseries
inputs=[]
"""
        result = parse_config(source)
        assert result['empty']['inputs'] == []
    
    def test_nested_list_strings(self):
        source = """:test
type=timeseries
parameters=["AAPL", "MSFT"]
"""
        result = parse_config(source)
        assert result['test']['parameters'] == ['AAPL', 'MSFT']
    
    def test_boolean_values(self):
        source = """:test
type=timeseries
tradable=true
active=false
"""
        result = parse_config(source)
        assert result['test']['tradable'] is True
        assert result['test']['active'] is False
    
    def test_numeric_values(self):
        source = """:test
type=timeseries
window=30
threshold=0.05
"""
        result = parse_config(source)
        assert result['test']['window'] == 30
        assert result['test']['threshold'] == 0.05


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
