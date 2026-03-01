import pytest
import sys
sys.path.insert(0, '.')

import tradsl
from tradsl import ConfigError, ResolutionError, CycleError


class TestE2E:
    def test_full_config_parsing(self):
        source = """mlparams:
lr=0.001
epochs=100

windowparams:
window=30

:yfinance
type=adapter
class=adapters.YFAdapter

:nvda
type=timeseries
adapter=yfinance
parameters=["nvda"]

:vix
type=timeseries
adapter=yfinance
parameters=["^VIX"]

:nvda_ma30
type=timeseries
function=rolling_mean
inputs=[nvda]
params=windowparams

:signal_model
type=model
class=RandomForest
inputs=[nvda, vix, nvda_ma30]
params=mlparams

:agent
type=agent
inputs=[signal_model, vix]
tradable=[nvda]
sizer=kelly_sizer
"""
        
        class RandomForest:
            pass
        
        class YFAdapter:
            pass
        
        def rolling_mean(data, window=30):
            pass
        
        def kelly_sizer(signals, tradable):
            pass
        
        result = tradsl.parse(source, context={
            'RandomForest': RandomForest,
            'adapters.YFAdapter': YFAdapter,
            'rolling_mean': rolling_mean,
            'kelly_sizer': kelly_sizer
        })
        
        assert result['_params']['mlparams']['lr'] == 0.001
        assert result['_params']['windowparams']['window'] == 30
        
        assert 'nvda' in result
        assert result['nvda']['adapter'] is not None
        
        assert result['nvda_ma30']['params'] == {'window': 30}
        
        assert result['signal_model']['params'] == {'lr': 0.001, 'epochs': 100}
        
        assert 'vix' in result['_execution_order']
        assert 'nvda' in result['_execution_order']
        assert 'signal_model' in result['_execution_order']
        assert 'agent' in result['_execution_order']
    
    def test_backtest_config(self):
        source = """:backtest
type=backtest
start=2020-01-01
end=2024-01-01
capital=100000
commission=0.001

:agent
type=agent
inputs=[spy]
tradable=[spy]

:spy
type=timeseries
inputs=[]
"""
        
        result = tradsl.parse(source, {})
        
        assert result['_backtest']['start'] == '2020-01-01'
        assert result['_backtest']['end'] == '2024-01-01'
        assert result['_backtest']['capital'] == 100000
        assert result['_backtest']['commission'] == 0.001
    
    def test_missing_type_field(self):
        source = """:agent
inputs=[spy]
"""
        
        with pytest.raises(ConfigError) as exc:
            tradsl.parse(source, {})
        
        assert "missing required 'type'" in str(exc.value)
    
    def test_unknown_type(self):
        source = """:agent
type=invalid_type
inputs=[]
"""
        
        with pytest.raises(ConfigError) as exc:
            tradsl.parse(source, {})
        
        assert "unknown type" in str(exc.value)
    
    def test_missing_required_field(self):
        source = """:agent
type=agent
"""
        
        with pytest.raises(ConfigError) as exc:
            tradsl.parse(source, {})
        
        assert "missing required field" in str(exc.value)
    
    def test_missing_dependency(self):
        source = """:agent
type=agent
inputs=[nonexistent]
tradable=[]
"""
        
        with pytest.raises(ValueError) as exc:
            tradsl.parse(source, {})
        
        assert "does not exist" in str(exc.value)
    
    def test_default_values(self):
        source = """:nvda
type=timeseries
inputs=[]
"""
        
        result = tradsl.parse(source, {})
        
        assert result['nvda']['tradable'] is False
        assert result['nvda']['inputs'] == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
