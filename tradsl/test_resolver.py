import pytest
import sys
sys.path.insert(0, '.')

from tradsl.parser import parse_config
from tradsl.schema import validate, ConfigError
from tradsl.resolver import resolve, ResolutionError


class MockAdapter:
    def __init__(self, params=None):
        self.params = params
    
    def load_historical(self, symbol, start, end):
        return []


class TestResolver:
    def test_resolve_param_ref(self):
        source = """mlparams:
lr=0.001
epochs=100

:agent
type=agent
inputs=[spy]
tradable=[spy]
params=mlparams
"""
        raw = parse_config(source)
        validated = validate(raw)
        
        context = {}
        result = resolve(validated, context)
        
        assert result['agent']['params'] == {'lr': 0.001, 'epochs': 100}
    
    def test_resolve_multiple_param_refs(self):
        source = """mlparams:
lr=0.001

windowparams:
window=30

:nvda
type=timeseries
inputs=[]
params=windowparams

:model
type=model
class=RandomForest
inputs=[nvda]
params=mlparams
"""
        raw = parse_config(source)
        validated = validate(raw)
        
        class RandomForest:
            pass
        
        result = resolve(validated, {'RandomForest': RandomForest})
        
        assert result['nvda']['params'] == {'window': 30}
        assert result['model']['params'] == {'lr': 0.001}
    
    def test_resolve_callable(self):
        source = """:agent
type=agent
inputs=[spy]
tradable=[spy]
sizer=kelly_sizer
"""
        raw = parse_config(source)
        validated = validate(raw)
        
        def kelly_sizer(signals, tradable):
            pass
        
        result = resolve(validated, {'kelly_sizer': kelly_sizer})
        
        assert result['agent']['sizer'] is kelly_sizer
    
    def test_resolve_timeseries_function(self):
        source = """:nvda_ma
type=timeseries
function=rolling_mean
inputs=[nvda]
"""
        raw = parse_config(source)
        validated = validate(raw)
        
        def rolling_mean(data, window=30):
            pass
        
        result = resolve(validated, {'rolling_mean': rolling_mean})
        
        assert result['nvda_ma']['function'] is rolling_mean
    
    def test_resolve_model_class(self):
        source = """:model
type=model
class=RandomForest
inputs=[nvda]
"""
        raw = parse_config(source)
        validated = validate(raw)
        
        class RandomForest:
            pass
        
        result = resolve(validated, {'RandomForest': RandomForest})
        
        assert result['model']['class'] is RandomForest
    
    def test_missing_param_ref(self):
        source = """mlparams:
lr=0.001

:agent
type=agent
inputs=[spy]
tradable=[spy]
params=nonexistent
"""
        raw = parse_config(source)
        validated = validate(raw)
        
        with pytest.raises(ResolutionError) as exc:
            resolve(validated, {})
        
        assert "Param ref 'nonexistent' not found" in str(exc.value)
    
    def test_missing_callable(self):
        source = """:agent
type=agent
inputs=[spy]
tradable=[spy]
sizer=kelly_sizer
"""
        raw = parse_config(source)
        validated = validate(raw)
        
        with pytest.raises(ResolutionError) as exc:
            resolve(validated, {})
        
        assert "kelly_sizer" in str(exc.value)
        assert "not found in context" in str(exc.value)
    
    def test_extracts_deps(self):
        source = """:agent
type=agent
inputs=[spy, nvda, model1]
tradable=[spy]

:model1
type=model
class=RandomForest
inputs=[nvda, vix]
"""
        raw = parse_config(source)
        validated = validate(raw)
        result = resolve(validated, {'RandomForest': type})
        
        assert result['agent']['_deps'] == ['spy', 'nvda', 'model1']
        assert result['model1']['_deps'] == ['nvda', 'vix']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
