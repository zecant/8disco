import pytest
import sys
sys.path.insert(0, '.')

from tradsl import parse_config, validate, resolve, build_dag, parse
from tradsl import ConfigError, ResolutionError, CycleError


class TestParserEdgeCases:
    def test_empty_source(self):
        result = parse_config("")
        
        assert result == {'_params': {}, '_adapters': {}, '_backtest': {}}
    
    def test_whitespace_only(self):
        result = parse_config("   \n\n   ")
        
        assert result == {'_params': {}, '_adapters': {}, '_backtest': {}}
    
    def test_only_comments(self):
        source = """
# This is a comment
# Another comment
"""
        result = parse_config(source)
        
        assert result == {'_params': {}, '_adapters': {}, '_backtest': {}}
    
    def test_list_values(self):
        source = """
:test
type=timeseries
inputs=[a, b, c]
"""
        result = parse_config(source)
        
        assert result['test']['inputs'] == ['a', 'b', 'c']
    
    def test_param_block_with_underscores(self):
        source = """
my_params:
lr=0.001

:agent
type=agent
inputs=[]
tradable=[]
"""
        result = parse_config(source)
        
        assert 'my_params' in result['_params']
    
    def test_multiple_param_refs_in_one_block(self):
        source = """
mlparams:
lr=0.001

windowparams:
window=30

:test
type=timeseries
params=mlparams
"""
        result = parse_config(source)
        
        assert result['test']['params'] == 'mlparams'
    
    def test_float_values(self):
        source = """
:test
type=timeseries
threshold=0.5
negative=-0.5
"""
        result = parse_config(source)
        
        assert result['test']['threshold'] == 0.5
        assert result['test']['negative'] == -0.5
    
    def test_scientific_notation(self):
        source = """
:test
type=timeseries
value=1e-5
"""
        result = parse_config(source)
        
        assert result['test']['value'] == 1e-5


class TestResolverEdgeCases:
    def test_param_ref_resolution(self):
        source = """mlparams:
lr=0.001

:agent
type=agent
inputs=[]
tradable=[]
params=mlparams
"""
        raw = parse_config(source)
        validated = validate(raw)
        result = resolve(validated, {})
        
        assert result['agent']['params'] == {'lr': 0.001}
    
    def test_adapter_with_params(self):
        source = """:yfinance
type=adapter
class=adapters.YFAdapter
api_key=secret123

:nvda
type=timeseries
adapter=yfinance
parameters=["nvda"]
"""
        raw = parse_config(source)
        
        class MockAdapter:
            def __init__(self, api_key=None):
                self.api_key = api_key
        
        validated = validate(raw)
        
        class FakeAdapter:
            pass
        
        result = resolve(validated, {'adapters.YFAdapter': MockAdapter})
        
        assert result['_adapters']['yfinance'] is not None
    
    def test_missing_callable_in_context(self):
        source = """:agent
type=agent
inputs=[]
tradable=[]
sizer=nonexistent_sizer
"""
        raw = parse_config(source)
        validated = validate(raw)
        
        with pytest.raises(ResolutionError):
            resolve(validated, {})


class TestDAGEdgeCases:
    def test_single_node(self):
        source = """:nvda
type=timeseries
inputs=[]
"""
        raw = parse_config(source)
        validated = validate(raw)
        resolved = resolve(validated, {})
        result = build_dag(resolved)
        
        assert result['_execution_order'] == ['nvda']
    
    def test_parallel_branches(self):
        source = """:agent
type=agent
inputs=[a, b, c]
tradable=[]

:a
type=timeseries
inputs=[]

:b
type=timeseries
inputs=[]

:c
type=timeseries
inputs=[]
"""
        raw = parse_config(source)
        validated = validate(raw)
        resolved = resolve(validated, {})
        result = build_dag(resolved)
        
        order = result['_execution_order']
        assert order.index('a') < order.index('agent')
        assert order.index('b') < order.index('agent')
        assert order.index('c') < order.index('agent')
    
    def test_deep_chain(self):
        source = """:agent
type=agent
inputs=[d]
tradable=[]

:d
type=timeseries
function=func
inputs=[c]

:c
type=timeseries
function=func
inputs=[b]

:b
type=timeseries
function=func
inputs=[a]

:a
type=timeseries
inputs=[]
"""
        raw = parse_config(source)
        validated = validate(raw)
        
        resolved = resolve(validated, {'func': lambda x: x})
        result = build_dag(resolved)
        
        order = result['_execution_order']
        assert order.index('a') < order.index('b')
        assert order.index('b') < order.index('c')
        assert order.index('c') < order.index('d')
        assert order.index('d') < order.index('agent')
    
    def test_model_chain(self):
        source = """:agent
type=agent
inputs=[model_c]
tradable=[]

:model_c
type=model
class=ModelC
inputs=[model_b]

:model_b
type=model
class=ModelB
inputs=[model_a]

:model_a
type=model
class=ModelA
inputs=[nvda]

:nvda
type=timeseries
inputs=[]
"""
        raw = parse_config(source)
        validated = validate(raw)
        
        class MockModel:
            pass
        
        resolved = resolve(validated, {'ModelA': MockModel, 'ModelB': MockModel, 'ModelC': MockModel})
        result = build_dag(resolved)
        
        order = result['_execution_order']
        assert order.index('nvda') < order.index('model_a')
        assert order.index('model_a') < order.index('model_b')
        assert order.index('model_b') < order.index('model_c')
        assert order.index('model_c') < order.index('agent')
    
    def test_diamond_dependency(self):
        source = """:agent
type=agent
inputs=[combined]
tradable=[]

:combined
type=model
class=Combiner
inputs=[a, b]

:a
type=timeseries
inputs=[base]

:b
type=timeseries
inputs=[base]

:base
type=timeseries
inputs=[]
"""
        raw = parse_config(source)
        validated = validate(raw)
        
        class MockModel:
            pass
        
        resolved = resolve(validated, {'Combiner': MockModel})
        result = build_dag(resolved)
        
        order = result['_execution_order']
        assert order.index('base') < order.index('a')
        assert order.index('base') < order.index('b')
        assert order.index('a') < order.index('combined')
        assert order.index('b') < order.index('combined')
        assert order.index('combined') < order.index('agent')


class TestFullIntegration:
    def test_full_workflow_with_all_features(self):
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

:vix
type=timeseries
adapter=yfinance
parameters=["^VIX"]

:nvda_ma30
type=timeseries
function=rolling_mean
inputs=[nvda]
params=mlparams

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

:backtest
type=backtest
start=2020-01-01
end=2024-01-01
capital=100000
"""
        
        class RandomForest:
            def __init__(self, lr=0.001, epochs=100):
                self.lr = lr
                self.epochs = epochs
            
            def predict(self, X, **params):
                return {'allocation': [0.5] * len(X)}
        
        def rolling_mean(x, **kwargs):
            return x
        
        def kelly_sizer(signals, tradable):
            return {t: 1.0 for t in tradable}
        
        class YFAdapter:
            pass
        
        result = parse(source, context={
            'RandomForest': RandomForest,
            'rolling_mean': rolling_mean,
            'kelly_sizer': kelly_sizer,
            'adapters.YFAdapter': YFAdapter
        })
        
        assert 'agent' in result
        assert result['agent']['tradable'] == ['nvda']
        assert result['_params']['mlparams']['lr'] == 0.001
        assert result['_backtest']['capital'] == 100000
        assert 'nvda' in result['_execution_order']
        assert 'signal_model' in result['_execution_order']
        assert 'agent' in result['_execution_order']
    
    def test_error_on_missing_dependency(self):
        source = """:agent
type=agent
inputs=[nonexistent]
tradable=[]
"""
        
        with pytest.raises(ValueError) as exc:
            parse(source, {})
        
        assert "does not exist" in str(exc.value)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
