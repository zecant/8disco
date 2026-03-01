import pytest
import sys
sys.path.insert(0, '.')

from tradsl.parser import parse_config
from tradsl.schema import validate
from tradsl.resolver import resolve
from tradsl.dag import build_dag, CycleError


class TestDAG:
    def test_simple_dag(self):
        source = """:agent
type=agent
inputs=[spy]
tradable=[spy]

:spy
type=timeseries
inputs=[]
"""
        raw = parse_config(source)
        validated = validate(raw)
        resolved = resolve(validated, {})
        result = build_dag(resolved)
        
        assert 'spy' in result['_execution_order']
        assert 'agent' in result['_execution_order']
        assert result['_execution_order'].index('spy') < result['_execution_order'].index('agent')
    
    def test_submodel_dag(self):
        source = """:agent
type=agent
inputs=[ensemble]
tradable=[nvda]

:ensemble
type=model
class=EnsembleModel
inputs=[signal_model, nvda_ma]

:signal_model
type=model
class=RandomForest
inputs=[nvda, vix]

:nvda_ma
type=timeseries
function=rolling_mean
inputs=[nvda]

:nvda
type=timeseries
inputs=[]

:vix
type=timeseries
inputs=[]
"""
        raw = parse_config(source)
        validated = validate(raw)
        
        class FakeModel:
            pass
        
        resolved = resolve(validated, {
            'EnsembleModel': FakeModel,
            'RandomForest': FakeModel,
            'rolling_mean': lambda x: x
        })
        result = build_dag(resolved)
        
        order = result['_execution_order']
        
        assert order.index('vix') < order.index('signal_model')
        assert order.index('nvda') < order.index('nvda_ma')
        assert order.index('nvda') < order.index('signal_model')
        assert order.index('signal_model') < order.index('ensemble')
        assert order.index('nvda_ma') < order.index('ensemble')
        assert order.index('ensemble') < order.index('agent')
    
    def test_cycle_detection(self):
        source = """:agent
type=agent
inputs=[model_a]
tradable=[spy]

:model_a
type=model
class=ModelA
inputs=[model_b]

:model_b
type=model
class=ModelB
inputs=[model_a]

:spy
type=timeseries
inputs=[]
"""
        raw = parse_config(source)
        validated = validate(raw)
        resolved = resolve(validated, {'ModelA': type, 'ModelB': type})
        
        with pytest.raises(CycleError) as exc:
            build_dag(resolved)
        
        assert 'cycle' in str(exc.value).lower()
    
    def self_cycle_detection(self):
        source = """:model
type=model
class=Model
inputs=[model]

:spy
type=timeseries
inputs=[]
"""
        raw = parse_config(source)
        validated = validate(raw)
        resolved = resolve(validated, {'Model': type})
        
        with pytest.raises(CycleError):
            build_dag(resolved)
    
    def test_skips_metadata_blocks(self):
        source = """mlparams:
lr=0.001

:yfinance
type=adapter
class=adapters.YFAdapter

:backtest
type=backtest
start=2020-01-01
end=2024-01-01
capital=100000

:agent
type=agent
inputs=[spy]
tradable=[spy]

:spy
type=timeseries
inputs=[]
"""
        raw = parse_config(source)
        validated = validate(raw)
        
        class MockAdapter:
            pass
        
        resolved = resolve(validated, {'adapters.YFAdapter': MockAdapter})
        result = build_dag(resolved)
        
        assert '_params' not in result['_execution_order']
        assert '_adapters' not in result['_execution_order']
        assert '_backtest' not in result['_execution_order']
        
        assert 'spy' in result['_execution_order']
        assert 'agent' in result['_execution_order']
    
    def test_graph_metadata(self):
        source = """:agent
type=agent
inputs=[spy, model1]
tradable=[spy]

:model1
type=model
class=Model1
inputs=[nvda]

:nvda
type=timeseries
inputs=[]

:spy
type=timeseries
inputs=[]
"""
        raw = parse_config(source)
        validated = validate(raw)
        resolved = resolve(validated, {'Model1': type})
        result = build_dag(resolved)
        
        graph = result['_graph']
        
        assert 'nodes' in graph
        assert 'deps' in graph
        assert 'reverse_deps' in graph
        
        assert graph['deps']['agent'] == ['spy', 'model1']
        assert 'agent' in graph['reverse_deps']['spy']
        assert 'agent' in graph['reverse_deps']['model1']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
