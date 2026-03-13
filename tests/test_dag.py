"""Tests for DAG Construction and Topological Sort."""
import pytest
from tradsl import parse_dsl, validate_config, build_execution_dag, CycleError


class TestDAG:
    def test_simple_linear_dag(self):
        dsl = """yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

a:
type=timeseries
adapter=yf

b:
type=timeseries
function=log_returns
inputs=[a]

c:
type=timeseries
function=sma
inputs=[b]
"""
        raw = parse_dsl(dsl)
        validated = validate_config(raw)
        dag = build_execution_dag(validated)

        assert dag.metadata.execution_order == ['a', 'b', 'c']

    def test_parallel_branches(self):
        dsl = """yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

source:
type=timeseries
adapter=yf

branch_a:
type=timeseries
function=log_returns
inputs=[source]

branch_b:
type=timeseries
function=sma
inputs=[source]

merge:
type=timeseries
function=subtract
inputs=[branch_a, branch_b]
"""
        raw = parse_dsl(dsl)
        validated = validate_config(raw)
        dag = build_execution_dag(validated)

        order = dag.metadata.execution_order
        assert order.index('source') < order.index('branch_a')
        assert order.index('source') < order.index('branch_b')
        assert order.index('branch_a') < order.index('merge')
        assert order.index('branch_b') < order.index('merge')

    def test_cycle_detection(self):
        dsl = """a:
type=timeseries
function=func_a
inputs=[c]

b:
type=timeseries
function=func_b
inputs=[a]

c:
type=timeseries
function=func_c
inputs=[b]
"""
        raw = parse_dsl(dsl)
        validated = validate_config(raw)
        
        with pytest.raises(CycleError) as exc:
            build_execution_dag(validated)
        
        assert 'a' in exc.value.cycle_path

    def test_self_loop_detection(self):
        dsl = """node:
type=timeseries
function=func
inputs=[node]
"""
        raw = parse_dsl(dsl)
        validated = validate_config(raw)
        
        with pytest.raises(CycleError) as exc:
            build_execution_dag(validated)
        
        assert 'node' in exc.value.cycle_path

    def test_source_nodes_identified(self):
        dsl = """yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

source_a:
type=timeseries
adapter=yf

source_b:
type=timeseries
adapter=yf

computed:
type=timeseries
function=combine
inputs=[source_a, source_b]
"""
        raw = parse_dsl(dsl)
        validated = validate_config(raw)
        dag = build_execution_dag(validated)

        assert 'source_a' in dag.metadata.source_nodes
        assert 'source_b' in dag.metadata.source_nodes
        assert 'computed' not in dag.metadata.source_nodes

    def test_model_nodes_identified(self):
        dsl = """yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

data:
type=timeseries
adapter=yf
tradable=true

model_node:
type=model
class=PPOAgent
inputs=[data]
action_space=discrete
reward_function=test_reward
training_window=504

strategy:
type=agent
inputs=[model_node]
tradable=[data]
sizer=fixed
"""
        raw = parse_dsl(dsl)
        validated = validate_config(raw)
        dag = build_execution_dag(validated)

        assert 'model_node' in dag.metadata.model_nodes

    def test_buffer_sizes_computed(self):
        dsl = """yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

data:
type=timeseries
adapter=yf
window=20

feature:
type=timeseries
function=sma
inputs=[data]
window=50
"""
        raw = parse_dsl(dsl)
        validated = validate_config(raw)
        dag = build_execution_dag(validated)

        assert dag.metadata.node_buffer_sizes['data'] == 20
        assert dag.metadata.node_buffer_sizes['feature'] == 50

    def test_warmup_bars_computed(self):
        dsl = """yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

source:
type=timeseries
adapter=yf
window=20

feature:
type=timeseries
function=sma
inputs=[source]
window=50
"""
        raw = parse_dsl(dsl)
        validated = validate_config(raw)
        dag = build_execution_dag(validated)

        assert dag.metadata.warmup_bars == 70

    def test_dag_edges(self):
        dsl = """yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

a:
type=timeseries
adapter=yf

b:
type=timeseries
function=func
inputs=[a]
"""
        raw = parse_dsl(dsl)
        validated = validate_config(raw)
        dag = build_execution_dag(validated)

        assert 'b' in dag.edges['a']
        assert 'a' in dag.reverse_edges['b']
