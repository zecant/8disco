"""
End-to-end integration tests for DAG construction.

Tests: DSL string -> Parser -> DAG -> Buffer sizes.
"""
import pytest
from tradsl.parser import parse
from tradsl.dag import DAG


class TestSimplePipeline:
    """Basic end-to-end tests."""

    def test_single_timeseries(self):
        dsl = """
ts1:
type=timeseries
adapter=yfinance
"""
        dag = DAG.from_config(parse(dsl)).build()
        assert "ts1" in dag.nodes
        assert dag.nodes["ts1"].buffer_size == 1
        assert dag.execution_order == ["ts1"]

    def test_timeseries_with_function(self):
        dsl = """
raw:
type=timeseries
adapter=yfinance

sma30:
type=function
function=sma
inputs=[raw]
window=30
"""
        dag = DAG.from_config(parse(dsl)).build()
        assert dag.nodes["sma30"].buffer_size == 1
        assert dag.nodes["raw"].buffer_size == 30

    def test_chain(self):
        dsl = """
source:
type=timeseries
adapter=yfinance

step1:
type=function
function=a
inputs=[source]
window=10

step2:
type=function
function=b
inputs=[step1]
window=20

step3:
type=function
function=c
inputs=[step2]
window=5
"""
        dag = DAG.from_config(parse(dsl)).build()
        assert dag.nodes["step3"].buffer_size == 1
        assert dag.nodes["step2"].buffer_size == 5
        assert dag.nodes["step1"].buffer_size == 20
        assert dag.nodes["source"].buffer_size == 10


class TestParallelBranches:
    """Tests for parallel branches."""

    def test_fast_slow_signal(self):
        dsl = """
data:
type=timeseries
adapter=yfinance

fast:
type=function
function=fast_ma
inputs=[data]
window=5

slow:
type=function
function=slow_ma
inputs=[data]
window=20

signal:
type=function
function=crossover
inputs=[fast, slow]
"""
        dag = DAG.from_config(parse(dsl)).build()
        assert dag.nodes["signal"].buffer_size == 1
        assert dag.nodes["fast"].buffer_size == 1
        assert dag.nodes["slow"].buffer_size == 1
        assert dag.nodes["data"].buffer_size == 20

    def test_diamond(self):
        dsl = """
a:
type=timeseries
adapter=data

b:
type=function
function=f
inputs=[a]
window=10

c:
type=function
function=g
inputs=[a]
window=5

d:
type=function
function=h
inputs=[b, c]
"""
        dag = DAG.from_config(parse(dsl)).build()
        assert dag.nodes["d"].buffer_size == 1
        assert dag.nodes["b"].buffer_size == 1
        assert dag.nodes["c"].buffer_size == 1
        assert dag.nodes["a"].buffer_size == 10


class TestComplexStrategies:
    """Tests for realistic strategies."""

    def test_mean_reversion(self):
        dsl = """
prices:
type=timeseries
adapter=yfinance

close:
type=function
function=get_close
inputs=[prices]

sma_20:
type=function
function=sma
inputs=[close]
window=20

sma_50:
type=function
function=sma
inputs=[close]
window=50

signal:
type=function
function=signal
inputs=[sma_20, sma_50]
"""
        dag = DAG.from_config(parse(dsl)).build()
        assert dag.nodes["signal"].buffer_size == 1
        assert dag.nodes["sma_20"].buffer_size == 1
        assert dag.nodes["sma_50"].buffer_size == 1
        assert dag.nodes["close"].buffer_size == 50
        assert dag.nodes["prices"].buffer_size == 1

    def test_momentum(self):
        dsl = """
prices:
type=timeseries
adapter=yfinance

close:
type=function
function=get_close
inputs=[prices]

volume:
type=function
function=get_volume
inputs=[prices]

returns:
type=function
function=pct_change
inputs=[close]

volume_sma:
type=function
function=sma
inputs=[volume]
window=20

momentum:
type=function
function=cumsum
inputs=[returns]
window=30

signal:
type=function
function=sign
inputs=[momentum, volume_sma]
"""
        dag = DAG.from_config(parse(dsl)).build()
        assert dag.nodes["signal"].buffer_size == 1
        assert dag.nodes["momentum"].buffer_size == 1
        assert dag.nodes["returns"].buffer_size == 30
        assert dag.nodes["volume_sma"].buffer_size == 1


class TestEdgeCases:
    """Edge cases."""

    def test_empty_dsl(self):
        dag = DAG.from_config(parse("")).build()
        assert dag.nodes == {}

    def test_whitespace_only(self):
        dag = DAG.from_config(parse("   \n   ")).build()
        assert dag.nodes == {}

    def test_large_window(self):
        dsl = """
data:
type=timeseries
adapter=yfinance

feature:
type=function
function=rolling
inputs=[data]
window=1000
"""
        dag = DAG.from_config(parse(dsl)).build()
        assert dag.nodes["feature"].buffer_size == 1
        assert dag.nodes["data"].buffer_size == 1000

    def test_window_one(self):
        dsl = """
data:
type=timeseries
adapter=yfinance

feature:
type=function
function=identity
inputs=[data]
window=1
"""
        dag = DAG.from_config(parse(dsl)).build()
        assert dag.nodes["data"].buffer_size == 1
        assert dag.nodes["feature"].buffer_size == 1

    def test_multiple_inputs_same_source(self):
        dsl = """
source:
type=timeseries
adapter=data

fn1:
type=function
function=a
inputs=[source]
window=10

fn2:
type=function
function=b
inputs=[source]
window=20

combined:
type=function
function=c
inputs=[fn1, fn2]
"""
        dag = DAG.from_config(parse(dsl)).build()
        assert dag.nodes["combined"].buffer_size == 1
        assert dag.nodes["fn1"].buffer_size == 1
        assert dag.nodes["fn2"].buffer_size == 1
        assert dag.nodes["source"].buffer_size == 20


class TestResolution:
    """Tests for resolution."""

    def test_resolve_function_names(self):
        dsl = """
ts:
type=timeseries
adapter=yfinance

fn:
type=function
function=my_function
inputs=[ts]
"""
        dag = DAG.from_config(parse(dsl)).build()

        registry = {"my_function": lambda *args: args}
        dag.resolve(registry)
        assert callable(dag.nodes["fn"].attrs["function"])

    def test_preserves_structure(self):
        dsl = """
data:
type=timeseries
adapter=yfinance

transform:
type=function
function=process
inputs=[data]
window=10
"""
        dag = DAG.from_config(parse(dsl)).build()
        registry = {"process": lambda x: x}
        dag.resolve(registry)
        
        assert "data" in dag.nodes
        assert "transform" in dag.nodes
        assert dag.nodes["data"].buffer_size == 10
        assert dag.nodes["transform"].buffer_size == 1


class TestStability:
    """Tests for deterministic ordering."""

    def test_alphabetical_stability(self):
        dsl = """
z_node:
type=timeseries
adapter=a

a_node:
type=timeseries
adapter=b

m_node:
type=timeseries
adapter=c
"""
        dag = DAG.from_config(parse(dsl)).build()
        assert dag.execution_order == ["a_node", "m_node", "z_node"]

    def test_same_input_multiple_times(self):
        dsl = """
source:
type=timeseries
adapter=data

use1:
type=function
function=f1
inputs=[source]
window=10

use2:
type=function
function=f2
inputs=[source]
window=20

final:
type=function
function=combine
inputs=[use1, use2]
"""
        dag = DAG.from_config(parse(dsl)).build()
        assert dag.nodes["final"].buffer_size == 1
        assert dag.nodes["use1"].buffer_size == 1
        assert dag.nodes["use2"].buffer_size == 1
        assert dag.nodes["source"].buffer_size == 20


class TestWideGraph:
    """Tests for wide graphs."""

    def test_many_leaves_one_sink(self):
        dsl = """
leaf0:
type=timeseries
adapter=a

leaf1:
type=timeseries
adapter=b

leaf2:
type=timeseries
adapter=c

leaf3:
type=timeseries
adapter=d

leaf4:
type=timeseries
adapter=e

middle0:
type=function
function=m0
inputs=[leaf0, leaf1]
window=10

middle1:
type=function
function=m1
inputs=[leaf2, leaf3]
window=15

root:
type=function
function=root
inputs=[middle0, middle1, leaf4]
window=5
"""
        dag = DAG.from_config(parse(dsl)).build()
        assert dag.nodes["root"].buffer_size == 1
        assert dag.nodes["middle0"].buffer_size == 5
        assert dag.nodes["middle1"].buffer_size == 5
        assert dag.nodes["leaf0"].buffer_size == 10
        assert dag.nodes["leaf1"].buffer_size == 10
        assert dag.nodes["leaf2"].buffer_size == 15
        assert dag.nodes["leaf3"].buffer_size == 15
        assert dag.nodes["leaf4"].buffer_size == 5


class TestDeepGraph:
    """Tests for deep graphs."""

    def test_deep_chain(self):
        dsl = "n0:\ntype=timeseries\nadapter=a\n"
        for i in range(1, 50):
            dsl += f"""
n{i}:
type=function
function=f{i}
inputs=[n{i-1}]
window=1
"""
        dsl += """
n49:
type=function
function=f49
inputs=[n48]
window=10
"""
        dag = DAG.from_config(parse(dsl)).build()
        assert len(dag.execution_order) == 50
        assert dag.nodes["n0"].buffer_size == 1
        assert dag.nodes["n49"].buffer_size == 1


class TestRealWorldPatterns:
    """Real-world usage patterns."""

    def test_technical_indicators(self):
        dsl = """
ohlcv:
type=timeseries
adapter=yfinance

close:
type=function
function=get_close
inputs=[ohlcv]

volume:
type=function
function=get_volume
inputs=[ohlcv]

change:
type=function
function=diff
inputs=[close]

gain:
type=function
function=positive_part
inputs=[change]

loss:
type=function
function=negative_part
inputs=[change]

avg_gain:
type=function
function=sma
inputs=[gain]
window=14

avg_loss:
type=function
function=sma
inputs=[loss]
window=14

rsi:
type=function
function=rsi
inputs=[avg_gain, avg_loss]

ema12:
type=function
function=ema
inputs=[close]
window=12

ema26:
type=function
function=ema
inputs=[close]
window=26

macd_line:
type=function
function=subtract
inputs=[ema12, ema26]

signal_line:
type=function
function=ema
inputs=[macd_line]
window=9

macd_hist:
type=function
function=subtract
inputs=[macd_line, signal_line]

features:
type=function
function=stack
inputs=[rsi, macd_hist, volume]

agent:
type=function
function=agent
inputs=[features]
"""
        dag = DAG.from_config(parse(dsl)).build()
        
        assert dag.nodes["agent"].buffer_size == 1
        assert dag.nodes["features"].buffer_size == 1
        assert dag.nodes["macd_hist"].buffer_size == 1
        assert dag.nodes["signal_line"].buffer_size == 1
        assert dag.nodes["macd_line"].buffer_size == 9
        assert dag.nodes["ema12"].buffer_size == 1
        assert dag.nodes["ema26"].buffer_size == 1
        assert dag.nodes["rsi"].buffer_size == 1
        assert dag.nodes["avg_gain"].buffer_size == 1
        assert dag.nodes["avg_loss"].buffer_size == 1
        assert dag.nodes["gain"].buffer_size == 14
        assert dag.nodes["loss"].buffer_size == 14
        assert dag.nodes["change"].buffer_size == 1
        assert dag.nodes["close"].buffer_size == 26
        assert dag.nodes["volume"].buffer_size == 1
        assert dag.nodes["ohlcv"].buffer_size == 1
