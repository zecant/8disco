"""Tests for Schema Validation."""
import pytest
from tradsl import parse_dsl, validate_config, ConfigError


class TestSchemaValidation:
    def test_valid_timeseries_with_adapter(self):
        dsl = """yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

spy_data:
type=timeseries
adapter=yf
tradable=true

strategy:
type=agent
inputs=[spy_data]
tradable=[spy_data]
sizer=fixed
"""
        raw = parse_dsl(dsl)
        validated = validate_config(raw)

        assert 'spy_data' in validated
        assert validated['spy_data']['adapter'] == 'yf'
        assert validated['spy_data']['tradable'] is True

    def test_valid_timeseries_with_function(self):
        dsl = """yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

price_data:
type=timeseries
adapter=yf
tradable=true

returns:
type=timeseries
function=log_returns
inputs=[price_data]

strategy:
type=agent
inputs=[returns]
tradable=[price_data]
sizer=fixed
"""
        raw = parse_dsl(dsl)
        validated = validate_config(raw)

        assert 'returns' in validated
        assert validated['returns']['function'] == 'log_returns'

    def test_timeseries_must_have_adapter_or_function(self):
        dsl = """node:
type=timeseries

strategy:
type=agent
inputs=[node]
tradable=[node]
sizer=fixed
"""
        raw = parse_dsl(dsl)
        with pytest.raises(ConfigError) as exc:
            validate_config(raw)
        assert 'must have either' in str(exc.value)

    def test_timeseries_cannot_have_both_adapter_and_function(self):
        dsl = """yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

node:
type=timeseries
adapter=yf
function=log_returns

strategy:
type=agent
inputs=[node]
tradable=[node]
sizer=fixed
"""
        raw = parse_dsl(dsl)
        with pytest.raises(ConfigError) as exc:
            validate_config(raw)
        assert 'cannot have both' in str(exc.value)

    def test_function_requires_inputs(self):
        dsl = """node:
type=timeseries
function=log_returns

strategy:
type=agent
inputs=[node]
tradable=[node]
sizer=fixed
"""
        raw = parse_dsl(dsl)
        with pytest.raises(ConfigError) as exc:
            validate_config(raw)
        assert "requires 'inputs'" in str(exc.value)

    def test_defaults_filled(self):
        dsl = """backtest:
type=backtest
start=2008-01-01
end=2024-12-31
test_start=2022-01-01

yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

data:
type=timeseries
adapter=yf
tradable=true

strategy:
type=agent
inputs=[data]
tradable=[data]
sizer=fixed
"""
        raw = parse_dsl(dsl)
        validated = validate_config(raw)

        assert validated['backtest']['capital'] == 100000.0
        assert validated['backtest']['commission'] == 0.001

    def test_backtest_date_validation(self):
        dsl = """backtest:
type=backtest
start=2024-01-01
end=2024-12-31
test_start=2023-01-01

yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

data:
type=timeseries
adapter=yf
tradable=true

strategy:
type=agent
inputs=[data]
tradable=[data]
sizer=fixed
"""
        raw = parse_dsl(dsl)
        with pytest.raises(ConfigError) as exc:
            validate_config(raw)
        assert 'test_start' in str(exc.value) and 'after' in str(exc.value)

    def test_block_size_validation(self):
        dsl = """backtest:
type=backtest
start=2008-01-01
end=2024-12-31
test_start=2022-01-01
block_size_min=120
block_size_max=30

yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

data:
type=timeseries
adapter=yf
tradable=true

strategy:
type=agent
inputs=[data]
tradable=[data]
sizer=fixed
"""
        raw = parse_dsl(dsl)
        with pytest.raises(ConfigError) as exc:
            validate_config(raw)
        assert 'block_size_min' in str(exc.value) and 'block_size_max' in str(exc.value)

    def test_capital_must_be_positive(self):
        dsl = """backtest:
type=backtest
start=2008-01-01
end=2024-12-31
test_start=2022-01-01
capital=0.0

yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

data:
type=timeseries
adapter=yf
tradable=true

strategy:
type=agent
inputs=[data]
tradable=[data]
sizer=fixed
"""
        raw = parse_dsl(dsl)
        with pytest.raises(ConfigError) as exc:
            validate_config(raw)
        assert 'capital' in str(exc.value) and 'positive' in str(exc.value)

    def test_multiple_errors_collected(self):
        dsl = """backtest:
type=backtest
start=2024-01-01
end=2024-12-31
test_start=2023-01-01
capital=-100

yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

data:
type=timeseries
adapter=yf
tradable=true

strategy:
type=agent
inputs=[data]
tradable=[data]
sizer=fixed
"""
        raw = parse_dsl(dsl)
        with pytest.raises(ConfigError) as exc:
            validate_config(raw)
        errors = exc.value.errors
        assert len(errors) >= 2

    def test_resolved_params(self):
        dsl = """my_params:
key=value

yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

block:
type=timeseries
adapter=yf
params=my_params
tradable=true

strategy:
type=agent
inputs=[block]
tradable=[block]
sizer=fixed
"""
        raw = parse_dsl(dsl)
        validated = validate_config(raw)
        assert validated['block']['params'] == 'my_params'

    def test_unresolved_params_raises(self):
        dsl = """yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

block:
type=timeseries
adapter=yf
params=nonexistent
tradable=true

strategy:
type=agent
inputs=[block]
tradable=[block]
sizer=fixed
"""
        raw = parse_dsl(dsl)
        with pytest.raises(ConfigError) as exc:
            validate_config(raw)
        assert 'not found' in str(exc.value)

    def test_multiple_errors_list(self):
        dsl = """backtest:
type=backtest
start=2024-01-01
end=2024-12-31
test_start=2023-01-01
capital=-50

yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

data:
type=timeseries
adapter=yf
tradable=true

strategy:
type=agent
inputs=[data]
tradable=[data]
sizer=fixed
"""
        raw = parse_dsl(dsl)
        with pytest.raises(ConfigError) as exc:
            validate_config(raw)
        assert exc.value.errors
