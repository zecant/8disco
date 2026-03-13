"""Tests for Public API and Registration."""
import pytest
from tradsl import (
    load, parse_dsl, validate_config, resolve_config, build_execution_dag,
    register_adapter, register_function, register_trainable_model,
    register_agent, register_sizer, register_reward, get_registry
)


class TestPublicAPI:
    def test_parse_dsl(self):
        dsl = """test:
type=timeseries
adapter=yf
"""
        result = parse_dsl(dsl)
        assert 'test' in result

    def test_full_pipeline_load(self):
        dsl = """yf:
type=adapter
class=tradsl.adapters.YFinanceAdapter

data:
type=timeseries
adapter=yf

backtest:
type=backtest
start=2008-01-01
end=2024-12-31
test_start=2022-01-01
"""
        result = load(dsl)
        assert result.config is not None
        assert result.dag is not None
        assert result.metadata is not None


class TestRegistration:
    def test_register_adapter(self):
        class DummyAdapter:
            pass

        register_adapter('test_adapter', DummyAdapter)
        registry = get_registry()
        assert 'test_adapter' in registry['adapters']
        assert registry['adapters']['test_adapter'] == DummyAdapter

    def test_register_function(self):
        def dummy_func(arr, **params):
            return 1.0

        register_function(
            'test_func',
            dummy_func,
            category='safe',
            description='Test function',
            min_lookback=10
        )
        registry = get_registry()
        assert 'test_func' in registry['functions']
        assert registry['functions']['test_func']['category'] == 'safe'
        assert registry['functions']['test_func']['min_lookback'] == 10

    def test_register_trainable_model(self):
        class DummyModel:
            pass

        register_trainable_model('test_model', DummyModel)
        registry = get_registry()
        assert 'test_model' in registry['trainable_models']

    def test_register_agent(self):
        class DummyAgent:
            pass

        register_agent('test_agent', DummyAgent)
        registry = get_registry()
        assert 'test_agent' in registry['agents']

    def test_register_sizer(self):
        class DummySizer:
            pass

        register_sizer('test_sizer', DummySizer)
        registry = get_registry()
        assert 'test_sizer' in registry['sizers']

    def test_register_reward(self):
        class DummyReward:
            pass

        register_reward('test_reward', DummyReward)
        registry = get_registry()
        assert 'test_reward' in registry['rewards']
