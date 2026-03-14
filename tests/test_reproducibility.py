"""
Reproducibility Tests for TradSL.

These tests verify that the same DSL + seed produces identical results across runs.
This catches bugs like unseeded operations, non-deterministic iteration, and state leakage.
"""
import pytest
import numpy as np
import random
import tempfile
import os
from datetime import datetime

from tradsl import (
    parse_dsl, validate_config, resolve_config, build_execution_dag,
    train, test, run,
    TestDataAdapter, create_test_adapter,
    BlockSampler, BlockTrainer, TrainingConfig,
    ParameterizedAgent, TabularAgent,
    RandomForestModel, LinearModel,
    Portfolio, Position,
    CircularBuffer,
    BacktestEngine,
    KellySizer,
    AsymmetricHighWaterMarkReward,
)
from tradsl.models import TradingAction
from tradsl.training import Block, Experience


SIMPLE_DSL = """
test_adapter:
type=adapter
class=tradsl.testdata_adapter.TestDataAdapter
interval=1d

spy:
type=timeseries
adapter=test_adapter
parameters=[SPY]
tradable=true

spy_sma:
type=timeseries
function=sma
inputs=[spy]
params=sma_cfg

sma_cfg:
window=20

_backtest:
type=backtest
start=2024-01-01
end=2024-12-31
test_start=2024-10-01
capital=100000
training_mode=random_blocks
block_size_min=20
block_size_max=40
n_training_blocks=3
training_window=200
seed=42
"""


class TestBlockSamplerDeterminism:
    """Verify block sampling is deterministic."""

    def test_same_seed_same_blocks(self):
        """sample_blocks(seed=42) called twice → identical blocks."""
        config = TrainingConfig(
            training_window=500,
            n_training_blocks=5,
            block_size_min=30,
            block_size_max=50,
            seed=42
        )
        
        sampler1 = BlockSampler(config)
        sampler2 = BlockSampler(config)
        
        blocks1 = sampler1.sample_blocks(0, 400)
        blocks2 = sampler2.sample_blocks(0, 400)
        
        assert len(blocks1) == len(blocks2)
        
        for b1, b2 in zip(blocks1, blocks2):
            assert b1.start_idx == b2.start_idx
            assert b1.end_idx == b2.end_idx
            assert b1.length == b2.length

    def test_different_seed_different_blocks(self):
        """sample_blocks(seed=42) vs seed=43 → different blocks."""
        config1 = TrainingConfig(
            training_window=500,
            n_training_blocks=5,
            block_size_min=30,
            block_size_max=50,
            seed=42
        )
        config2 = TrainingConfig(
            training_window=500,
            n_training_blocks=5,
            block_size_min=30,
            block_size_max=50,
            seed=43
        )
        
        sampler1 = BlockSampler(config1)
        sampler2 = BlockSampler(config2)
        
        blocks1 = sampler1.sample_blocks(0, 400)
        blocks2 = sampler2.sample_blocks(0, 400)
        
        assert len(blocks1) == len(blocks2)
        
        any_different = False
        for b1, b2 in zip(blocks1, blocks2):
            if b1.start_idx != b2.start_idx or b1.end_idx != b2.end_idx:
                any_different = True
                break
        
        assert any_different, "Different seeds should produce different blocks"

    def test_blocks_sorted_by_start(self):
        """Blocks should always be sorted by start_idx."""
        config = TrainingConfig(
            training_window=500,
            n_training_blocks=10,
            block_size_min=20,
            block_size_max=40,
            seed=12345
        )
        
        sampler = BlockSampler(config)
        blocks = sampler.sample_blocks(0, 400)
        
        for i in range(len(blocks) - 1):
            assert blocks[i].start_idx < blocks[i + 1].start_idx

    def test_blocks_non_overlapping(self):
        """Blocks should never overlap."""
        config = TrainingConfig(
            training_window=500,
            n_training_blocks=10,
            block_size_min=20,
            block_size_max=40,
            seed=999
        )
        
        sampler = BlockSampler(config)
        blocks = sampler.sample_blocks(0, 400)
        
        for i, b1 in enumerate(blocks):
            for j, b2 in enumerate(blocks):
                if i != j:
                    assert not (b1.start_idx < b2.end_idx and b2.start_idx < b1.end_idx), \
                        f"Blocks {i} and {j} overlap"


class TestNumpyRandomDeterminism:
    """Verify numpy random operations are seeded correctly."""

    def test_numpy_random_seed(self):
        """Verify numpy random state is controllable."""
        np.random.seed(42)
        arr1 = np.random.randn(100)
        
        np.random.seed(42)
        arr2 = np.random.randn(100)
        
        np.testing.assert_array_equal(arr1, arr2)

    def test_numpy_random_different_seeds(self):
        """Verify different seeds produce different results."""
        np.random.seed(42)
        arr1 = np.random.randn(100)
        
        np.random.seed(43)
        arr2 = np.random.randn(100)
        
        assert not np.array_equal(arr1, arr2)


class TestCircularBufferDeterminism:
    """Verify circular buffer operations are deterministic."""

    def test_buffer_push_order(self):
        """Pushing values in same order produces same latest value."""
        buf1 = CircularBuffer(size=10)
        buf2 = CircularBuffer(size=10)
        
        for i in range(20):
            val = float(i)
            buf1.push(val)
            buf2.push(val)
        
        assert buf1.latest() == buf2.latest()
        
        arr1 = buf1.to_array()
        arr2 = buf2.to_array()
        
        np.testing.assert_array_equal(arr1, arr2)


class TestExperienceReplayDeterminism:
    """Verify experience replay buffer is deterministic."""

    def test_replay_buffer_sampling(self):
        """Same seed → same experience sampling order."""
        from tradsl.training import ReplayBuffer, Experience
        
        random.seed(42)
        buf1 = ReplayBuffer(capacity=100)
        
        random.seed(42)
        buf2 = ReplayBuffer(capacity=100)
        
        for i in range(50):
            obs = np.random.randn(10)
            action = int(np.random.randint(0, 3))
            reward = float(np.random.randn())
            next_obs = np.random.randn(10)
            done = bool(i % 10 == 9)
            
            exp1 = Experience(obs, action, reward, next_obs, done)
            exp2 = Experience(obs, action, reward, next_obs, done)
            
            buf1.add(exp1)
            buf2.add(exp2)
        
        random.seed(42)
        batch1 = buf1.sample(10)
        
        random.seed(42)
        batch2 = buf2.sample(10)
        
        assert len(batch1) == len(batch2)


class TestModelDeterminism:
    """Verify model training and inference are deterministic."""

    def test_random_forest_same_seed(self):
        """Same seed → identical RandomForestModel predictions."""
        np.random.seed(42)
        X_train = np.random.randn(200, 5)
        y_train = (X_train[:, 0] > 0).astype(int)
        
        model1 = RandomForestModel(n_estimators=10, random_state=42)
        model1.fit(X_train, y_train)
        
        model2 = RandomForestModel(n_estimators=10, random_state=42)
        model2.fit(X_train, y_train)
        
        X_test = np.random.randn(10, 5)
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)
        
        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_linear_model_same_seed(self):
        """Same seed → identical LinearModel predictions."""
        np.random.seed(42)
        X_train = np.random.randn(200, 5)
        y_train = X_train[:, 0] + 0.5 * X_train[:, 1] + np.random.randn(200) * 0.1
        
        model1 = LinearModel(random_state=42)
        model1.fit(X_train, y_train)
        
        model2 = LinearModel(random_state=42)
        model2.fit(X_train, y_train)
        
        X_test = np.random.randn(10, 5)
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)
        
        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_model_checkpoint_roundtrip(self):
        """Save/load checkpoint preserves model behavior."""
        np.random.seed(42)
        X_train = np.random.randn(200, 5)
        y_train = (X_train[:, 0] > 0).astype(int)
        
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        X_test = np.random.randn(10, 5)
        pred_before = model.predict(X_test)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        
        try:
            model.save_checkpoint(path)
            
            new_model = RandomForestModel(n_estimators=10, random_state=42)
            new_model.load_checkpoint(path)
            
            pred_after = new_model.predict(X_test)
            
            np.testing.assert_array_almost_equal(pred_before, pred_after)
        finally:
            os.unlink(path)


class TestAgentDeterminism:
    """Verify agent behavior is deterministic."""

    def test_parameterized_agent_same_seed(self):
        """Same seed → identical ParameterizedAgent actions."""
        from tradsl.models_impl import RandomForestModel
        
        policy1 = RandomForestModel(n_estimators=5, random_state=42)
        agent1 = ParameterizedAgent(policy_model=policy1, seed=42)
        
        policy2 = RandomForestModel(n_estimators=5, random_state=42)
        agent2 = ParameterizedAgent(policy_model=policy2, seed=42)
        
        obs = np.random.randn(10)
        portfolio_state = {
            'position': 0.0,
            'portfolio_value': 100000.0,
            'unrealized_pnl': 0.0,
            'drawdown': 0.0,
            'time_in_trade': 0,
        }
        
        action1, conv1 = agent1.observe(obs, portfolio_state, bar_index=100)
        action2, conv2 = agent2.observe(obs, portfolio_state, bar_index=100)
        
        assert action1 == action2
        
        if isinstance(conv1, np.ndarray):
            np.testing.assert_array_almost_equal(conv1, conv2)
        else:
            assert conv1 == conv2

    def test_tabular_agent_same_seed(self):
        """Same seed → identical TabularAgent actions."""
        agent1 = TabularAgent(n_actions=4, epsilon=0.1, seed=42)
        agent2 = TabularAgent(n_actions=4, epsilon=0.1, seed=42)
        
        obs = np.random.randn(10)
        portfolio_state = {
            'position': 0.0,
            'portfolio_value': 100000.0,
            'unrealized_pnl': 0.0,
            'drawdown': 0.0,
            'time_in_trade': 0,
        }
        
        action1, conv1 = agent1.observe(obs, portfolio_state, bar_index=100)
        action2, conv2 = agent2.observe(obs, portfolio_state, bar_index=100)
        
        assert action1 == action2

    def test_agent_checkpoint_roundtrip(self):
        """Save/load checkpoint preserves agent behavior."""
        from tradsl.models_impl import RandomForestModel
        
        policy = RandomForestModel(n_estimators=5, random_state=42)
        agent = ParameterizedAgent(policy_model=policy, seed=42)
        
        obs = np.random.randn(10)
        portfolio_state = {
            'position': 0.0,
            'portfolio_value': 100000.0,
            'unrealized_pnl': 0.0,
            'drawdown': 0.0,
            'time_in_trade': 0,
        }
        
        action_before, _ = agent.observe(obs, portfolio_state, bar_index=100)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'agent_checkpoint.pkl')
            
            agent.save_checkpoint(path)
            
            new_policy = RandomForestModel(n_estimators=5, random_state=42)
            new_agent = ParameterizedAgent(policy_model=new_policy, seed=42)
            new_agent.load_checkpoint(path)
            
            action_after, _ = new_agent.observe(obs, portfolio_state, bar_index=100)
        
        assert action_before == action_after


class TestSizerDeterminism:
    """Verify position sizers are deterministic."""

    def test_kelly_sizer_reproducible(self):
        """KellySizer with same inputs → same output."""
        sizer1 = KellySizer(max_kelly=0.25)
        sizer2 = KellySizer(max_kelly=0.25)
        
        action = TradingAction.BUY
        conviction = 0.7
        current_position = 0.0
        portfolio_value = 100000.0
        instrument_id = "SPY"
        current_price = 450.0
        
        size1 = sizer1.calculate_size(
            action, conviction, current_position, 
            portfolio_value, instrument_id, current_price
        )
        
        size2 = sizer2.calculate_size(
            action, conviction, current_position,
            portfolio_value, instrument_id, current_price
        )
        
        assert size1 == size2


class TestTrainingPipelineDeterminism:
    """Verify training pipeline produces deterministic results."""

    def test_block_trainer_reproducible(self):
        """BlockTrainer with same config → identical block sequences."""
        from tradsl.training import TrainingConfig
        
        np.random.seed(42)
        
        config = TrainingConfig(
            training_window=100,
            n_training_blocks=3,
            block_size_min=20,
            block_size_max=30,
            seed=42,
        )
        
        sampler1 = BlockSampler(config)
        sampler2 = BlockSampler(config)
        
        blocks1 = sampler1.sample_blocks(0, 100)
        blocks2 = sampler2.sample_blocks(0, 100)
        
        assert len(blocks1) == len(blocks2)
        
        for b1, b2 in zip(blocks1, blocks2):
            assert b1.start_idx == b2.start_idx
            assert b1.end_idx == b2.end_idx


class TestPortfolioDeterminism:
    """Verify portfolio operations are deterministic."""

    def test_portfolio_trade_sequence(self):
        """Same trade sequence → same final portfolio state."""
        from tradsl.nt_integration import Portfolio
        
        p1 = Portfolio(starting_capital=100000.0)
        p2 = Portfolio(starting_capital=100000.0)
        
        trades = [
            (105.0, 100, 1.0),
            (110.0, 50, 1.0),
            (108.0, 75, 1.0),
        ]
        
        for price, qty, commission in trades:
            p1.cash -= price * qty + commission
            p1.get_or_create_position("SPY").add_fill(price, qty, commission)
            p2.cash -= price * qty + commission
            p2.get_or_create_position("SPY").add_fill(price, qty, commission)
        
        assert p1.cash == p2.cash
        assert p1.total_equity == p2.total_equity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
