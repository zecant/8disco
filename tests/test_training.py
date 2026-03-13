"""
Tests for training module.
"""
import pytest
import numpy as np
from tradsl.training import (
    Block,
    TrainingConfig,
    BlockSampler,
    Experience,
    ReplayBuffer,
    PolicyUpdateResult,
    BlockTrainer,
    WalkForwardTester
)


class TestBlock:
    """Tests for Block dataclass."""
    
    def test_create_block(self):
        """Test creating a block."""
        block = Block(start_idx=100, end_idx=200)
        assert block.start_idx == 100
        assert block.end_idx == 200
        assert block.length == 100
    
    def test_block_with_dates(self):
        """Test block with date strings."""
        block = Block(start_idx=100, end_idx=200, start_date="2020-01-01", end_date="2020-02-01")
        assert block.start_date == "2020-01-01"
        assert block.end_date == "2020-02-01"


class TestTrainingConfig:
    """Tests for TrainingConfig."""
    
    def test_default_values(self):
        """Test default configuration."""
        config = TrainingConfig(training_window=504)
        
        assert config.training_window == 504
        assert config.retrain_schedule == "every_n_bars"
        assert config.retrain_n == 252
        assert config.n_training_blocks == 40
        assert config.block_size_min == 30
        assert config.block_size_max == 120
        assert config.seed == 42
    
    def test_validation_valid(self):
        """Test validation with valid config."""
        config = TrainingConfig(
            training_window=1000,
            n_training_blocks=5,
            block_size_min=30,
            block_size_max=60
        )
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validation_block_size_order(self):
        """Test validation catches reversed block sizes."""
        config = TrainingConfig(
            training_window=504,
            block_size_min=60,
            block_size_max=30
        )
        errors = config.validate()
        assert any("block_size_min must be less than block_size_max" in e for e in errors)
    
    def test_validation_coverage(self):
        """Test validation catches excessive coverage."""
        config = TrainingConfig(
            training_window=100,
            n_training_blocks=50,
            block_size_min=30,
            block_size_max=30
        )
        errors = config.validate()
        assert any("exceeds 80%" in e for e in errors)


class TestBlockSampler:
    """Tests for BlockSampler."""
    
    def test_sample_blocks(self):
        """Test sampling blocks."""
        config = TrainingConfig(
            training_window=500,
            n_training_blocks=5,
            block_size_min=30,
            block_size_max=50,
            seed=42
        )
        
        sampler = BlockSampler(config)
        blocks = sampler.sample_blocks(0, 400)
        
        assert len(blocks) > 0
        assert all(isinstance(b, Block) for b in blocks)
    
    def test_blocks_are_sorted(self):
        """Test blocks are sorted by start index."""
        config = TrainingConfig(
            training_window=500,
            n_training_blocks=5,
            block_size_min=30,
            block_size_max=50,
            seed=123
        )
        
        sampler = BlockSampler(config)
        blocks = sampler.sample_blocks(0, 400)
        
        if len(blocks) > 1:
            for i in range(len(blocks) - 1):
                assert blocks[i].start_idx < blocks[i + 1].start_idx
    
    def test_blocks_non_overlapping(self):
        """Test blocks don't overlap."""
        config = TrainingConfig(
            training_window=500,
            n_training_blocks=5,
            block_size_min=30,
            block_size_max=50,
            seed=456
        )
        
        sampler = BlockSampler(config)
        blocks = sampler.sample_blocks(0, 400)
        
        for i, b1 in enumerate(blocks):
            for j, b2 in enumerate(blocks):
                if i != j:
                    assert not (b1.start_idx < b2.end_idx and b2.start_idx < b1.end_idx)


class TestExperience:
    """Tests for Experience dataclass."""
    
    def test_create_experience(self):
        """Test creating experience."""
        obs = np.array([1.0, 2.0])
        exp = Experience(
            observation=obs,
            action=0,
            reward=1.0,
            next_observation=obs * 1.1,
            done=False
        )
        
        assert np.array_equal(exp.observation, obs)
        assert exp.action == 0
        assert exp.reward == 1.0
        assert exp.done is False


class TestReplayBuffer:
    """Tests for ReplayBuffer."""
    
    def test_init(self):
        """Test buffer initialization."""
        buffer = ReplayBuffer(capacity=100)
        assert len(buffer) == 0
        assert buffer.capacity == 100
    
    def test_add_experience(self):
        """Test adding experiences."""
        buffer = ReplayBuffer(capacity=10)
        
        obs = np.array([1.0])
        exp = Experience(
            observation=obs,
            action=0,
            reward=1.0,
            next_observation=obs,
            done=False
        )
        
        buffer.add(exp)
        assert len(buffer) == 1
    
    def test_capacity_limit(self):
        """Test capacity is limited."""
        buffer = ReplayBuffer(capacity=2)
        
        for i in range(5):
            obs = np.array([float(i)])
            exp = Experience(
                observation=obs,
                action=i,
                reward=float(i),
                next_observation=obs,
                done=False
            )
            buffer.add(exp)
        
        assert len(buffer) == 2
    
    def test_sample_uniform(self):
        """Test uniform sampling."""
        buffer = ReplayBuffer(capacity=100, prioritized=False)
        
        for i in range(50):
            obs = np.array([float(i)])
            exp = Experience(
                observation=obs,
                action=i,
                reward=float(i),
                next_observation=obs,
                done=False
            )
            buffer.add(exp)
        
        batch = buffer.sample(10)
        assert len(batch) == 10
    
    def test_sample_prioritized(self):
        """Test prioritized sampling."""
        buffer = ReplayBuffer(capacity=100, prioritized=True, alpha=0.6, beta=0.4)
        
        for i in range(50):
            obs = np.array([float(i)])
            exp = Experience(
                observation=obs,
                action=i,
                reward=float(i),
                next_observation=obs,
                done=False,
                priority=float(i + 1)
            )
            buffer.add(exp)
        
        batch = buffer.sample(10)
        assert len(batch) == 10
    
    def test_clear(self):
        """Test clearing buffer."""
        buffer = ReplayBuffer(capacity=10)
        
        obs = np.array([1.0])
        exp = Experience(
            observation=obs,
            action=0,
            reward=1.0,
            next_observation=obs,
            done=False
        )
        
        buffer.add(exp)
        buffer.clear()
        
        assert len(buffer) == 0


class TestBlockTrainer:
    """Tests for BlockTrainer."""
    
    def test_init(self):
        """Test trainer initialization."""
        config = TrainingConfig(training_window=504)
        
        class MockAgent:
            pass
        
        class MockReward:
            pass
        
        trainer = BlockTrainer(config, MockAgent(), MockReward())
        
        assert trainer.config == config
    
    def test_train_returns_dict(self):
        """Test train returns expected dict."""
        config = TrainingConfig(
            training_window=500,
            n_training_blocks=3,
            block_size_min=30,
            block_size_max=40,
            seed=42
        )
        
        class MockAgent:
            replay_buffer = ReplayBuffer(capacity=100)
            def reset(self): pass
            def should_update(self, idx, recent_performance):
                return False
        
        class MockReward:
            pass
        
        trainer = BlockTrainer(config, MockAgent(), MockReward())
        
        data = np.random.randn(400, 5)
        
        def execute_bar_fn(bar_idx, bar_data):
            return {'portfolio_value': 100000.0, 'rolling_sharpe': 0.0}
        
        result = trainer.train(data, 20, execute_bar_fn)
        
        assert 'n_blocks' in result
        assert 'total_bars' in result


class TestWalkForwardTester:
    """Tests for WalkForwardTester."""
    
    def test_init(self):
        """Test tester initialization."""
        config = TrainingConfig(training_window=504)
        
        class MockAgent:
            pass
        
        tester = WalkForwardTester(config, MockAgent())
        
        assert tester.config == config
    
    def test_test_returns_dict(self):
        """Test test returns expected dict."""
        config = TrainingConfig(training_window=504, seed=42)
        
        class MockAgent:
            def should_update(self, idx, recent_performance):
                return False
        
        tester = WalkForwardTester(config, MockAgent())
        
        data = np.random.randn(200, 5)
        
        def execute_bar_fn(bar_idx, bar_data):
            return {'portfolio_value': 100000.0, 'rolling_sharpe': 0.0}
        
        result = tester.test(data, 100, 20, execute_bar_fn)
        
        assert 'equity_curve' in result
        assert 'n_bars' in result
        assert result['n_bars'] == 100
