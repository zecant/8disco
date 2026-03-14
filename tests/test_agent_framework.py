"""Tests for agent framework."""
import pytest
import numpy as np
import os
import tempfile

from tradsl.agent_framework import (
    ParameterizedAgent, TabularAgent, PPOUpdate, DQNUpdate, PolicyGradientUpdate
)
from tradsl.models import TradingAction
from tradsl.training import Experience


class TestParameterizedAgent:
    def test_init_default(self):
        from tradsl.models_impl import RandomForestModel
        policy = RandomForestModel(n_estimators=5, random_state=42)
        agent = ParameterizedAgent(policy_model=policy)
        
        assert agent.action_space == "discrete"
        assert agent.algorithm_name == "ppo"
        assert agent.n_actions == 4
        assert agent.is_trained is False
    
    def test_init_custom(self):
        from tradsl.models_impl import RandomForestModel
        policy = RandomForestModel(n_estimators=5, random_state=42)
        agent = ParameterizedAgent(
            policy_model=policy,
            action_space="discrete",
            algorithm="ppo",
            update_schedule="every_n_bars",
            update_n=100,
            entropy_coef=0.02,
            seed=42
        )
        
        assert agent.update_schedule == "every_n_bars"
        assert agent.update_n == 100
        assert agent.entropy_coef == 0.02
    
    def test_observe_not_trained_returns_hold(self):
        from tradsl.models_impl import RandomForestModel
        policy = RandomForestModel(n_estimators=5, random_state=42)
        agent = ParameterizedAgent(policy_model=policy, seed=42)
        
        observation = np.random.randn(10)
        portfolio_state = {'position': 0.0, 'portfolio_value': 100000.0}
        
        action, conviction = agent.observe(observation, portfolio_state, bar_index=10)
        
        assert action == TradingAction.HOLD
        assert conviction == 0.0
    
    def test_observe_after_training(self):
        from tradsl.models_impl import RandomForestModel
        policy = RandomForestModel(n_estimators=5, random_state=42)
        agent = ParameterizedAgent(policy_model=policy, seed=42)
        
        agent._is_trained = True
        
        observation = np.random.randn(10)
        portfolio_state = {'position': 0.0, 'portfolio_value': 100000.0}
        
        action, conviction = agent.observe(observation, portfolio_state, bar_index=10)
        
        assert isinstance(action, TradingAction)
        assert 0.0 <= conviction <= 1.0
    
    def test_record_experience(self):
        from tradsl.models_impl import RandomForestModel
        policy = RandomForestModel(n_estimators=5, random_state=42)
        agent = ParameterizedAgent(policy_model=policy, seed=42)
        
        obs = np.random.randn(10)
        next_obs = np.random.randn(10)
        
        agent.record_experience(
            observation=obs,
            portfolio_state={'portfolio_value': 100000.0},
            action=TradingAction.BUY,
            conviction=0.5,
            reward=0.01,
            next_observation=next_obs,
            next_portfolio_state={'portfolio_value': 101000.0},
            done=False
        )
        
        assert len(agent.replay_buffer) == 1
    
    def test_should_update_every_n_bars(self):
        from tradsl.models_impl import RandomForestModel
        policy = RandomForestModel(n_estimators=5, random_state=42)
        agent = ParameterizedAgent(
            policy_model=policy,
            update_schedule="every_n_bars",
            update_n=100,
            seed=42
        )
        
        assert agent.should_update(50) is False
        assert agent.should_update(100) is True
        assert agent.should_update(150) is False
        assert agent.should_update(200) is True
    
    def test_should_update_performance_degradation(self):
        from tradsl.models_impl import RandomForestModel
        policy = RandomForestModel(n_estimators=5, random_state=42)
        agent = ParameterizedAgent(
            policy_model=policy,
            update_schedule="performance_degradation",
            update_n=100,
            update_threshold=0.1,
            seed=42
        )
        
        assert agent.should_update(50, recent_performance=1.0) is False
        
        agent._last_sharpe = 1.0
        assert agent.should_update(100, recent_performance=0.05) is True
    
    def test_update_policy_insufficient_buffer(self):
        from tradsl.models_impl import RandomForestModel
        policy = RandomForestModel(n_estimators=5, random_state=42)
        agent = ParameterizedAgent(policy_model=policy, seed=42)
        
        agent.update_policy(100)
        
        assert agent.is_trained is False
    
    def test_update_policy_with_experiences(self):
        from tradsl.models_impl import RandomForestModel
        policy = RandomForestModel(n_estimators=5, random_state=42)
        agent = ParameterizedAgent(policy_model=policy, seed=42, update_n=10)
        
        for i in range(50):
            obs = np.random.randn(10)
            next_obs = np.random.randn(10)
            agent.record_experience(
                observation=obs,
                portfolio_state={'portfolio_value': 100000.0},
                action=TradingAction.BUY,
                conviction=0.5,
                reward=np.random.randn() * 0.01,
                next_observation=next_obs,
                next_portfolio_state={'portfolio_value': 100000.0},
                done=False
            )
        
        agent.update_policy(50)
        
        assert agent._update_count == 1
    
    def test_save_and_load_checkpoint(self):
        from tradsl.models_impl import RandomForestModel
        policy = RandomForestModel(n_estimators=5, random_state=42)
        agent = ParameterizedAgent(policy_model=policy, seed=42)
        
        for i in range(50):
            obs = np.random.randn(10)
            next_obs = np.random.randn(10)
            agent.record_experience(
                observation=obs,
                portfolio_state={'portfolio_value': 100000.0},
                action=TradingAction.BUY,
                conviction=0.5,
                reward=0.01,
                next_observation=next_obs,
                next_portfolio_state={'portfolio_value': 101000.0},
                done=False
            )
        
        agent.update_policy(50)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'agent.joblib')
            agent.save_checkpoint(path)
            
            new_policy = RandomForestModel(n_estimators=5, random_state=42)
            new_agent = ParameterizedAgent(policy_model=new_policy, seed=42)
            new_agent.load_checkpoint(path)
            
            assert new_agent.is_trained is True
            assert new_agent._update_count == 1
    
    def test_reset(self):
        from tradsl.models_impl import RandomForestModel
        policy = RandomForestModel(n_estimators=5, random_state=42)
        agent = ParameterizedAgent(policy_model=policy, seed=42)
        
        agent._bar_index = 100
        agent._action_history = [0, 1, 2]
        agent._reward_history = [0.1, 0.2, -0.05]
        
        agent.reset()
        
        assert agent._bar_index == 0
        assert len(agent._action_history) == 0
        assert len(agent._reward_history) == 0


class TestTabularAgent:
    def test_init(self):
        agent = TabularAgent(n_actions=4, epsilon=0.1, seed=42)
        
        assert agent.n_actions == 4
        assert agent.epsilon == 0.1
        assert agent.is_trained is False
    
    def test_observe_not_trained(self):
        agent = TabularAgent(epsilon=0.1, seed=42)
        
        observation = np.random.randn(5)
        portfolio_state = {'position': 0.0, 'portfolio_value': 100000.0}
        
        action, conviction = agent.observe(observation, portfolio_state, bar_index=10)
        
        assert action == TradingAction.HOLD
        assert conviction == 0.0
    
    def test_observe_after_training(self):
        agent = TabularAgent(epsilon=0.0, seed=42)
        agent._is_trained = True
        
        observation = np.random.randn(5)
        portfolio_state = {'position': 0.0, 'portfolio_value': 100000.0}
        
        action, conviction = agent.observe(observation, portfolio_state, bar_index=10)
        
        assert isinstance(action, TradingAction)
        assert 0.0 <= conviction <= 1.0
    
    def test_record_experience_updates_q_table(self):
        agent = TabularAgent(epsilon=0.1, seed=42)
        
        obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        next_obs = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        
        agent.record_experience(
            observation=obs,
            portfolio_state={'portfolio_value': 100000.0},
            action=TradingAction.BUY,
            conviction=0.5,
            reward=0.1,
            next_observation=next_obs,
            next_portfolio_state={'portfolio_value': 101000.0},
            done=False
        )
        
        assert agent.is_trained is True
        
        state = agent._discretize(obs)
        assert state in agent._q_table
    
    def test_save_and_load(self):
        agent = TabularAgent(n_actions=4, seed=42)
        
        obs = np.random.randn(5)
        next_obs = np.random.randn(5)
        
        agent.record_experience(
            observation=obs,
            portfolio_state={'portfolio_value': 100000.0},
            action=TradingAction.BUY,
            conviction=0.5,
            reward=0.1,
            next_observation=next_obs,
            next_portfolio_state={'portfolio_value': 101000.0},
            done=False
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'tabular.joblib')
            agent.save_checkpoint(path)
            
            new_agent = TabularAgent(n_actions=4)
            new_agent.load_checkpoint(path)
            
            assert new_agent.is_trained is True


class TestPPOUpdate:
    def test_init_default(self):
        update = PPOUpdate()
        
        assert update.clip_epsilon == 0.2
        assert update.value_coef == 0.5
        assert update.entropy_coef == 0.01
        assert update.gamma == 0.99
    
    def test_init_custom(self):
        update = PPOUpdate(
            clip_epsilon=0.1,
            value_coef=0.5,
            entropy_coef=0.02,
            learning_rate=1e-4,
            gamma=0.95
        )
        
        assert update.clip_epsilon == 0.1
        assert update.learning_rate == 1e-4
    
    def test_update_insufficient_experiences(self):
        from tradsl.models_impl import RandomForestModel
        policy = RandomForestModel(n_estimators=5, random_state=42)
        value = RandomForestModel(n_estimators=5, random_state=42)
        
        update = PPOUpdate()
        
        experiences = [
            Experience(
                observation=np.random.randn(10),
                action=0,
                reward=0.01,
                next_observation=np.random.randn(10),
                done=False
            )
        ]
        
        result = update.update(experiences, policy, value)
        
        assert result['loss'] == 0.0
    
    def test_update_with_experiences(self):
        from tradsl.models_impl import RandomForestModel
        policy = RandomForestModel(n_estimators=5, random_state=42)
        value = RandomForestModel(n_estimators=5, random_state=42)
        
        update = PPOUpdate()
        
        experiences = [
            Experience(
                observation=np.random.randn(10),
                action=i % 4,
                reward=np.random.randn() * 0.1,
                next_observation=np.random.randn(10),
                done=False
            )
            for i in range(20)
        ]
        
        result = update.update(experiences, policy, value)
        
        assert 'loss' in result
        assert 'policy_loss' in result
        assert 'entropy' in result
    
    def test_compute_returns(self):
        update = PPOUpdate(gamma=0.99)
        
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        returns = update._compute_returns(rewards)
        
        assert len(returns) == len(rewards)
        assert returns[0] > returns[-1]
    
    def test_compute_advantages(self):
        update = PPOUpdate(gamma=0.99, lam=0.95)
        
        rewards = np.array([0.1, -0.1, 0.2, -0.05, 0.15])
        values = np.array([1.0, 0.9, 0.8, 0.85, 0.9, 0.0])
        dones = np.array([False, False, False, False, False])
        
        advantages = update._compute_advantages(rewards, values, dones)
        
        assert len(advantages) == len(rewards)


class TestDQNUpdate:
    def test_init_default(self):
        update = DQNUpdate()
        
        assert update.gamma == 0.99
        assert update.learning_rate == 1e-4
        assert update.target_update_freq == 100
    
    def test_update(self):
        from tradsl.models_impl import RandomForestModel
        policy = RandomForestModel(n_estimators=5, random_state=42)
        
        update = DQNUpdate()
        
        experiences = [
            Experience(
                observation=np.random.randn(10),
                action=i % 4,
                reward=np.random.randn() * 0.1,
                next_observation=np.random.randn(10),
                done=False
            )
            for i in range(20)
        ]
        
        result = update.update(experiences, policy)
        
        assert 'loss' in result
        assert 'q_value' in result


class TestPolicyGradientUpdate:
    def test_init_default(self):
        update = PolicyGradientUpdate()
        
        assert update.learning_rate == 1e-3
        assert update.entropy_coef == 0.01
        assert update.value_coef == 0.5
    
    def test_update(self):
        from tradsl.models_impl import RandomForestModel
        policy = RandomForestModel(n_estimators=5, random_state=42)
        
        update = PolicyGradientUpdate()
        
        experiences = [
            Experience(
                observation=np.random.randn(10),
                action=i % 4,
                reward=np.random.randn() * 0.1,
                next_observation=np.random.randn(10),
                done=False
            )
            for i in range(10)
        ]
        
        result = update.update(experiences, policy)
        
        assert 'loss' in result
        assert 'policy_loss' in result


class TestAgentActionConversion:
    def test_action_to_idx(self):
        from tradsl.models_impl import RandomForestModel
        policy = RandomForestModel(n_estimators=5, random_state=42)
        agent = ParameterizedAgent(policy_model=policy, seed=42)
        
        assert agent._action_to_idx(TradingAction.BUY) == 0
        assert agent._action_to_idx(TradingAction.SELL) == 1
        assert agent._action_to_idx(TradingAction.HOLD) == 2
        assert agent._action_to_idx(TradingAction.FLATTEN) == 3
    
    def test_idx_to_action(self):
        from tradsl.models_impl import RandomForestModel
        policy = RandomForestModel(n_estimators=5, random_state=42)
        agent = ParameterizedAgent(policy_model=policy, seed=42)
        
        assert agent._idx_to_action(0) == TradingAction.BUY
        assert agent._idx_to_action(1) == TradingAction.SELL
        assert agent._idx_to_action(2) == TradingAction.HOLD
        assert agent._idx_to_action(3) == TradingAction.FLATTEN
