# TradSL Phase 5.5: Critical Issues Analysis and Remediation Plan

> **Document Purpose:** This document catalogs all critical, serious, and design-gap issues discovered in the current TradSL codebase against the authoritative Technical Specification v1.0. Each issue is documented with: (1) exact location in code, (2) why it violates the specification, (3) what the correct behavior should be.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Reinforcement Learning Implementation Failures](#2-reinforcement-learning-implementation-failures)
3. [Training Pipeline Disconnections](#3-training-pipeline-disconnections)
4. [Label Function Bugs](#4-label-function-bugs)
5. [Backtest Engine Non-Execution](#5-backtest-engine-non-execution)
6. [DAG Construction Errors](#6-dag-construction-errors)
7. [Position Sizer Defects](#7-position-sizer-defects)
8. [Reward Function Isolation](#8-reward-function-isolation)
9. [Configuration System Bypass](#9-configuration-system-bypass)
10. [Code Quality and Architecture Issues](#10-code-quality-and-architecture-issues)
11. [Missing Components](#11-missing-components)
12. [Remediation Priority Matrix](#12-remediation-priority-matrix)

---

## 1. Executive Summary

The current TradSL codebase contains foundational architectural elements (DSL parser, schema validation, DAG construction, modular component design) but suffers from **critical functional failures** in the ML training pipeline, trading execution, and component integration. The RL algorithms are non-functional stubs, the backtest engine never executes trades, and the full pipeline from DSL to trained model is disconnected.

**Critical Finding:** The codebase has 202 passing tests, yet the core trading functionality is largely non-operational. This is a textbook case of test coverage without correctness — the tests validate individual components in isolation but do not verify that the components work together to produce a functioning trading system.

**Root Cause:** The implementation follows the specification's architectural structure superficially but implements the core algorithmic logic as stubs or placeholders. This matches exactly the "vibecoded" failure mode the specification was designed to prevent: "plausible-looking structure with subtly broken internals."

---

## 2. Reinforcement Learning Implementation Failures

### 2.1 PPOUpdate Is Not Actual PPO

**Location:** `tradsl/agent_framework.py:18-151`

**Issue:** The `PPOUpdate` class claims to implement Proximal Policy Optimization but is functionally a stub that never performs gradient-based learning.

**Specific Defects:**

1. **No Gradient Computation (lines 68-106):**
   ```python
   # Current code (lines 93-94):
   policy_loss -= log_prob * advantage
   entropy += 0.5 * (np.exp(log_prob) * log_prob).sum()
   ```
   This computes a loss value but never uses it to update model weights. There is no backpropagation, no optimizer step, no gradient computation whatsoever.

2. **Mock Logits Extraction (lines 68-81):**
   ```python
   # Current code attempts to extract "logits" from sklearn models:
   if hasattr(policy_model, 'predict') and len(states) > 0:
       try:
           all_logits = []
           for state in states:
               logits = policy_model.predict(state)
               if isinstance(logits, (int, float)):
                   all_logits.append([logits] * 4)  # Fabricates 4-class logits from scalar
       except Exception:
           action_logits = np.zeros((len(states), 4))
   ```
   sklearn models (RandomForest, LinearRegression) do not produce logits. They produce regression values or class probabilities. The code treats a scalar prediction as if it were a 4-dimensional action probability vector — this is architecturally wrong.

3. **Nonsensical Entropy Computation (line 94):**
   ```python
   entropy += 0.5 * (np.exp(log_prob) * log_prob).sum()
   ```
   Standard entropy is `-Σ p·log(p)`. This formula computes `0.5 * Σ exp(log_prob) * log_prob` which equals `0.5 * Σ p * log_prob` (since `exp(log_prob) = probs`). This is not entropy in any standard formulation.

**Why This Violates the Specification:**

- Section 10.6.3 specifies that agent architectures must "update policy using replay buffer" — this implementation does not update any policy.
- Section 10.9 requires "entropy bonus" for policy gradient updates — the entropy computed here is not used for any learning purpose.
- Section 10.8 specifies PPO's clipped surrogate objective — there is no clipping, no surrogate objective, no PPO whatsoever.

**Correct Behavior:** The `PPOUpdate.update()` method must:
1. Extract actual action probabilities from the policy model (or use a model designed to produce them)
2. Compute clipped surrogate objective: `L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]`
3. Compute value function loss: `L^VF = (V(s_t) - V_target)^2`
4. Add entropy bonus: `-entropy_coeff * H(π)`
5. Combine losses and perform backpropagation through the policy and value networks

---

### 2.2 DQNUpdate Is Completely Empty

**Location:** `tradsl/agent_framework.py:154-208`

**Issue:** The `DQNUpdate` class claims to implement Deep Q-Network but returns zero loss always and performs no learning.

**Specific Defects:**

```python
# Current code (lines 198-207):
current_q = 0.0  # Hardcoded to zero
loss = 0.0       # Hardcoded to zero
self._update_count += 1
return {
    'loss': loss,
    'q_value': current_q,
    'update_count': self._update_count
}
```

**Why This Violates the Specification:**

- Section 10.6 specifies DQN with "experience replay and target network" — there is no target network, no Q-value computation, no experience replay integration.
- Section 10.8 specifies DQN's update: `Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]` — no such computation occurs.

**Correct Behavior:** The `DQNUpdate.update()` method must:
1. Extract current Q-values: `Q(s,a; θ)` for all actions
2. Compute target: `y = r + γ * max_a' Q(s',a'; θ_target)` (or double DQN: `γ * Q(s', argmax_a Q(s',a; θ); θ_target)`)
3. Compute TD error: `δ = y - Q(s,a; θ)`
4. Compute loss: `L = δ²`
5. Backpropagate through the Q-network
6. Periodically sync target network: `θ_target ← τ * θ + (1-τ) * θ_target`

---

### 2.3 PolicyGradientUpdate Is Fake REINFORCE

**Location:** `tradsl/agent_framework.py:211-270`

**Issue:** The `PolicyGradientUpdate` class claims to implement vanilla policy gradient (REINFORCE) but the "gradient" is fabricated.

**Specific Defects:**

```python
# Current code (lines 255-259):
policy_loss = 0.0
for i, exp in enumerate(experiences):
    policy_loss -= rewards[i] * 0.1  # Hardcoded coefficient, no actual gradient

entropy = 1.0  # Fixed entropy, not computed from policy
total_loss = policy_loss - self.entropy_coef * entropy
return {
    'loss': abs(total_loss),  # Absolute value — gradient direction lost
    ...
}
```

**Why This Violates the Specification:**

- REINFORCE computes: `∇J ≈ ∇log π(a|s) * G_t` where `G_t` is the return. This implementation uses raw rewards directly multiplied by a hardcoded `0.1`.
- The gradient direction is lost by taking `abs()` of the loss.
- No backpropagation occurs.

**Correct Behavior:** Must compute `∇θ J = E[∇θ log π_θ(a|s) * G_t]` and perform gradient ascent on policy parameters.

---

### 2.4 Advantage Estimation (GAE) Is Fundamentally Broken

**Location:** `tradsl/agent_framework.py:117-130`

**Issue:** The Generalized Advantage Estimation implementation is mathematically incorrect.

**Current Code:**
```python
def _compute_advantages(self, rewards: np.ndarray) -> np.ndarray:
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = rewards[t + 1]  # BUG: Uses reward, not value function!
        
        delta = rewards[t] + self.gamma * next_value - rewards[t]  # BUG: simplifies to gamma * next_value
        last_gae = delta + self.gamma * self.lam * last_gae
        advantages[t] = last_gae
```

**Mathematical Error:**

1. **Line 127:** `delta = rewards[t] + self.gamma * next_value - rewards[t]`
   - This simplifies to: `delta = self.gamma * next_value`
   - The `rewards[t]` terms cancel out completely
   - GAE delta should be: `δ_t = r_t + γ * V(s_t+1) - V(s_t)` — it needs a value function estimate, not the reward

2. **Using rewards as value estimates (line 125):**
   ```python
   next_value = rewards[t + 1]  # Reward, not value!
   ```
   - A value estimate `V(s)` is the expected return from state `s`, not a single reward
   - Using `rewards[t+1]` as `V(s_t+1)` is an extremely biased single-sample estimate

**Why This Violates the Specification:**

- Section 10.6 specifies GAE as: `A_t = Σ (γλ)^l * δ_t+l` where `δ_t = r_t + γV(s_t+1) - V(s_t)`
- The implementation uses `δ_t = γ * r_t+1` which is neither GAE nor any valid advantage estimation

**Correct Behavior:** Must maintain a learned value function `V(s)` and compute:
```python
for t in reversed(range(T)):
    next_value = V(s_{t+1}) if not done else 0.0
    delta = r_t + gamma * next_value - V(s_t)  # TD error
    advantages[t] = delta + gamma * lam * advantages[t+1]  # GAE
    V(s_t) += alpha * delta  # Value function update
```

---

## 3. Training Pipeline Disconnections

### 3.1 BlockTrainer Never Calls Model.fit()

**Location:** `tradsl/training.py:229-307`

**Issue:** The `BlockTrainer` class iterates through training blocks but never actually trains any supervised models.

**Current Code (lines 309-345):**
```python
def _train_block(self, block, data, warmup_bars, execute_bar_fn, block_idx):
    if hasattr(self.agent, 'replay_buffer'):
        self.agent.replay_buffer.clear()
    if hasattr(self.agent, 'reset'):
        self.agent.reset()
    
    portfolio_value = 100000.0
    
    for bar_idx in range(block.start_idx, block.end_idx):
        relative_idx = bar_idx - block.start_idx
        
        portfolio_state = execute_bar_fn(bar_idx, data[bar_idx])
        portfolio_value = portfolio_state.get('portfolio_value', portfolio_value)
        
        if relative_idx >= warmup_bars and hasattr(self.agent, 'should_update'):
            if self.agent.should_update(relative_idx, ...):
                self.agent.update_policy(relative_idx)  # Calls fake RL update
    
    # No model.fit() call anywhere
```

**Why This Violates the Specification:**

- Section 10.4.2 requires trainable models to implement `fit(features, labels)`
- Section 14.7 specifies that trainable models must be retrained within each block using historical data
- The training loop calls `update_policy()` but never trains supervised models

**Correct Behavior:** The training loop must:
1. At block start: extract training window features and generate labels using the registered label function
2. Before each bar: check if retrain is due for any trainable model node
3. If retrain due: call `model.fit(features_window, labels_window)`
4. Then call `model.predict(current_features)` for the current bar

---

### 3.2 No Feature or Label Computation in Pipeline

**Location:** `tradsl/__init__.py:200-281` (the `train()` function)

**Issue:** The top-level `train()` function creates hardcoded models and executes an empty training loop.

**Current Code (lines 247-267):**
```python
# Hardcoded model - ignores DSL config
policy = RandomForestModel(n_estimators=10, random_state=42)
agent = ParameterizedAgent(
    policy_model=policy,
    update_schedule=training_config.retrain_schedule,
    ...
)

# Dummy execute_bar_fn - returns constant values
def execute_bar_fn(bar_idx, bar_data):
    return {'portfolio_value': 100000.0, 'position': 0.0}  # Hardcoded!
```

**Why This Violates the Specification:**

- Section 19.1 specifies the public API must execute the full pipeline
- Section 3.3 example shows a complete strategy with timeseries nodes, trainable_model nodes, and agent nodes all working together
- The current implementation ignores the resolved config and executes nothing

**Correct Behavior:** The `train()` function must:
1. Instantiate the DAG from resolved config
2. Initialize all node buffers to required sizes
3. For each block in shuffled order:
   - Execute DAG to compute features for each bar
   - Generate labels using registered label functions
   - Call `model.fit()` for trainable_model nodes
   - Execute agent.observe() to generate signals
   - Call agent.record_experience() to populate replay buffer
   - Call agent.update_policy() on schedule

---

## 4. Label Function Bugs

### 4.1 Triple Barrier — NOT A BUG (Retracted)

**Location:** `tradsl/labels.py:141-168`

**Analysis:** After careful re-examination, the triple barrier implementation is actually **correct**.

**Current Code:**
```python
entry_price = prices[0]  # line 141 - initial assignment

for i in range(n):      # line 143
    # ... check barriers against entry_price ...
    
    if i + 1 < n:
        entry_price = prices[i + 1]  # line 168 - updates for NEXT iteration
```

**Why It's Correct:**
- i=0: entry_price = prices[0], checks barriers, then updates to prices[1]
- i=1: entry_price = prices[1], checks barriers, then updates to prices[2]
- i=2: entry_price = prices[2], etc.

The code properly uses `prices[i]` as the entry price for each bar i via the update mechanism at line 168. This was an incorrect bug report — the implementation matches Section 10.5.2 of the specification correctly.

**Status:** REMOVED FROM PRIORITY LIST

---

## 4B. Configuration System Bypass — Complete Inventory

The DSL parser, schema validator, and DAG builder all work correctly. However, the resolved configuration values are ignored at execution time, replaced with hardcoded defaults. This section documents every instance.

### 4B.1 Hardcoded Model Parameters

**Location:** `tradsl/__init__.py:251`
```python
policy = RandomForestModel(n_estimators=10, random_state=42)
```
- DSL config key: `trainable_model.params.n_estimators` (from `rf_cfg` block)
- Hardcoded value: `n_estimators=10`
- Should extract: `config['rf_cfg'].get('n_estimators', 10)` or fail if missing

---

**Location:** `tradsl/__init__.py:316` (in `test()` function)
```python
policy = RandomForestModel(n_estimators=10, random_state=42)
```
- Same issue as above, hardcoded in test path

---

### 4B.2 Hardcoded Replay Buffer Size

**Location:** `tradsl/__init__.py:256`
```python
agent = ParameterizedAgent(
    policy_model=policy,
    replay_buffer_size=4096,  # HARDCODED
    ...
)
```
- DSL config key: `model.replay_buffer_size`
- Hardcoded value: `4096`
- Should extract: `config['model'].get('replay_buffer_size', 4096)`

---

### 4B.3 Hardcoded Portfolio Initial Value

**Location:** `tradsl/training.py:324` (in `BlockTrainer._train_block()`)
```python
portfolio_value = 100000.0  # HARDCODED
```
- DSL config key: `backtest.capital`
- Hardcoded value: `100000.0`
- Should extract: `self.config.get('capital', 100000.0)`

---

**Location:** `tradsl/training.py:378` (in `WalkForwardTester.test()`)
```python
portfolio_value = 100000.0  # HARDCODED
```
- Same issue as above

---

**Location:** `tradsl/backtest.py:152` (in `BacktestEngine._reset()`)
```python
self.portfolio_value = self.config.get('capital', 100000.0)
```
- This one actually reads from config correctly
- Listed for completeness

---

### 4B.4 Hardcoded Sizer

**Location:** `tradsl/__init__.py:261`
```python
sizer = KellySizer()  # HARDCODED, ignores DSL sizer= field
```
- DSL config key: `agent.sizer`
- Hardcoded value: Always `KellySizer()`
- Should extract: `config['agent'].get('sizer', 'kelly')` and instantiate appropriate sizer

---

### 4B.5 Hardcoded Reward Function

**Location:** `tradsl/__init__.py:260`
```python
reward_fn = AsymmetricHighWaterMarkReward()  # HARDCODED
```
- DSL config key: `model.reward_function`
- Hardcoded value: Always `AsymmetricHighWaterMarkReward()`
- Should extract: `config['model'].get('reward_function')` and instantiate from registry

---

### 4B.6 Hardcoded Training Config

**Location:** `tradsl/__init__.py:234-242`
```python
training_config = TrainingConfig(
    training_window=training_config_dict.get('training_window', 504),
    retrain_schedule=training_config_dict.get('retrain_schedule', 'every_n_bars'),
    retrain_n=training_config_dict.get('retrain_n', 252),
    n_training_blocks=training_config_dict.get('n_training_blocks', 40),
    block_size_min=training_config_dict.get('block_size_min', 30),
    block_size_max=training_config_dict.get('block_size_max', 120),
    seed=training_config_dict.get('seed', 42)
)
```
- This section DOES read from config correctly
- Listed for completeness — this is the CORRECT pattern other sections should follow

---

### 4B.7 Complete Config Bypass Summary Table

| Line | File | Hardcoded Value | DSL Config Key | Correct? |
|------|------|-----------------|----------------|----------|
| 251 | `__init__.py` | `n_estimators=10` | `trainable_model.params.n_estimators` | NO |
| 256 | `__init__.py` | `replay_buffer_size=4096` | `model.replay_buffer_size` | NO |
| 260 | `__init__.py` | `AsymmetricHighWaterMarkReward()` | `model.reward_function` | NO |
| 261 | `__init__.py` | `KellySizer()` | `agent.sizer` | NO |
| 316 | `__init__.py` | `n_estimators=10` | `trainable_model.params.n_estimators` | NO |
| 324 | `training.py` | `portfolio_value=100000.0` | `backtest.capital` | NO |
| 378 | `training.py` | `portfolio_value=100000.0` | `backtest.capital` | NO |
| 234-242 | `__init__.py` | (reads from dict) | All training params | **YES** |

---

**Why This Violates the Specification:**

- Section 3.3: Every DSL field must be respected
- Section 19.1: `tradsl.run()` must execute "the full pipeline: train + test + bootstrap"
- Section 20.3: "Silently discarding DSL parameters" is a prohibited pattern
- Section 20.4: "Silently discarding DSL parameters" pattern — sizer is wrong signature, `KellySizer` always used

**Correct Behavior:** Every component must be instantiated using values from the resolved config dict, with sensible defaults only as fallback.

---

## 5. Backtest Engine Non-Execution

### 5.1 BacktestEngine.run() Never Executes Trades

**Location:** `tradsl/backtest.py:159-198`

**Issue:** The backtest engine iterates through bars but never executes any trades or updates portfolio value.

**Current Code:**
```python
def run(self, data, execute_bar_fn):
    self._reset()
    
    for bar_idx in range(len(data)):
        portfolio_state = execute_bar_fn(bar_idx, data[bar_idx])
        
        self.equity_curve.append(self.portfolio_value)  # Never changes!
        
        if self.portfolio_value > self.high_water_mark:
            self.high_water_mark = self.portfolio_value
        
        current_dd = (self.high_water_mark - self.portfolio_value) / self.high_water_mark
        self.drawdown = max(self.drawdown, current_dd)
    
    # Returns equity_curve of identical values
```

**Why This Violates the Specification:**

- Section 15.1 specifies backtest must "execute trades and track P&L"
- Section 15.2 requires transaction costs to be "applied and reflected in post_fill_portfolio before reward function"
- The current implementation produces an equity curve of constant values

**Correct Behavior:** The `run()` method must:
1. On each bar: call `execute_bar_fn` which should return actual portfolio state after trades
2. Call `self.execute_trade()` when agent generates non-HOLD action
3. Update `self.portfolio_value` after each trade based on fills
4. Apply transaction costs from `TransactionCosts` class
5. Record trades in `self.trades` list

---

### 5.2 execute_trade() Never Called from run()

**Location:** `tradsl/backtest.py:200-236`

**Issue:** The `execute_trade()` method is implemented but never called from the main loop.

**Current Architecture:**
- `execute_trade()` exists and correctly computes commission, slippage, market impact
- It updates portfolio_value and records trades
- But `run()` never calls it — the execution flow is broken

**Why This Violates the Specification:**

- Section 15.2 contract: "All transaction costs must be applied and reflected in post_fill_portfolio"
- The method exists but is unreachable

---

## 6. DAG Construction Errors

### 6.1 Topological Sort In-Degree Calculation Is Backwards

**Location:** `tradsl/dag.py:148-153`

**Issue:** The in-degree calculation for Kahn's algorithm increments the wrong nodes.

**Current Code:**
```python
in_degree = {name: 0 for name in nodes}
for name in nodes:
    for dep in edges.get(name, []):  # edges[name] = nodes that depend on name
        if dep in in_degree:
            in_degree[dep] += 1  # BUG: Increments dependents, should increment name!
```

**Why This Is Wrong:**

Kahn's algorithm requires `in_degree[node]` = number of edges pointing TO `node` (dependencies of `node`).

The current code:
- Iterates through each node
- For each node, looks at its dependents (edges pointing FROM it)
- Increments the dependent's in-degree

This produces the reverse: `in_degree[node]` = number of nodes that depend on `node`.

**Correct Behavior:**
```python
in_degree = {name: 0 for name in nodes}
for name in nodes:
    for dep in edges.get(name, []):  # dep is a dependency of name
        if dep in in_degree:
            in_degree[name] += 1  # Correct: name has one more dependency
```

**Why This Violates the Specification:**

- Section 6.2 specifies Kahn's algorithm with `collections.deque`
- Section 6.4 guarantees "all nodes in N's transitive dependency set appear before N"
- The current implementation may produce incorrect execution order or fail on valid DAGs

---

## 7. Position Sizer Defects

### 7.1 KellySizer Has Duplicate Code

**Location:** `tradsl/sizers.py:195-197`

**Issue:** The `calculate_size` method contains duplicate code blocks.

**Current Code:**
```python
def calculate_size(self, action, conviction, current_position, portfolio_value, instrument_id, current_price):
    ...
    if action == TradingAction.FLATTEN:
        return 0.0
    if action == TradingAction.HOLD:
        return current_position
    if action == TradingAction.HOLD:  # DUPLICATE - does nothing
        return current_position
    
    p = conviction * self.estimated_win_rate
    ...
```

**Why This Violates the Specification:**

- Section 20.6 explicitly prohibits "Duplicate code blocks" as a copy-paste artifact
- This is a direct violation of the prohibited patterns catalogue

---

### 7.2 KellySizer Uses Conviction Incorrectly

**Location:** `tradsl/sizers.py:199`

**Issue:** The Kelly calculation uses `conviction` as if it were a win probability.

**Current Code:**
```python
p = conviction * self.estimated_win_rate  # Incorrect
```

**Why This Is Wrong:**

- `conviction` is defined in Section 10.6.2 as "certainty of the action" in [0, 1]
- It is NOT a probability — it is a confidence measure from the model
- Kelly criterion requires true win probability `p`, not model confidence
- Using conviction inflates the Kelly fraction inappropriately

**Correct Behavior:**
```python
# Option 1: Use estimated_win_rate as prior, update with conviction as Bayesian posterior
p = self.estimated_win_rate  # Or Bayesian update: p = (conviction * p) / normalization

# Option 2: Separate Kelly calculation from conviction scaling
kelly = (self.estimated_win_rate * self.win_loss_ratio - (1 - self.estimated_win_rate)) / self.win_loss_ratio
kelly = max(0, min(kelly, self.max_kelly))
scaled_kelly = kelly * conviction  # Conviction modulates position, not Kelly fraction
```

---

### 7.3 Sizer Interface Never Called with Correct Signature

**Location:** `tradsl/__init__.py` (no explicit call exists)

**Issue:** The specification Section 12.1 requires the sizer to receive `portfolio_value`, but no call path exists that passes all required parameters.

**Why This Violates the Specification:**

- Section 12.1 requires: `calculate_size(action, conviction, current_position, portfolio_value, instrument_id, current_price)`
- Section 20.4 specifies the prohibited pattern: calling sizer with wrong signature `position_sizer(predictions, symbols)`
- The current codebase has no integration that calls the sizer at all

---

## 8. Reward Function Isolation

### 8.1 Reward Functions Are Implemented But Never Used

**Location:** `tradsl/rewards.py` and `tradsl/__init__.py:266-267`

**Issue:** The reward function classes are well-implemented but no code path computes rewards during training or backtesting.

**Current Code:**
```python
# In __init__.py train():
reward_fn = AsymmetricHighWaterMarkReward()  # Created but never called

def execute_bar_fn(bar_idx, bar_data):
    return {'portfolio_value': 100000.0, 'position': 0.0}  # Dummy, no reward
```

**Why This Violates the Specification:**

- Section 11.1 specifies that reward functions "are scalar feedback signals computed after a trading action's consequence is known"
- Section 11.2 specifies the `RewardContext` with all available information at fill time
- Section 13.2 Step 1-2 requires computing reward from portfolio state change and calling `model.record_experience()`

**Correct Behavior:** The training/backtest loop must:
1. After each fill: assemble `RewardContext` with pre/post fill portfolio values
2. Call `reward_fn.compute(ctx)` to get scalar reward
3. Call `agent.record_experience(..., reward=reward, ...)`
4. Never call `update_policy()` reactively — updates happen on schedule (Section 13.2)

---

## 9. Configuration System Bypass

### 9.1 Complete Inventory in Section 4B

All configuration bypass issues are documented in detail in **Section 4B: Configuration System Bypass — Complete Inventory**.

That section contains:
- Exact line numbers for every hardcoded value
- The corresponding DSL config key that should be used
- Current code snippets
- A summary table of all 8 locations (7 incorrect, 1 correct)

**Summary:** 7 locations hardcode values that should be extracted from resolved DSL config.

---

## 10. Code Quality and Architecture Issues

### 10.1 Type Inconsistencies

| Location | Issue |
|----------|-------|
| `models_impl.py:64` | `predict()` declares return `float` but can return numpy array |
| `training.py:262` | Type hint `Callable[[int, np.ndarray], Dict[str, float]]` but actual usage varies |
| `agent_framework.py:389` | `observation: np.ndarray` but can be scalar |

### 10.2 Magic Numbers Without Constants

| Location | Magic Number | Should Be |
|----------|-------------|-----------|
| `agent_framework.py:59` | `< 8` experiences minimum | `MIN_EXPERIENCES_FOR_UPDATE = 8` |
| `agent_framework.py:480` | `< 32` for policy update | `MIN_BUFFER_FOR_UPDATE = 32` |
| `training.py:53` | `0.8` for 80% block coverage | `MAX_BLOCK_COVERAGE = 0.8` |
| `bootstrap.py:229` | `int(np.sqrt(T))` block length | Document as standard choice |

### 10.3 Poor Error Handling

| Location | Issue |
|----------|-------|
| `models_impl.py:49` | "Model not initialized: model_class is None" — not actionable |
| `models_impl.py:79` | `except Exception: return 0.0` — silently masks all errors |
| Multiple locations | Generic exception handling without logging |

### 10.4 Missing Input Validation

- No validation of array shapes in model `fit()`/`predict()`
- No bounds checking on indices
- No NaN handling in many computation paths (some exists but inconsistent)

---

## 11. Missing Components

### 11.1 Not Implemented

| Component | Specification Section | Status |
|----------|---------------------|--------|
| Neural Network models | 10.4.4 | Not implemented |
| Deep RL (PPO/DQN) | 10.6 | Stub implementations |
| Live trading execution | 15 | Not implemented |
| Real-time data ingestion | 8 | Only YFinance exists |
| Portfolio optimization | 12 | Partial only |
| Risk management | — | Not implemented |
| Order management | — | Not implemented |
| Regime detection | 14.5 | Not implemented |
| Feature selection | — | Not implemented |
| Hyperparameter tuning | — | Not implemented |

### 11.2 Incomplete Implementations

| Component | Issue |
|-----------|-------|
| `DQNUpdate` | Completely empty — returns zero loss |
| `TabularAgent` | Works but not integrated |
| `BlockBootstrap` | Has code but not used in training |
| `WalkForwardTester` | Defined but not called from pipeline |

### 11.3 Test Coverage Gaps

- No tests for actual model training (fit + predict producing correct outputs)
- No tests for RL learning (updates actually improving policy)
- No integration tests for full pipeline
- No tests for trade execution

---

## 12. Remediation Priority Matrix

### Priority 1: Critical (System Non-Functional)

| Issue | Location | Fix Required |
|-------|----------|--------------|
| RL algorithms are stubs | agent_framework.py:18-270 | Implement actual PPO, DQN, PolicyGradient with real gradient updates |
| GAE broken | agent_framework.py:117-130 | Compute δ = r + γV(s') - V(s), maintain value function |
| Training never calls model.fit() | training.py:229-307 | Add trainable model retraining loop |
| Backtest never executes trades | backtest.py:159-198 | Connect execute_trade() in main loop |
| DAG topological sort backwards | dag.py:148-153 | Fix in_degree calculation |

### Priority 2: High (Major Features Missing)

| Issue | Location | Fix Required |
|-------|----------|--------------|
| Reward functions disconnected | rewards.py + __init__.py | Compute RewardContext, call reward_fn.compute() |
| Walk-forward testing unused | training.py:348-400 | Connect WalkForwardTester in pipeline |
| Config values ignored (7 locations) | See Section 4B | Extract and use resolved config values |
| Feature normalization missing | models_impl.py | Add StandardScaler to pipeline |

### Priority 3: Medium (Code Quality)

| Issue | Location | Fix Required |
|-------|----------|--------------|
| KellySizer duplicate code | sizers.py:195-197 | Remove duplicate condition |
| KellySizer conviction misuse | sizers.py:199 | Separate Kelly from conviction scaling |
| Magic numbers | Multiple | Create constants module |
| Type hints incomplete | Multiple | Add proper typing throughout |
| Error messages unhelpful | models_impl.py | Add actionable error messages |

---

## Appendix A: Specification Reference

This analysis references the following sections of the Technical Specification v1.0:

- **Section 10 (RL Model Architecture):** All agent implementations must perform actual gradient-based learning
- **Section 10.5.2 (Triple Barrier):** Implementation verified CORRECT — uses prices[i] via update mechanism
- **Section 11 (Reward Functions):** Must compute rewards from RewardContext after fills
- **Section 12 (Position Sizers):** Must receive full parameter set including portfolio_value
- **Section 13 (NT Integration):** Must follow strict execution order (trainable model retrain → policy update → observe)
- **Section 14 (Training Framework):** Randomized block training with proper replay buffer flushing
- **Section 15 (Backtest Engine):** Must execute trades, apply transaction costs, track P&L
- **Section 20 (Prohibited Patterns):** All identified issues map to catalogued anti-patterns

---

## Appendix B: Mapping to Original Vibecode Audit

Many issues in this codebase match the audit findings from Section 19 of the specification:

| Audit Finding | Current Code Finding |
|--------------|---------------------|
| A.1.2 Retraining after prediction | training.py: BlockTrainer calls update_policy after iterate (but both are fake) |
| A.1.4 Model params discarded | __init__.py:247-258 hardcoded values |
| A.1.5 Wrong sizer signature | sizers.py:196-197 but no integration |
| A.2.9 Bare exception handling | models_impl.py:79 `except Exception: return 0.0` |
| A.2.10 O(N²) list.pop(0) | dag.py uses deque but wrong in-degree calc |
| A.2.17 Position.weight returns 0.0 | Not implemented |

---

*Document Version: 5.5*
*Created: 2026-03-13*
*Purpose: Comprehensive issue analysis for TradSL remediation planning*
