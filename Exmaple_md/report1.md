# Multi-Armed Bandit Problem - Agent Comparison Report

## Executive Summary

This report presents a comprehensive comparison of three different bandit algorithms implemented to solve the multi-armed bandit problem:

1. **Epsilon-Greedy Agent** (ε = 0.1)
2. **Optimistic Agent** (ε = 0.1, initial Q = 5.0)
3. **Upper-Confidence-Bound (UCB) Agent** (c = 2.0)

The experiments were conducted over 2000 runs with 1000 steps each, using 10-armed bandit problems with Gaussian rewards.

## Results

### Final Performance Statistics

| Agent | Final Reward (Last 100 Steps) | Total Reward (1000 Steps) |
|-------|-------------------------------|---------------------------|
| **UCB (c=2.0)** | **1.4798** | **1380.49** |
| Optimistic (e=0.1, Q=5.0) | 1.3593 | 1323.13 |
| Epsilon-Greedy (e=0.1) | 1.3575 | 1296.88 |
| Purely Greedy (e=0) | 1.0283 | 1019.56 |
| Purely Random (e=1.0) | 0.0048 | 1.39 |

### Key Findings

1. **UCB Agent performs best**: The UCB agent achieved the highest final reward (1.4798) and total reward (1380.49), demonstrating superior exploration-exploitation balance.

2. **Optimistic Agent shows improvement**: The optimistic agent slightly outperformed the standard epsilon-greedy agent, showing the benefit of optimistic initialization.

3. **Epsilon-Greedy provides solid baseline**: The standard epsilon-greedy agent performed well, achieving 1.3575 final reward.

4. **Greedy strategy fails**: The purely greedy agent (ε=0) performed poorly (1.0283), getting stuck in suboptimal arms due to lack of exploration.

5. **Random strategy is ineffective**: The purely random agent (ε=1.0) performed worst (0.0048), confirming that learning is essential.

## Algorithm Analysis

### 1. Epsilon-Greedy Agent

**Strategy**: Balances exploration and exploitation with a fixed probability ε.

**Implementation**:
- With probability ε: explore (choose random arm)
- With probability (1-ε): exploit (choose best arm based on Q-values)
- Updates Q-values using sample-average rule

**Strengths**:
- Simple and intuitive
- Consistent performance
- Easy to tune with ε parameter

**Weaknesses**:
- Fixed exploration rate regardless of uncertainty
- May explore too much or too little depending on problem

### 2. Optimistic Agent

**Strategy**: Inherits from Epsilon-Greedy but starts with optimistic initial Q-values.

**Implementation**:
- Initializes Q-values to 5.0 instead of 0.0
- Uses same epsilon-greedy strategy as base class
- Encourages early exploration through high initial values

**Strengths**:
- Promotes early exploration
- Slightly better performance than standard epsilon-greedy
- Simple modification of existing algorithm

**Weaknesses**:
- Still uses fixed exploration rate
- Performance gain is modest
- May explore too much initially

### 3. UCB Agent

**Strategy**: Uses confidence bounds to intelligently balance exploration and exploitation.

**Implementation**:
- Ensures all arms are tried at least once
- Calculates UCB values: `Q + c * sqrt(log(total_steps) / N)`
- Chooses arm with highest UCB value
- Uses same sample-average update rule

**Strengths**:
- **Best performance** in our experiments
- Adapts exploration based on uncertainty
- Theoretically optimal under certain conditions
- No hyperparameter tuning needed for exploration rate

**Weaknesses**:
- More complex than epsilon-greedy
- Requires tracking total steps
- May be computationally more expensive

## Learning Curves Analysis

The learning curves show that:

1. **UCB agent** learns fastest and reaches the highest performance
2. **Optimistic agent** shows good early exploration but converges to similar performance as epsilon-greedy
3. **Epsilon-greedy agent** shows steady improvement throughout training
4. **Greedy agent** quickly converges to suboptimal performance
5. **Random agent** shows no learning (flat line around 0)

## Discussion

### Why UCB Performs Best

The UCB agent's superior performance can be attributed to:

1. **Intelligent exploration**: It explores arms with high uncertainty (low visit count) more than arms with low uncertainty
2. **Adaptive strategy**: The exploration decreases naturally as confidence in estimates increases
3. **Theoretical foundation**: UCB is derived from confidence interval theory, providing theoretical guarantees

### Practical Implications

1. **For real-world applications**: UCB should be preferred when computational complexity is not a major concern
2. **For simple systems**: Epsilon-greedy provides a good balance of simplicity and performance
3. **For early exploration**: Optimistic initialization can be beneficial in scenarios where early exploration is crucial

### Limitations and Future Work

1. **Parameter sensitivity**: The c-value in UCB could be tuned for better performance
2. **Non-stationary environments**: These algorithms assume stationary reward distributions
3. **Computational cost**: UCB requires more computation per step
4. **Memory requirements**: UCB needs to track total steps

## Conclusion

The experiments demonstrate that:

1. **UCB is the clear winner** for this multi-armed bandit problem, achieving the highest final reward of 1.4798
2. **Optimistic initialization** provides modest improvements over standard epsilon-greedy
3. **Exploration is crucial** - both greedy and random strategies fail
4. **Intelligent exploration** (UCB) outperforms fixed exploration (epsilon-greedy)

The results validate the theoretical understanding that intelligent exploration strategies, which adapt based on uncertainty, generally outperform fixed exploration strategies in multi-armed bandit problems.

## Code Implementation

All agents were implemented in Python with the following key features:

- **EpsilonGreedyAgent**: Standard epsilon-greedy with sample-average updates
- **OptimisticAgent**: Inherits from EpsilonGreedyAgent with optimistic initialization
- **UCBAgent**: Implements UCB algorithm with confidence bounds

The complete implementation can be found in `Agents.py` and the comparison script in `test_all_agents.py`.
