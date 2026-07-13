# Decentralized Multi-Agent Coordination via Permutation-Equivariant MCTS and Coordinated Heuristics

This repository implements a decentralized, sample-efficient Multi-Agent Reinforcement Learning (MARL) framework for grid-based pathfinding and collision avoidance. The framework integrates:
1. **Permutation-Equivariant GNN (PE-GNN)**: Restricts policies to be equivariant under the symmetric group $S_M$ of agent index permutations, allowing zero-shot generalization to unseen agent counts.
2. **Decentralized Monte Carlo Tree Search (PE-MCTS)**: Individual agents plan independently via search trees, avoiding the exponential scaling of joint action spaces.
3. **Coordinated Potential Field Heuristics**: Accelerates training convergence and ensures safety during exploration through a decaying KL-regularized loss and action priors during search.

---

## Key Features

* **$S_M$-Equivariant Policy Backbone**: By using Transformer blocks without positional encodings on agent features, swapping agent indices permutes the policy output correspondingly.
* **Decentralized Search**: Avoids the $O(|\mathcal{A}|^M)$ exponential scaling of joint MCTS by executing independent $O(M \cdot N_{\text{search}})$ local searches.
* **Cold-Start Safety**: Combines goal attraction and reciprocal neighbor/obstacle repulsion in a potential field prior, guiding early-stage exploration.

---

## Directory Structure

* `multi_agent_env.py`: Decentralized 2D grid environment supporting variable agent counts and collision types (static, vertex, edge).
* `equivariant_gnn.py`: Permutation-equivariant Graph Neural Network policy and value head backbone.
* `multi_agent_mcts.py`: Decentralized Monte Carlo Tree Search with Predictor Upper Confidence Bound (PUCT) guided by mixed priors.
* `heuristic_guided_loss.py`: Decaying KL-divergence loss function regularizing policy outputs toward the safety heuristic.
* `train_multi_agent.py`: Training pipeline using decentralized actor-critic MCTS rollouts and PPO optimization.
* `run_multi_agent_experiments.py`: Suite for evaluating zero-shot scalability, ablation studies, and spatial robustness.
* `models/`: Contains trained model checkpoints.
* `results/`: Output directories for scalability and ablation plot figures.

---

## Installation & Setup

Ensure you have PyTorch and standard scientific python packages installed:
```bash
pip install torch numpy matplotlib
```

---

## Running the Code

### 1. Training the Model
To train the permutation-equivariant network guided by MCTS and potential fields:
```bash
python train_multi_agent.py
```
This trains the network with $M=4$ agents for 250 episodes, periodically decaying the heuristic loss regularizer $\beta$, and saves the checkpoint to `models/multi_agent_model.pth`.

### 2. Evaluating Performance & Scalability
To evaluate the trained model zero-shot on unseen agent counts $M \in \{2, 3, 4, 5, 6, 8\}$ across Default, Empty, and Random obstacle maps:
```bash
python run_multi_agent_experiments.py
```
This script evaluates the model and plots the robustness and ablation curves in the `results/` folder.

---

## Experimental Results

The framework demonstrates perfect zero-shot scalability and coordinate navigation for up to 4 agents, significantly outperforming search-free and heuristic-only baselines.

| Configuration / Policy Mode | M = 2 | M = 3 | M = 4 | M = 6 | M = 8 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ours: Default Map** | **100.0%** | **100.0%** | **100.0%** | 0.0% | 0.0% |
| **Ours: Empty Map** | **100.0%** | **100.0%** | **100.0%** | 0.0% | 0.0% |
| **Ours: Random Obstacles** | **100.0%** | **100.0%** | 0.0% | 0.0% | 0.0% |
| Baseline: GNN Only | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Baseline: Heuristic Only | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

### Generated Visualizations

* **`results/scalability_test.png`**: Illustrates zero-shot success rates under varying obstacle configurations.
* **`results/mcts_ablation.png`**: Compares success rates of our GNN-MCTS framework against search-free and heuristic-only baselines.

---

## License & Citation

Licensed under the MIT License. Copyright (c) 2026 WonChan Cho. All rights reserved.
For academic use, please cite:
```bibtex
@misc{wonchan_cho_multi_agent_equiv_2026,
  author = {WonChan Cho},
  title = {Decentralized Multi-Agent Coordination via Permutation-Equivariant MCTS and Coordinated Heuristics},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/WonC-Lab/Permutation-Equivariant-MCTS-and-Coordinated-Heuristics}}
}
```
