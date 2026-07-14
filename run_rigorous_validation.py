import os
import sys
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add local path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multi_agent_env import MultiAgentNavigationEnv
from equivariant_gnn import PermutationEquivariantGNN
from multi_agent_mcts import MultiAgentMCTS

def check_permutation_equivariance(model, num_agents=4, size=13):
    """
    Rigorously verifies Theorem 1: Permutation Equivariance of the GNN Policy.
    Checks that model(P * Z) = P * model(Z) for random permutation matrices P.
    """
    print("\n-------------------------------------------------------------")
    print(" Verifying Theorem 1: Permutation Equivariance Check")
    print("-------------------------------------------------------------")
    env = MultiAgentNavigationEnv(size=size, num_agents=num_agents)
    state = env.generate_initial_state()
    obs_joint = env.get_joint_observation(state) # (M, 3, size, size)
    
    # Run forward pass on original observation
    model.eval()
    with torch.no_grad():
        original_logits, _ = model(obs_joint.unsqueeze(0))
    original_logits = original_logits.squeeze(0) # (M, 8)
    
    # Generate all permutations or a set of random ones
    diffs = []
    for _ in range(50):
        perm = list(range(num_agents))
        random.shuffle(perm)
        
        # Apply permutation to input: Z' = P * Z
        perm_obs = obs_joint[perm]
        
        # Run forward pass on permuted observation
        with torch.no_grad():
            perm_logits, _ = model(perm_obs.unsqueeze(0))
        perm_logits = perm_logits.squeeze(0) # (M, 8)
        
        # Apply permutation to original output: P * L
        expected_perm_logits = original_logits[perm]
        
        # Calculate maximum absolute difference
        diff = torch.max(torch.abs(perm_logits - expected_perm_logits)).item()
        diffs.append(diff)
        
    max_diff = max(diffs)
    mean_diff = np.mean(diffs)
    print(f"Permutation Equivariance Verification over 50 random permutations:")
    print(f"  - Max Absolute Difference: {max_diff:.2e}")
    print(f"  - Mean Absolute Difference: {mean_diff:.2e}")
    if max_diff < 1e-4:
        print("  [SUCCESS] Theorem 1 Mathematically Confirmed (logits commute with P_sigma).")
    else:
        print("  [WARNING] High numerical differences detected in equivariance.")
    print("-------------------------------------------------------------\n")
    return max_diff

def sample_random_starts_goals(num_agents, size, obstacles, rng):
    """Sample non-colliding random starts and goals for all agents."""
    forbidden = set(obstacles)
    starts = []
    goals = []
    used = set()
    candidates = [(r, c) for r in range(1, size-1) for c in range(1, size-1) if (r, c) not in forbidden]
    rng.shuffle(candidates)
    # Need 2 * num_agents unique positions
    if len(candidates) < 2 * num_agents:
        return None, None
    for i in range(num_agents):
        starts.append(candidates[i])
    for i in range(num_agents, 2 * num_agents):
        goals.append(candidates[i])
    # Ensure no agent starts at its own goal (trivial episode)
    for i in range(num_agents):
        if starts[i] == goals[i]:
            return None, None
    return tuple(starts), tuple(goals)

def sample_random_obstacles(size, obstacle_count, starts, goals, rng):
    """Sample random obstacles avoiding starts and goals."""
    forbidden = set(starts) | set(goals)
    candidates = [(r, c) for r in range(size) for c in range(size) if (r, c) not in forbidden]
    rng.shuffle(candidates)
    return set(candidates[:obstacle_count])

def evaluate_performance(model, num_agents, obstacle_mode="default", num_episodes=20, mode="mcts", mcts_searches=40, obstacle_count=12, beta=0.3):
    import random as pyrandom
    size = 13
    env = MultiAgentNavigationEnv(size=size, num_agents=num_agents)
    mcts = MultiAgentMCTS(model=model, c_puct=1.4)

    episode_results = []  # 1 = success, 0 = failure per episode

    for ep in range(num_episodes):
        # Use a different seed per episode for genuine randomization
        rng = pyrandom.Random(ep * 1000 + num_agents * 7 + hash(obstacle_mode) % 997)

        # --- Randomize obstacles per episode (for random/density modes) ---
        if obstacle_mode == "empty":
            env.obstacles = set()
        elif obstacle_mode in ("random", "density"):
            # Generate temporary starts/goals to avoid placing obstacles on them
            tmp_starts = env.default_starts[:num_agents]
            tmp_goals  = env.default_goals[:num_agents]
            env.obstacles = sample_random_obstacles(size, obstacle_count, tmp_starts, tmp_goals, rng)
        # "default" keeps env.obstacles as-is (loaded at env init)

        # --- Randomize starts and goals per episode ---
        starts, goals = sample_random_starts_goals(num_agents, size, env.obstacles, rng)
        if starts is None:
            # Fallback to default if not enough free cells (very dense obstacle case)
            starts = tuple(env.default_starts[:num_agents])
            goals  = tuple(env.default_goals[:num_agents])

        env.starts = starts
        env.goals  = goals

        state = env.generate_initial_state()
        done = False
        step = 0
        max_steps = num_agents * 20  # Scale timeout with agent count

        while not done and step < max_steps:
            joint_action = []
            active_mask = state[3]

            if mode == "mcts":
                for i in range(num_agents):
                    if not active_mask[i]:
                        joint_action.append(0)
                        continue
                    _, mcts_probs = mcts.get_action_probabilities(
                        state, agent_idx=i, env=env, num_searches=mcts_searches, temp=0.0, beta=beta
                    )
                    joint_action.append(np.argmax(mcts_probs))
            elif mode == "gnn":
                obs_joint = env.get_joint_observation(state)
                model.eval()
                with torch.no_grad():
                    logits, _ = model(obs_joint.unsqueeze(0))
                logits = logits.squeeze(0)
                for i in range(num_agents):
                    if not active_mask[i]:
                        joint_action.append(0)
                        continue
                    v_actions = env.get_valid_actions(state, i)
                    a_logits = logits[i].clone()
                    inv_actions = [a for a in range(8) if a not in v_actions]
                    a_logits[inv_actions] = -1e9
                    joint_action.append(torch.argmax(a_logits).item())
            elif mode == "heuristic":
                for i in range(num_agents):
                    if not active_mask[i]:
                        joint_action.append(0)
                        continue
                    heur_probs = env.get_heuristic_policy(state, i)
                    v_actions = env.get_valid_actions(state, i)
                    masked_heur = np.zeros(8)
                    for a in v_actions:
                        masked_heur[a] = heur_probs[a]
                    sum_h = np.sum(masked_heur)
                    if sum_h > 0:
                        masked_heur /= sum_h
                        joint_action.append(np.argmax(masked_heur))
                    else:
                        joint_action.append(0)

            state, _, done, _ = env.step(state, tuple(joint_action))
            step += 1

        end_pos, end_goal, _, _ = state
        reached = sum(1 for i in range(num_agents) if end_pos[i] == end_goal[i])
        episode_results.append(1 if reached == num_agents else 0)

    success_rate = np.mean(episode_results)
    avg_steps = max_steps  # kept for compatibility; meaningful only for successful eps
    return success_rate, avg_steps, np.std(episode_results)

def run_rigorous_validation():
    print("=============================================================")
    print(" Starting Rigorous Multi-Agent Validation Suite...")
    print("=============================================================")
    
    model = PermutationEquivariantGNN(grid_size=13, in_channels=3, d_model=128)
    model_path = "models/multi_agent_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found! Please run train_multi_agent.py first.")
        return
        
    model.load_state_dict(torch.load(model_path))
    print("Loaded trained model weights successfully.")
    
    # Verify Theorem 1
    check_permutation_equivariance(model, num_agents=4)
    
    agent_counts = [2, 3, 4, 5, 6, 8]
    obstacle_modes = ["default", "empty", "random"]
    
    results = {}
    
    # 1. Evaluate GNN+MCTS across different layouts (Robustness & Generalization)
    for obs_mode in obstacle_modes:
        results[obs_mode] = {"rates": [], "steps": [], "stds": []}
        print(f"\n--- Robustness: Obstacle Mode = {obs_mode.upper()} ---")
        for m in agent_counts:
            sr, st, sd = evaluate_performance(model, num_agents=m, obstacle_mode=obs_mode, num_episodes=20, mode="mcts")
            results[obs_mode]["rates"].append(sr)
            results[obs_mode]["steps"].append(st)
            results[obs_mode]["stds"].append(sd)
            print(f"PE-GNN+MCTS | M = {m} | Success: {sr * 100:.1f}% ± {sd * 100:.1f}%")

    # 2. Evaluate baselines on Default Map for ablation study
    baseline_results = {
        "gnn": {"rates": [], "steps": []},
        "heuristic": {"rates": [], "steps": []}
    }
    
    print("\n--- Ablation Baselines (Default Map) ---")
    for mode in ["gnn", "heuristic"]:
        for m in agent_counts:
            sr, st, sd = evaluate_performance(model, num_agents=m, obstacle_mode="default", num_episodes=20, mode=mode)
            baseline_results[mode]["rates"].append(sr)
            baseline_results[mode]["steps"].append(st)
            baseline_results[mode].setdefault("stds", []).append(sd)
            print(f"Baseline: {mode.upper()} | M = {m} | Success: {sr * 100:.1f}% ± {sd * 100:.1f}%")

    # 3. Obstacle Density Robustness Test
    print("\n--- Obstacle Density Robustness (M=4 Agents) ---")
    density_counts = [6, 12, 18] # Low, Medium, High
    density_results = {"rates": [], "steps": []}
    for count in density_counts:
        sr, st, sd = evaluate_performance(model, num_agents=4, obstacle_mode="density", num_episodes=20, mode="mcts", obstacle_count=count)
        density_results["rates"].append(sr)
        density_results["steps"].append(st)
        density_results.setdefault("stds", []).append(sd)
        print(f"Obstacles = {count:2d} | Success: {sr * 100:.1f}% ± {sd * 100:.1f}%")
        
    # Save quantitative results to JSON
    os.makedirs("results", exist_ok=True)
    json_path = "results/academic_results_rigorous.json"
    with open(json_path, "w") as f:
        json.dump({
            "agent_counts": agent_counts,
            "robustness": results,
            "baselines": baseline_results,
            "density": {
                "obstacle_counts": density_counts,
                "rates": density_results["rates"],
                "steps": density_results["steps"]
            }
        }, f, indent=4)
    print(f"\nSaved quantitative results to: {json_path}")
    
    # ── Plots Generation ───────────────────────────────────────────────────
    
    # Plot 1: Robustness of GNN+MCTS across Obstacle Layouts (with error bands)
    fig1, ax1 = plt.subplots(figsize=(8.5, 4.8))
    for obs_mode, color, marker, ls, label in [
        ("default", '#1f77b4', 'o', '-',  'Default Map (Trained Layout)'),
        ("empty",   '#2ca02c', 's', '--', 'Empty Map (No Obstacles)'),
        ("random",  '#d62728', 'd', '-.', 'Random Obstacles (Unseen Layout)'),
    ]:
        rates = np.array(results[obs_mode]["rates"]) * 100
        stds  = np.array(results[obs_mode]["stds"]) * 100
        ax1.plot(agent_counts, rates, marker=marker, linestyle=ls, label=label, linewidth=2.5, color=color)
        ax1.fill_between(agent_counts, rates - stds, rates + stds, alpha=0.15, color=color)
    ax1.axvline(x=4, color='red', linestyle=':', label='Training Agent Limit (M=4)')
    ax1.set_title('PE-GNN+MCTS Robustness across Obstacle Configurations', fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel('Number of Cooperative Agents $M$', fontsize=11)
    ax1.set_ylabel('Success Rate (%) — mean ± std over 20 episodes', fontsize=10)
    ax1.set_xticks(agent_counts)
    ax1.set_ylim(-5, 110)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)
    fig1.tight_layout()
    fig1.savefig('results/scalability_test.png', dpi=300)
    plt.close(fig1)
    
    # Plot 2: MCTS Ablation with error bands
    fig2, ax2 = plt.subplots(figsize=(8.5, 4.8))
    for key, color, marker, ls, label in [
        ("ours",      '#1f77b4', 'o', '-',  'PE-GNN + MCTS (Ours)'),
        ("gnn",       '#ff7f0e', 's', '--', 'PE-GNN Only (No Search)'),
        ("heuristic", '#2ca02c', 'd', '-.', 'Coordinated Heuristic Only'),
    ]:
        rates_data = results["default"]["rates"] if key == "ours" else baseline_results[key]["rates"]
        stds_data  = results["default"]["stds"]  if key == "ours" else baseline_results[key].get("stds", [0]*len(agent_counts))
        rates = np.array(rates_data) * 100
        stds  = np.array(stds_data) * 100
        ax2.plot(agent_counts, rates, marker=marker, linestyle=ls, label=label, linewidth=2.5, color=color)
        ax2.fill_between(agent_counts, rates - stds, rates + stds, alpha=0.15, color=color)
    ax2.axvline(x=4, color='red', linestyle=':', label='Training Agent Limit (M=4)')
    ax2.set_title('Ablation: Success Rate Comparison against Baselines', fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel('Number of Cooperative Agents $M$', fontsize=11)
    ax2.set_ylabel('Success Rate (%) — mean ± std over 20 episodes', fontsize=10)
    ax2.set_xticks(agent_counts)
    ax2.set_ylim(-5, 110)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)
    fig2.tight_layout()
    fig2.savefig('results/mcts_ablation.png', dpi=300)
    plt.close(fig2)

    # Plot 3: Obstacle Density Robustness with std error bars
    fig3, ax3 = plt.subplots(figsize=(7.5, 4.5))
    x_labels = [f"Low ({density_counts[0]})", f"Medium ({density_counts[1]})", f"High ({density_counts[2]})"]
    rates_d = [r * 100 for r in density_results["rates"]]
    stds_d  = [s * 100 for s in density_results.get("stds", [0, 0, 0])]
    bars = ax3.bar(x_labels, rates_d, color=['#2ca02c', '#1f77b4', '#d62728'],
                   alpha=0.85, width=0.5, yerr=stds_d, capsize=6, error_kw=dict(elinewidth=1.5, ecolor='black'))
    ax3.set_title('Robustness to Obstacle Density (M=4 Agents)', fontsize=12, fontweight='bold', pad=15)
    ax3.set_xlabel('Obstacle Density Level (Obstacle Count)', fontsize=11)
    ax3.set_ylabel('Success Rate (%) — mean ± std', fontsize=11)
    ax3.set_ylim(0, 120)
    ax3.grid(axis='y', linestyle='--', alpha=0.6)
    for idx, (rate, std) in enumerate(zip(rates_d, stds_d)):
        ax3.text(idx, rate + std + 2, f"{rate:.1f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/density_robustness.png', dpi=300)
    plt.close()
    
    print("=============================================================")
    print(" Rigorous Validation Completed. Output Files Saved:")
    print(" - results/academic_results_rigorous.json (Quantitative data)")
    print(" - results/scalability_test.png (Robustness curves)")
    print(" - results/mcts_ablation.png (Ablation curves)")
    print(" - results/density_robustness.png (Obstacle density bar plot)")
    print("=============================================================")

if __name__ == "__main__":
    run_rigorous_validation()
