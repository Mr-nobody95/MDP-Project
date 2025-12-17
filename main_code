import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from itertools import product
import os
import scipy.stats as stats
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ==============================
# GRID WORLD CLASS
# ==============================
class GridWorld:
    def __init__(self, size, seed=None):
        """Initialize a grid world with random rewards"""
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.grid = self.rng.choice(REWARD_LEVELS, (size, size))
        all_pos = [(i, j) for i in range(size) for j in range(size)]
        start_end = self.rng.choice(all_pos, 2, replace=False)
        self.start, self.end = tuple(start_end[0]), tuple(start_end[1])
        self.grid[self.end] = 10
        self.grid[self.start] = 0

    def step(self, state, action_idx):
        """Take action in current state, return next state, reward, done"""
        if state == self.end:
            return state, 0, True
        i, j = state
        di, dj = ACTIONS[action_idx]
        ni, nj = np.clip(i + di, 0, self.size - 1), np.clip(j + dj, 0, self.size - 1)
        next_state = (ni, nj)
        reward = self.grid[next_state]
        done = next_state == self.end
        return next_state, reward, done

    def get_all_states(self):
        """Return all possible states in the grid"""
        return [(i, j) for i in range(self.size) for j in range(self.size)]


# ==============================
# CONSTANTS
# ==============================
GRID_SIZE = 32
REWARD_LEVELS = [-5, -2, -1, 0, 1, 2, 5]
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NUM_ACTIONS = len(ACTIONS)

GAMMA_CANDIDATES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
THETA_CANDIDATES = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

HYPERPARAMS = list(product(GAMMA_CANDIDATES, THETA_CANDIDATES))


# ==============================
# ALGORITHM CORE
# ==============================
def run_experiment(seed, gamma, theta, return_details=False):
    """Run a complete experiment with given hyperparameters"""
    env = GridWorld(GRID_SIZE, seed=seed)
    all_states = env.get_all_states()
    NUM_STATES = len(all_states)
    state_to_idx = {s: i for i, s in enumerate(all_states)}
    idx_to_state = {i: s for s, i in state_to_idx.items()}

    # Build transition and reward matrices
    P = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))
    R = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))
    for state in all_states:
        s_idx = state_to_idx[state]
        for a in range(NUM_ACTIONS):
            ns, r, _ = env.step(state, a)
            ns_idx = state_to_idx[ns]
            P[s_idx, a, ns_idx] = 1.0
            R[s_idx, a, ns_idx] = r
    R_exp = np.sum(P * R, axis=2)

    # Value Iteration
    def value_iteration():
        """Value Iteration algorithm implementation"""
        V = np.zeros(NUM_STATES)
        iters = 0
        start_t = time.time()
        deltas = []
        while True:
            delta = 0
            iters += 1
            for s_idx in range(NUM_STATES):
                s = idx_to_state[s_idx]
                if s == env.end:
                    continue
                v_old = V[s_idx]
                q = R_exp[s_idx] + gamma * np.sum(P[s_idx] * V, axis=1)
                V[s_idx] = np.max(q)
                delta = max(delta, abs(v_old - V[s_idx]))
            deltas.append(delta)
            if delta < theta:
                break
        policy = np.zeros((NUM_STATES, NUM_ACTIONS))
        for s_idx in range(NUM_STATES):
            s = idx_to_state[s_idx]
            if s == env.end:
                continue
            q = R_exp[s_idx] + gamma * np.sum(P[s_idx] * V, axis=1)
            policy[s_idx, np.argmax(q)] = 1.0
        return V, policy, iters, time.time() - start_t, deltas

    # Policy Iteration
    def policy_iteration():
        """Policy Iteration algorithm implementation"""
        policy = np.ones((NUM_STATES, NUM_ACTIONS)) / NUM_ACTIONS
        V = np.zeros(NUM_STATES)
        updates = 0
        start_t = time.time()
        change_counts = []
        while True:
            updates += 1
            # Policy Evaluation
            while True:
                delta = 0
                for s_idx in range(NUM_STATES):
                    s = idx_to_state[s_idx]
                    if s == env.end:
                        continue
                    v_old = V[s_idx]
                    v_new = np.sum(
                        policy[s_idx] * (R_exp[s_idx] + gamma * np.sum(P[s_idx] * V, axis=1))
                    )
                    V[s_idx] = v_new
                    delta = max(delta, abs(v_old - V[s_idx]))
                if delta < theta:
                    break
            old_policy = policy.copy()
            change = 0
            # Policy Improvement
            for s_idx in range(NUM_STATES):
                s = idx_to_state[s_idx]
                if s == env.end:
                    continue
                q = R_exp[s_idx] + gamma * np.sum(P[s_idx] * V, axis=1)
                new_a = np.argmax(q)
                if policy[s_idx, new_a] != 1.0:
                    change += 1
                policy[s_idx].fill(0)
                policy[s_idx, new_a] = 1.0
            change_counts.append(change)
            if change == 0:
                break
        return V, policy, updates, time.time() - start_t, change_counts

    def extract_path(policy):
        """Extract optimal path from policy using DFS"""
        path = []
        visited = set()

        def dfs(s):
            if len(path) > 2000:
                return False
            path.append(s)
            visited.add(s)
            if s == env.end:
                return True
            s_idx = state_to_idx[s]
            best_a = np.argmax(policy[s_idx])
            for a in [best_a] + [a for a in range(NUM_ACTIONS) if a != best_a]:
                ns, _, _ = env.step(s, a)
                if ns in visited or abs(ns[0] - s[0]) + abs(ns[1] - s[1]) != 1:
                    continue
                if dfs(ns):
                    return True
            path.pop()
            return False

        dfs(env.start)
        return path

    # Run both algorithms
    _, policy_vi, iters_vi, t_vi, deltas_vi = value_iteration()
    _, policy_pi, updates_pi, t_pi, changes_pi = policy_iteration()
    path_vi = extract_path(policy_vi)
    path_pi = extract_path(policy_pi)
    reward_vi = sum(env.grid[s] for s in path_vi) if path_vi else float('-inf')
    reward_pi = sum(env.grid[s] for s in path_pi) if path_pi else float('-inf')

    if return_details:
        return {
            'env': env,
            'path_vi': path_vi, 'reward_vi': reward_vi, 'iters_vi': iters_vi,
            'time_vi': t_vi, 'deltas_vi': deltas_vi,
            'path_pi': path_pi, 'reward_pi': reward_pi, 'updates_pi': updates_pi,
            'time_pi': t_pi, 'changes_pi': changes_pi
        }
    else:
        return {
            'reward_vi': reward_vi, 'iters_vi': iters_vi, 'time_vi': t_vi,
            'reward_pi': reward_pi, 'updates_pi': updates_pi, 'time_pi': t_pi,
            'len_vi': len(path_vi), 'len_pi': len(path_pi)
        }


# ==============================
# STATISTICAL ANALYSIS FUNCTIONS
# ==============================
def calculate_statistics(data_list, name=""):
    """Calculate comprehensive statistics for a dataset"""
    if not data_list:
        return {}

    data_array = np.array(data_list)
    n = len(data_list)

    stats_dict = {
        'mean': np.mean(data_array),
        'median': np.median(data_array),
        'std': np.std(data_array, ddof=1),
        'var': np.var(data_array, ddof=1),
        'min': np.min(data_array),
        'max': np.max(data_array),
        'q1': np.percentile(data_array, 25),
        'q3': np.percentile(data_array, 75),
        'iqr': np.percentile(data_array, 75) - np.percentile(data_array, 25),
        'n': n
    }

    # Calculate 95% confidence interval
    if n > 1 and stats_dict['std'] > 0:
        sem = stats_dict['std'] / np.sqrt(n)
        ci = stats.t.interval(0.95, n - 1, loc=stats_dict['mean'], scale=sem)
        stats_dict['ci_lower'] = ci[0]
        stats_dict['ci_upper'] = ci[1]
        stats_dict['ci_width'] = ci[1] - ci[0]
    else:
        stats_dict['ci_lower'] = stats_dict['mean']
        stats_dict['ci_upper'] = stats_dict['mean']
        stats_dict['ci_width'] = 0

    # Coefficient of variation
    if stats_dict['mean'] != 0:
        stats_dict['cv'] = stats_dict['std'] / abs(stats_dict['mean'])
    else:
        stats_dict['cv'] = 0

    return stats_dict


def print_statistics_table(stats_dict, metric_name):
    """Print statistics in a formatted table"""
    print(f"\n{'=' * 60}")
    print(f"STATISTICS FOR {metric_name}")
    print(f"{'=' * 60}")

    rows = [
        ("Sample Size (n)", f"{stats_dict['n']}"),
        ("Mean", f"{stats_dict['mean']:.4f}"),
        ("Median", f"{stats_dict['median']:.4f}"),
        ("Std Dev", f"{stats_dict['std']:.4f}"),
        ("Variance", f"{stats_dict['var']:.4f}"),
        ("Min", f"{stats_dict['min']:.4f}"),
        ("Max", f"{stats_dict['max']:.4f}"),
        ("Q1 (25th %ile)", f"{stats_dict['q1']:.4f}"),
        ("Q3 (75th %ile)", f"{stats_dict['q3']:.4f}"),
        ("IQR", f"{stats_dict['iqr']:.4f}"),
        ("95% CI Lower", f"{stats_dict['ci_lower']:.4f}"),
        ("95% CI Upper", f"{stats_dict['ci_upper']:.4f}"),
        ("95% CI Width", f"{stats_dict['ci_width']:.4f}"),
        ("CV (Std/Mean)", f"{stats_dict['cv']:.4f}")
    ]

    for label, value in rows:
        print(f"{label:20s}: {value}")


def main():
    """Main execution function"""
    print("=" * 100)
    print(" " * 30 + "MDP Grid Search and Stability Analysis")
    print("=" * 100)
    
    # Grid Search
    print("\nStarting grid search...")
    results = []
    N_SEEDS = 3
    for gamma, theta in HYPERPARAMS:
        for seed in range(N_SEEDS):
            print(f"Running: γ={gamma}, θ={theta}, seed={seed}...")
            res = run_experiment(seed, gamma, theta, return_details=False)
            results.append({
                'γ': gamma, 'θ': theta, 'seed': seed,
                'VI_iters': res['iters_vi'], 'VI_time': res['time_vi'],
                'PI_updates': res['updates_pi'], 'PI_time': res['time_pi'],
                'reward_vi': res['reward_vi'], 'len_vi': res['len_vi'],
                'reward_pi': res['reward_pi'], 'len_pi': res['len_pi'],
            })

    df = pd.DataFrame(results)
    summary = df.groupby(['γ', 'θ']).agg({
        'VI_iters': 'mean', 'VI_time': 'mean',
        'PI_updates': 'mean', 'PI_time': 'mean',
        'reward_vi': 'mean', 'len_vi': 'mean',
        'reward_pi': 'mean', 'len_pi': 'mean',
    }).round(3)

    # Save results
    os.makedirs('data', exist_ok=True)
    csv_path = 'data/gridsearch_results_32x32.csv'
    summary.to_csv(csv_path)
    print(f"\nGrid search results saved to: {csv_path}")

    # Find best configuration
    best_row = summary[['reward_vi', 'reward_pi']].max(axis=1).idxmax()
    best_gamma, best_theta = best_row
    print(f"\nBest hyperparameters: γ={best_gamma}, θ={best_theta}")

    # Stability analysis
    print(f"\nRunning stability analysis with best configuration (10 runs)...")
    N_STABILITY_RUNS = 10
    stability_results = []
    
    for seed in range(100, 100 + N_STABILITY_RUNS):
        print(f"  Run {seed-99}/{N_STABILITY_RUNS} with seed={seed}...")
        res = run_experiment(seed, best_gamma, best_theta, return_details=False)
        stability_results.append({
            'seed': seed,
            'reward_vi': res['reward_vi'], 'iters_vi': res['iters_vi'],
            'time_vi': res['time_vi'], 'len_vi': res['len_vi'],
            'reward_pi': res['reward_pi'], 'updates_pi': res['updates_pi'],
            'time_pi': res['time_pi'], 'len_pi': res['len_pi']
        })

    stability_df = pd.DataFrame(stability_results)
    stability_csv_path = 'data/stability_analysis_best_config.csv'
    stability_df.to_csv(stability_csv_path, index=False)
    print(f"\nStability analysis saved to: {stability_csv_path}")

    # Calculate and print statistics
    metrics = ['reward_vi', 'iters_vi', 'time_vi', 'len_vi',
               'reward_pi', 'updates_pi', 'time_pi', 'len_pi']
    all_stats = {}
    for metric in metrics:
        all_stats[metric] = calculate_statistics(stability_df[metric].tolist(), metric)

    print("\n" + "=" * 100)
    print(" " * 35 + "COMPREHENSIVE STATISTICS")
    print("=" * 100)

    # VI Statistics
    print("\nVALUE ITERATION (VI)")
    print("-" * 50)
    print_statistics_table(all_stats['reward_vi'], "VI Reward")
    print_statistics_table(all_stats['iters_vi'], "VI Iterations")
    print_statistics_table(all_stats['time_vi'], "VI Time (seconds)")
    print_statistics_table(all_stats['len_vi'], "VI Path Length")

    # PI Statistics
    print("\nPOLICY ITERATION (PI)")
    print("-" * 50)
    print_statistics_table(all_stats['reward_pi'], "PI Reward")
    print_statistics_table(all_stats['updates_pi'], "PI Policy Updates")
    print_statistics_table(all_stats['time_pi'], "PI Time (seconds)")
    print_statistics_table(all_stats['len_pi'], "PI Path Length")

    # Statistical test
    if N_STABILITY_RUNS > 1:
        vi_rewards = stability_df['reward_vi'].values
        pi_rewards = stability_df['reward_pi'].values
        t_stat, p_value = stats.ttest_rel(vi_rewards, pi_rewards)
        
        print(f"\nPaired t-test for Reward (VI vs PI):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        if p_value < 0.05:
            print(f"  Result: SIGNIFICANT difference (p < 0.05)")
        else:
            print(f"  Result: NO significant difference (p ≥ 0.05)")

    print("\n" + "=" * 100)
    print("Analysis complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()
