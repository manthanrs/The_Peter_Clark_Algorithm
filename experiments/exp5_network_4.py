"""
Experiment 5: 10-Node Complex Nonlinear System (Network 4)
The ultimate stress test. Evaluates the algorithms on a highly entangled 
10-node network with nested non-linear combinations (sinh, tanh, log1p) 
simulating multi-stage cascading effects.
"""

import numpy as np
import pandas as pd

from src.pc_baseline import pc_scratch as pc_conventional
from src.nr_dcov import pc_scratch as pc_modified
from src.utils import plot_cpdag, decompose_shd

def generate_network4_data(n_samples=5000, seed=42):
    """Generates the 10-node complex non-linear SEM data."""
    np.random.seed(seed)
    n = n_samples
    noise_std = 0.8
    
    # Generate structural noise for all 10 variables
    noise = {f'X{i}': np.random.normal(0, noise_std, n) for i in range(1, 11)}

    # --- Root node ---
    X1 = np.random.normal(0, 1, n)

    # --- Level 1: direct children of X1 ---
    X4 = np.tanh(1.2 * X1) + noise['X4']
    X8 = np.arctan(1.5 * X1) + noise['X8']

    # --- Level 2: children of X4 ---
    X5 = np.sinh(0.7 * X4) + noise['X5']
    X7 = np.sign(X4) * np.log1p(np.abs(1.5 * X4)) + noise['X7']

    # --- Level 2: children of X4 and X8 ---
    X2 = np.tanh(0.9 * X4 + 0.7 * X8) + noise['X2']
    X9 = np.arctan(0.9 * X4 + 1.0 * X8) + noise['X9']

    # --- Level 3: children of X2 and X7 ---
    X3 = np.sign(X2 + X7) * np.log1p(np.abs(X2 + X7)) + noise['X3']

    # --- Level 4: X6 <- X1, X5, X9 ---
    X6 = np.tanh(0.6 * X1 + 0.6 * X5 + 0.6 * X9) + noise['X6']

    # --- Level 5: X10 <- X6, X7 ---
    X10 = np.sinh(0.6 * X6 + 0.5 * X7) + noise['X10']

    # Combine into a DataFrame (ensuring correct column order X1 to X10)
    df = pd.DataFrame({
        'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 
        'X6': X6, 'X7': X7, 'X8': X8, 'X9': X9, 'X10': X10
    })
    return df

def get_ground_truth_graph():
    """
    Returns the ground truth adjacency matrix for the 10-node Network 4.
    Based on the structural equations defined in the data generation process.
    """
    # Nodes: 0:X1, 1:X2, ..., 9:X10
    true_graph = np.zeros((10, 10), dtype=int)
    
    # Define the true directed edges
    edges = [
        (0, 3),   # X1 -> X4
        (0, 7),   # X1 -> X8
        (3, 4),   # X4 -> X5
        (3, 6),   # X4 -> X7
        (3, 1),   # X4 -> X2
        (7, 1),   # X8 -> X2
        (3, 8),   # X4 -> X9
        (7, 8),   # X8 -> X9
        (1, 2),   # X2 -> X3
        (6, 2),   # X7 -> X3
        (0, 5),   # X1 -> X6
        (4, 5),   # X5 -> X6
        (8, 5),   # X9 -> X6
        (5, 9),   # X6 -> X10
        (6, 9)    # X7 -> X10
    ]
    
    for parent, child in edges:
        true_graph[parent, child] = 1
        
    return true_graph

def main():
    print("--- Running Experiment 5: 10-Node Complex System (Network 4) ---")
    
    # 1. Generate Data
    df = generate_network4_data(n_samples=5000)
    labels = df.columns.tolist()
    true_graph = get_ground_truth_graph()
    alpha = 0.05
    
    # 2. Run Conventional PC Algorithm (Fisher-Z)
    print("\nRunning Conventional PC Algorithm (Fisher-Z)...")
    cpdag_conventional = pc_conventional(df, alpha=alpha, labels=labels)
    
    # 3. Run Modified PC Algorithm (NR-dCov)
    print("Running Modified PC Algorithm (NR-dCov)...")
    # Note: 10 nodes with deep nesting and random forest permutations will take some time!
    cpdag_modified = pc_modified(df, alpha=alpha, labels=labels)
    
    # 4. Evaluate and Compare (Decomposed SHD)
    str_shd_conv, dir_shd_conv, tot_shd_conv = decompose_shd(true_graph, cpdag_conventional)
    str_shd_mod, dir_shd_mod, tot_shd_mod = decompose_shd(true_graph, cpdag_modified)
    
    print("\n" + "="*50)
    print("RESULTS: DECOMPOSED SHD")
    print("="*50)
    print(f"{'Algorithm':<25} | {'Structural SHD':<14} | {'Directional SHD':<15} | {'Total SHD'}")
    print("-" * 75)
    print(f"{'Conventional PC':<25} | {str_shd_conv:<14} | {dir_shd_conv:<15} | {tot_shd_conv}")
    print(f"{'Modified PC (NR-dCov)':<25} | {str_shd_mod:<14} | {dir_shd_mod:<15} | {tot_shd_mod}")
    print("="*50)
    print("* Expected Modified Structural SHD: 0")
    print("* Expected Conventional Structural SHD: 5 (False Positives)")
    
    # 5. Plot the Results
    plot_cpdag(cpdag_conventional, labels, title="Example 5: Conventional PC (Network 4)", figsize=(10, 8))
    plot_cpdag(cpdag_modified, labels, title="Example 5: Modified PC (Network 4)", figsize=(10, 8))

if __name__ == "__main__":
    main()