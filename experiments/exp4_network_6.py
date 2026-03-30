"""
Experiment 4: 8-Node Nonlinear System (Network 6)
Validates the robustness of the NR-dCov modification in handling 
high-dimensional, nested nonlinearities (trigonometric, exponential, hyperbolic).
"""

import numpy as np
import pandas as pd

from src.pc_baseline import pc_scratch as pc_conventional
from src.nr_dcov import pc_scratch as pc_modified
from src.utils import plot_cpdag, decompose_shd

def generate_network6_data(n_samples=10000, seed=42):
    """Generates the 8-node non-linear system data."""
    np.random.seed(seed)
    n = n_samples

    # --- Exogenous root nodes ---
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)

    # --- Noise terms ---
    noise_X3 = np.random.normal(0, 0.5, n)
    noise_X4 = np.random.normal(0, 0.5, n)
    noise_X5 = np.random.normal(0, 0.5, n)
    noise_X6 = np.random.normal(0, 0.5, n)
    noise_X7 = np.random.normal(0, 0.5, n)
    noise_X8 = np.random.normal(0, 0.5, n)

    # --- Structural Equations ---
    # X3: function of X1, X2 only
    X3 = np.tanh(0.6 * X1 + 0.6 * X2) + noise_X3

    # X4: child of X3
    X4 = np.sin(0.7 * X3) + 0.3 * X3**2 + noise_X4

    # X5: child of X4 ONLY (logarithmic and squared terms)
    X5 = np.log1p(np.abs(X4)) * np.sign(X4) + 0.5 * X4**2 + noise_X5

    # X6: child of X5
    X6 = np.sin(X5) + 0.3 * X5 + noise_X6

    # X7: child of X5 (sigmoid-like exponential)
    X7 = np.exp(0.3 * X5) / (1 + np.exp(0.3 * X5)) * 3 + noise_X7

    # X8: child of X5 (hyperbolic sine)
    X8 = np.sinh(0.3 * X5) + noise_X8

    # Combine into a DataFrame
    df = pd.DataFrame({
        'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 
        'X5': X5, 'X6': X6, 'X7': X7, 'X8': X8
    })
    return df

def get_ground_truth_graph():
    """
    Returns the ground truth adjacency matrix for the 8-node Network 6.
    Edges: X1->X3, X2->X3, X3->X4, X4->X5, X5->X6, X5->X7, X5->X8
    """
    # Nodes: 0:X1, 1:X2, 2:X3, 3:X4, 4:X5, 5:X6, 6:X7, 7:X8
    true_graph = np.zeros((8, 8), dtype=int)
    true_graph[0, 2] = 1  # X1 -> X3
    true_graph[1, 2] = 1  # X2 -> X3
    true_graph[2, 3] = 1  # X3 -> X4
    true_graph[3, 4] = 1  # X4 -> X5
    true_graph[4, 5] = 1  # X5 -> X6
    true_graph[4, 6] = 1  # X5 -> X7
    true_graph[4, 7] = 1  # X5 -> X8
    return true_graph

def main():
    print("--- Running Experiment 4: 8-Node Nonlinear System (Network 6) ---")
    
    # 1. Generate Data
    # 10,000 samples as defined in the paper's data generating process
    df = generate_network6_data(n_samples=10000)
    labels = df.columns.tolist()
    true_graph = get_ground_truth_graph()
    alpha = 0.05
    
    # 2. Run Conventional PC Algorithm (Fisher-Z)
    print("\nRunning Conventional PC Algorithm (Fisher-Z)...")
    cpdag_conventional = pc_conventional(df, alpha=alpha, labels=labels)
    
    # 3. Run Modified PC Algorithm (NR-dCov)
    print("Running Modified PC Algorithm (NR-dCov)...")
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
    print("* Expected Conventional Total SHD: 8")
    print("* Expected Modified Total SHD: 2")
    
    # 5. Plot the Results
    plot_cpdag(cpdag_conventional, labels, title="Example 4: Conventional PC (Network 6)")
    plot_cpdag(cpdag_modified, labels, title="Example 4: Modified PC (Network 6)")

if __name__ == "__main__":
    main()