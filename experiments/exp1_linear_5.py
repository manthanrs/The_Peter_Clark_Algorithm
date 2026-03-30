"""
Experiment 1: 5-Node Linear-Gaussian SEM
Reproduces the sanity check demonstrating that the NR-dCov modification 
does not degrade the performance of the baseline PC algorithm on purely linear data.
"""

import numpy as np
import pandas as pd
from src.utils import plot_cpdag, decompose_shd

from src.pc_baseline import pc_scratch as pc_conventional
from src.nr_dcov import pc_scratch as pc_modified

def generate_linear_data(n_samples=5000, seed=42):
    """Generates the 5-node linear SEM data."""
    np.random.seed(seed)
    
    # Root node
    A = np.random.normal(0, 1, n_samples)
    
    # Exogenous noise terms
    noise_B = np.random.normal(0, 1, n_samples)
    noise_C = np.random.normal(0, 1, n_samples)
    noise_D = np.random.normal(0, 1, n_samples)
    noise_E = np.random.normal(0, 1, n_samples)
    
    # Linear structural equations
    B = 0.7 * A + noise_B
    C = 0.6 * B + noise_C
    D = 0.5 * B + noise_D
    E = 0.6 * C + 0.5 * D + noise_E
    
    # Combine into a DataFrame
    df = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D, 'E': E})
    return df

def get_ground_truth_graph():
    """Returns the ground truth adjacency matrix for the 5-node linear SEM."""
    # Nodes: 0:A, 1:B, 2:C, 3:D, 4:E
    true_graph = np.zeros((5, 5), dtype=int)
    true_graph[0, 1] = 1  # A -> B
    true_graph[1, 2] = 1  # B -> C
    true_graph[1, 3] = 1  # B -> D
    true_graph[2, 4] = 1  # C -> E
    true_graph[3, 4] = 1  # D -> E
    return true_graph

def main():
    print("--- Running Experiment 1: 5-Node Linear SEM ---")
    
    # 1. Generate Data
    df = generate_linear_data(n_samples=5000)
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
    
    # 5. Plot the Results
    plot_cpdag(cpdag_conventional, labels, title="Example 1: Conventional PC (Fisher-Z)")
    plot_cpdag(cpdag_modified, labels, title="Example 1: Modified PC (NR-dCov)")

if __name__ == "__main__":
    main()