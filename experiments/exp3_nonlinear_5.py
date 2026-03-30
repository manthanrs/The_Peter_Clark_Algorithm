"""
Experiment 3: 5-Node Non-Linear SEM
Demonstrates the breakdown of the conventional PC algorithm (Fisher-Z) 
when confronted with non-linear conditional dependencies, and validates 
the perfect skeleton recovery of the proposed NR-dCov modification.
"""

import numpy as np
import pandas as pd

from src.pc_baseline import pc_scratch as pc_conventional
from src.nr_dcov import pc_scratch as pc_modified
from src.utils import plot_cpdag, decompose_shd

def generate_nonlinear_data(n_samples=5000, seed=42):
    """Generates the 5-node non-linear SEM data."""
    np.random.seed(seed)
    
    # Exogenous noise terms
    noise_B = np.random.normal(0, 1, n_samples)
    noise_C = np.random.normal(0, 1, n_samples)
    noise_D = np.random.normal(0, 1, n_samples)
    noise_E = np.random.normal(0, 1, n_samples)
    
    # Root node
    A = np.random.normal(0, 1, n_samples)
    
    # Non-linear structural equations
    B = np.tanh(2.0 * A) + noise_B
    C = np.sinh(np.tanh(B)) + noise_C                             
    D = np.arctan(2.5 * B) + noise_D                              
    E = np.tanh(1.5 * C) + np.sign(D) * np.log1p(np.abs(1.5 * D)) + noise_E  
    
    # Combine into a DataFrame
    df = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D, 'E': E})
    return df

def get_ground_truth_graph():
    """
    Returns the ground truth adjacency matrix for the 5-node non-linear SEM.
    Edges: A->B, B->C, B->D, C->E, D->E
    """
    # Nodes: 0:A, 1:B, 2:C, 3:D, 4:E
    true_graph = np.zeros((5, 5), dtype=int)
    true_graph[0, 1] = 1  # A -> B
    true_graph[1, 2] = 1  # B -> C
    true_graph[1, 3] = 1  # B -> D
    true_graph[2, 4] = 1  # C -> E
    true_graph[3, 4] = 1  # D -> E
    return true_graph

def main():
    print("--- Running Experiment 3: 5-Node Non-Linear SEM ---")
    
    # 1. Generate Data
    df = generate_nonlinear_data(n_samples=5000)
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
    print("* Expected Modified Structural SHD: 0")
    print("* Expected Conventional Structural SHD: 3 (False Positives)")
    
    # 5. Plot the Results
    plot_cpdag(cpdag_conventional, labels, title="Example 3: Conventional PC (Fisher-Z)")
    plot_cpdag(cpdag_modified, labels, title="Example 3: Modified PC (NR-dCov)")

if __name__ == "__main__":
    main()