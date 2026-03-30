"""
Experiment 2: 20-Node Linear-Gaussian Benchmark
Reproduces the evaluation on the 20-node dataset sourced from the 
causal-learn repository to confirm scalability and baseline parity.
"""

import os
import sys
import numpy as np
import pandas as pd

from src.pc_baseline import pc_scratch as pc_conventional
from src.nr_dcov import pc_scratch as pc_modified
from src.utils import plot_cpdag, decompose_shd

def get_ground_truth_graph():
    """Returns the hardcoded 20-node ground truth adjacency matrix."""
    ground_truth_20node = [ 
        [0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1],
        [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
        [0,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
        [1,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]
    return np.array(ground_truth_20node, dtype=int)

def main():
    print("--- Running Experiment 2: 20-Node Linear Benchmark ---")
    
    # 1. Setup paths and Data
    # Assuming the script is run from the root of the repository
    data_path = os.path.join("data", "data_linear_10.txt")
    
    if not os.path.exists(data_path):
        print(f"\n[ERROR] Dataset not found at {data_path}")
        print("Please ensure 'data_linear_10.txt' is placed in the 'data/' folder.")
        sys.exit(1)
        
    print(f"Loading data from {data_path}...")
    # Load the space-separated values from the text file
    raw_data = np.loadtxt(data_path)
    
    # Define labels and wrap in DataFrame
    node_labels = [f"X{i}" for i in range(1, 21)]
    df = pd.DataFrame(raw_data, columns=node_labels)
    
    true_graph = get_ground_truth_graph()
    alpha = 0.05
    
    # 2. Run Conventional PC Algorithm (Fisher-Z)
    print("\nRunning Conventional PC Algorithm (Fisher-Z)...")
    # This may take a minute depending on your CPU
    cpdag_conventional = pc_conventional(df, alpha=alpha, labels=node_labels)
    
    # 3. Run Modified PC Algorithm (NR-dCov)
    print("Running Modified PC Algorithm (NR-dCov)...")
    # This will take longer due to Random Forest and Permutations
    cpdag_modified = pc_modified(df, alpha=alpha, labels=node_labels)
    
    # 4. Evaluate and Compare (Decomposed SHD)
    str_shd_conv, dir_shd_conv, tot_shd_conv = decompose_shd(true_graph, cpdag_conventional)
    str_shd_mod, dir_shd_mod, tot_shd_mod = decompose_shd(true_graph, cpdag_modified)
    
    print("\n" + "="*50)
    print("RESULTS: DECOMPOSED SHD (20-Node Network)")
    print("="*50)
    print(f"{'Algorithm':<25} | {'Structural SHD':<14} | {'Directional SHD':<15} | {'Total SHD'}")
    print("-" * 75)
    print(f"{'Conventional PC':<25} | {str_shd_conv:<14} | {dir_shd_conv:<15} | {tot_shd_conv}")
    print(f"{'Modified PC (NR-dCov)':<25} | {str_shd_mod:<14} | {dir_shd_mod:<15} | {tot_shd_mod}")
    print("="*50)
    
    # 5. Plot the Results
    # Using a slightly larger figsize for the 20-node network to prevent overlap
    plot_cpdag(cpdag_conventional, node_labels, title="Example 2: Conventional PC (20-Node)", figsize=(12, 10))
    plot_cpdag(cpdag_modified, node_labels, title="Example 2: Modified PC (20-Node)", figsize=(12, 10))

if __name__ == "__main__":
    main()