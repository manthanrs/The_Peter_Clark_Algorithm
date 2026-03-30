import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple

def calculate_shd_for_arrays(truth, est):
    """
    Calculates SHD between two 0/1 adjacency matrices.
    0: No edge
    1: Edge exists (i -> j)
    If both adj[i,j] and adj[j,i] are 1, it's treated as undirected.
    """
    shd = 0
    N = len(truth)
    
    # We only need to look at each pair of nodes (i, j) once
    for i in range(N):
        for j in range(i + 1, N):
            # Get the edge states for both graphs
            # (i->j, j->i)
            truth_edge = (truth[i][j], truth[j][i])
            est_edge = (est[i][j], est[j][i])
            
            # If the edge patterns don't match exactly, it's an error
            if truth_edge != est_edge:
                shd += 1
                
    return shd

def audit_graph_differences(truth, est, node_names=None):
    """
    Compares two 0/1 adjacency matrices and lists specific edits needed.
    0: No edge, 1: Edge exists.
    """
    if node_names is None:
        node_names = [f"Node {i}" for i in range(len(truth))]
    
    edits = []
    shd_check = 0
    N = len(truth)

    # We iterate through the upper triangle to evaluate each pair (i, j) once
    for i in range(N):
        for j in range(i + 1, N):
            t_ij, t_ji = truth[i][j], truth[j][i]
            e_ij, e_ji = est[i][j], est[j][i]
            
            # 1. No Edge in Truth
            if t_ij == 0 and t_ji == 0:
                if e_ij == 1 or e_ji == 1:
                    shd_check += 1
                    edits.append(f"DELETE extra edge: {node_names[i]} -- {node_names[j]}")
            
            # 2. Directed Edge in Truth: i -> j
            elif t_ij == 1 and t_ji == 0:
                if e_ij == 0 and e_ji == 0:
                    shd_check += 1
                    edits.append(f"ADD missing edge: {node_names[i]} -> {node_names[j]}")
                elif e_ij == 0 and e_ji == 1:
                    shd_check += 1
                    edits.append(f"REVERSE direction: {node_names[j]} -> {node_names[i]} should be {node_names[i]} -> {node_names[j]}")
                elif e_ij == 1 and e_ji == 1:
                    shd_check += 1
                    edits.append(f"DIRECT the undirected edge: {node_names[i]} - {node_names[j]} should be {node_names[i]} -> {node_names[j]}")

            # 3. Directed Edge in Truth: j -> i
            elif t_ij == 0 and t_ji == 1:
                if e_ij == 0 and e_ji == 0:
                    shd_check += 1
                    edits.append(f"ADD missing edge: {node_names[j]} -> {node_names[i]}")
                elif e_ij == 1 and e_ji == 0:
                    shd_check += 1
                    edits.append(f"REVERSE direction: {node_names[i]} -> {node_names[j]} should be {node_names[j]} -> {node_names[i]}")
                elif e_ij == 1 and e_ji == 1:
                    shd_check += 1
                    edits.append(f"DIRECT the undirected edge: {node_names[i]} - {node_names[j]} should be {node_names[j]} -> {node_names[i]}")

            # 4. Undirected Edge in Truth: i - j
            elif t_ij == 1 and t_ji == 1:
                if e_ij == 0 and e_ji == 0:
                    shd_check += 1
                    edits.append(f"ADD missing undirected edge: {node_names[i]} - {node_names[j]}")
                elif (e_ij == 1 and e_ji == 0) or (e_ij == 0 and e_ji == 1):
                    shd_check += 1
                    edits.append(f"REMOVE arrowhead: {node_names[i]} and {node_names[j]} should be undirected (-)")

    return edits, shd_check

def plot_cpdag(adj_matrix, labels, title="Recovered CPDAG", figsize=(10, 8)):
    """
    Plots a Completed Partially Directed Acyclic Graph (CPDAG) from an adjacency matrix.

    Parameters:
    - adj_matrix (np.ndarray): The adjacency matrix (e.g., output from PC algorithm).
                               adj_matrix[i, j] == 1 means an edge from i to j.
    - labels (list): List of node labels (e.g., column names from the dataframe).
    - title (str): Title of the plot.
    - figsize (tuple): Dimensions of the plot.
    """
    # Initialize a directed graph
    G = nx.DiGraph()
    G.add_nodes_from(labels)

    directed_edges = []
    undirected_edges = []

    n = len(labels)
    # Parse the adjacency matrix directly to cleanly separate edge types
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] == 1:
                if adj_matrix[j, i] == 1:
                    # Undirected edge: avoid duplicates by only adding when i < j
                    if i < j:
                        undirected_edges.append((labels[i], labels[j]))
                else:
                    # Directed edge: i -> j
                    directed_edges.append((labels[i], labels[j]))

    # Add edges to the graph for the layout calculation
    G.add_edges_from(directed_edges + undirected_edges)

    # Generate layout
    pos = nx.circular_layout(G)
    plt.figure(figsize=figsize)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color='skyblue')

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Draw undirected edges (blue, no arrows)
    if undirected_edges:
        nx.draw_networkx_edges(G, pos, edgelist=undirected_edges, edge_color='blue', 
                               arrows=False, style='solid', width=1.25)

    # Draw directed edges (green, with arrows)
    if directed_edges:
        nx.draw_networkx_edges(G, pos, edgelist=directed_edges, edge_color='green', 
                               arrows=True, arrowsize=15, width=1.25)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def decompose_shd(true_graph: np.ndarray, pred_graph: np.ndarray) -> Tuple[int, int, int]:
    """
    Calculates the Total Structural Hamming Distance (SHD) and decomposes it 
    into Structural SHD (skeleton errors) and Directional SHD (orientation errors).

    Args:
        true_graph (np.ndarray): The ground truth adjacency matrix.
        pred_graph (np.ndarray): The predicted adjacency matrix (CPDAG).

    Returns:
        Tuple[int, int, int]: (Structural SHD, Directional SHD, Total SHD)
    """
    # 1. Create purely undirected skeletons (symmetric matrices)
    true_skeleton = np.maximum(true_graph, true_graph.T)
    pred_skeleton = np.maximum(pred_graph, pred_graph.T)
    
    # 2. Calculate Structural SHD (Missing or extra edges)
    # Divide by 2 because symmetric matrices double-count edges
    structural_diff = np.abs(true_skeleton - pred_skeleton)
    structural_shd = int(np.sum(structural_diff) / 2)
    
    # 3. Calculate Directional SHD
    # Only evaluate orientation on edges where the skeleton was correctly identified
    correct_skeleton_mask = (true_skeleton == 1) & (pred_skeleton == 1)
    
    directional_shd = 0
    n = true_graph.shape[0]
    
    # Iterate through the upper triangle to check edge pairs
    for i in range(n):
        for j in range(i + 1, n):
            if correct_skeleton_mask[i, j]:
                # Compare the exact (i->j and j->i) states
                true_state = (true_graph[i, j], true_graph[j, i])
                pred_state = (pred_graph[i, j], pred_graph[j, i])
                
                if true_state != pred_state:
                    directional_shd += 1
                    
    total_shd = structural_shd + directional_shd
    
    return structural_shd, directional_shd, total_shd