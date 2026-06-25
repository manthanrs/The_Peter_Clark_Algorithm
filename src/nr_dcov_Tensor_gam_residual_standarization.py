# This block generates the meta file which consists of all conditional independence tests that have been carried out and
# corresponding p-values (this will be saved in the home directory of this file)
class MetaLogger:
    def __init__(self, filepath="meta.txt"):
        self.filepath = filepath
        self.logs = []

    def log_test(self, x, y, S, p_val, alpha):
        result = "ind" if p_val > alpha else "dep"
        line = f"{x} {result} {y} | {tuple(S)} with p-value {p_val:.6f}"
        self.logs.append(line)

    def log_orientation(self, src, dst, rule_name):
        line = f"Oriented: {src} → {dst} by rule: {rule_name}"
        self.logs.append(line)

    def write(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            for line in self.logs:
                f.write(line + "\n")

# Importing essential libraries 

import dcor
import itertools
import copy
import scipy.stats as stats
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from math import log, sqrt
from collections.abc import Iterable
from scipy.stats import chi2, norm
from causallearn.search.ConstraintBased.PC import pc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from pygam import LinearGAM, s, te
# Fix the pyGAM + Scikit-Learn cloning compatibility bug
def pygam_sklearn_clone(self):
    return copy.deepcopy(self)

LinearGAM.__sklearn_clone__ = pygam_sklearn_clone
# Fix the pyGAM + Scikit-Learn cloning compatibility bug
def pygam_sklearn_clone(self):
    return copy.deepcopy(self)

LinearGAM.__sklearn_clone__ = pygam_sklearn_clone
logger = MetaLogger()

"""
    Find neighbors of node x excluding node y (i.e., Adjacencies(x) / {y})
    
    param Graph: Adjacency matrix of the graph (2D NumPy array), where Graph[i][j] = 1 indicates an edge from node i to node j
    param x: Index of the node x (integer)
    param y: Index of the node y (integer to exclude from x's neighbors)
    
    return: List of node indices that are direct neighbors of x, excluding y
"""
# Given node, it will return list of neighboring nodes (based on connections)
def neighborsFind(Graph, x, y): 
    neighbors = np.where(Graph[x] == 1)[0]
    neighbors = neighbors[neighbors != y]
    return neighbors.tolist()

"""
    Find all undirected edges in the graph (i.e., bidirectional connections)
    
    param G: Adjacency matrix of the graph (2D NumPy array), 
             where G[i][j] = 1 and G[j][i] = 1 indicates an undirected edge between node i and node j
    
    return: List of tuples (i, j), where each tuple represents an undirected edge (i --- j)
"""
# only undirected edges (which are all edges in the beginning)
def undirectNodeBrother(G):  
    ind = []
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            if G[i][j] == 1 and G[j][i] == 1:
                ind.append((i, j))
    return ind


"""
    Non-parametric permutation test for conditional independence using
    distance covariance / partial distance covariance.
"""

def nonlinear_permutation_test(df, X, Y, condition_set=None, n_permutations=999, random_seed=42):
    """
    Nonlinear residualization + permutation-based distance covariance test.
    Mirrors your current permutation_test but uses Random Forest regression
    for nonlinear residualization when a conditioning set is provided.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with variables as columns.
    X, Y : str
        Names of the two variables being tested.
    condition_set : list[str] or None
        List of conditioning variable names.
    n_permutations : int, default=1000
        Number of permutations for p-value estimation.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    p_value : float
        Estimated permutation-based p-value.
    """

    rng = np.random.default_rng(random_seed)

    A = df[X].values.reshape(-1, 1)
    C = df[Y].values.reshape(-1, 1)

    # --- Case 1: No conditioning set (marginal dependence)
    if not condition_set:
        r_observed = dcor.distance_covariance(A, C)
        r_permuted = np.zeros(n_permutations)
        for i in range(n_permutations):
            C_perm = rng.permutation(C)
            r_permuted[i] = dcor.distance_covariance(A, C_perm)

    # --- Case 2: Conditional dependence via nonlinear residualization
    else:
 
        # 1. Prepare your conditioning set Z
        Z = df[condition_set].values
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        conditioning_set_size = Z.shape[1]

        # 2. Dynamically build the GAM formula structure
        # We automatically add a smooth spline term 's(i)' for every variable in Z
        # 1. DYNAMIC TERM SELECTION (Mean Function)
        # If conditioning on 1 variable, use a simple spline.
        # If conditioning on multiple, use a Tensor Product to capture physical cross-terms.
        if conditioning_set_size == 1:
            gam_terms = s(0)
        else:
            gam_terms = te(*range(conditioning_set_size))
        

        # 3. Instantiate the GAM models
        # n_splines=15 is plenty smooth for tank physics. lam controls regularization.
        gam_x = LinearGAM(gam_terms, max_iter=2000, lam=0.5)
        gam_y = LinearGAM(gam_terms, max_iter=2000, lam=0.5)

        # 4. Generate K-Fold Cross-Validated predictions 
        # This replaces the .oob_prediction_ mechanism perfectly
        cv_folds = 5  # 5-fold CV is fast and gives clean, generalized residuals
        A_ravel = A.ravel()
        C_ravel = C.ravel()

        pred_X = cross_val_predict(gam_x, Z, A_ravel, cv=cv_folds, n_jobs=-1)
        pred_Y = cross_val_predict(gam_y, Z, C_ravel, cv=cv_folds, n_jobs=-1)

        raw_res_X = A_ravel - pred_X
        raw_res_Y = C_ravel - pred_Y
    
        # 3. FIX THE VOLATILITY (Heteroscedasticity Adjustment)
        # Fit a secondary GAM to the squared residuals to model how the variance scales with Z
        #First, we take our raw residuals and square them (raw_res_X2). Why? Because an error of 
        #−5 cm and +5 cm represent the exact same amount of volatility. Squaring them makes everything 
        #positive so we are only looking at the pure magnitude of the noise.
        var_gam_x = LinearGAM(gam_terms).fit(Z, raw_res_X**2)
        var_gam_y = LinearGAM(gam_terms).fit(Z, raw_res_Y**2)
    
        # Predict variance and clip at a tiny positive number to avoid division by zero
        pred_var_X = np.clip(var_gam_x.predict(Z), 1e-6, None)
        pred_var_Y = np.clip(var_gam_y.predict(Z), 1e-6, None)
    
        # Standardize the residuals by their predicted standard deviation
        res_X = raw_res_X / np.sqrt(pred_var_X)
        res_Y = raw_res_Y / np.sqrt(pred_var_Y)
        
        
        # 5. Compute the clean, smooth residuals for Distance Covariance
        #res_X = A_ravel - pred_X
        #res_Y = C_ravel - pred_Y
        

        '''plt.figure()
        plt.plot(res_X)
        plt.plot(res_Y)
        plt.title("GAM Residuals")
        plt.xlabel("Index")
        plt.ylabel("Residuals")'''

        
        
        # Observed distance covariance between residuals
        r_observed = dcor.distance_covariance(res_X, res_Y)
        """plt.figure()
        plt.title("Distance covariances")
        plt.xlabel("Index")
        plt.ylabel("dCov(Rx,Ry)")        
        plt.pause(5)
        print(r_observed)"""
        
        # Permutation null distribution
        r_permuted = np.zeros(n_permutations)
        for i in range(n_permutations):
            res_Y_perm = np.random.permutation(res_Y)
            r_permuted[i] = dcor.distance_covariance(res_X, res_Y_perm)

    # --- Compute p-value
    extreme_count = np.sum(np.abs(r_permuted) >= np.abs(r_observed))
    p_value = (1 + extreme_count) / (1 + n_permutations)

    return p_value

def skeleton_stable(df, alpha, labels):
    n = len(labels)
    G = np.ones((n, n)) - np.eye(n)  # fully connected undirected graph (no self-loops)
    sepset = [[None for _ in range(n)] for _ in range(n)]

    ord = 0
    done = False

    while not done:
        done = True
        edge_removal = []

        for x in range(n):
            for y in range(n):
                if x == y: #Skip self-loops and already removed edges
                    continue
                if G[x][y] == 0: #Skip self-loops and already removed edges
                    continue

                neighbors_x = [i for i in range(n) if G[x][i] == 1 and i != y] #neighbors of x excluding y 
                if len(neighbors_x) < ord:
                    continue

                found_sep = False
                for S in itertools.combinations(neighbors_x, ord):
                    S_labels = [labels[i] for i in S]
                    #p_val = permutation_test(df, labels[x], labels[y], S_labels)
                    p_val = nonlinear_permutation_test(df, labels[x], labels[y], S_labels)

                    result = "ind" if p_val > alpha else "dep"
                    logger.log_test(x, y, S, p_val, alpha) #for generating meta file 
                    print(f"{x} {result} {y} | {tuple(S)} with p-value {p_val:.6f}")

                    if p_val > alpha:
                        sepset[x][y] = list(S)
                        sepset[y][x] = list(S)
                        edge_removal.append((x, y))
                        found_sep = True
                        break

                if not found_sep and len(neighbors_x) >= ord:
                    done = False  # More combinations possible at higher order

        for x, y in edge_removal:
            G[x][y] = 0
            G[y][x] = 0

        ord += 1

    ind = [(i, j) for i in range(n) for j in range(i + 1, n) if G[i][j] == 1]
    return {'sk': G, 'sepset': sepset, 'ind': ind}

"""
    Identify and orient V-structures in the graph
    
    param Graph: Dictionary output from skeleton_stable() containing:
                 - 'sk': adjacency matrix of the undirected skeleton
                 - 'sepset': separation sets from conditional independence tests
    param labels: List of variable names (used for readable logging and print statements)
    
    return: Updated adjacency matrix with v-structures oriented:
            If x - y - z and x and z are not connected, and y ∉ SepSet(x,z), then orient as x → y ← z
"""

def V_Structure(Graph, labels): #final version with debug print
    G = Graph['sk']
    Sep = Graph['sepset']
    n = len(labels)

    for y in range(n):
        neighbors = [i for i in range(n) if G[y][i] == 1 and G[i][y] == 1]
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                x = neighbors[i]
                z = neighbors[j]

                # Enforce consistent order (x < z) to avoid processing twice
                if x > z:
                    x, z = z, x

                # Check unshielded triple: x - y - z, but x and z not connected
                if G[x][z] == 0 and G[z][x] == 0:
                    sep_xz = Sep[x][z] if Sep[x][z] is not None else []
                    sep_zx = Sep[z][x] if Sep[z][x] is not None else []

                    if y not in sep_xz and y not in sep_zx:
                        # Only orient if still undirected
                        if G[x][y] == 1 and G[y][x] == 1:
                            G[x][y], G[y][x] = 1, 0
                            logger.log_orientation(labels[x], labels[y], "V-Structure")
                        if G[z][y] == 1 and G[y][z] == 1:
                            G[z][y], G[y][z] = 1, 0
                            logger.log_orientation(labels[z], labels[y], "V-Structure")
                        print(f"Orienting v-structure: {labels[x]} → {labels[y]} ← {labels[z]}")

    return G

"""
    Check if there is a directed path from 'start' to 'end' in the graph

    param graph: Adjacency matrix representing a partially directed graph
                 - graph[i][j] = 1 and graph[j][i] = 0 ⇒ i → j (directed edge)
    param start: Index of the starting node (integer)
    param end: Index of the ending node (integer)

    return: True if a directed path exists from 'start' to 'end', else False
"""
 
def has_directed_path(graph, start, end): #Depth-First Search (DFS) 
    n = len(graph)
    visited = [False] * n
    stack = [start]

    while stack:
        node = stack.pop()
        if node == end:
            return True
        for neighbor in range(n):
            if graph[node][neighbor] == 1 and graph[neighbor][node] == 0 and not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)
    return False

"""
    Apply Rule 1: If A → B — C and A is not connected to C, then orient B → C

    param Graph: Adjacency matrix representing a partially directed graph:
                 - Graph[i][j] = 1 and Graph[j][i] = 0 ⇒ i → j (directed edge)
                 - Graph[i][j] = Graph[j][i] = 1 ⇒ i — j (undirected edge)
                 - Graph[i][j] = Graph[j][i] = 0 ⇒ no edge
    param labels: List of variable names (used for logging and readable printout)

    return: 
        - Updated adjacency matrix with new orientations applied
        - Boolean flag 'changed' indicating whether any edge was oriented in this pass
"""

def rule1(Graph, labels):
    changed = False
    n = len(Graph)
    for a in range(n):
        for b in range(n):
            if Graph[a][b] == 1 and Graph[b][a] == 0:  # A → B
                for c in range(n):
                    if (
                        Graph[b][c] == 1 and Graph[c][b] == 1  # B — C
                        and Graph[a][c] == 0 and Graph[c][a] == 0  # A not connected to C
                        and len({a, b, c}) == 3
                    ):
                        Graph[b][c] = 1
                        Graph[c][b] = 0
                        logger.log_orientation(labels[b], labels[c], "Rule 1")
                        changed = True
    return Graph, changed

"""
    Apply Rule 2: If A — B and there exists a directed path A → ... → B, 
    then orient the undirected edge A — B as A → B to maintain acyclicity.

    param Graph: Adjacency matrix of a partially directed graph
                 - Graph[i][j] = 1 and Graph[j][i] = 0 ⇒ i → j (directed)
                 - Graph[i][j] = Graph[j][i] = 1 ⇒ i — j (undirected)
    param labels: List of variable names (for readable logging)

    return:
        - Updated adjacency matrix after applying Rule 2
        - Boolean flag 'changed' indicating whether any orientation was made
"""

def rule2(Graph, labels):
    changed = False
    n = len(Graph)
    for a in range(n):
        for b in range(n):
            if Graph[a][b] == 1 and Graph[b][a] == 1:  # A — B
                if has_directed_path(Graph, a, b) and not has_directed_path(Graph, b, a):
                    Graph[a][b] = 1
                    Graph[b][a] = 0
                    logger.log_orientation(labels[a], labels[b], "Rule 2")
                    changed = True
    return Graph, changed

"""
    Apply Rules 1 and 2 iteratively until no further orientations are possible.
"""

def extend_cpdag(graph, labels):
    changed = True
    pdag = graph.copy()
    while changed:
        changed = False
        pdag, ch1 = rule1(pdag, labels)
        pdag, ch2 = rule2(pdag, labels)
        changed = ch1 or ch2
    return pdag

def pc_scratch(df, alpha, labels):
    skeletonGraph = skeleton_stable(df, alpha, labels)
    Ved_Graph = V_Structure(skeletonGraph, labels)
    CPDAG = extend_cpdag(Ved_Graph, labels)
    logger.write()
    return CPDAG
