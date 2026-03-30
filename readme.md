# Non-Linear Residualization and Distance Covariance (NR-dCov) for Causal Discovery

This repository contains the official Python implementation of the NR-dCov modified PC Algorithm. 

Traditional constraint-based causal discovery methods, such as the standard Peter-Clark (PC) algorithm, rely heavily on parametric tests like Fisher's Z. While effective for linear-Gaussian data, these methods often fail to conditionally separate variables in complex industrial systems characterized by nested non-linearities (e.g., hyperbolic, trigonometric, and logarithmic relationships).

This project proposes a robust modification to the PC algorithm's skeleton discovery phase. By replacing linear partial correlation with **Non-Parametric Permutation Testing via Distance Covariance**, combined with **Random Forest-based Non-Linear Residualization**, this method successfully isolates true conditional dependencies in highly entangled non-linear networks.

## Features
* **Standard PC Algorithm Implementation:** A baseline implementation using the classical Fisher-Z test.
* **NR-dCov CI Test:** A custom conditional independence testing pipeline robust to non-linear confounding.
* **Meek Rules Engine:** Deterministic edge orientation logic for V-structure detection and cycle prevention.
* **Evaluation Metrics:** Built-in Structural Hamming Distance (SHD) calculators, decomposed into Structural and Directional error components.

## Repository Structure
* `src/`: Contains the core implementation of the conventional PC algorithm and the proposed NR-dCov modification.
* `experiments/`: Data generation scripts and execution pipelines to reproduce the 5-node, 8-node, and 10-node network evaluations discussed in the research.
* `data/`: Includes the 20-node linear benchmark dataset sourced from the `causal-learn` repository.

## Installation
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/YOUR_USERNAME/NR-dCov-Causal-Discovery.git](https://github.com/YOUR_USERNAME/NR-dCov-Causal-Discovery.git)
cd NR-dCov-Causal-Discovery
pip install -r requirements.txt
