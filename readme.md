# Data-driven Causal Discovery In Nonlinear Systems: A Generalized PC Algorithm

This repository contains the official Python implementation of the Generalized PC (gPC) Algorithm, introducing a non-parametric Conditional Independence (CI) testing pipeline designed for non-linear industrial systems.

Traditional constraint-based causal discovery methods, such as the standard Peter-Clark (PC) algorithm, rely heavily on parametric tests like Fisher's Z. While effective for linear-Gaussian data, these methods often fail to conditionally separate variables in complex systems characterized by coupled non-linearities (e.g., the Quadruple Tank Benchmark system).

This project proposes a robust modification to the PC algorithm's skeleton discovery phase. By replacing linear partial correlation with **Non-Parametric Permutation Testing via Distance Covariance**, combined with **Non-Linear Residualization**, this method successfully isolates true conditional dependencies in highly entangled non-linear networks.

## Features & Implementation Variants
To capture complex non-linear interactions, the non-linear residualization step is implemented using two primary regression frameworks, resulting in five distinct pipeline variants provided in this repository:

1. **`nr_dcov_oob.py`**: Random Forest residualization utilizing highly efficient Out-of-Bag (OOB) predictions.
2. **`nr_dcov_kfoldCV.py`**: Random Forest residualization using a rigorous Repeated K-Fold (5 splits, 5 repeats) cross-validation scheme.
3. **`nr_dcov_oob_hyperparameter_tuning.py`**: Random Forest implementation with an embedded grid search over `max_depth` and `ccp_alpha` for optimal tree configuration prior to residual extraction.
4. **`nr_dcov_gam.py`**: Generalized Additive Models (GAM) utilizing 1D cubic splines to map additive non-linear effects.
5. **`nr_dcov_Tensor_gam.py`**: Advanced GAM implementation utilizing tensor products to capture multi-variable physical cross-terms, coupled with a secondary GAM to correct for residual heteroscedasticity.

## Evaluated Benchmarks
The algorithms have been benchmarked against the conventional PC algorithm (Fisher-Z) using Structural Hamming Distance (SHD) across three datasets:
* **5-Node Synthetic Network:** A pure mathematical benchmark stacking non-linear functions (tanh, sinh, arctan).
* **Quadruple Tank System (6 Nodes):** A highly coupled physical benchmark simulating gravity drainage and pump states.
* **10-Node Complex Network:** A multi-stage cascading non-linear system.

## Repository Structure
* `src/`: Contains the core implementations of the gPC algorithm variants (the 5 scripts listed above) and the conventional baseline.
* `utils/`: Contains `utils.py` for graph auditing, exact SHD calculation (decomposed into structural and directional errors), and CPDAG visualization.
* `data/`: Datasets representing the evaluated benchmarks.

## Installation & Usage
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/manthanrs/The_Generalized_Peter_Clark_Algorithm.git](https://github.com/manthanrs/The_Generalized_Peter_Clark_Algorithm.git)
cd The_Generalized_Peter_Clark_Algorithm
