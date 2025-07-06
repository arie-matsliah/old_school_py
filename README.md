# FlyWire VNC Matching Challenge – Winning Solution Code

This repository contains code from Team Old School (Dan Lee and Lawrence Saul) to discover a 
top-scoring solution to the [Flywire VNC Matching Challenge](https://codex.flywire.ai/app/vnc_matching_challenge). 
The challenge involves matching connectomes of male and female *Drosophila melanogaster* (fruit fly) ventral nervous 
cords (VNCs).

## Overview

The code implements an algorithm to find the optimal matching between two sets of neurons based on their 
connectivity graphs. It utilizes techniques like the Frank-Wolfe algorithm and permutation matching to 
achieve high accuracy.

This repository includes:
- The original MATLAB implementation (`src_m/`).
- A Python translation of the code (`src_py/`) provided by Arie Matsliah.
- Data:
    - Full dataset from the FlyWire challenge (`data/`)
    - Smaller dataset for quicker evaluation (`data/compact/`)
    - Tiny dataset for fast integrity checks (`data/tiny/`)

## MATLAB Version

### Files

The `src_m` directory contains the following MATLAB files:

- `main.m`: The main script to run the optimization.
- `compute_gradient.m`: Computes the gradient for the optimization.
- `do_frank_wolfe.m`: Implements the Frank-Wolfe algorithm steps.
- `do_mult_updates.m`: Performs multiplicative updates.
- `do_swaps.m`: Implements swap operations for refinement.
- `permutation_match.m`: Solves the assignment problem for permutation matching.
- `read_connectome.m`: Reads the connectome graph data.
- `read_solution.m`: Reads a previously saved solution.
- `save_solution.m`: Saves the current solution.

### Running the MATLAB Code

1.  Navigate to the `src_m` directory in MATLAB.
2.  Run the main script by typing:
    ```matlab 
    main
    ```

The code was developed on a MacBook Pro (2021 Apple M1 Max) with 64 GB of RAM. The optimization can be initialized from 
a warm start (at the benchmark solution) or a cold start (at the barycenter of the simplex of all doubly stochastic matrices).
- From a warm start, it takes less than 20 minutes to reach a score of 5850K.
- From a cold start, it takes less than 45 minutes.

More details of these runs can be found in these files:
- `log/log_warm_start.txt`
- `log/log_cold_start.txt`

The initialization method is determined by the setting in line 3 of `main.m`. 
Other settings for the code can be changed in lines 4-6 of `main.m`.

## Python Version

### Files

The `src_py` directory contains the Python implementation:

- `main.py`: The main script to run the optimization.
- `compute_gradient.py`: Computes the gradient.
- `do_frank_wolfe.py`: Implements the Frank-Wolfe algorithm.
- `do_mult_updates.py`: Performs multiplicative updates.
- `do_swaps.py`: Implements swap operations.
- `permutation_match.py`: Solves the assignment problem.
- `util.py`: Utility functions for reading data and saving solutions.

### Setup and Dependencies

1.  Ensure you have Python 3 installed.
2.  Install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file lists the following dependencies:
    - `numpy`
    - `scipy`
    - `pandas`

### Running the Python Code

1.  Navigate to the root directory of the repository.
2.  Run the main script:
    ```bash
    python src_py/main.py
    ```
    You can modify `src_py/main.py` to change settings such as the dataset used 
3. (e.g., `tiny`, `compact`, or full dataset) and whether to start from a warm or cold start.

### Running Python Tests

Unit tests are provided to verify the correctness of the Python implementation.

1.  Navigate to the root directory of the repository.
2.  Run the tests using the `unittest` module:
    ```bash
    python -m unittest test/unit.py
    ```

## Contact

For questions about the original MATLAB code, contact Lawrence Saul (lsaul@flatironinstitute.org).
For questions about the Python translation, contact Arie Matsliah (arie@princeton.edu).
