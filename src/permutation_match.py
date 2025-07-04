import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from src.util import log

# Computes a permutation matrix P that maximizes sum(P * W).
# This is a linear assignment problem, solved with scipy's optimized function.
# If an initial guess P0 is provided, it uses the preconditioning heuristic.
def permutation_match(W, P0=None):
    if P0 is None:
        row_ind, col_ind = min_weight_full_bipartite_matching(csr_matrix(W))
        log("permutation_match [1a]")
        P = csr_matrix((np.ones_like(row_ind), (row_ind, col_ind)), shape=W.shape)
    else:
        W_perm = W @ P0.T  # PERMUTE TO NEARLY DIAGONAL
        log("permutation_match [2a]")
        D = W_perm.diagonal()[:, np.newaxis]  # Ensure D is a column vector
        log("permutation_match [2b]")
        # SHIFT step: W = W - sum(W,1) - sum(W,2) + D + D'
        # This makes the matrix more diagonally dominant to speed up the solver.
        W_shifted = W_perm - W_perm.sum(axis=0) - W_perm.sum(axis=1)[:, np.newaxis] + D + D.T
        log("permutation_match [2c]")
        P_match = permutation_match(W_shifted)  # Solve on the preconditioned matrix
        log("permutation_match [2d]")
        P = P_match @ P0  # UN-PERMUTE to get the final result

    log("permutation_match done")
    return P
