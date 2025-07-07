import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from .util import dbg


# Computes a permutation matrix P that maximizes sum(P * W).
# This is a linear assignment problem, solved with scipy's optimized function.
# If an initial guess P0 is provided, it uses the preconditioning heuristic.
def permutation_match(W, P0=None):
    if P0 is None:
        row_ind, col_ind = min_weight_full_bipartite_matching(csr_matrix(-W))
        dbg("permutation_match [1a]")
        P = csr_matrix((np.ones_like(row_ind), (row_ind, col_ind)), shape=W.shape)
    else:
        W_perm = W @ P0.T  # PERMUTE TO NEARLY DIAGONAL

        # D is a dense ndarray column vector
        D = np.diag(W_perm)
        D_col = D[:, np.newaxis] # D is 1D array, D_col is (N,1) 2D array

        # Ensure sums broadcast correctly. W_perm.sum(axis=0) is (N,), broadcasts over rows.
        # W_perm.sum(axis=1) is (N,), needs reshaping to (N,1) to broadcast over columns.
        sum_axis0 = W_perm.sum(axis=0, keepdims=True) # Shape (1,N)
        sum_axis1 = W_perm.sum(axis=1, keepdims=True) # Shape (N,1)

        # The formula used in the script:
        W_shifted = W_perm - sum_axis0 - sum_axis1 + D_col + D_col.T

        # Recursive call, P0=None ensures it hits the base case using min_weight_full_bipartite_matching
        P_match = permutation_match(W_shifted, P0=None)  # Solve on the preconditioned matrix
        P = P_match @ P0  # UN-PERMUTE to get the final result. P_match and P0 are sparse.

    dbg("permutation_match done")
    return P
