import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from src_py.util import dbg


# Computes a permutation matrix P that maximizes sum(P * W).
# This is a linear assignment problem, solved with scipy's optimized function.
# If an initial guess P0 is provided, it uses the preconditioning heuristic.
def permutation_match(W, P0=None):
    if P0 is None:
        row_ind, col_ind = min_weight_full_bipartite_matching(csr_matrix(W), maximize=True)
        dbg(f"Match score: {W[row_ind, col_ind].sum()}")
        P = csr_matrix((np.ones_like(row_ind), (row_ind, col_ind)), shape=W.shape)
    else:
        W_perm = W @ P0.T  # PERMUTE TO NEARLY DIAGONAL
        # D is a dense ndarray column vector
        D = np.diag(W_perm)
        row_sum = W_perm.sum(axis=1, keepdims=True)  # shape (N, 1)
        col_sum = W_perm.sum(axis=0, keepdims=True)  # shape (1, N)
        W_shifted = W_perm - col_sum - row_sum + D[:, None] + D[None, :]
        P_match = permutation_match(W_shifted)  # Solve on the preconditioned matrix
        P = P_match @ P0  # UN-PERMUTE to get the final result

    dbg("permutation_match done")
    return P
