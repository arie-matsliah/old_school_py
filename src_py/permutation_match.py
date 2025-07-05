import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from src_py.util import log


# Computes a permutation matrix P that maximizes sum(P * W).
# This is a linear assignment problem, solved with scipy's optimized function.
# If an initial guess P0 is provided, it uses the preconditioning heuristic.
def permutation_match(W, P0=None):
    if P0 is None:
        row_ind, col_ind = min_weight_full_bipartite_matching(csr_matrix(-W))
        log("permutation_match [1a]")
        P = csr_matrix((np.ones_like(row_ind), (row_ind, col_ind)), shape=W.shape)
    else:
        W_perm = W @ P0.T  # PERMUTE TO NEARLY DIAGONAL

        # D is a dense ndarray column vector
        D = np.diag(W_perm)
        D_col = D[:, np.newaxis]

        W_shifted = W_perm - W_perm.sum(axis=0) - W_perm.sum(axis=1) + D_col + D_col.T
        P_match = permutation_match(W_shifted)  # Solve on the preconditioned matrix
        P = P_match @ P0  # UN-PERMUTE to get the final result

    log("permutation_match done")
    return P
