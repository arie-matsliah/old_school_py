import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment


def permutation_match(W, P0=None):
    """
    Computes a permutation matrix P that maximizes sum(P * W).
    This is a linear assignment problem, solved with scipy's optimized function.
    If an initial guess P0 is provided, it uses the preconditioning heuristic.
    """
    if P0 is None:
        # The linear_sum_assignment function finds a minimum cost matching.
        # To find a maximum value matching, we use the negative of the weight matrix.
        # The input W must be dense for this function.
        W_dense = W if isinstance(W, np.ndarray) else W.toarray()
        row_ind, col_ind = linear_sum_assignment(-W_dense)
        P = csr_matrix((np.ones_like(row_ind), (row_ind, col_ind)), shape=W.shape)
        return P
    else:
        # Preconditioning heuristic from the paper and MATLAB code
        W_perm = W @ P0.T  # PERMUTE TO NEARLY DIAGONAL
        D = W_perm.diagonal()[:, np.newaxis]  # Ensure D is a column vector

        # SHIFT step: W = W - sum(W,1) - sum(W,2) + D + D'
        # This makes the matrix more diagonally dominant to speed up the solver.
        W_shifted = W_perm - W_perm.sum(axis=0) - W_perm.sum(axis=1)[:, np.newaxis] + D + D.T

        P_match = permutation_match(W_shifted)  # Solve on the preconditioned matrix
        P = P_match @ P0  # UN-PERMUTE to get the final result
        return P
