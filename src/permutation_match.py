import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment

def permutation_match(W, P0=None):
    """
    Computes a permutation matrix P that maximizes sum(P * W).
    This is a linear assignment problem.
    """
    if P0 is None:
        # The linear_sum_assignment function finds a minimum weight matching.
        # To find a maximum, we negate the weight matrix.
        row_ind, col_ind = linear_sum_assignment(-W)
        P = csr_matrix((np.ones_like(row_ind), (row_ind, col_ind)), shape=W.shape)
        return P
    else:
        # Preconditioning heuristic from the paper
        W_perm = W @ P0.T
        D = W_perm.diagonal()
        
        # The SHIFT step
        # W_shifted = W_perm - sum(W_perm, axis=0) - sum(W_perm, axis=1).T + D + D.T
        # This seems complex to translate directly. Let's stick to the paper's formula
        # Ω = Λ + diag(Λ)1^T + 1diag(Λ)^T - 11^T Λ - Λ 11^T
        # This is not what the MATLAB code does. The MATLAB code is simpler:
        # W = W - sum(W,1) - sum(W,2) + D + D'
        W_shifted = W_perm - W_perm.sum(axis=0) - W_perm.sum(axis=1) + D[:, np.newaxis] + D
        
        P_match = permutation_match(W_shifted)
        P = P_match @ P0
        return P

