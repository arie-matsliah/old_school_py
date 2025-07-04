import unittest

import numpy as np
from scipy.sparse import csr_matrix
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


class TestPermutationMatch(unittest.TestCase):

    def setUp(self):
        # A weight matrix where the minimum cost is clearly the anti-diagonal
        self.W = np.array([
            [100, 10, 1],
            [10, 1, 100],
            [1, 100, 10]
        ])
        # The optimal permutation P should pick the 1s.
        # This corresponds to the assignment: 0->2, 1->1, 2->0 (total cost 3)
        optimal_rows = np.array([0, 1, 2])
        optimal_cols = np.array([2, 1, 0])
        optimal_data = np.ones(3)
        self.P_optimal = csr_matrix((optimal_data, (optimal_rows, optimal_cols)), shape=(3, 3))

    def assertSparseMatrixEqual(self, p1, p2, msg=None):
        self.assertEqual((p1 != p2).nnz, 0, msg)

    def test_no_initial_guess(self):
        P_result = permutation_match(self.W)
        self.assertSparseMatrixEqual(P_result, self.P_optimal,
                                     "Failed to find the correct permutation without P0.")

    def test_with_bad_initial_guess(self):
        P0_guess = csr_matrix(np.identity(3))
        P_result = permutation_match(self.W, P0=P0_guess)
        self.assertSparseMatrixEqual(P_result, self.P_optimal,
                                     "Failed to find the correct permutation when using bad P0.")

    def test_with_good_initial_guess(self):
        P0_guess = self.P_optimal
        P_result = permutation_match(self.W, P0=P0_guess)
        self.assertSparseMatrixEqual(P_result, self.P_optimal,
                                     "Failed to find the correct permutation when using good P0.")

    def test_diagonal_case(self):
        W_diag = np.array([
            [1, 10, 20],
            [10, 2, 30],
            [20, 30, 3]
        ])
        P_identity = csr_matrix(np.identity(3))
        P_result = permutation_match(W_diag)
        self.assertSparseMatrixEqual(P_result, P_identity,
                                     "Failed on a simple diagonal-dominant case.")
