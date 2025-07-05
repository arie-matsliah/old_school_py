import unittest
import numpy as np
from scipy.sparse import csr_matrix, find

from src_py.compute_gradient import compute_gradient
from src_py.do_frank_wolfe import frank_wolfe_update, do_frank_wolfe
from src_py.permutation_match import permutation_match


class TestPermutationMatch(unittest.TestCase):

    def setUp(self):
        # A weight matrix where the maximum cost is clearly the anti-diagonal
        self.W = np.array([
            [100, 10, 200],
            [10, 500, 100],
            [1000, 100, 10]
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
            [100, 10, 20],
            [10, 200, 30],
            [20, 30, 300]
        ])
        P_identity = csr_matrix(np.identity(3))
        P_result = permutation_match(W_diag)
        self.assertSparseMatrixEqual(P_result, P_identity,
                                     "Failed on a simple diagonal-dominant case.")


class TestComputeGradient(unittest.TestCase):
    def setUp(self):
        """Set up a simple, non-trivial test case for the gradient."""
        # A is a simple sparse matrix
        self.A = csr_matrix(np.array([
            [0, 5, 0],
            [5, 0, 1],
            [0, 1, 0]
        ]))
        # B is another simple sparse matrix
        self.B = csr_matrix(np.array([
            [0, 2, 8],
            [2, 0, 0],
            [8, 0, 0]
        ]))
        # P is a permutation matrix (doubly-stochastic)
        self.P = csr_matrix(np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ]))

    def test_gradient_calculation(self):
        """
        Tests the optimized gradient function against a direct, unoptimized
        (but clear) implementation of the same formula.
        """
        # Use the provided optimized function
        G_optimized = compute_gradient(self.P, self.A, self.B)

        # Ground truth calculation: G = A.T @ P @ B + A @ P @ B.T
        # This is a less efficient but direct way to compute the same value.
        P_dense, A_dense, B_dense = self.P.toarray(), self.A.toarray(), self.B.toarray()

        # The formula in the docstring involves np.minimum, which is different
        # from the simple matrix multiplication. Let's implement the formula
        # from the docstring directly as the ground truth.
        n = self.P.shape[0]
        G_ground_truth = np.zeros((n, n))

        rows, cols, vals = find(self.P)
        for r, c, v in zip(rows, cols, vals):  # r=i, c=k, v=P_ik
            for j in range(n):
                for l in range(n):
                    term1 = np.minimum(A_dense[r, j], B_dense[c, l])
                    term2 = np.minimum(A_dense[j, r], B_dense[l, c])
                    G_ground_truth[j, l] += (term1 + term2) * v

        # Compare the optimized result with the ground truth
        np.testing.assert_allclose(G_optimized, G_ground_truth,
                                   err_msg="Optimized gradient does not match ground truth.")


class TestFrankWolfe(unittest.TestCase):
    def setUp(self):
        # The optimal alignment is clearly the identity matrix.
        self.A_ident = csr_matrix(np.identity(3))
        self.B_ident = csr_matrix(np.identity(3))

        # The optimal alignment is the anti-diagonal permutation.
        self.A_mix = csr_matrix(np.identity(3))
        self.B_mix = csr_matrix(np.fliplr(np.identity(3)))

        # An initial doubly-stochastic matrix (uniform probabilities)
        self.Ps_initial = csr_matrix(np.full((3, 3), 1 / 3))

    def test_frank_wolfe_update_moves_towards_target(self):
        P0 = self.Ps_initial
        # Let's assume the gradient suggests P1 is the identity matrix
        G0 = -np.identity(3)  # Gradient pushes towards maximizing diagonal
        Pm = permutation_match(-P0.toarray())

        P_updated = frank_wolfe_update(P0, G0, Pm, self.A_ident, self.B_ident)

        # The updated matrix should not be the same as the start
        self.assertNotEqual((P_updated - P0).nnz, 0, "Update step did not change the matrix.")

        # The diagonal elements should have increased
        self.assertTrue(P_updated.diagonal().sum() > P0.diagonal().sum(),
                        "Update step did not move towards the diagonal.")

    def test_do_frank_wolfe_convergence_simple(self):
        """
        Tests that the full FW algorithm converges towards the identity
        matrix when A and B are identity matrices.
        """
        Ps_final = do_frank_wolfe(self.Ps_initial, self.A_ident, self.B_ident, num_updates=5)

        # After several updates, the matrix should be more diagonal-dominant
        # than the initial uniform matrix.
        initial_diag_sum = self.Ps_initial.diagonal().sum()
        final_diag_sum = Ps_final.diagonal().sum()

        self.assertTrue(final_diag_sum > initial_diag_sum,
                        "FW did not converge towards a more diagonal matrix.")

        # Check if row sums are still close to 1 (property of doubly-stochastic)
        row_sums = Ps_final.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones((3, 1)), atol=1e-7,
                                   err_msg="Matrix is not row-stochastic after updates.")

    def test_do_frank_wolfe_convergence_mixed(self):
        """
        Tests that FW converges towards an anti-diagonal matrix
        when A is identity and B is anti-diagonal.
        """
        Ps_final = do_frank_wolfe(self.Ps_initial, self.A_mix, self.B_mix, num_updates=10)

        # The anti-diagonal sum should be greater than the diagonal sum
        diag_sum = Ps_final.diagonal().sum()
        anti_diag_sum = np.fliplr(Ps_final.toarray()).diagonal().sum()
        self.assertTrue(anti_diag_sum > diag_sum,
                        "FW did not converge towards an anti-diagonal solution.")
