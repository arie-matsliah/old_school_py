import numpy as np
from scipy.sparse import find, csc_matrix, csr_matrix

from .util import dbg # Assuming dbg is available for debugging


def compute_gradient(P, A, B):
    """
    Computes the gradient G for the matrix matching problem.
    Optimized version using efficient sparse slicing and vectorized operations.
    """
    n = P.shape[0]
    G = np.zeros((n, n))

    # Ensure A and B are in desired formats for efficient slicing
    # Convert only if not already in the correct sparse format
    A_csc = A.tocsc() if not isinstance(A, csc_matrix) else A
    B_csc = B.tocsc() if not isinstance(B, csc_matrix) else B
    A_csr = A.tocsr() if not isinstance(A, csr_matrix) else A
    B_csr = B.tocsr() if not isinstance(B, csr_matrix) else B

    P_rows, P_cols, P_vals = find(P)

    for p_r, p_c, p_val in zip(P_rows, P_cols, P_vals):
        # p_val is P[p_r, p_c]. If P is a permutation matrix, p_val is 1.
        if p_val == 0: # Optimization: skip if P_val is zero
            continue

        # Term 1: Contribution from A[:, p_r] and B[:, p_c]
        # A_col_pr_sparse is the sparse column vector A[:, p_r]
        # B_col_pc_sparse is the sparse column vector B[:, p_c]
        A_col_pr_sparse = A_csc[:, p_r]
        B_col_pc_sparse = B_csc[:, p_c]

        if A_col_pr_sparse.nnz > 0 and B_col_pc_sparse.nnz > 0:
            # For A_col_pr_sparse (a column vector slice from CSC):
            # find returns (actual_row_indices_in_A, column_indices_in_slice (all 0), values_a)
            row_indices_A, _, values_a = find(A_col_pr_sparse)

            # For B_col_pc_sparse (a column vector slice from CSC):
            # find returns (actual_row_indices_in_B, column_indices_in_slice (all 0), values_b)
            row_indices_B, _, values_b = find(B_col_pc_sparse)

            # Broadcast to compute outer minimum:
            # values_a becomes (len(values_a), 1)
            # values_b becomes (1, len(values_b))
            # outer_min_vals becomes (len(values_a), len(values_b))
            outer_min_vals = np.minimum(values_a[:, np.newaxis], values_b[np.newaxis, :])

            # Add to G using advanced indexing. np.ix_ creates meshgrid-like indexers.
            # G[ia, jb] in MATLAB, where ia are row_indices_A and jb are row_indices_B
            G[np.ix_(row_indices_A, row_indices_B)] += p_val * outer_min_vals
            # dbg(f"Term 1 for (p_r,p_c)=({p_r},{p_c}): added to G at A rows {row_indices_A}, B rows {row_indices_B}")

        # Term 2: Contribution from A[p_r, :] and B[p_c, :]
        # MATLAB: G(ia,jb) = G(ia,jb) + val(k) * min(a,b'); where a = A(row(k),:)' and b = B(col(k),:)'
        # This means 'a' are values from row p_r of A (shaped as a column vector for outer min).
        # 'b' are values from row p_c of B (shaped as a column vector, then transposed for outer min).
        # The indices ia,jb for G correspond to the original column indices of A and B respectively.

        A_row_pr_sparse = A_csr[p_r, :] # This is a sparse row vector from CSR
        B_row_pc_sparse = B_csr[p_c, :] # This is a sparse row vector from CSR

        if A_row_pr_sparse.nnz > 0 and B_row_pc_sparse.nnz > 0:
            # For A_row_pr_sparse (a row vector slice from CSR):
            # find returns (row_indices_in_slice (all 0), actual_col_indices_in_A, values_at)
            _, col_indices_A, values_at = find(A_row_pr_sparse)

            # For B_row_pc_sparse (a row vector slice from CSR):
            # find returns (row_indices_in_slice (all 0), actual_col_indices_in_B, values_bt)
            _, col_indices_B, values_bt = find(B_row_pc_sparse)

            # Broadcast to compute outer minimum:
            # values_at becomes (len(values_at), 1)
            # values_bt becomes (1, len(values_bt))
            # outer_min_vals_T becomes (len(values_at), len(values_bt))
            outer_min_vals_T = np.minimum(values_at[:, np.newaxis], values_bt[np.newaxis, :])

            # Add to G. Indices for G are col_indices_A (as rows of G) and col_indices_B (as columns of G).
            G[np.ix_(col_indices_A, col_indices_B)] += p_val * outer_min_vals_T
            # dbg(f"Term 2 for (p_r,p_c)=({p_r},{p_c}): added to G at A cols {col_indices_A} (as G_rows), B cols {col_indices_B} (as G_cols)")

    return G