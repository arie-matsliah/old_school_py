from collections import defaultdict

import numpy as np
from scipy.sparse import find

from src_py.util import dbg


def compute_gradient(P, A, B):
    """
    Translates the MATLAB gradient computation to Python.

    Args:
        P (scipy.sparse.spmatrix): A sparse permutation matrix.
        A (scipy.sparse.spmatrix): A sparse matrix.
        B (scipy.sparse.spmatrix): A sparse matrix.

    Returns:
        numpy.ndarray: The dense gradient matrix G.
    """
    n = P.shape[0]
    G = np.zeros((n, n))

    # Convert matrices to COO format for efficient iteration over non-zero elements.
    # This is equivalent to MATLAB's `find()` but gives 0-indexed results.
    P_coo = P.tocoo()
    A_coo = A.tocoo()
    B_coo = B.tocoo()

    # Pre-cache column data to avoid slow slicing inside the loop.
    # This is an optimization over the direct MATLAB translation.
    A_cols = {}
    for r, c, v in zip(A_coo.row, A_coo.col, A_coo.data):
        A_cols.setdefault(c, []).append((r, v))

    B_cols = {}
    for r, c, v in zip(B_coo.row, B_coo.col, B_coo.data):
        B_cols.setdefault(c, []).append((r, v))

    # Pre-cache transposed data as well.
    At_cols = {}
    for r, c, v in zip(A_coo.row, A_coo.col, A_coo.data):
        At_cols.setdefault(r, []).append((c, v))  # Swap row and col for transpose

    Bt_cols = {}
    for r, c, v in zip(B_coo.row, B_coo.col, B_coo.data):
        Bt_cols.setdefault(r, []).append((c, v))  # Swap row and col for transpose

    # Iterate over the non-zero entries of the permutation matrix P.
    for p_r, p_c, p_v in zip(P_coo.row, P_coo.col, P_coo.data):
        # This loop corresponds to `for k=1:length(row)` in MATLAB.

        # --- First Term: Corresponds to A and B ---
        # Get all non-zero entries from column p_r of A and column p_c of B.
        a_entries = A_cols.get(p_r, [])
        b_entries = B_cols.get(p_c, [])

        if a_entries and b_entries:
            # ia, a = find(A(:,row(k)))
            ia, a = zip(*a_entries)
            # jb, b = find(B(:,col(k)))
            jb, b = zip(*b_entries)

            # Reshape for broadcasting to mimic MATLAB's min(a, b').
            a_col = np.array(a)[:, np.newaxis]
            b_row = np.array(b)

            # G(ia,jb) += val(k) * min(a,b')
            # np.ix_ is the NumPy equivalent for indexing with two lists.
            G[np.ix_(ia, jb)] += p_v * np.minimum(a_col, b_row)

        # --- Second Term: Corresponds to A' and B' ---
        # Get all non-zero entries from column p_r of A' and column p_c of B'.
        at_entries = At_cols.get(p_r, [])
        bt_entries = Bt_cols.get(p_c, [])

        if at_entries and bt_entries:
            ia_t, a_t = zip(*at_entries)
            jb_t, b_t = zip(*bt_entries)

            a_t_col = np.array(a_t)[:, np.newaxis]
            b_t_row = np.array(b_t)

            G[np.ix_(ia_t, jb_t)] += p_v * np.minimum(a_t_col, b_t_row)

    return G