from src.util import log

import numpy as np
from collections import defaultdict
from scipy.sparse import find


def compute_gradient(P, A, B):
    """
    Computes the gradient G for the graph alignment score:
    G[j,l] = sum_ik [min(A[i,j], B[k,l]) + min(A[j,i], B[l,k])] * P[i,k]
    """

    log("compute_gradient: Preprocessing A and B...")

    # Convert A and B to COO format for fast row/col access
    A_coo = A.tocoo()
    B_coo = B.tocoo()

    # Precompute A[:, r] for all columns r
    A_cols = defaultdict(list)
    for i, j, v in zip(A_coo.row, A_coo.col, A_coo.data):
        A_cols[j].append((i, v))

    # Precompute B[:, c] for all columns c
    B_cols = defaultdict(list)
    for i, j, v in zip(B_coo.row, B_coo.col, B_coo.data):
        B_cols[j].append((i, v))

    # Same for transposes (A.T[:, r] and B.T[:, c])
    At_cols = defaultdict(list)
    for j, i, v in zip(A_coo.col, A_coo.row, A_coo.data):
        At_cols[j].append((i, v))

    Bt_cols = defaultdict(list)
    for j, i, v in zip(B_coo.col, B_coo.row, B_coo.data):
        Bt_cols[j].append((i, v))

    log("compute_gradient: Starting gradient computation...")

    n = P.shape[0]
    G = np.zeros((n, n))

    # Extract non-zero entries of P
    rows, cols, vals = find(P)

    for idx, (r, c, v) in enumerate(zip(rows, cols, vals)):
        if idx % 6000 == 1:
            log(f"compute_gradient: Processing entry {idx}/{len(rows)}")

        # A.T @ P @ B term
        a_entries = A_cols.get(r, [])
        b_entries = B_cols.get(c, [])

        if a_entries and b_entries:
            a_rows, a_vals = zip(*a_entries)
            b_rows, b_vals = zip(*b_entries)
            a_vals = np.array(a_vals)[:, np.newaxis]  # shape (m, 1)
            b_vals = np.array(b_vals)[np.newaxis, :]  # shape (1, k)
            min_matrix = np.minimum(a_vals, b_vals) * v
            G[np.ix_(a_rows, b_rows)] += min_matrix

        # A @ P @ B.T term
        at_entries = At_cols.get(r, [])
        bt_entries = Bt_cols.get(c, [])

        if at_entries and bt_entries:
            at_rows, at_vals = zip(*at_entries)
            bt_rows, bt_vals = zip(*bt_entries)
            at_vals = np.array(at_vals)[:, np.newaxis]
            bt_vals = np.array(bt_vals)[np.newaxis, :]
            min_matrix_t = np.minimum(at_vals, bt_vals) * v
            G[np.ix_(at_rows, bt_rows)] += min_matrix_t

    log("compute_gradient: Finished gradient computation.")

    return G

