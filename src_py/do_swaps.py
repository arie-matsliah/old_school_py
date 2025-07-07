import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from .compute_gradient import compute_gradient
from .util import log, dbg


# Computes S_ij, the gain in score from swapping nodes i and j.
# P is expected in LIL format for efficient diagonal access if P were dense,
# but for sparse P, conversion to CSR for products is better.
def compute_swap_gains(P_lil, A_orig, B_orig):
    # Convert P to CSR for efficient products and gradient computation
    P_csr = P_lil.tocsr()

    # LINEAR TERM
    # compute_gradient benefits from P_csr if its input P is CSR/CSC
    G = compute_gradient(P_csr, A_orig, B_orig) # G is dense

    # P_lil.nonzero() is efficient for LIL to get indices for sumGP2
    # However, G is dense. P_csr.nonzero() is also fine.
    # Using P_csr as it's already computed and consistent.
    P_csr_rows, P_csr_cols = P_csr.nonzero()
    sumGP2 = G[P_csr_rows, P_csr_cols] # Direct indexing into dense G

    sumGP2_full = np.zeros(P_csr.shape[0])
    # np.add.at needs row indices to sum up contributions for each row of P
    # If P is a permutation matrix, P_csr_rows will be unique (0 to n-1 permuted).
    # If P is not strictly permutation (e.g. doubly stochastic), then P_csr_rows might have duplicates.
    # Assuming P is close to permutation, P_csr_rows are the row indices.
    np.add.at(sumGP2_full, P_csr_rows, sumGP2)


    # G is dense, P_csr.T is sparse CSR (transpose of CSR is CSC, but P_csr.T operation handles it)
    L = G @ P_csr.T + P_csr @ G.T # Sparse @ Dense and Dense @ Sparse products
    L = L - sumGP2_full[:, np.newaxis] - sumGP2_full

    # QUADRATIC TERM
    # Use P_csr for products
    Bp = P_csr @ B_orig @ P_csr.T # Bp is sparse

    dA = A_orig.diagonal() # dA is dense
    dB = Bp.diagonal()   # dB is dense (diagonal of a sparse matrix)

    # These .toarray() calls are potential bottlenecks if A_orig or Bp are large.
    # Keeping them for now to maintain algorithm semantics, as optimizing custom sparse minimums
    # without changing the formula's meaning is complex.
    A_dense = A_orig.toarray()
    Bp_dense = Bp.toarray()
    Bp_T_dense = Bp.T.toarray() # Or Bp_dense.T

    term1 = np.minimum(dA[:, np.newaxis], dB) + np.minimum(dA[:, np.newaxis], dB.T)
    term2 = np.minimum(dA[:, np.newaxis], Bp_dense) + np.minimum(dA[:, np.newaxis], Bp_T_dense)
    Q_diag_part = term1 - term2

    term3 = np.minimum(A_dense, Bp_dense) + np.minimum(A_dense, Bp_T_dense)
    term4 = np.minimum(A_dense, dB) + np.minimum(A_dense, dB.T) # A_dense min dB (vector)
    Q_offdiag_part = term3 - term4

    Q = Q_diag_part + Q_offdiag_part

    # COMBINE TERMS
    return L + Q + Q.T # Returns a dense matrix S


# Tests all swaps (i<->j) for which S_ij > 0 and performs them if they result in a gain.
def swap_check(P_lil, S, A_orig, B_orig): # P_lil is in LIL format, S is dense
    A_dense = A_orig.toarray() # Densify A once

    P_csr = P_lil.tocsr()
    M_csr = P_csr @ B_orig @ P_csr.T
    scoreP = np.minimum(A_dense, M_csr.toarray()).sum() # M_csr.toarray() still needed for now

    num_swap = 0
    if S.max() <= 0: # S is dense gain matrix
        return P_lil, scoreP, num_swap

    np.fill_diagonal(S, -np.inf)  # Exclude diagonal swaps (no gain from i<->i)

    # Iterate while there's a potential positive gain estimated by S
    while S.max() > 0:
        idx = np.unravel_index(np.argmax(S, axis=None), S.shape)
        i, j = idx

        # Create the swapped permutation matrix (LIL format is efficient for this)
        P_swapped_lil = P_lil.copy()
        P_swapped_lil[[i, j], :] = P_swapped_lil[[j, i], :] # Swap rows i and j

        # Convert to CSR for efficient matrix products
        P_swapped_csr = P_swapped_lil.tocsr()
        M_swapped_csr = P_swapped_csr @ B_orig @ P_swapped_csr.T

        # Recalculate score with the swapped matrix. .toarray() is a known bottleneck.
        current_swap_score = np.minimum(A_dense, M_swapped_csr.toarray()).sum()

        if current_swap_score > scoreP:
            dbg(f"Swapped {i} and {j}: {scoreP} -> {current_swap_score}")
            P_lil = P_swapped_lil  # Accept the swap, update P_lil for next iteration
            scoreP = current_swap_score
            num_swap += 1

        # Mark this swap (and its symmetric counterpart) as tested by setting gain to -inf
        S[i, j] = -np.inf
        S[j, i] = -np.inf

    return P_lil, scoreP, num_swap


def do_swaps(P_initial, A_orig, B_orig, max_iter):
    # Ensure P starts in LIL format as swap_check and compute_swap_gains expect it
    if not isinstance(P_initial, lil_matrix):
        P_lil = lil_matrix(P_initial)
    else:
        P_lil = P_initial

    log('iter     score    swaps')
    for i in range(max_iter):
        # compute_swap_gains expects P_lil, A_orig, B_orig (A, B are original sparse matrices)
        S = compute_swap_gains(P_lil, A_orig, B_orig) # S is dense

        # swap_check expects P_lil, S (dense), A_orig, B_orig
        P_lil, scoreP, num_swap = swap_check(P_lil, S, A_orig, B_orig)

        log(f'    {i + 1:02d}   {int(scoreP):07d}    {num_swap:5d}')
        if num_swap == 0: # No improvement found in this iteration
            break

    return P_lil.tocsr() # Convert final result to CSR
