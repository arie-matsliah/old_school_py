import numpy as np
from compute_gradient import compute_gradient
from src.util import log


# Computes S_ij, the gain in score from swapping nodes i and j, before applying the permutation P.
def compute_swap_gains(P, A, B):
    # LINEAR TERM
    G = compute_gradient(P, A, B)
    sumGP2 = G[P.nonzero()]
    sumGP2_full = np.zeros(P.shape[0])
    np.add.at(sumGP2_full, P.nonzero()[0], sumGP2)

    L = G @ P.T + P @ G.T
    L = L - sumGP2_full[:, np.newaxis] - sumGP2_full
    
    # QUADRATIC TERM
    Bp = P @ B @ P.T
    dA = A.diagonal()
    dB = Bp.diagonal()

    term1 = np.minimum(dA[:, np.newaxis], dB) + np.minimum(dA[:, np.newaxis], dB.T)
    term2 = np.minimum(dA[:, np.newaxis], Bp.toarray()) + np.minimum(dA[:, np.newaxis], Bp.T.toarray())
    Q_diag_part = term1 - term2

    term3 = np.minimum(A.toarray(), Bp.toarray()) + np.minimum(A.toarray(), Bp.T.toarray())
    term4 = np.minimum(A.toarray(), dB) + np.minimum(A.toarray(), dB.T)
    Q_offdiag_part = term3 - term4
    
    Q = Q_diag_part + Q_offdiag_part
    
    # COMBINE TERMS
    return L + Q + Q.T


# Tests all swaps (i<->j) for which S_ij > 0 and performs them if they result in a gain.
def swap_check(P, S, A, B):
    scoreP = np.minimum(A.toarray(), (P @ B @ P.T).toarray()).sum()
    num_swap = 0
    if S.max() <= 0:
        return P, scoreP, num_swap

    np.fill_diagonal(S, -np.inf)  # Exclude diagonal swaps
    while S.max() > 0:  # Test swaps in order of potential gain
        idx = np.unravel_index(np.argmax(S, axis=None), S.shape)
        i, j = idx
        P_swapped = P.copy()
        P_swapped[[i, j], :] = P_swapped[[j, i], :]
        swap_score = np.minimum(A.toarray(), (P_swapped @ B @ P_swapped.T).toarray()).sum()
        if swap_score > scoreP:
            P = P_swapped
            scoreP = swap_score
            num_swap += 1
        # Mark this swap as tested
        S[i, j] = -np.inf
        S[j, i] = -np.inf
        
    return P, scoreP, num_swap


def do_swaps(P, A, B, max_iter):
    scoreP = np.minimum(A.toarray(), (P @ B @ P.T).toarray()).sum()
    log('\nPairwise swaps:\n  iter     score    swaps   tMin')
    for i in range(max_iter):
        log(f'    {i+1:02d}   {int(scoreP):07d}    ')
        S = compute_swap_gains(P, A, B)
        P, scoreP, num_swap = swap_check(P, S, A, B)
        log(f'{num_swap:5d} swaps..')
        if num_swap == 0:
            break
    return P

