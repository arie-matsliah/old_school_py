import time
import numpy as np
from scipy.sparse import find
from compute_gradient import compute_gradient

def compute_swap_gains(P, A, B):
    """
    Computes S_ij, the gain in score from swapping nodes i and j
    before applying the permutation P.
    """
    # LINEAR TERM
    G = compute_gradient(P, A, B)
    # In MATLAB, .* is element-wise multiplication
    sumGP2 = G[P.nonzero()] 
    sumGP2_full = np.zeros(P.shape[0])
    np.add.at(sumGP2_full, P.nonzero()[0], sumGP2)
    
    # G*P' + P*G'
    L = G @ P.T + P @ G.T
    # -sumGP2-sumGP2'
    L = L - sumGP2_full[:, np.newaxis] - sumGP2_full
    
    # QUADRATIC TERM
    Bp = P @ B @ P.T
    dA = A.diagonal()
    dB = Bp.diagonal()
    
    # This part is tricky to translate directly due to broadcasting rules
    # Let's break it down
    # min(dA,dB) + min(dA,dB') - min(dA,Bp) - min(dA,Bp')
    term1 = np.minimum(dA[:, np.newaxis], dB) + np.minimum(dA[:, np.newaxis], dB.T)
    term2 = np.minimum(dA[:, np.newaxis], Bp.toarray()) + np.minimum(dA[:, np.newaxis], Bp.T.toarray())
    Q_diag_part = term1 - term2
    
    # min(A,Bp) + min(A,Bp') - min(A,dB) - min(A,dB')
    term3 = np.minimum(A.toarray(), Bp.toarray()) + np.minimum(A.toarray(), Bp.T.toarray())
    term4 = np.minimum(A.toarray(), dB) + np.minimum(A.toarray(), dB.T)
    Q_offdiag_part = term3 - term4
    
    Q = Q_diag_part + Q_offdiag_part
    
    # COMBINE TERMS
    S = L + Q + Q.T
    return S

def swap_check(P, S, A, B):
    """
    Tests all swaps (i<->j) for which S_ij > 0 and performs them if they result in a gain.
    """
    scoreP = np.minimum(A.toarray(), (P @ B @ P.T).toarray()).sum()
    num_swap = 0
    n = S.shape[0]

    if S.max() <= 0:
        return P, scoreP, num_swap

    # Exclude diagonal swaps
    np.fill_diagonal(S, -np.inf)

    # Test swaps in order of potential gain
    while S.max() > 0:
        idx = np.unravel_index(np.argmax(S, axis=None), S.shape)
        i, j = idx
        
        # Perform swap
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
    """
    Iteratively computes gains from pairwise swaps and performs them.
    """
    t_start = time.time()
    scoreP = np.minimum(A.toarray(), (P @ B @ P.T).toarray()).sum()
    print('\nPairwise swaps:')
    print('  iter     score    swaps   tMin')

    for i in range(max_iter):
        print(f'    {i+1:02d}   {int(scoreP):07d}    ', end='')
        S = compute_swap_gains(P, A, B)
        P, scoreP, num_swap = swap_check(P, S, A, B)
        t_min = (time.time() - t_start) / 60
        print(f'{num_swap:5d}   {t_min:04.1f}')
        if num_swap == 0:
            break
            
    return P

