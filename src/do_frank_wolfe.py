import time
import numpy as np
from compute_gradient import compute_gradient
from permutation_match import permutation_match

def frank_wolfe_update(P0, G0, Pm, A, B):
    """Performs a single Frank-Wolfe update step."""
    # Project gradient to permutation matrix
    P1 = permutation_match(G0, Pm)

    if np.all(P0.toarray() == P1.toarray()):
        return P0

    # Compute step size
    G1 = compute_gradient(P1, A, B)
    
    # numer = full(sum((G1-G0).*P0 + (P1-P0).*G0,'all'));
    numer = np.sum((G1 - G0) * P0.toarray() + (P1.toarray() - P0.toarray()) * G0)
    
    # denom = full(sum((G1-G0).*(P1-P0),'all'));
    denom = np.sum((G1 - G0) * (P1.toarray() - P0.toarray()))
    
    if denom == 0:
        step = 1.0
    else:
        step = -0.5 * (numer / denom)

    # Stay within simplex
    step = min(max(step, 0), 1)

    # Interpolate
    Ps = P0 + step * (P1 - P0)
    return Ps

def do_frank_wolfe(Ps, A, B, num_updates):
    """
    Performs Frank-Wolfe updates on the sparse doubly stochastic matrix Ps.
    """
    t_start = time.time()
    print('\nFrank-Wolfe updates:')
    print('  iter    vertex   simplex   tMin')
    
    for i in range(1, num_updates + 1):
        Pm = permutation_match(Ps.toarray())
        Gs = compute_gradient(Ps, A, B)
        
        scorePm = np.minimum(A.toarray(), (Pm @ B @ Pm.T).toarray()).sum()
        scorePs = 0.5 * np.sum(Gs * Ps.toarray())
        
        t_min = (time.time() - t_start) / 60
        print(f'    {i:02d}   {int(scorePm):07d}   {int(scorePs):07d}   {t_min:04.1f}')
        
        if i < num_updates:
            Ps = frank_wolfe_update(Ps, Gs, Pm, A, B)
            
    return Ps

