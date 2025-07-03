import time
import numpy as np
from compute_gradient import compute_gradient
from permutation_match import permutation_match
from scipy.sparse import csr_matrix


def frank_wolfe_update(P0, G0, Pm, A, B):
    """Performs a single Frank-Wolfe update step."""
    # Project gradient to permutation matrix using the preconditioner
    P1 = permutation_match(G0, Pm)

    # If the projection doesn't change, we're done
    if (P0.toarray() == P1.toarray()).all():
        return P0

    # Compute step size
    G1 = compute_gradient(P1, A, B)

    P0_dense = P0.toarray()
    P1_dense = P1.toarray()

    numer = np.sum((G1 - G0) * P0_dense + (P1_dense - P0_dense) * G0)
    denom = np.sum((G1 - G0) * (P1_dense - P0_dense))

    if abs(denom) < 1e-9:  # Avoid division by zero
        step = 1.0
    else:
        step = -0.5 * (numer / denom)

    # Stay within simplex, setting to 1 if outside [0,1] per the paper
    if not (0 <= step <= 1):
        step = 1.0

    # Interpolate and return sparse matrix
    Ps = P0 + step * (P1 - P0)
    return Ps


def do_frank_wolfe(Ps, A, B, num_updates):
    """
    Performs Frank-Wolfe updates on the sparse doubly stochastic matrix Ps.
    """
    t_start = time.time()
    print('\nFrank-Wolfe updates:')
    print('  iter    vertex   simplex   tMin')

    # Ensure Ps is sparse for the loop
    if isinstance(Ps, np.ndarray):
        Ps = csr_matrix(Ps)

    for i in range(1, num_updates + 1):
        # Pm is the closest permutation matrix to the current doubly-stochastic Ps
        Pm = permutation_match(Ps.toarray())
        Gs = compute_gradient(Ps, A, B)

        scorePm = np.minimum(A.toarray(), (Pm @ B @ Pm.T).toarray()).sum()
        scorePs = 0.5 * np.sum(Gs * Ps.toarray())

        t_min = (time.time() - t_start) / 60
        print(f'    {i:02d}   {int(scorePm):07d}   {int(scorePs):07d}   {t_min:04.1f}')

        if i < num_updates:
            Ps = frank_wolfe_update(Ps, Gs, Pm, A, B)

    return Ps
