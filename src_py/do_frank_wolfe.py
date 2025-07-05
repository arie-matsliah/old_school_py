import numpy as np
from src_py.permutation_match import permutation_match
from scipy.sparse import csr_matrix

from src_py.compute_gradient import compute_gradient
from src_py.util import log


def frank_wolfe_update(P0, G0, Pm, A, B):
    log("frank_wolfe_update [1]")
    P1 = permutation_match(G0, Pm)
    log("frank_wolfe_update [2]")

    if (P0 != P1).nnz == 0:
        log("frank_wolfe_update no changes.. return")
        return P0

    G1 = compute_gradient(P1, A, B)
    log("frank_wolfe_update [3]")

    delta_P, delta_G = P1 - P0, G1 - G0
    log(f"frank_wolfe_update {delta_P=} {delta_G=} [4]")

    numer = (delta_G * P0).sum() + (G0 * delta_P).sum()
    denom = (delta_G * delta_P).sum()

    step = -0.5 * (numer / denom) if abs(denom) >= 1e-9 else 1.0
    if step > 1 or step < 0:
        step = 1.0

    log("frank_wolfe_update [5]")
    Ps = P0 + delta_P * step
    return Ps


# Performs Frank-Wolfe updates on the sparse doubly stochastic matrix Ps.
def do_frank_wolfe(Ps, A, B, num_updates):
    log('\nFrank-Wolfe updates:\n  iter    vertex   simplex   tMin')
    if isinstance(Ps, np.ndarray):
        Ps = csr_matrix(Ps)

    for i in range(1, num_updates + 1):
        # Pm is the closest permutation matrix to the current doubly-stochastic Ps
        Pm = permutation_match(Ps)
        scorePm = np.minimum((A @ Pm).toarray(), (Pm @ B).toarray()).sum()

        Gs = compute_gradient(Ps, A, B)
        scorePs = round(0.5 * np.sum(Gs * Ps.toarray()))
        log(f'    {i:02d}   {int(scorePm):07d}   {int(scorePs):07d} updating..')

        if i < num_updates:
            Ps = frank_wolfe_update(Ps, Gs, Pm, A, B)
        log(f'    {i:02d}   {int(scorePm):07d}   {int(scorePs):07d} done')

    return Ps
