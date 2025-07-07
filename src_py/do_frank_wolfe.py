import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from .compute_gradient import compute_gradient
from .permutation_match import permutation_match
from .util import log, dbg


def frank_wolfe_update(P0, G0, Pm, A, B): # P0 is sparse, G0 is dense
    dbg("frank_wolfe_update [1]")
    P1 = permutation_match(G0, Pm) # G0 dense, Pm sparse. P1 sparse.
    dbg("frank_wolfe_update [2]")

    if (P0 != P1).nnz == 0: # P0, P1 are sparse
        dbg("frank_wolfe_update no changes.. return")
        return P0

    G1 = compute_gradient(P1, A, B) # P1 sparse. G1 dense.
    dbg("frank_wolfe_update [3]")

    # delta_P is sparse, delta_G is dense
    delta_P, delta_G = P1 - P0, G1 - G0
    dbg(f"frank_wolfe_update {delta_P.nnz=} {delta_G.shape=} [4]")

    # Optimized calculation for numer and denom
    # P0 and delta_P are sparse, G0 and delta_G are dense.
    # Avoid dense intermediate products like (delta_G * P0_dense).

    P0_coo = P0.tocoo() # Convert to COO for easy access to data and indices
    delta_P_coo = delta_P.tocoo()

    numer1 = np.sum(delta_G[P0_coo.row, P0_coo.col] * P0_coo.data)
    numer2 = np.sum(G0[delta_P_coo.row, delta_P_coo.col] * delta_P_coo.data)
    numer = numer1 + numer2

    denom = np.sum(delta_G[delta_P_coo.row, delta_P_coo.col] * delta_P_coo.data)

    step = -0.5 * (numer / denom) if abs(denom) >= 1e-9 else 1.0
    if step > 1 or step < 0: # Ensure step is in [0, 1]
        step = 1.0

    # dbg(f"frank_wolfe_update step: {step} numer: {numer} denom: {denom}")

    dbg("frank_wolfe_update [5]")
    # Ps remains sparse as P0 and delta_P are sparse
    Ps_updated = P0 + delta_P * step
    return Ps_updated


# Performs Frank-Wolfe updates on the sparse doubly stochastic matrix Ps.
def do_frank_wolfe(Ps, A, B, num_updates):
    log('\nFrank-Wolfe updates:\n  iter    vertex   simplex   tMin')
    if isinstance(Ps, np.ndarray): # If Ps is dense initially, convert to sparse
        Ps = csr_matrix(Ps)
    elif not isinstance(Ps, coo_matrix): # Ensure Ps is COO for efficient data access later if needed
        Ps = Ps.tocoo() # Or keep as CSR/CSC if direct data access pattern changes

    for i in range(1, num_updates + 1):
        # Pm is the closest permutation matrix to the current doubly-stochastic Ps
        # Ps is sparse (COO or CSR/CSC). permutation_match returns sparse.
        Pm = permutation_match(Ps)

        # scorePm calculation: This remains a potential bottleneck due to .toarray()
        # For now, leaving as is, as optimizing sparse element-wise minimum sum is complex.
        scorePm_val = np.minimum((A @ Pm).toarray(), (Pm @ B).toarray()).sum()

        # Gs is dense (gradient of Ps)
        Gs = compute_gradient(Ps, A, B)

        # Optimized scorePs calculation:
        # Ps is sparse (e.g., COO from previous step or initial conversion)
        # Gs is dense. Avoid Ps.toarray().
        if not isinstance(Ps, coo_matrix): # Ensure Ps is COO for row, col, data access
             Ps_coo = Ps.tocoo()
        else:
             Ps_coo = Ps

        scorePs_val = round(0.5 * np.sum(Gs[Ps_coo.row, Ps_coo.col] * Ps_coo.data))
        log(f'    {i:02d}   {int(scorePm_val):07d}   {int(scorePs_val):07d}')

        if i < num_updates:
            Ps = frank_wolfe_update(Ps, Gs, Pm, A, B) # Ps will be updated, potentially changing type if frank_wolfe_update doesn't preserve COO
            if not isinstance(Ps, coo_matrix): # Ensure COO for next iteration's scorePs
                Ps = Ps.tocoo()
        dbg(f'    {i:02d}   {int(scorePm_val):07d}   {int(scorePs_val):07d} updated')

    return Ps.tocsr() # Convert final result to CSR as commonly used format
