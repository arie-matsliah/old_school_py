import numpy as np

from src.util import log


def dense_mult_update(P, G, u, v):
    """Performs the inner multiplicative update loop."""
    numer = P * G
    if u is None or v is None:
        u = 0.5 * np.sum(numer, axis=1)
        v = 0.5 * np.sum(numer, axis=0)
    
    max_iter1 = 200
    max_iter2 = 250

    for _ in range(max_iter1):
        P = numer / (u[:, np.newaxis] + v)
        u = u * np.sum(P, axis=1)
        v = v * np.sum(P, axis=0)
        shift = 0.5 * (np.min(u) - np.min(v))
        v = np.maximum(np.finfo(float).eps, v + shift)
        u = np.maximum(np.finfo(float).eps, u - shift)

    for _ in range(max_iter1, max_iter2):
        P = numer / (u[:, np.newaxis] + v)
        u = u * np.maximum(1, np.sum(P, axis=1))
        v = v * np.maximum(1, np.sum(P, axis=0))
        
    return P, u, v


def do_mult_updates(P, A, B, max_iter):
     # Convert sparse to dense for this part
    A_dense = A.toarray()
    B_dense = B.toarray()

    maxA = A_dense[A_dense > 0].max()
    maxB = B_dense[B_dense > 0].max()
    minMax = min(maxA, maxB)

    A_dense = np.minimum(A_dense, minMax)
    B_dense = np.minimum(B_dense, minMax)
    
    sqrtA = np.sqrt(A_dense)
    sqrtB = np.sqrt(B_dense)
    u, v = None, None

    log('\nDense multiplicative updates on bounded objective:\n  iter   midPoint   lowerBnd   upperBnd   tMin')

    for i in range(1, max_iter + 1):
        # Gradients
        Gl = (A_dense.T @ P @ B_dense + A_dense @ P @ B_dense.T) / minMax
        Gu = (sqrtA.T @ P @ sqrtB + sqrtA @ P @ sqrtB.T)
        Gm = 0.5 * (Gl + Gu)
        
        # Bounds
        lower_bound = 0.5 * np.sum(P * Gl)
        upper_bound = 0.5 * np.sum(P * Gu)
        mid_point = 0.5 * (lower_bound + upper_bound)

        log(f'    {i:02d}    {mid_point:07.0f}    {lower_bound:07.0f}    {upper_bound:07.0f}')
        
        # Update
        P, u, v = dense_mult_update(P, Gm, u, v)
        
    return P

