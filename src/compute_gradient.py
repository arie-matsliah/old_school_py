import numpy as np
from scipy.sparse import find

def compute_gradient(P, A, B):
    """
    Computes dJ/dP where P is a sparse doubly stochastic matrix.
    J(P) = sum_ijkl min(A_ij, B_kl) P_ik P_jl
    (dJ/dP)_jl = sum_ik [min(A_ij, B_kl) + min(A_ji, B_lk)] P_ik
    
    The MATLAB code implements a different formula which seems to be
    dJ/dP_ik = sum_jl [min(A_ij, B_kl) + min(A_ji, B_lk)] P_jl
    Let's stick to the paper's formula for the gradient:
    [gradS(P)]_jl = sum_ik [min(A_ij, B_kl) + min(A_ji, B_lk)] P_ik
    
    The MATLAB code seems to compute:
    G_ia,jb += val(k) * min(a, b')
    where P(row(k), col(k)) = val(k)
    This is tricky. Let's re-implement based on the paper's math.
    G_jl = sum_ik (min(A_ij, B_kl) + min(A_ji, B_lk)) * P_ik
    
    Let's re-implement the MATLAB logic directly first.
    It seems to calculate G(ia, jb) += P(row(k), col(k)) * min(A(ia, row(k)), B(jb, col(k)))
    This is essentially (A.T @ P @ B) + (A @ P @ B.T)
    """
    # This is the direct, but potentially slow, translation from the paper's formula
    # G = A.T @ P @ B + A @ P @ B.T
    
    # Direct translation of the MATLAB code's efficient loop
    n = P.shape[0]
    G = np.zeros((n, n))
    rows, cols, vals = find(P)
    At = A.T.tocsr()
    Bt = B.T.tocsr()

    for r, c, v in zip(rows, cols, vals):
        # Term 1: A.T @ P @ B
        # For a given P(r,c)=v, we add contributions to G
        # from A(:,r) and B(:,c)
        a_rows, _, a_vals = find(A[:, r])
        b_rows, _, b_vals = find(B[:, c])
        
        # Create outer product of minimums
        if a_rows.size > 0 and b_rows.size > 0:
            min_matrix = np.minimum(a_vals[:, np.newaxis], b_vals)
            G[np.ix_(a_rows, b_rows)] += v * min_matrix
            
        # Term 2: A @ P @ B.T
        at_rows, _, at_vals = find(At[:, r])
        bt_rows, _, bt_vals = find(Bt[:, c])

        if at_rows.size > 0 and bt_rows.size > 0:
             min_matrix_t = np.minimum(at_vals[:, np.newaxis], bt_vals)
             G[np.ix_(at_rows, bt_rows)] += v * min_matrix_t
             
    return G
