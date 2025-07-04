import time
import numpy as np
from do_mult_updates import do_mult_updates
from do_frank_wolfe import do_frank_wolfe
from do_swaps import do_swaps
from permutation_match import permutation_match
from src.util import read_connectome, read_solution, save_solution


def main():
    """
    Main script to run the FlyWire VNC Matching Challenge solution.
    """
    # --- Configuration ---
    t_start = time.time()
    do_cold_start = False  # Set to True to initialize from barycenter
    num_mult_updates = 25
    num_frank_wolfe = 15
    num_swap_checks = 20
    
    # --- Read in Connectomes ---
    print('Loading connectomes ... ', end='', flush=True)
    A = read_connectome('../data/male_connectome_graph.csv')
    B = read_connectome('../data/female_connectome_graph.csv')
    print(f'{time.time() - t_start:.1f} sec.')

    # --- Initialization ---
    if do_cold_start:
        print('Initializing at barycenter.')
        n = A.shape[0]
        P = np.ones((n, n)) / n
        P = do_mult_updates(P, A, B, num_mult_updates)
        P = permutation_match(P)
    else:
        print('Initializing at submission benchmark.')
        P = read_solution('../data/vnc_matching_submission_benchmark_5154247.csv')

    # --- Main Optimization Loop ---
    # Search in simplex, project to vertex, then swap
    P = do_frank_wolfe(P, A, B, num_frank_wolfe)
    P = permutation_match(P)
    P = do_swaps(P, A, B, num_swap_checks)

    # --- Save Solution ---
    # Note: The scoring function is slightly different in Python due to matrix representations
    # A is (from, to), P is (male, female). We want to match A's 'to' with B's 'to' (permuted)
    # and A's 'from' with B's 'from' (permuted).
    # The MATLAB expression `min(A*P, P*B)` is equivalent to `min(A @ P, P @ B)`
    # but let's be explicit about the graph matching objective.
    # S(P) = sum_{i,j} min(A_ij, B_{pi_i, pi_j})
    # This is equivalent to sum(min(A, P @ B @ P.T))
    scoreP = np.minimum(A.toarray(), (P @ B @ P.T).toarray()).sum()
    
    filename = f'vnc_matching_submission_{int(scoreP)}.csv'
    save_solution(filename, P)

    # --- Done ---
    t_min = (time.time() - t_start) / 60
    print(f'\nTotal elapsed time is {t_min:.1f} minutes.')

if __name__ == '__main__':
    main()

