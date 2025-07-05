import numpy as np

from do_frank_wolfe import do_frank_wolfe
from do_mult_updates import do_mult_updates
from do_swaps import do_swaps
from permutation_match import permutation_match
from src_py.util import read_connectome, read_solution, save_solution, log


def main():
    do_cold_start = False  # Set to True to initialize at barycenter
    num_mult_updates, num_frank_wolfe, num_swap_checks = 25, 15, 20
    A = read_connectome('../data/male_connectome_graph.csv')
    B = read_connectome('../data/female_connectome_graph.csv')
    log("data loaded")

    # --- Initialization ---
    if do_cold_start:
        n = A.shape[0]
        P = np.ones((n, n)) / n
        P = do_mult_updates(P, A, B, num_mult_updates)
        P = permutation_match(P)
        log('initialized at barycenter')
    else:
        P = read_solution('../data/vnc_matching_submission_benchmark_5154247.csv')
        log('initialized at submission benchmark.')

    # --- Main Optimization Loop ---
    # Search in simplex, project to vertex, then swap
    P = do_frank_wolfe(P, A, B, num_frank_wolfe)
    P = permutation_match(P)
    P = do_swaps(P, A, B, num_swap_checks)

    # --- Save Solution ---
    scoreP = np.minimum(A.toarray(), (P @ B @ P.T).toarray()).sum()
    save_solution(f'vnc_matching_submission_{int(scoreP)}.csv', P)

    log(f'\n=== DONE ===')


if __name__ == '__main__':
    main()
