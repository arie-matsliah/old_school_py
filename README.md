This directory contains MATLAB code from Team Old School (Dan Lee and Lawrence Saul) to discover a top-scoring solution to the Flywire VNC Matching Challenge. 

The src directory contains these files:

     compute_gradient.m
     main.m
     do_frank_wolfe.m
     do_mult_updates.m
     do_swaps.m
     permutation_match.m
     read_connectome.m
     read_solution.m
     save_solution.m

Before running the code, you must first download and decompress the data files from the challenge web site. In particular, MATLAB will look on its path for these files:

     female_connectome_graph.csv
     male_connectome_graph.csv
     vnc_matching_submission_benchmark_5154247.csv

To run the code in MATLAB, simply type
    > main

The code was developed on a MacBook Pro (2021 Apple M1 Max) with 64 GB of RAM. The optimization can be initialized from a warm start (at the benchmark solution) or a cold start (at the barycenter of the simplex of all doubly stochastic matrices). From a warm start, it takes less than 20 min to reach a score of 5850K, and from a cold start, it takes less than 45 min. More details of these runs can be found in these files:

      log/log_warm_start.txt
      log/log_cold_start.txt

The initialization is determined by the setting in line 3 of main.m, and other settings for the code can be changed in lines 4-6.

Lawrence Saul (lsaul@flatironinstitute.org)
