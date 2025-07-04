% SETUP
tStart = tic;
do_cold_start = 0;
num_mult_updates = 25;
num_frank_wolfe = 15;
num_swap_checks = 20;
secs_per_minute = 60;

% READ IN CONNECTOMES
fprintf(1,'Loading connectomes ... ');
A = read_connectome('male_connectome_graph.csv');
B = read_connectome('female_connectome_graph.csv');
fprintf(1,'%3.1f sec.\n',toc(tStart));

% COLD_START?
if (do_cold_start)
  fprintf(1,'Initializing at barycenter.\n');
  P = ones(size(A))/size(A,1);
  P = do_mult_updates(P,A,B,num_mult_updates);
  P = permutation_match(P);
else
  fprintf(1,'Initializing at submission benchmark.\n');
  P = read_solution('vnc_matching_submission_benchmark_5154247.csv');
end

% SEARCH IN SIMPLEX, PROJECT TO VERTEX, THEN SWAP
P = do_frank_wolfe(P,A,B,num_frank_wolfe);
P = permutation_match(P);
P = do_swaps(P,A,B,num_swap_checks);

% SAVE SOLUTION
scoreP = full(sum(min(A*P,P*B),'all'));
filename = sprintf('vnc_matching_submission_%07d.csv',scoreP);
save_solution(filename,P);

% DONE
tMin = toc(tStart)/secs_per_minute;
fprintf('\nTotal elapsed time is %.1f minutes.\n',tMin);