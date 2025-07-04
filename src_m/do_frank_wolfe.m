%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% function Ps = do_frank_wolfe(Ps,A,B,num_updates)
%
% Performs Frank_Wolfe updates on the sparse doubly stochastic matrix Ps.
% Stores the nearest matching permutations as Pm.
% Tracks the scores of Pm (on a vertex) and Ps (in the simplex).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Ps = do_frank_wolfe(Ps,A,B,num_updates)

tStart = tic;
secs_per_minute = 60;
fprintf(1,'\nFrank-Wolfe updates:\n');
fprintf(1,'  iter    vertex   simplex   tMin\n');
for iter=1:num_updates
  Pm = permutation_match(Ps);
  Gs = compute_gradient(Ps,A,B);
  scorePm = full(sum(min(A*Pm,Pm*B),'all'));
  scorePs = round(full(0.5*(sum(Gs.*Ps,'all'))));
  tMin = toc(tStart)/secs_per_minute;
  fprintf(1,'    %02d   %07d   %07d   %04.1f\n',iter,scorePm,scorePs,tMin);
  if (iter<num_updates)
    Ps = frank_wolfe_update(Ps,Gs,Pm,A,B);
  end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Ps = frank_wolfe_update(P0,G0,Pm,A,B)

% PROJECT GRADIENT TO PERMUTATION MATRIX
P1 = permutation_match(G0,Pm);

% DONE?
if (isequal(P0,P1))
  Ps = P0;
  return;
end

% COMPUTE STEP SIZE
G1 = compute_gradient(P1,A,B);
numer = full(sum((G1-G0).*P0 + (P1-P0).*G0,'all'));
denom = full(sum((G1-G0).*(P1-P0),'all'));
step = -0.5*(numer/denom);

% STAY WITHIN SIMPLEX
step(step>1) = 1;     % CONCAVE BUT MAXIMUM AT step>1
step(step<0) = 1;     % CONVEX WITH MINIMUM AT step<0

% INTERPOLATE
Ps = P0 + step*(P1-P0);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%