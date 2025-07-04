%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% function P = do_swaps(P,A,B,max_iter)
%
% Iteratively computes the gains from all pairwise
% swaps and performs these swaps until there are no
% more gains.
%
% P is the current permutation matrix.
% A and B are the graphs being matched.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function P = do_swaps(P,A,B,max_iter)

% SETUP
tStart = tic;
secs_per_minute = 60;
scoreP = full(sum(min(A*P,P*B),'all'));
fprintf(1,'\nPairwise swaps:\n');
fprintf(1,'  iter     score    swaps   tMin\n');

% LOOP
for iter=1:max_iter
  fprintf(1,'    %02d   %07d    ',iter,scoreP);
  S = compute_swap_gains(P,A,B);
  [P,scoreP,num_swap] = swap_check(P,S,A,B);
  fprintf(1,'%5d   %04.1f\n',num_swap,toc(tStart)/secs_per_minute);
  if (num_swap==0)
    break;
  end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% function S = compute_swap_gains(P,A,B)
%
% Computes S_ij, the gain in score from swapping nodes 
% i and j before applying the permutation stored in
% the sparse matrix P
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function S = compute_swap_gains(P,A,B)

% LINEAR TERM
G = compute_gradient(P,A,B);
sumGP2 = sum(G.*P,2);
L = full(G*P'+P*G'-sumGP2-sumGP2');

% QUADRATIC TERM
Bp = P*B*P';
dA = diag(A);
dB = diag(Bp);
Q = min(dA,dB) + min(dA,dB') - min(dA,Bp) - min(dA,Bp') + ...
    min(A,Bp) + min(A,Bp') - min(A,dB) - min(A,dB');
    
% COMBINE TERMS
S = L+Q+Q';

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% function [P,scoreP,num_swap] = swap_check(P,S,A,B)
%
% Tests all swaps (i<-->j) for which S_ij > 0 and
% performs each swap if it results in a gain.
%
% Returns the new permutation matrix P, its score,
% and the number of performed swaps.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [P,scoreP,num_swap] = swap_check(P,S,A,B)

% INITIALIZE
scoreP = full(sum(min(A*P,P*B),'all'));
num_swap = 0;

% NOTHING TO SWAP?
if (max(S,[],'all')==0)
  return;
end
  
% EXCLUDE DIAGONAL SWAPS
n = size(S,1);
S(1:n+1:end) = -Inf;

% TEST ALL SWAPS WITH POSITIVE DIFFERENTIALS
[maxS,idx] = max(S,[],'all');
while (maxS>0)
  [i,j] = ind2sub(size(S),idx);
  P([i j],:) = P([j i],:);
  swap_score = full(sum(min(A*P,P*B),'all'));
  if (swap_score>scoreP)
    scoreP = swap_score;
    num_swap = num_swap+1;
  else
    P([j i],:) = P([i j],:);
  end
  S(i,j) = -Inf;
  S(j,i) = -Inf;
  [maxS,idx] = max(S,[],'all');
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%