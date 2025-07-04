%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% P = permutation_match(W,P0)
%
% Computes a permutation matrix P that maximizes sum(P.*W,'all').
%
% Uses an internal Matlab routine (in matchpairs) for perfect matching.
% Reference: Duff and Koster, "On algorithms for permuting large entries 
% to the diagonal of a sparse matrix" (1999).
%
% If an initial guess P0 (optional) is available for P, it is used to
% accelerate the perfect matching.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function P = permutation_match(W,P0)

if (nargin==1)
  [m,n] = size(W);
  matchIdx = matlab.internal.graph.perfectMatching(-full(W));
  P = sparse(matchIdx,1:n,1,m,n);
else
  W = W*P0';                    % PERMUTE TO NEARLY DIAGONAL
  D = diag(W);
  W = W-sum(W,1)-sum(W,2)+D+D'; % NO CHANGE IF W IS DIAGONAL
  P = permutation_match(W);     
  P = P*P0;                     % UN-PERMUTE
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%