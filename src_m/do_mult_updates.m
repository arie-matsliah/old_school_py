%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% function P = do_mult_updates(P,A,B,max_iter)
%
% Performs multiplicative updates to maximize a surrogate for the 
% objective function
%   J(P) = sum_ijkl min(A_ij,B_kl) P_ik P_jl,
% where P is a dense stochastic matrix. The surrogate is obtained
% from lower and upper bounds on J(P), which are in turn derived
% from the inequalities
%   min(a,b) >= a*b/maxVal,
%   min(a,b) <= sqrt(a*b).
% The multiplicative updates here use the average of these lower 
% and upper bounds as a surrogate for J(P). Note that the 
% gradients of these bounds can be cheaply computed, even for 
% dense stochastic matrices.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function P = do_mult_updates(P,A,B,max_iter)

% SETUP
tStart = tic;
maxA = max(nonzeros(A));
maxB = max(nonzeros(B));
minMax = min(maxA,maxB);
A = min(A,minMax); % TIGHTENS UPPER BOUND BUT HAS NO
B = min(B,minMax); % EFFECT ON OBJECTIVE FUNCTION
sqrtA = sqrt(A);
sqrtB = sqrt(B);
[u,v] = deal([]);
secs_per_minute = 60;

% LOOP
fprintf(1,'\nDense multiplicative updates on bounded objective:\n');
fprintf(1,'  iter   midPoint   lowerBnd   upperBnd   tMin\n');
for iter=1:max_iter
  % GRADIENTS
  Gl = ((A'*P)*B + (A*P)*B')/minMax;         % min(a,b) >= a*b/minMax      
  Gu = (sqrtA'*P)*sqrtB + (sqrtA*P)*sqrtB';  % min(a,b) <= sqrt(a*b)
  Gm = 0.5*(Gl+Gu);
  % BOUNDS
  lower_bound = full(0.5*sum(P.*Gl,'all')); 
  upper_bound = full(0.5*sum(P.*Gu,'all'));
  mid_point = 0.5*(lower_bound+upper_bound);
  % RATE OF PROGRESS
  tMin = toc(tStart)/secs_per_minute;
  fprintf(1,'    %02d    %07.0f    %07.0f    %07.0f   %04.1f\n',...
    iter,mid_point,lower_bound,upper_bound,tMin);
  % UPDATE
  [P,u,v] = dense_mult_update(P,Gm,u,v);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [P,u,v] = dense_mult_update(P,G,u,v)

% SETUP
numer = P.*G;
if (isempty(u))
  u = 0.5*sum(numer,2);
  v = 0.5*sum(numer,1);
end
max_iter1 = 200;
max_iter2 = 250;

% MULTIPLICATIVE UPDATES
for iter=1:max_iter1
  P = numer./(u+v);
  u = u.*sum(P,2);
  v = v.*sum(P,1);
  shift = 0.5*(min(u)-min(v));
  v = max(eps,v+shift);
  u = max(eps,u-shift);
end

% STABILIZE SO THAT NO ROW OR COLUMN SUMS EXCEED ONE
for iter=(1+max_iter1):max_iter2
  P = numer./(u+v);  
  u = u.*max(1,sum(P,2));
  v = v.*max(1,sum(P,1));
end

% CONVERGENCE DIAGNOSTIC
verbose = 0;
if (verbose)
  sum_err = abs([sum(P,1)';sum(P,2)]-1);
  errQ = quantile(sum_err,[0.99 0.995 0.999 1]);
  fprintf(1,'  iter    err99    err995   err999    errMax\n');
  fprintf(1,'   %03d   %6.5f   %6.5f   %6.5f   %6.5f\n',...
    iter,errQ(1),errQ(2),errQ(3),errQ(4));
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%