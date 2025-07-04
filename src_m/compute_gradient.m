%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% function G = compute_gradient(P,A,B)
%
% Computes dJ/dP where P is a sparse doubly stochastic matrix,
%            J(P) = sum_ijkl min(A_ij,B_kl) P_ik P_jl,
% and
%      (dJ/dP)_jl = sum_ik [min(A_ij,B_kl) + min(A_ji,B_lk)] P_ik
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
function G = compute_gradient(P,A,B)

n = size(P,1);
G = zeros(n);
[row,col,val] = find(P);
At = A';
Bt = B'; 

for k=1:length(row)
  [ia,~,a] = find(A(:,row(k)));
  [jb,~,b] = find(B(:,col(k)));
  G(ia,jb) = G(ia,jb) + val(k)*min(a,b');
  [ia,~,a] = find(At(:,row(k)));
  [jb,~,b] = find(Bt(:,col(k)));
  G(ia,jb) = G(ia,jb) + val(k)*min(a,b');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
