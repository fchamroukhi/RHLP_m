function [XBeta, XW] = designmatrix_RHLP(x,p,q)
%
%
%
%
%
%
%
%
%
%
%
%
%
%
% Faicel Chamroukhi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin > 2
    ordre_max = max(p,q);
else
    ordre_max = p;
end

if size(x,2) ~= 1;
    x=x'; % a column vector
end

X=[];
for ord = 0:ordre_max
    X =[X x.^(ord)];% X = [1 t t.^2 t.^3 t.^p;......;...]
end
XBeta= X(:,1:p+1); %matrice de regresseurs pour Beta

if nargin > 2
   XW = X(:,1:q+1);% matrice de regresseurs pour w
else
    XW =[];
end