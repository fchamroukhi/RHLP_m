function [klas, Z] = MAP(post_probas)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [classes Z_MAP] = MAP(post_probas)
%   
% calcule une partition d'un echantillon par la regle du Maximum A Posteriori ?? partir des
%
% probabilites a posteriori 
%
% Entrees : post_probas , Matrice de dimensions [n x K] des probabibiltes a
% posteriori (matrice de la partition floue)
%
%       n : taille de l'echantillon
%
%       K : nombres de classes
%
%       klas(i) = arg   max (post_probas(i,k)) , for all i=1,...,n
%                     1<=k<=K
%               = arg   max  p(zi=k|xi;theta)
%                     1<=k<=K
%               = arg   max  p(zi=k;theta)p(xi|zi=k;theta)/sum{l=1}^{K}p(zi=l;theta) p(xi|zi=l;theta)
%                     1<=k<=K
%
% Sorties : classes : vecteur collones contenant les classe (1:K)
%
%       Z : Matrice de dimension [nxK] de la partition dure : ses elements sont zik, avec zik=1 si xi
%       appartient ?? la classe k (au sens du MAP) et zero sinon.
%
%%%%%%%%%%%%%%%%%%%%%%%%% Faicel Chamroukhi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[n, K] = size(post_probas);
[~,klas] = max(post_probas,[],2);
partition_MAP = (klas*ones(1,K))==(ones(n,1)*[1:K]);
% double   
Z = double(partition_MAP);

