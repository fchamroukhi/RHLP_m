function param = init_regression_param(data,K,phi,type_variance, try_EM)
% init_modele_regression estime les parametres de regression initiaux d'un
% modele de regression à processus logistique cache,où la loi conditionnelle
% des observations est une gaussienne.
%
% Entrees :
%       
%        data: signal
%        nsignal (notez que pour la partie parametrisation des signaux les 
%        observations sont monodimentionnelles)
%        K : nbre d'états (classes) cachés
%        duree_signal : = duree du signal en secondes
%        fs : fréquence d'échantiloonnage des signaux en Hz
%        ordre_reg : ordre de regression olynomiale
%
% Sorties :
%
%      
%         param : parametres initiaux du modele de 
%         regression : structure contenant les champs :
%         1. betak : le vecteur parametre de regression associe a la classe k.
%         vecteur colonne de dim [(p+1)x1]
%         2. sigmak(k) = variance de x(i) sachant z(i)=k; sigmak(j) =
%         sigma^2_k.
%
%
%
% Faicel Chamroukhi, Novembre 2008
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
if strcmp(type_variance,'homoskedastic')
   homoskedastic = 1;
else
    homoskedastic = 0;
end
x = data;
m = length(x);
 
 
if  try_EM ==1% découpage uniforme en K segments
     zi = round(m/K)-1;
     for k=1:K
         xk = x((k-1)*zi+1:k*zi);
         phik = phi((k-1)*zi+1:k*zi,:);

         param.betak(:,k) = regress(xk,phik);
         
         muk = phik*param.betak(:,k);
         if homoskedastic
             param.sigma = 1;%var(x);
         else
             sigmak = var(xk);       
             % param.sigmak(k) =  1;
             % muk = phik* param.betak(:,k);
             % sigmak ((xk-muk)'*(xk-muk))/zi;%
             param.sigmak(k) =  sigmak;                 
         end
     end
     
 else % decoupage aléatoire
     Lmin=10;% 
     tk_init = zeros(K,1);
     tk_init(1) = 0;         
     K_1=K;
     for k = 2:K
         K_1 = K_1-1;
         temp = tk_init(k-1)+Lmin:m-K_1*Lmin;
         ind = randperm(length(temp));
         tk_init(k)= temp(ind(1));                      
     end
     tk_init(K+1) = m;
     for k=1:K
         i = tk_init(k)+1;
         j = tk_init(k+1);
         xij = x(i:j);
         xk = xij;%x((k-1)*zi+1:k*zi);
         phik = phi(i:j,:);%phi((k-1)*zi+1:k*zi,:);
         param.betak(:,k) = inv(phik'*phik)*phik'*xk;%regress(xk,phik); % for a use in octave, where regress doesnt exist
         sigmak = var(xk);
         
         % muk = phik* param.betak(:,k);
         % sigmak ((xk-muk)'*(xk-muk))/zi;%
         if homoskedastic
             param.sigma = var(x);
         else
             param.sigmak(k) =  sigmak;
             param.sigmak(k) =  1;
         end
     end
 end
 
 
 