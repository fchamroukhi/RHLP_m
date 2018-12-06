function res = IRLS(Y,X,Winit,verbose)
% res = IRLS(Winit,M,Y,verbose) : estime le parametre et les pobas d'un processus logistique temporel
%        dans un contexte multi-classes ???tant donn???s une partion et une variable 
%        explicative, par maximum de vraisembmance (ou MAP)mise en oeuvre par 
%        l'algorithme Iteratively Reweighted Least Squares. Le modele est le suivant:
%        probas(i,k) = p(zi=k;W)
%                    = \pi_{ik}(W)
%                    =       exp(wk'vi)
%                      -----------------------
%                      sum_{l=1}^{K} exp(wl'vi)
%                                  
%       avec :
%            * probas(i,k) est la proba de la classe k au temps t_i :
%            i=1,...n et k=1,...,K.
%            * vi = [1,ti,ti^2,...,ti^q]^T;
%            * q : l'ordre du mod???le logistique
%       Le paramtere W=[w1,...,wK] (dans l'algo wK est suppose nul);
%
% Entrees :
%
%         Winit : parametre initial W(0). dim(Winit) = [(q+1)x(K-1)]
%         M : desgin matric for the logistic weights.  dim(X) = [nx(q+1)]
%                            X = [1 t1 t1^2 ... t1^q
%                                 1 t2 t2^2 ... t2^q
%                                      ..
%                                 1 ti ti^2 ... ti^q
%                                      ..
%                                 1 tn tn^2 ... tn^q]
%         Y : matrice de la partion dure ou floue (ici floue : les pro a 
%           posteiori (tik) obtenus par EM). 
%         verbose : a mettre ??? zero si on veut afficher le critere (la
%         vraisemblance) aucours des iterations de l'algorithme (par defaut
%         verbose = 0)
%
% Sorties :
%
%          res : structure contenant les resultats. les champs de la
%          structure sont:
%              wk : le parametre W. matrice de dim [(q+1)x(K-1)]
%                  (le dernier etant nul)
%              piik :les probabilites
%              loglik : la vraisemblance ??? la convergence de l'algorithme
%              LL : vecteur conetant la vraisemblance a chaque iteration de
%              l'IRLS.
%              reg_irls : log de la proba a priori de W (pour l'injecter
%              dans l'EM dans le cas du MAP dans l'IRLS)
%
%  voir article "Improved learning algorithms for mixture of experts in
%  multiclass classification" K. Chen, L. Xu & H. Chi. Neural Networks 1999
%
% Faicel 31 octobre 2008 (mise ??? jour)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n, K] = size(Y);
[n,q] = size(X);% q here is (q+1)

if nargin<4
   verbose = 0;
end
if nargin<3
   Winit = zeros(q,K-1);
end
lambda = 1e-9;% if a MAP regularization (a gaussian prior on W) (L_2 penalization)
I = eye(q*(K-1));

%% IRLS Initialization (iter = 0)
W_old = Winit;

[piik_old, loglik_old] =logit_model(W_old,X,Y);
loglik_old = loglik_old - lambda*(norm(W_old(:),2))^2;
iter = 0;
converge = 0;
max_iter = 300;
LL = [];
if verbose
 fprintf(1,'IRLS : Iteration : %d "Log-lik": %f\n', iter, loglik_old);
end
%% IRLS
while ~converge && (iter<max_iter)
     % Hw_old matrice carree de dimensions hx x hx
     hx = q*(K-1);
     Hw_old = zeros(hx,hx);
     gw_old = zeros(q,K-1,1);
     
     % Gradient :
        for k=1:K-1
            gwk = Y(:,k) - piik_old(:,k);
            for qq=1:q
            vq = X(:,qq);        
            gw_old(qq,k) = gwk'*vq;
            end
        end
        gw_old = reshape(gw_old,q*(K-1),1);
    % Hessienne
    for k=1:K-1
        for ell=1:K-1
            delta_kl =(k==ell);% kronecker delta        
            gwk = piik_old(:,k).*(ones(n,1)*delta_kl - piik_old(:,ell));   
            Hkl = zeros(q,q);
            for qqa=1:q            
                vqa=X(:,qqa);               
                for qqb=1:q
                    vqb=X(:,qqb);  
                    hwk = vqb'*(gwk.*vqa);
                    Hkl(qqa,qqb) = hwk;
                 end
            end         
              Hw_old((k-1)*q +1 : k*q, (ell-1)*q +1 : ell*q) = -Hkl;
        end      
    end


%% si a priori gaussien sur W (lambda ~ 1e-9)
Hw_old = Hw_old + lambda*I;
gw_old = gw_old - lambda*W_old(:);
%% Newton Raphson : W(c+1) = W(c) - H(W(c))^(-1)g(W(c))
w = W_old(:) - inv(Hw_old)*gw_old ;%[(q+1)x(K-1),1]
W = reshape(w,q,K-1);%[(q+1)*(K-1)] 
% mise a jour des probas et de la loglik
[piik, loglik] =logit_model(W,X,Y);
loglik = loglik - lambda*(norm(W(:),2))^2;

%% Verifier si Qw1(w^(c+1),w^(c))> Qw1(w^(c),w^(c)) 
%(adaptation) de Newton Raphson : W(c+1) = W(c) - pas*H(W)^(-1)*g(W)
 pas = 1; % initialisation pas d'adaptation de l'algo Newton raphson
 alpha = 2;
 %ll = loglik_old;
 while (loglik < loglik_old)
        pas = pas/alpha; % pas d'adaptation de l'algo Newton raphson
        %recalcul du parametre W et de la loglik
        w = W_old(:) - pas* inv(Hw_old)*gw_old ;
        W = reshape(w,q,K-1);
        [piik loglik] =logit_model(W,X,Y); 
        loglik = loglik - lambda*(norm(W(:),2))^2;
 end     
    converge1 = abs((loglik-loglik_old)/loglik_old) <= 1e-7;
    converge2 = abs(loglik-loglik_old) <= 1e-6;
    
    converge = converge1| converge2 ; 
    
    piik_old = piik;
    W_old = W;
    iter = iter+1;
    LL = [LL loglik_old];
    loglik_old = loglik;
    if verbose
 fprintf(1,'IRLS : Iteration : %d "Log-lik": %f\n', iter, loglik_old);
    end
end % end of IRLS

if converge 
   if verbose
    fprintf('\n');
    fprintf(1,'IRLS : convergence  OK ; nbr of iterations %d \n', iter);
    fprintf('\n'); 
   end
else
    fprintf('\n');
    fprintf('IRLS : doesn''t converged (augment the number of iterations > %d\n', max_iter) ;
end
% resultat
res.wk = W;
res.LL= LL;
res.loglik = loglik;
res.piik = piik;

if lambda~=0 % calculaye the value of the MAP criterion in case of regulrization
res.reg_irls = - lambda*(norm(W(:),2))^2;
else
    res.reg_irls = 0;
end

