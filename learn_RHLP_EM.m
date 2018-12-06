function solution = learn_RHLP_EM(x, y, K, p, dim_w, type_variance, total_EM_tries, max_iter_EM,...
    threshold, verbose_EM, verbose_IRLS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function solution = learn_RHLP_EM(x, y, K, p, dim_w, type_variance, total_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS)%
%
% Learn a Regression model with a Hidden Logistic Process (RHLP) for modeling and segmentatio nf of a time series with regime changes.
% The learning is performed by the EM algorithm.
%
% Inputs :
%
%          1. y : a time series composed of m points : dim(Y)=[m 1]
%                * Each curve is observed during the interval [0,T], i.e x =[t_1,...,t_m]
%                * t{j}-t_{j-1} = dt (sampling period)
%
%          2. K : Number of polynomial regression components (regimes)
%          3. p : degree of the polynomials
%          4. q :  order of the logistic regression (choose 1 for
%          convex segmentation)
%          5. total_EM_tries :  (the solution providing the highest log-lik
%          is chosen
%          6. verbose_EM : set to 1 for printing the "log-lik"  values during
%          EM iterations (by default verbose_EM = 0)
%          7. verbose_IRLS : set to 1 for printing the values of the criterion
%             optimized by IRLS at each IRLS iteration. (IRLS is used at
%             each M step of EM). (By default: verbose_EM = 0)
%
% Outputs :
%
%          1. solution : structure containing mainly the following fields:
%                      1.1 param : the model parameters:(W,beta1,...,betaK,sigma1,...,sigmaK).
%                          param is a structure containing the following
%                          fields:
%                          1.1.1 wk = (w1,...,wK-1) parameters of the logistic process:
%                          matrix of dimension [(q+1)x(K-1)] with q the
%                          order of logistic regression.
%                          1.1.2 betak = (beta1,...,betaK) polynomial
%                          regression coefficient vectors: matrix of
%                          dimension [(p+1)xK] p being the polynomial
%                          degree.
%                          1.1.3 sigmak = (sigma1,...,sigmak) : the
%                          variances for the K regmies. vector of dimension [Kx1]
%
%          4. tjk : post prob (fuzzy segmentation matrix of dim [mxK])
%          5. Zjk : Hard segmentation matrix of dim [mxK] obtained by the
%          MAP rule :  z_{jk} = 1 if and only z_j = arg max_k tjk
%          (k1,...,K)
%          appartient ??? la classe k et zero sinon.
%          6. klas : column vector of the labels issued from Zjk, its
%          elements are klas(j)= k (k=1,...,K.)
%          8. theta : parameter vector of the model: theta=(wk,betak,sigmak).
%              column vector of dim [nu x 1] with nu = nbr of free parametres
%
%          9. Ey: curve expectation : sum of the polynomial components betak ri weighted by
%             the logitic probabilities pijk: Ey(j) = sum_{k=1}^K pijk betak rj, j=1,...,m. Ey
%              is a column vector of dimension m
%          13. loglik : log-lik at convergence of EM
%          14. stored_loglik : vector of stored valued of the log-lik at each EM
%          iteration
%
%          17. BIC : valeur du critre BIC.  BIC = loglik - nu*log(nm)/2.
%          17. ICL : valeur du critre ICL.  BIC = complete_loglik - nu*log(nm)/2.
%          18. AIC : valeur du critere AIC. AIC = loglik - nu.
%          20. nu : nbr of free model parametres

%          21. X :  Regression (covariate, Vendermond, design) matrices
%          22. Xw : design matrix for the logistic regression: matrix of dim [mx(q+1)].
%          23. XBeta : design matrix for the polynomial regression: matrix of dim [mx(p+1)].
%
%          2. log_fxj : logarithme des probabilites des
%          observations : log_fxj = log(sum_{i=1}^n sum_{k=1}^K pijk
%          fk(xi)). vecteur colonne de dim n
%          3. fxj : probas des observations : sum_{j=1}^m sum_{k=1}^K pijk
%          fk(xj)). vecteur colonne de dim n

%% References
%  If you are using this code, please cite the following papers:
%
% @article{chamroukhi_et_al_NN2009,
% 	Address = {Oxford, UK, UK},
% 	Author = {Chamroukhi, F. and Sam\'{e}, A. and Govaert, G. and Aknin, P.},
% 	Date-Added = {2014-10-22 20:08:41 +0000},
% 	Date-Modified = {2014-10-22 20:08:41 +0000},
% 	Journal = {Neural Networks},
% 	Number = {5-6},
% 	Pages = {593--602},
% 	Publisher = {Elsevier Science Ltd.},
% 	Title = {Time series modeling by a regression approach based on a latent process},
% 	Volume = {22},
% 	Year = {2009},
% 	url  = {https://chamroukhi.users.lmno.cnrs.fr/papers/Chamroukhi_Neural_Networks_2009.pdf}
% 	}
% 
% @INPROCEEDINGS{Chamroukhi-IJCNN-2009,
%   AUTHOR =       {Chamroukhi, F. and Sam\'e,  A. and Govaert, G. and Aknin, P.},
%   TITLE =        {A regression model with a hidden logistic process for feature extraction from time series},
%   BOOKTITLE =    {International Joint Conference on Neural Networks (IJCNN)},
%   YEAR =         {2009},
%   month = {June},
%   pages = {489--496},
%   Address = {Atlanta, GA},
%  url = {https://chamroukhi.users.lmno.cnrs.fr/papers/chamroukhi_ijcnn2009.pdf}
% }
% 
% @article{Chamroukhi-FDA-2018,
% 	Journal = {},
% 	Author = {Faicel Chamroukhi and Hien D. Nguyen},
% 	Volume = {},
% 	Title = {Model-Based Clustering and Classification of Functional Data},
% 	Year = {2018},
% 	eprint ={arXiv:1803.00276v2},
% 	url =  {https://chamroukhi.users.lmno.cnrs.fr/papers/MBCC-FDA.pdf}
% 	}
%
%
% Faicel CHAMROUKHI
% Mise ??? jour (01 Novembre 2008)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

warning off

switch type_variance
    case 'homoskedastic'
        homoskedastic =1;
    case 'hetereskedastic'
        homoskedastic=0;
    otherwise
        error('The type of the model variance should be : ''homoskedastic'' ou ''hetereskedastic''');
end

if nargin<11; verbose_IRLS = 0; end
if nargin<10; verbose_IRLS = 0; verbose_EM = 0; end
if nargin<9;  verbose_IRLS = 0; verbose_EM = 0;   threshold = 1e-6; end
if nargin<8;  verbose_IRLS = 0; verbose_EM = 0;   threshold = 1e-6; max_iter_EM = 1000; end
if nargin<7;  verbose_IRLS = 0; verbose_EM = 0;   threshold = 1e-6; max_iter_EM = 1000; total_EM_tries=1;end

if size(y,2)~=1, y=y'; end % 

m = length(y);

q = dim_w;

% number of free model parameters
if homoskedastic
    nu = (p+q+3)*K-(q+1) - (K-1) ;
else
    nu = (p+q+3)*K-(q+1);
end


[XBeta, Xw] = designmatrix_RHLP(x,p,q);


best_loglik = -inf;
nb_good_try=0;
total_nb_try=0;
cputime_total = [];
try_EM = 1;
while (nb_good_try < total_EM_tries)
    if total_EM_tries>1,fprintf(1, 'EM try n°  %d \n ',nb_good_try+1); end
    total_nb_try=total_nb_try+1;
    time = cputime;
    %% EM Initializaiton sept
    
    %   1. Initialization of W
    if try_EM ==1
        W0 = zeros(q+1,K-1);
    else
        W0 = rand(q+1,K-1);
    end
    param.wk = W0;
    %   2. Initialization of betak et de sigmak (from one curve)
    param = init_regression_param(y,K,XBeta,type_variance, try_EM);
    
    %%%
    theta = zeros(nu,1);% model parammeter vector
    iter = 0;
    converge = 0;
    prev_loglik=-inf;
    top=0;
    Winit = W0;%
    %% EM %%%%
    
    while ~converge && ~isempty(param) && (iter< max_iter_EM)
        %% E-Step
        param.piik = logit_model(Winit,Xw);
        
        log_piik_fik =zeros(m,K);
        for k = 1:K
            muk = XBeta*param.betak(:,k);
            if homoskedastic
                sigmak =  param.sigma;
            else
                sigmak = param.sigmak(k);
            end
            z=((y-muk).^2)/sigmak;
            log_piik_fik(:,k) = log(param.piik(:,k)) -0.5*ones(m,1).*(log(2*pi)+log(sigmak)) - 0.5*z;
        end
        
        % % log_piik_fik  = min(log_piik_fik,log(realmax));
        log_piik_fik = max(log_piik_fik ,log(realmin));
        piik_fik = exp(log_piik_fik);
        fxi =sum(piik_fik,2);
        log_fxi=log(fxi);
        log_sum_piik_fik = log(sum(piik_fik,2));
        log_tik = log_piik_fik - log_sum_piik_fik*ones(1,K);
        tik = normalize(exp(log_tik),2);
        
        %% M-Step
        % Maximization w.r.t betak and sigmak (the variances)
        % --------------------------------------------------%
        
        if homoskedastic,  s = 0;   end
        %
        for k=1:K
            weights = tik(:,k);% post prob of each component k (dimension nx1)
            nk = sum(weights);% expected cardinal numnber of class k
            
            Xk = XBeta.*(sqrt(weights)*ones(1,p+1));%[m*(p+1)]
            yk=y.*(sqrt(weights));% dimension :(nx1).*(nx1) = (nx1)
            M = Xk'*Xk ;
            epps = 1e-9;
            %if rcond(M)<1e-16
            M=M+epps*eye(p+1);
            betak = inv(M)*Xk'*yk; % Maximization w.r.t betak
            param.betak(:,k)=betak;
            z = sqrt(weights).*(y-XBeta*betak);
            % Maximisation w.r.t sigmak (the variances)
            priorsigma =  0;%1e-5;
            if homoskedastic
                sk = z'*z;
                s = s+sk;
                param.sigma = s/m;
            else
                param.sigmak(k)=z'*z/nk  + priorsigma ;
            end
        end
        % Maximization w.r.t W
        % ----------------------------------%
        %%  IRLS : Iteratively Reweighted Least Squares (for IRLS, see the IJCNN 2009 paper)
        res = IRLS(tik,Xw,Winit,verbose_IRLS);
        param.piik = res.piik;
        param.wk = res.wk;
        Winit = res.wk;
        
        
        %% End of EM
        iter=iter+1;
        
        %% log-likelihood
        
        if (priorsigma~=0); regEM = log(priorsigma); else regEM = 0; end
        
        loglik = sum(log_sum_piik_fik) + res.reg_irls;% + regEM;
        %%
        if prev_loglik-loglik > 1e-4
            top = top+1;
            if (top==10)
                %fprintf(1, '!!!!! The loglikelihood is decreasing from %6.4f to %6.4f!\n', prev_loglik, loglik);
                break;
            end
        end
        %%
        if verbose_EM,fprintf(1, 'EM   : Iteration : %d   Log-likelihood : %f \n',  iter,loglik); end
        converge = abs((loglik-prev_loglik)/prev_loglik) <= threshold;
        prev_loglik = loglik;
        stored_loglik(iter) = loglik;
        
    end% end of an EM run
    try_EM = try_EM +1;
    cputime_total = [cputime_total cputime-time];
    
    
    solution.loglik = loglik;
    solution.stored_loglik = stored_loglik;
    solution.param = param;
    solution.log_piik_fik = log_piik_fik;
    
    %% estimated parameter vector
    if homoskedastic; theta = [param.wk(:); param.betak(:); param.sigma];
    else; theta = [param.wk(:); param.betak(:); param.sigmak(:)];
    end
    solution.theta = theta;
    % solution.param.piik = param.piik(1:m,:);
    solution.tik = tik(1:m,:);
    
    if total_EM_tries>1
        fprintf(1,'loglik = %f \n',solution.loglik);
    end
    if ~isempty(solution.param)
        nb_good_try=nb_good_try+1;
        total_nb_try=0;
        
        if loglik > best_loglik
            best_solution = solution;
            best_loglik = loglik;
        end
    end
    
    if total_nb_try > 500
        fprintf('Impossible to obtain the requested number of classes \n');
        solution=[];
        return   % ou continue
    end
end%fin de la premi???re boucle while


solution = best_solution;
%
if total_EM_tries>1;   fprintf(1,'best loglik:  %f\n',solution.loglik); end

% % for the best solution

solution.param.piik = solution.param.piik(1:m,:);
solution.tik = solution.tik(1:m,:);

%% classsification pour EM : classes = argmax(piik) (here to ensure a convex segmentation of the curve(s)).
[klas, Zik] = MAP(solution.param.piik);

solution.klas = klas;
%
if homoskedastic; theta = [param.wk(:); param.betak(:); param.sigma];
else; theta = [param.wk(:); param.betak(:); param.sigmak(:)];
end
solution.theta = theta;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

solution.polynomials = XBeta*solution.param.betak;
solution.weighted_polynomials = solution.param.piik.*(XBeta*solution.param.betak);
solution.Ex =sum(solution.weighted_polynomials,2);

solution.cputime = mean(cputime_total);
solution.cputime_total = cputime_total;

% solution.log_fxi = log_fxi(1:m,:);
% solution.fxi = fxi(1:m,:);

%%% BIC AIC et ICL
solution.BIC = solution.loglik - (nu*log(m)/2);
solution.AIC = solution.loglik - nu;
%% CL(theta) : vraisemblance classificante
zik_log_piik_fk = Zik.*solution.log_piik_fik;
sum_zik_log_fik = sum(zik_log_piik_fk,2);
comp_loglik = sum(sum_zik_log_fik);
solution.comp_loglik = comp_loglik;
solution.ICL = solution.comp_loglik - (nu*log(m)/2);
%warning on
%
solution.nu = nu;
%solution.X = Phi(1:m,:);
solution.XBeta = XBeta;
solution.Xw = Xw;



