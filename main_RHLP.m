% User-freindly and flexible algorithm for time series segmentation with a Regression
% model with a Hidden Logistic Process (RHLP).
%
%% Please cite the following papers for this code:
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
% 	Journal = {Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery},
% 	Author = {Faicel Chamroukhi and Hien D. Nguyen},
% 	Note = {DOI: 10.1002/widm.1298.},
% 	Volume = {},
% 	Title = {Model-Based Clustering and Classification of Functional Data},
% 	Year = {2019},
% 	Month = {to appear},
% 	url =  {https://chamroukhi.com/papers/MBCC-FDA.pdf}
% 	}
%
%
% by Faicel Chamroukhi (2008)


%%

clear;
close all;
clc;


% model specification
K = 5; % nomber of regimes (mixture components)
p = 3; % dimension of beta' (order of the polynomial regressors)
q = 1; % dimension of w (ordre of the logistic regression: to be set to 1 for segmentation)

% options
%type_variance = 'homoskedastic';
type_variance = 'hetereskedastic';
nbr_EM_tries = 1;
max_iter_EM = 1500;
threshold = 1e-6;
verbose_EM = 1;
verbose_IRLS = 0;


%% toy time series with regime changes
% y =[randn(100,1); 7+randn(120,1);4+randn(200,1); -2+randn(100,1); 3.5+randn(150,1);]';
% n = length(y);
% x = linspace(0,1,n);

load simulated_time_series;

rhlp =  learn_RHLP_EM(x, y, K, p, q, ...
    type_variance,nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);

%if model selection
current_BIC = -inf;verbose_EM = 0;
for K=1:10
    for p=0:4
        
        rhlp_Kp = learn_RHLP_EM(x, y, K, p, 1, type_variance,nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
        fprintf(1,'Number of segments K: %d | polynomial degree p: %d | BIC %f \n', K, p, rhlp_Kp.BIC);
        if rhlp_Kp.BIC>current_BIC
            rhlp=rhlp_Kp;
            current_BIC = rhlp_Kp.BIC;
        end
        bic(K, p+1) = rhlp_Kp.BIC;
    end
end

show_RHLP_results(x,y, rhlp)

%% some real time series with regime changes

%load real_time_series_1
load real_time_series_2

rhlp =  learn_RHLP_EM(x, y, K, p, q,...
    type_variance,nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);

%if model selection
current_BIC = -inf;verbose_EM = 0;
for K=1:10
    for p=0:4
        rhlp_Kp = learn_RHLP_EM(x, y, K, p, 1, type_variance,nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
        fprintf(1,'Number of segments K: %d | polynomial degree p: %d | BIC %f \n', K, p, rhlp_Kp.BIC);
        if rhlp_Kp.BIC>current_BIC
            rhlp=rhlp_Kp;
            current_BIC = rhlp_Kp.BIC;
        end
        bic(K,p+1) = rhlp_Kp.BIC;
    end
end

yaxislim = [240 600];
show_RHLP_results(x,y,rhlp, yaxislim)%yaxislim is optional





