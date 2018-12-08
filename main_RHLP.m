% User-freindly and flexible algorithm for time series segmentation with a Regression
% model with a Hidden Logistic Process (RHLP).
%
%%  If you are using this code, please cite the following papers:
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

solution =  learn_RHLP_EM(x, y, K, p, q, ...
    type_variance,nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
show_RHLP_results(x,y,solution)


%% some real time series with regime changes

%load real_time_series_1
load real_time_series_2

solution =  learn_RHLP_EM(x, y, K, p, q,...
    type_variance,nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);

yaxislim = [240 600];
show_RHLP_results(x,y,solution, yaxislim)%yaxislim is optional





