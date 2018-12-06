function show_RHLP_results(x,y,solution, yaxislim)
%

if size(x,1)~=1
    x = x'; %y=y';
end
set(0,'defaultaxesfontsize',14);
%colors = {'b','g','r','c','m','k','y'};
colors = {[0.8 0 0],[0 0 0.8],[0 0.8 0],'m','c','k','y'};
style =  {'r.','b.','g.','m.','c.','k.','y.'};

if (nargin<4)||isempty(yaxislim)
    yaxislim = [mean(y)-2*std(y), mean(y)+2*std(y)];
end

%% data, regressors, and segmentation

scrsz = get(0,'ScreenSize');
figr = figure('Position',[scrsz(4)/6 scrsz(4)/2 560 scrsz(4)/1.4]);
axes1 = axes('Parent',figr,'Position',[0.1 0.45 0.8 0.48],'FontSize',14);
box(axes1,'on'); hold(axes1,'all');
title('Time series, RHLP regimes, and process probabilites')
plot(x,y,'Color',[0.5 0.5 0.5]);%black')%
[~, K] = size(solution.param.piik);
for k=1:K
    model_k = solution.polynomials(:,k);
    %prob_model_k = solution.param.piik(:,k);
    
    active_model_k = model_k(solution.klas==k);%prob_model_k >= prob);
    active_period_model_k = x(solution.klas==k);%prob_model_k >= prob);
    
    inactive_model_k = model_k(solution.klas ~= k);%prob_model_k >= prob);
    inactive_period_model_k = x(solution.klas ~= k);%prob_model_k >= prob);
    % clf
    % plot(model_k(solution.klas ~= k))
    % pause
    %plot(x,solution.polynomials(solution.param.piik >= prob),'linewidth',3);
    hold on,
    plot(inactive_period_model_k,inactive_model_k,style{k},'markersize',0.001);
    hold on,
    plot(active_period_model_k, active_model_k,'Color', colors{k},'linewidth',3.5);
end
ylabel('y');
ylim(yaxislim);

% Probablities of the hidden process (segmentation)
axes2 = axes('Parent',figr,'Position',[0.1 0.06 0.8 0.35],'FontSize',14);
box(axes2,'on'); hold(axes2,'all');
%subplot(212),
for k=1:K
    plot(x,solution.param.piik(:,k),'Color', colors{k},'linewidth',1.5);
    hold on
end
set(gca,'ytick',[0:0.2:1]);
xlabel('t');
ylabel('Probability \pi_{k}(t,w)');

%% data, regression model, and segmentation
scrsz = get(0,'ScreenSize');
figr = figure('Position',[scrsz(4)/1.2 scrsz(4)/2 560 scrsz(4)/1.4]);
axes1 = axes('Parent',figr,'Position',[0.1 0.45 0.8 0.48],'FontSize',14);
box(axes1,'on'); hold(axes1,'all');
title('Time series, estimated RHLP model, and segmentation')
ylabel('y');
plot(x,y,'Color',[0.5 0.5 0.8]);%black'%
hold on, plot(x,solution.Ex,'r','linewidth',2);

% transition time points
tk = find(diff(solution.klas)~=0);
hold on, plot([x(tk); x(tk)], [ones(length(tk),1)*[min(y)-2*std(y) max(y)+2*std(y)]]','--','color','k','linewidth',1.5);
ylabel('y');
ylim(yaxislim);
 
% Probablities of the hidden process (segmentation)
axes2 = axes('Parent',figr,'Position',[0.1 0.06 0.8 0.35],'FontSize',14);
box(axes2,'on'); hold(axes2,'all');
plot(solution.klas,'k.','linewidth',1.5);

xlabel('t');
ylabel('class labels');

%% model log-likelihood during EM
% %
% figure,
% plot(solution.stored_loglik,'-','linewidth',1.5)
% xlabel('EM iteration number')
% ylabel('log-likelihood')
end
