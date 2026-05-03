% qsm_dl_graphs.m
%   Display results of similarity metrics from deep-learning models of head and
%   neck QSM data.
%
% MT Cherukara, University College London
% Created: 2025-09-25
% Last Update: 2026-04-23

clearvars; close all;

% Specify metric
i_metric = 'RMSE';

% Load the data from a saved table
t_data = readtable(['DL_results_',i_metric,'.csv']);

% Re-order table variables
t_data = t_data(:,[1:3,5:7]);

% Number of subjects and variables
n_sub = height(t_data);
n_var = width(t_data) - 1;

% Pull out method names
names_meth = {'StarQSM';'QSMnet';'xQSM';'xQSM-HN';'SqE-HN'};

% Extract the data to a matrix
mat_data = table2array(t_data(:,2:end));

% Metric-specific plot limits
% Set new ylimit
switch i_metric
    case 'PSNR'
        ylims = [0,50];
        hscale = 2;
    case 'RMSE'
        i_metric = 'NRMSE';
        ylims = [0,1.2];
        hscale = 0.02;
    otherwise
        ylims = [0,1];
        hscale = 0.02;
end

% Calculate mean and standard deviation
av_data = mean(mat_data,1);
sd_data = std(mat_data,[],1);

% Perform ANOVA analysis
[p1, t_anova, stats] = anova1(mat_data,[],'off');
mat_results = multcompare(stats,'Display','off');

% Calculate maximum heights needed for each bar
hx1 = av_data(mat_results(:,1)) + sd_data(mat_results(:,1));
hx2 = av_data(mat_results(:,2)) + sd_data(mat_results(:,2));

% Find min height needed for each pair
hmax = max(hx1,hx2);

% Pre-allocate vector of heights
h1 = zeros(1,size(mat_results,1)+1);

% Colours
mat_col = [188, 228, 229; ...   % Nile Blue
           100,  80, 161; ...   % Blue Violet
           121,  51,  39; ...   % Hay's Russet
           238, 180, 128; ...   % Pinkish Cinnamon
            98, 198, 191; ...   % Venice Green
           193, 196, 148] ...   % Olive Buff
                        ./255;     

% Plot bar chart
f1 = figure(11);
set(f1,'WindowStyle','normal','Position',[-1700,300,550,500]);
co1 = colororder(mat_col);

b1 = bar(1:n_var,av_data,0.6,'FaceColor','flat');
b1.LineWidth = 1;
hold on;
errorbar(1:n_var,av_data,sd_data,'k.','LineWidth',1);
ylabel(i_metric);
xticks(1:n_var);
xticklabels(names_meth);
ylim(ylims);
set(gca,'FontSize',16);

b1.CData(4,:) = mat_col(2,:);
b1.CData(5,:) = mat_col(2,:);

% % Loop through and add lines for significant variables
% for rr = 1:size(mat_results,1)
% 
%     if mat_results(rr, 6) < 0.05 % Check if the p-value is significant
% 
%         % Pull out x coords of line
%         x1 = mat_results(rr,1);
%         x2 = mat_results(rr,2);
% 
%         % Compare previous height
%         if (hmax(rr) + hscale) > (h1(rr) + (2*hscale))
%             h1(rr+1) = hmax(rr) + hscale;
%         else
%             h1(rr+1) = h1(rr) + (2*hscale);
%         end
% 
%         % Draw a line
%         plot([x1,x2],[h1(rr+1),h1(rr+1)],'k-','LineWidth',1.5);
% 
%         % Number of significance stars
%         nstar = sum(mat_results(rr,6) < [0.001,0.01,0.05]);
% 
%         % Add text of the stars
%         text((x1+x2)/2,h1(rr+1)+0.01,sprintf('%s', repmat('*',1,nstar)),...
%              'FontSize',14,'HorizontalAlignment','center');
% 
%     end
% 
% end % for rr = 1:size(mat_results,1)

% ylim([0,max(max(h1)+0.1,1)])

