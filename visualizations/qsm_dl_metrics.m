% qsm_dl_metrics.m
%   Calculate similarity metrics (RMSE, PSNR, SSIM, XSIM) for deep-learning head
%   and neck QSM results.
%
% Requires COMPUTE_RMSE, COMPUTE_SSIM, and COMPUTE_XSIM from the FANSI toolbox
%
% MT Cherukara, University College London
% Created: 2025-09-22
% Last Update: 2026-04-23

clearvars;

%% Specify the data

% data directory
dir_data = '..\Data_DLQSM\derivatives\qsm\';
dir_rois = '..\Data_DLQSM\rois\rater-01\';

% Subjects
subs = 1:5;
n_subs = length(subs);

% "Ground truth" data
name_gt = 'autoNDI';

% DL comparison
name_dl = 'xQSMSE';

% Mask name
name_mask = '_desc-mask';

% Storage (in this order: RMSE, XSIM, SSIM, PSNR)
name_metrics = {'RMSE','XSIM','SSIM','PSNR'};
res_metrics = zeros(n_subs,4);


%% Loop Over Subjects and Calculate the Metrics
for sub = subs

    % Construct scan name
    subname = sprintf('sub-%0.2d',sub);
    scanname = strcat(subname,'_ses-01');

    % Directory for the present scan
    dir_curr = fullfile(dir_data,subname,'ses-01','qsm');

    % Load the data and convert to single precision
    arr_gt = single(niftiread(strcat(dir_curr,'\',scanname,'_unwrapped-SEGUE_bfr-PDF_susc-',name_gt,'_Chimap')));
    arr_dl = single(niftiread(strcat(dir_curr,'\',scanname,'_unwrapped-SEGUE_bfr-PDF_susc-',name_dl,'_Chimap')));

    % Load the mask and convert to logical
    arr_mask = niftiread(strcat(dir_curr,'\',scanname,name_mask,'_mask')) == 1;

    % Apply mask
    arr_gt = arr_gt.*arr_mask;
    arr_dl = arr_dl.*arr_mask;

    % Calculate metrics
    res_metrics(sub,1) = 0.01.*compute_rmse(arr_dl,arr_gt);
    res_metrics(sub,2) = compute_xsim(arr_gt,arr_dl,arr_mask);
    res_metrics(sub,3) = compute_ssim(arr_gt,arr_dl);
    res_metrics(sub,4) = psnr(arr_dl,arr_gt);

end % for sub = subs


%% Display results
for mm = 1:4
    vec_metric = res_metrics(:,mm);
    fprintf('%s results: %.3f +/- %.3f \n',name_metrics{mm},mean(vec_metric),std(vec_metric));
end


%% Save the results into the tables
for mm = 1:4
    t_data = readtable(['DL_resultsROI_',name_metrics{mm},'.csv']);
    t_data = addvars(t_data,res_metrics(:,mm),'NewVariableNames',name_dl);
    writetable(t_data,['DL_resultsROI_',name_metrics{mm},'.csv']);
end

