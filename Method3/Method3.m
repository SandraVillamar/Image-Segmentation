% load data
D1_FG = matfile('../data/TrainingSamplesDCT_subsets_8.mat').D1_FG;
D1_BG = matfile('../data/TrainingSamplesDCT_subsets_8.mat').D1_BG;
D2_FG = matfile('../data/TrainingSamplesDCT_subsets_8.mat').D2_FG;
D2_BG = matfile('../data/TrainingSamplesDCT_subsets_8.mat').D2_BG;
D3_FG = matfile('../data/TrainingSamplesDCT_subsets_8.mat').D3_FG;
D3_BG = matfile('../data/TrainingSamplesDCT_subsets_8.mat').D3_BG;
D4_FG = matfile('../data/TrainingSamplesDCT_subsets_8.mat').D4_FG;
D4_BG = matfile('../data/TrainingSamplesDCT_subsets_8.mat').D4_BG;

% for each dataset and strategy, perform Part A,B,C
%[fig, errors_pd, errors_mle, errors_map] = method3(D1_FG, D1_BG, 1, 'D1, Strategy 1');
%[fig, errors_pd, errors_mle, errors_map] = method3(D2_FG, D2_BG, 1, 'D2, Strategy 1');
%[fig, errors_pd, errors_mle, errors_map] = method3(D3_FG, D3_BG, 1, 'D3, Strategy 1');
%fig, errors_pd, errors_mle, errors_map] = method3(D4_FG, D4_BG, 1, 'D4, Strategy 1');
%[fig, errors_pd, errors_mle, errors_map] = method3(D1_FG, D1_BG, 2, 'D1, Strategy 2');
%[fig, errors_pd, errors_mle, errors_map] = method3(D2_FG, D2_BG, 2, 'D2, Strategy 2');
%[fig, errors_pd, errors_mle, errors_map] = method3(D3_FG, D3_BG, 2, 'D3, Strategy 2');
[fig, errors_pd, errors_mle, errors_map] = method3(D4_FG, D4_BG, 2, 'D4, Strategy 2');


function [fig, errors_pd, errors_mle, errors_map] = method3(D_FG, D_BG, strategy, curr_title)
% D_FG, D_BG: datasets for cheetah and grass classes
% strategy: 1 or 2
% curr_title: title of plot, contains data subset and strategy number

% Add path to access multi_gaussian fn
addpath('..')

% Load variables
if strategy == 1
    m = matfile('../data/Prior_1.mat');
else
    m = matfile('../data/Prior_2.mat');
end
alpha = matfile('../data/Alpha.mat').alpha;

% PART A:
% Compute sigma for each class: c=cheetah, g=grass
sigma_c = cov(D_FG);
sigma_g = cov(D_BG);

% FOR EACH ALPHA
errors_pd = zeros([1,length(alpha)]);
errors_mle = zeros([1,length(alpha)]);
errors_map = zeros([1,length(alpha)]);
for k = 1:length(alpha)
    % Compute sigma0
    sigma0 = diag(alpha(k)*m.W0);
    
    % Compute mu1 and sigma1 for each class
    n_c = size(D_FG, 1);
    mu1_c = sigma0*(inv(sigma0 + sigma_c/n_c))*mean(D_FG)' ...
        + sigma_c*(inv(sigma0 + sigma_c/n_c))*m.mu0_FG'/n_c;
    sigma1_c = sigma0*(inv(sigma0 + sigma_c/n_c))*sigma_c/n_c;
    n_g = size(D_BG, 1);
    mu1_g = sigma0*(inv(sigma0 + sigma_g/n_g))*mean(D_BG)' ...
        + sigma_g*(inv(sigma0 + sigma_g/n_g))*m.mu0_BG'/n_g;
    sigma1_g = sigma0*(inv(sigma0 + sigma_g/n_g))*sigma_g/n_g;
    
    % Compute parameters of predictive distribution
    % pd mean = mu1
    sigma_pd_c = sigma_c + sigma1_c;
    sigma_pd_g = sigma_g + sigma1_g;
    
    % Compute ML estimates of class priors
    p_c = size(D_FG, 1) / (size(D_FG, 1) + size(D_BG, 1));
    p_g = size(D_BG, 1) / (size(D_FG, 1) + size(D_BG, 1));

    % (intertwining PART B): Compute MLE of mu and sigma
    mu_mle_c = mean(D_FG)';
    mu_mle_g = mean(D_BG)';
    % mle sigma = sigma
    
    % CLASSIFY IMAGE
    % read zigzag text file
    zigzag = readmatrix('../data/Zig-Zag Pattern.txt');
    zigzag = reshape(zigzag+1, [64,1]);
    % read cheetah.bmp and reformat to range [0,1]
    img = imread('../data/cheetah.bmp');
    img = im2double(img);
    % divide into 8x8 blocks, sliding window = 248*263 blocks
    mask_prediction = zeros(248,263);
    mask_prediction_mle = zeros(248,263);
    mask_prediction_map = zeros(248,263);
    for i = 1:(size(img,1)-7)
        for j = 1:(size(img,2)-7)
            block = img(i:i+7, j:j+7);
            dct_matrix = dct2(block);
            % transform dct matrix into vector, wrt zigzag indices
            dct_vector = reshape(dct_matrix, [64,1]);
            [~, sorted_ind] = sort(zigzag, 'ascend');
            dct_vector = dct_vector(sorted_ind);
    
            % PREDICTIVE DISTRIBUTION
            % Compute p(X|Y=cheetah, T=D1)
            pd_c = multi_gaussian(dct_vector, mu1_c, sigma_pd_c);
            % Compute p(X|Y=grass, T=D1)
            pd_g = multi_gaussian(dct_vector, mu1_g, sigma_pd_g);
            % BDR
            if pd_c*p_c > pd_g*p_g
                mask_prediction(i,j) = 1;
            end

            % MLE
            % Compute p(X|Y=cheetah, T=D1)
            mle_c = multi_gaussian(dct_vector, mu_mle_c, sigma_c);
            % Compute p(X|Y=grass, T=D1)
            mle_g = multi_gaussian(dct_vector, mu_mle_g, sigma_g);
            % BDR:
            if mle_c*p_c > mle_g*p_g
                mask_prediction_mle(i,j) = 1;
            end

            % PART C: MAP
            % Compute p(X|Y=cheetah, T=D1)
            map_c = multi_gaussian(dct_vector, mu1_c, sigma_c);
            % Compute p(X|Y=grass, T=D1)
            map_g = multi_gaussian(dct_vector, mu1_g, sigma_g);
            % BDR
            if map_c*p_c > map_g*p_g
                mask_prediction_map(i,j) = 1;
            end
        end
    end

    % pad mask_prediction to match original image shape
    mask_prediction = padarray(mask_prediction, [7, 7], 0, 'post');
    mask_prediction_mle = padarray(mask_prediction_mle, [7, 7], 0, 'post');
    mask_prediction_map = padarray(mask_prediction_map, [7, 7], 0, 'post');
    % plot predicted mask
    figure(1)
    colormap(gray(255));
    mask_prediction = imagesc(mask_prediction);
    figure(2)
    colormap(gray(255));
    mask_prediction_mle = imagesc(mask_prediction_mle);
    figure(3)
    colormap(gray(255));
    mask_prediction_map = imagesc(mask_prediction_map);
    
    % PROBABILITY OF ERROR
    % read cheetah_mask.bmp and reformat to range [0,1]
    y_truth = imread('../data/cheetah_mask.bmp');
    y_truth = im2double(y_truth);

    % error for pd
    prob_error = y_truth ~= mask_prediction.CData;
    errors_pd(k) = sum(prob_error,'all') / numel(prob_error);

    % error for mle
    prob_error_mle = y_truth ~= mask_prediction_mle.CData;
    errors_mle(k) = sum(prob_error_mle,'all') / numel(prob_error_mle);

    % error for map
    prob_error_map = y_truth ~= mask_prediction_map.CData;
    errors_map(k) = sum(prob_error_map,'all') / numel(prob_error_map);

end

fig = figure;
h(1) = semilogx(alpha, errors_pd, 'color','b', 'DisplayName', 'Predictive Dist');
hold on
h(2) = semilogx(alpha, errors_mle, 'color','r', 'DisplayName', 'MLE');
hold on
h(3) = semilogx(alpha, errors_map, 'color','g', 'DisplayName', 'MAP');
legend(h, 'Location', 'east');
title(curr_title);

end



