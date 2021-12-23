% load data
TrainsampleDCT_BG = matfile('../data/TrainingSamplesDCT_8_new.mat').TrainsampleDCT_BG;
TrainsampleDCT_FG = matfile('../data/TrainingSamplesDCT_8_new.mat').TrainsampleDCT_FG;

% PART A:
% Compute number of samples
rows_fg = size(TrainsampleDCT_FG,1);
rows_bg = size(TrainsampleDCT_BG,1);

% Compute MLE for pY_cheetah and pY_grass
pY_cheetah_mle = rows_fg/(rows_fg+rows_bg);
pY_grass_mle = rows_bg/(rows_fg+rows_bg);

% PART B:
% Compute MLE of mean and covariance for p(x|cheetah) and p(x|grass)
mean_mle_cheetah = sum(TrainsampleDCT_FG) / rows_fg;
mean_mle_grass = sum(TrainsampleDCT_BG) / rows_bg;

sum1 = 0;
for i = 1:rows_fg
    sum1 = sum1 + (TrainsampleDCT_FG(i,:) - mean_mle_cheetah)' * (TrainsampleDCT_FG(i,:) - mean_mle_cheetah);
end
cov_mle_cheetah = sum1/rows_fg;

sum2 = 0;
for i = 1:rows_bg
    sum2 = sum2 + (TrainsampleDCT_BG(i,:) - mean_mle_grass)' * (TrainsampleDCT_BG(i,:) - mean_mle_grass);
end
cov_mle_grass = sum2/rows_bg;

% Compute p(Xk|cheetah)
p_Xk_cheetah = zeros(size(TrainsampleDCT_FG));
% iterate through cols
for k = 1:64
    % iterate through samples of Xk
    for n = 1:rows_fg
        p_Xk_cheetah(n,k) = (1/sqrt(2*pi*cov_mle_cheetah(k,k))) * exp(-((TrainsampleDCT_FG(n,k) - mean_mle_cheetah(k))^2) / (2*cov_mle_cheetah(k,k)));
    end
end

% Compute p(Xk|grass)
p_Xk_grass = zeros(size(TrainsampleDCT_BG));
% iterate through cols
for k = 1:64
    % iterate through samples of Xk
    for n = 1:rows_bg
        p_Xk_grass(n,k) = (1/sqrt(2*pi*cov_mle_grass(k,k))) * exp(-((TrainsampleDCT_BG(n,k) - mean_mle_grass(k))^2) / (2*cov_mle_grass(k,k)));
    end
end

% Plot 64 marginal densities P(Xk|Y)
figure(1)
for k = 1:64
    subplot(8,8,k)

    [sorted_x1, sorted_ind] = sort(TrainsampleDCT_FG(:,k));
    y1 = p_Xk_cheetah(:,k);
    sorted_y1 = y1(sorted_ind);
    plot(sorted_x1, sorted_y1)
    hold on
    [sorted_x0, sorted_ind] = sort(TrainsampleDCT_BG(:,k));
    y0 = p_Xk_grass(:,k);
    sorted_y0 = y0(sorted_ind);
    plot(sorted_x0, sorted_y0)
    
    title(sprintf('X %i',k));
end

% 8 best features:
features = [1,11,14,23,25,27,32,40];

% PART C:
% Using 8x8 block and sliding window, compute DCT vector per block
% and use vector as input into p(X|Y=i)

% read zigzag text file
zigzag = readmatrix('../data/Zig-Zag Pattern.txt');
zigzag = reshape(zigzag+1, [64,1]);
% read cheetah.bmp and reformat to range [0,1]
img = imread('../data/cheetah.bmp');
img = im2double(img);
% divide into 8x8 blocks, sliding window = 248*263 blocks
mask_prediction = zeros(248,263);
mask_prediction_8f = zeros(248,263);
for i = 1:(size(img,1)-7)
    for j = 1:(size(img,2)-7)
        block = img(i:i+7, j:j+7);
        dct_matrix = dct2(block);
        % transform dct matrix into vector, wrt zigzag indices
        dct_vector = reshape(dct_matrix, [64,1]);
        [sorted_row, sorted_ind] = sort(zigzag, 'ascend');
        dct_vector = dct_vector(sorted_ind);

        % 64 FEATURES:
        % compute p(x|y=cheetah)
        px_cheetah = (1/sqrt(((2*pi)^64)*det(cov_mle_cheetah))) * exp(-0.5*(dct_vector-mean_mle_cheetah')'*(inv(cov_mle_cheetah))*(dct_vector-mean_mle_cheetah'));
        % compute p(x|y=grass)
        px_grass = (1/sqrt(((2*pi)^64)*det(cov_mle_grass))) * exp(-0.5*(dct_vector-mean_mle_grass')'*(inv(cov_mle_grass))*(dct_vector-mean_mle_grass'));
        % BDR
        if px_cheetah*pY_cheetah_mle > px_grass*pY_grass_mle
            mask_prediction(i,j) = 1;
        end 

        % 8 FEATURES:
        dct_vector = dct_vector(features);
        % compute p(x|y=cheetah)
        px_cheetah = (1/sqrt(((2*pi)^64)*det(cov_mle_cheetah(features,features)))) * exp(-0.5*(dct_vector-mean_mle_cheetah(features)')'*(inv(cov_mle_cheetah(features,features)))*(dct_vector-mean_mle_cheetah(features)'));
        % compute p(x|y=grass)
        px_grass = (1/sqrt(((2*pi)^64)*det(cov_mle_grass(features,features)))) * exp(-0.5*(dct_vector-mean_mle_grass(features)')'*(inv(cov_mle_grass(features,features)))*(dct_vector-mean_mle_grass(features)'));
        % BDR
        if px_cheetah*pY_cheetah_mle > px_grass*pY_grass_mle
            mask_prediction_8f(i,j) = 1;
        end 
    end
end

% pad mask_prediction to match original image shape
mask_prediction = padarray(mask_prediction, [7, 7], 0, 'post');
mask_prediction_8f = padarray(mask_prediction_8f, [7, 7], 0, 'post');
% plot predicted mask
figure(2)
colormap(gray(255));
mask_prediction = imagesc(mask_prediction);
figure(3)
colormap(gray(255));
mask_prediction_8f = imagesc(mask_prediction_8f);

% Compute probability of error:
% read cheetah_mask.bmp and reformat to range [0,1]
y_truth = imread('../data/cheetah_mask.bmp');
y_truth = im2double(y_truth);

% error for 64 dimensions
error64 = y_truth ~= mask_prediction.CData;
error64 = sum(error64,'all') / numel(error64);

% error for 8 dimensions
error8 = y_truth ~= mask_prediction_8f.CData;
error8 = sum(error8,'all') / numel(error8);


















