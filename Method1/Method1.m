%{ 
STEP 1: reduce vector to scalar
- for each row of TrainsampleDCT_BG and TrainsampleDCT_FG, find 2nd largest
value and record the index position in X_bg and X_fg

STEP 2:
- create histogram for X_bg and X_fg (64 values so 64 bins)

STEP 3:
- estimate priors based on # of samples from each category

STEP 4:
- separate cheetah image into blocks of 8x8 pixels
    - use sliding window that moves by 1 pixel each step
- apply dct2() function to each block --> 64x1 dct vector per block
- transform dct vector into one value: index (by zigzag scan) of 2nd
largest value
- store in X for each block

STEP 5:
- use BDR to find class Y for each block for '0-1 loss'
    - g*(x) = argmax_i (P(X=x|Y=i)*P(Y=i) 
- store classes in array A

STEP 6:
- create picture of array

STEP 7:
- read cheetah_mask image
- compute prob of error between my output and cheetah_mask.bmp
%}

% load data
TrainsampleDCT_BG = matfile('../data/TrainingSamplesDCT_8.mat').TrainsampleDCT_BG;
TrainsampleDCT_FG = matfile('../data/TrainingSamplesDCT_8.mat').TrainsampleDCT_FG;

% STEP 1
rows_bg = size(TrainsampleDCT_BG,1);
rows_fg = size(TrainsampleDCT_FG,1);
X_bg = zeros(rows_bg,1);
X_fg = zeros(rows_fg,1);
for i = 1:rows_bg
    % find index of 2nd largest value, add to X_bg
    [sorted_row, sorted_ind] = sort(TrainsampleDCT_BG(i,:), 'descend');
    X_bg(i) = sorted_ind(2);
end
for i = 1:rows_fg
    % find index of 2nd largest value, add to X_fg
    [sorted_row, sorted_ind] = sort(TrainsampleDCT_FG(i,:), 'descend');
    X_fg(i) = sorted_ind(2);
end

% STEP 2
xbins = 1:65;
figure(1)
hist_grass = histogram(X_bg, xbins, 'Normalization', 'probability');
px_grass = hist_grass.Values;  % store counts of each bin
hold on
hist_cheetah = histogram(X_fg, xbins, 'Normalization', 'probability');
px_cheetah = hist_cheetah.Values;  % store counts of each bin
legend('P(x|grass)', 'P(x|cheetah)')
hold off

% STEP 3 
pY_cheetah = rows_fg/(rows_fg+rows_bg);
pY_grass = rows_bg/(rows_fg+rows_bg);

% STEP 4
% read zigzag text file
zigzag = readmatrix('../data/Zig-Zag Pattern.txt');
zigzag = reshape(zigzag+1, [64,1]);
% read cheetah.bmp and reformat to range [0,1]
img = imread('../data/cheetah.bmp');
img = im2double(img);
% divide into 8x8 blocks, sliding window = 248*263 blocks
X = zeros(248*263,1);
for i = 1:(size(img,1)-7)
    for j = 1:(size(img,2)-7)
        block = img(i:i+7, j:j+7);
        dct_matrix = abs(dct2(block));
        % transform dct matrix into vector
        dct_vector = reshape(dct_matrix, [64,1]);
        % find index of 2nd largest value
        [sorted_row, sorted_ind] = sort(dct_vector, 'descend');
        % get ind according to zigzag pattern
        ind = zigzag(sorted_ind(2));
        % store in X
        X( (i-1)*263 + j ) = ind;
    end
end

% STEP 5
% choose Y=1 (cheetah) if P(x|cheetah)*P(cheetah) > P(x|grass)*P(grass)
A = zeros(size(X));
for i = 1:size(A,1)
    % BDR for 0-1 loss
    if px_cheetah(X(i))*pY_cheetah > px_grass(X(i))*pY_grass
        A(i) = 1;
    end
end

% STEP 6
% reshape array A into pixels of image
A = reshape(A,[263,248])';
% pad A to match original image shape
A = padarray(A, [7, 7], 0, 'post');
% plot predicted mask
figure(2)
colormap(gray(255));
mask_prediction = imagesc(A);

% STEP 7
% read cheetah_mask.bmp and reformat to range [0,1]
y_truth = imread('../data/cheetah_mask.bmp');
y_truth = im2double(y_truth);

error = y_truth ~= mask_prediction.CData;
error = sum(error,'all') / numel(error);

