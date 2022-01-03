% PART A:
% (5 times) randomly initialize parameters for BG
    % (5 times) randomly initialize parameters for FG 
        % call EM_gaussian(d=64, C=8, pi, mu, sigma, X=Trainsample_FG)
        % call EM_gaussian(d=64, C=8, pi, mu, sigma, X=Trainsample_BG) 
        % --> returns 64-dim final parameters
        % for each d:
            % use final parameters to classify cheetah image 
            % (marginalize parameters and dct vector according to dim)
            % compute prob of error and store: 5x10 matrix (5 curves, 10 dim)
    % plot error vs. dim for one BG with five FG initializations

% add path to access multi_gaussian fn
addpath('..')
% variables throughout part A:
d = 64;  % # of features to use in EM
%C = 8;
dimensions = [1,2,4,8,16,32,40,48,56,64];
% read zigzag text file
zigzag = readmatrix('../data/Zig-Zag Pattern.txt');
zigzag = reshape(zigzag+1, [64,1]);
% read cheetah.bmp and reformat to range [0,1]
img = imread('../data/cheetah.bmp');
img = im2double(img);
% read cheetah_mask.bmp and reformat to range [0,1]
y_truth = imread('../data/cheetah_mask.bmp');
y_truth = im2double(y_truth);
% class probabilites
p_FG = size(TrainsampleDCT_FG, 1) / (size(TrainsampleDCT_FG, 1) + size(TrainsampleDCT_BG, 1));
p_BG = size(TrainsampleDCT_BG, 1) / (size(TrainsampleDCT_FG, 1) + size(TrainsampleDCT_BG, 1));

% error per plot (1 row for each FG parameter set)
%error1 = zeros(5,length(dimensions));

% error for part B: 1 row for each C
components = [1,2,4,8,16,32];
errorC = zeros(length(components), length(dimensions));

% PART A:
% initialize BG parameters
%{
% set 1
pi_BG_0 = (1/C)*ones(C,1);
mu_BG_0 = ones(d,C);
sigma_BG_0 = repmat(diag(ones(d,1)), 1, C);
% set 2
pi_BG_0 = .1*ones(C,1);
pi_BG_0(1) = .3;
mu_BG_0 = zeros(d,C);
mu_BG_0(1,:) = 1;
sigma_BG_0 = repmat(diag(.1*ones(d,1)), 1, C);
% set 3
pi_BG_0 = .05*ones(C,1);
pi_BG_0([1,2,3,4]) = .2;
mu_BG_0 = zeros(d,C);
mu_BG_0(:,[2,3]) = -5;
mu_BG_0(:,[1,5]) = 2;
sigma_BG_0 = repmat(diag(2*ones(d,1)), 1, C);
% set 4
pi_BG_0 = .05*ones(C,1);
pi_BG_0([1,2]) = .35;
mu_BG_0 = zeros(d,C);
mu_BG_0(3,:) = 5;
sigma_BG_0 = repmat(diag(5*ones(d,1)), 1, C);
% set 5
pi_BG_0 = .05*ones(C,1);
pi_BG_0([1,C]) = .35;
mu_BG_0 = zeros(d,C);
mu_BG_0(:,1) = 2;
mu_BG_0(2,:) = 5;
sigma_BG_0 = repmat(diag(10*ones(d,1)), 1, C);

% 5 sets of initial FG parameters
% set 1
pi_FG_01 = (1/C)*ones(C,1);
mu_FG_01 = ones(d,C);
sigma_FG_01 = repmat(diag(ones(d,1)), 1, C);
% set 2
pi_FG_02 = .1*ones(C,1);
pi_FG_02(1) = .3;
mu_FG_02 = zeros(d,C);
mu_FG_02(1,:) = 1;
sigma_FG_02 = repmat(diag(.1*ones(d,1)), 1, C);
% set 3
pi_FG_03 = .05*ones(C,1);
pi_FG_03([1,2,3,4]) = .2;
mu_FG_03 = zeros(d,C);
mu_FG_03(:,[2,3]) = -5;
mu_FG_03(:,[1,5]) = 2;
sigma_FG_03 = repmat(diag(2*ones(d,1)), 1, C);
% set 4
pi_FG_04 = .05*ones(C,1);
pi_FG_04([1,2]) = .35;
mu_FG_04 = zeros(d,C);
mu_FG_04(3,:) = 5;
sigma_FG_04 = repmat(diag(5*ones(d,1)), 1, C);
% set 5
pi_FG_05 = .05*ones(C,1);
pi_FG_05([1,C]) = .35;
mu_FG_05 = zeros(d,C);
mu_FG_05(:,1) = 2;
mu_FG_05(2,:) = 5;
sigma_FG_05 = repmat(diag(10*ones(d,1)), 1, C);


pi_FG_0set = [pi_FG_01, pi_FG_02, pi_FG_03, pi_FG_04, pi_FG_05];
mu_FG_0set = [mu_FG_01, mu_FG_02, mu_FG_03, mu_FG_04, mu_FG_05];
sigma_FG_0set = [sigma_FG_01, sigma_FG_02, sigma_FG_03, sigma_FG_04, sigma_FG_05];
%}

% show progress of loop
bar = waitbar(0, 'In Progress');

for C_ind = 1:length(components)
    
    C = components(C_ind);
    waitbar(C_ind/length(components)) 

    % PART B: initialize one BG and FG set
    % BG
    r = rand(C,1);
    pi_BG_0 = r/sum(r);
    %pi_BG_0 = (1/C)*ones(C,1);
    %r2 = rand(d,C);
    mu_BG_0 = rand(d,C);
    %mu_BG_0 = zeros(d,C);
    %mu_BG_0(1,:) = 3;
    sigma_BG_0 = repmat(diag(.1*ones(d,1)), 1, C);
    % FG
    r = rand(C,1);
    pi_FG_0 = r/sum(r);
    %pi_FG_0 = (1/C)*ones(C,1);
    %r2 = rand(d,C);
    mu_FG_0 = rand(d,C);
    %mu_FG_0 = zeros(d,C);
    %mu_FG_0(1,:) = 1;
    sigma_FG_0 = repmat(diag(.1*ones(d,1)), 1, C);


%for FG_classifier = 1:5   

    % initialize FG parameters
    %pi_FG_0 = pi_FG_0set(:,FG_classifier);
    %mu_FG_0 = mu_FG_0set(:,((FG_classifier-1)*C+1):(C*FG_classifier));
    %sigma_FG_0 = sigma_FG_0set(:,((FG_classifier-1)*d*C+1):(d*C*FG_classifier));
    
    % perform EM and get updated parameters
    [pi_BG, mu_BG, sigma_BG] = EM_gaussian(d, C, pi_BG_0, mu_BG_0, sigma_BG_0, TrainsampleDCT_BG);
    [pi_FG, mu_FG, sigma_FG] = EM_gaussian(d, C, pi_FG_0, mu_FG_0, sigma_FG_0, TrainsampleDCT_FG);
    
    % classify cheetah image
    mask_prediction = zeros(248,263*length(dimensions));  % regular 248x263 mask_pred matrix but horizontally stacked for each dim
    for i = 1:(size(img,1)-7)
        for j = 1:(size(img,2)-7)
            block = img(i:i+7, j:j+7);
            dct_matrix = dct2(block);
            % transform dct matrix into vector, wrt zigzag indices
            dct_vector = reshape(dct_matrix, [64,1]);
            [sorted_row, sorted_ind] = sort(zigzag, 'ascend');
            dct_vector = dct_vector(sorted_ind);
    
            % for each dimension
            for dim_ind = 1:length(dimensions)
                dim = dimensions(dim_ind);
                p_xgivenFG = 0;
                p_xgivenBG = 0;
                for c = 1:C
                    % Compute p(X|Y=cheetah)
                    p_xgivenFG = p_xgivenFG + ( pi_FG(c) * multi_gaussian(...
                        dct_vector(1:dim), ...
                        mu_FG(1:dim,c),...
                        sigma_FG(1:dim,((c-1)*d+1):((c-1)*d+dim))) );
                    % Compute p(X|Y=grass)
                    p_xgivenBG = p_xgivenBG + ( pi_BG(c) * multi_gaussian(...
                        dct_vector(1:dim), ...
                        mu_BG(1:dim,c),...
                        sigma_BG(1:dim,((c-1)*d+1):((c-1)*d+dim))) );
                end
                % BDR
                if p_xgivenFG*p_FG > p_xgivenBG*p_BG
                    mask_prediction(i,263*(dim_ind-1)+j) = 1;
                end
            end
        end
    end
    
    % for each dim, compute error
    for dim_ind = 1:length(dimensions)
        dim = dimensions(dim_ind);
        % get corresponding mask_pred
        curr_mask_pred = mask_prediction(:,(263*(dim_ind-1)+1):(263*dim_ind));
        % pad to match original image shape
        curr_mask_pred = padarray(curr_mask_pred, [7, 7], 0, 'post');
        % compute prob of error
        error = y_truth ~= curr_mask_pred;
        error = sum(error,'all') / numel(error);
        %error1(FG_classifier, dim_ind) = error;
        errorC(C_ind, dim_ind) = error;
    end
    
    %error1(FG_classifier,:)
    errorC(C_ind,:)
end

close(bar)

%{
% plot error vs. log(dim) for all 5 FG initializations
fig = figure;
h(1) = semilogx(dimensions, error1(1,:), 'color','b');
hold on
h(2) = semilogx(dimensions, error1(2,:), 'color','r');
hold on
h(3) = semilogx(dimensions, error1(3,:), 'color','g');
hold on
h(4) = semilogx(dimensions, error1(4,:), 'color','c');
hold on
h(5) = semilogx(dimensions, error1(5,:), 'color','m');
title('BG classifier 5, against 5 FG classifiers');
%}

% plot error vs. log(dim) for all C's
fig = figure;
colors = ['b','r','g','c','m', 'k'];
disp_name = ['1','2','4','8','16','32'];
for comp = 1:length(components)
    h(comp) = semilogx(dimensions, errorC(comp,:), 'color',colors(comp), 'DisplayName', disp_name(comp));
    hold on
end
legend(h, 'Location', 'east');
title("Prob of Error vs. Dim for various C's");
