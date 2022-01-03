function [pi_n1, mu_n1, sigma_n1] = EM_gaussian(d, C, pi, mu, sigma, X)
% Perform EM algorithm on Gaussian mixture model with 
%   - d: dimension of sample X(i,:) i.e. # of features to keep from dct vec
%   - C: number of components in Gaussian mixture model
%   - initial parameters: 
%       - pi: Cx1 vector representing component probability
%       - mu: dxC matrix where each col represents mu for that component
%       - sigma: dx(d*C) matrix where each dxd sub-matrix represents sigma
%                for that component
%   - X: matrix of training samples (TrainsampleDCT_BG or FG) 

num_obs = size(X,1);
pi_n = pi; mu_n = mu; sigma_n = sigma;
pi_n1 = zeros(C,1); mu_n1 = zeros(d,C); sigma_n1 = zeros(d,d*C);


for iterate = 1:50
    
    % E-step
    h = zeros(num_obs, C);
    for i = 1:num_obs
        denom = 0;
        for k = 1:C
            denom = denom + ...
                (multi_gaussian(X(i,1:d)', mu_n(:,k), sigma_n(:,((k-1)*d+1):(k*d))) ...
                * pi_n(k) );
        end
        for j = 1:C
            h(i,j) = multi_gaussian(X(i,1:d)', mu_n(:,j), sigma_n(:,((j-1)*d+1):(j*d))) ...
            * pi_n(j) / denom;
        end
    end
    
    
    % M-step
    for j = 1:C
        pi_n1(j) = mean(h(:,j));
        mu_n1(:,j) = X(:,1:d)' * h(:,j) / sum(h(:,j));
        temp_sigma = zeros(d,d);
        for i = 1:num_obs
            temp_sigma = temp_sigma + h(i,j)*(X(i,1:d)'-mu_n(:,j))*(X(i,1:d)'-mu_n(:,j))';
        end
        temp_sigma = temp_sigma / sum(h(:,j));
        % sigma must be diagonal
        temp_sigma = temp_sigma .* eye(size(temp_sigma));
        temp_sigma(isnan(temp_sigma)) = 0;
        % ensure no diag elem < .01
        for row = 1:d
            temp_sigma(row,row) = max([.01, temp_sigma(row,row)]);
        end   
        % update sigma parameter
        sigma_n1(:,((j-1)*d+1):(j*d)) = temp_sigma;
        
    end

    % move new psi to old psi for next iteration
    if iterate ~= 50
        pi_n = pi_n1;
        mu_n = mu_n1;
        sigma_n = sigma_n1;
    end

    % Continue EM iterations or stop?
    %{
    % compute Q(psi^(n+1);psi^(n)) and Q(psi^(n);psi^(n))
    Q_n1_n = 0;
    Q_n_n = 0;
    for i = 1:num_obs
        for j = 1:C
            Q_n1_n = Q_n1_n + h(i,j)*log( ...
                multi_gaussian(X(i,1:d)', mu_n1(:,j), sigma_n1(:,((j-1)*d+1):(j*d))) ...
                * pi_n1(j) );
            Q_n_n = Q_n_n + h(i,j)*log( ...
                multi_gaussian(X(i,1:d)', mu_n(:,j), sigma_n(:,((j-1)*d+1):(j*d))) ...
                * pi_n(j) );
        end
    end

    % if Q's are close enough, then stop EM iterations
    if abs(Q_n1_n-Q_n_n) < .01
        break
    else
        % otherwise continue EM and set new psi to old psi
        pi_n = pi_n1;
        mu_n = mu_n1;
        sigma_n = sigma_n1;
    end
    %}

end


