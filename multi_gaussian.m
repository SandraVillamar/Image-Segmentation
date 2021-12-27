function [output] = multi_gaussian(x, mu, sigma)
% Compute multivariate Gaussian(x, mu, sigma)
d = length(x);
output = (1/sqrt(((2*pi)^d)*det(sigma))) ...
    * exp(-0.5*(x-mu)'*(inv(sigma))*(x-mu));

end
