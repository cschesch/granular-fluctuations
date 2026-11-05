function [theta] = x2theta(x,N)

theta    = nan(size(x));
theta(1) = exp(x(1)) ./ (1 + exp(x(1))); % gmmma -- between 0 and 1
theta(2) = exp(x(2)); % sig0 -- positive
theta(3) = 5 * exp(x(3)) ./ (1 + exp(x(3))); % sig1 -- positive, between 0 and 5
theta(4) = exp(x(4)); % Z -- positive
theta(5) = 1  * exp(x(5)) ./ (1 + exp(x(5))); % psi -- positive, between 0 and 1
theta(6) = 18 * exp(x(6)) ./ (1 + exp(x(6))); % mu_So -- between 0 and 18
theta(7) = 8 * exp(x(7)) ./ (1 + exp(x(7))); % sigma_So -- positive, between 0 and 8
theta(8) = ceil(N * exp(x(8)) ./ (1 + exp(x(8)))); % Ntop -- positive integer, between 1 and N


if length(theta) == 9
    theta(9) = exp(x(9));
end