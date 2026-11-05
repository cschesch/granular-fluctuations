function [x] = theta2x(theta,N)

x    = nan(size(theta));
x(1) = log(theta(1) / (1-theta(1)));
x(2) = log(theta(2));
x(3) = log( (theta(3) /  5) / (1 - (theta(3) /  5)));
x(4) = log(theta(4));
x(5) = log( (theta(5) /  1) / (1 - (theta(5) /  1)));
x(6) = log( (theta(6) / 18) / (1 - (theta(6) / 18)));
x(7) = log( (theta(7) /  8) / (1 - (theta(7) /  8)));
x(8) = log( (theta(8) /  N) / (1 - (theta(8) /  N))) - 1e-4;

if length(x) == 9
    x(9) = log(theta(9));
end


