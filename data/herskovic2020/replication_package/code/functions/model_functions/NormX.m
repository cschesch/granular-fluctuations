function [out] = NormX(x)
% x is a 54 by 1 vector of moments
[M, K] = size(x);

if M ~=54
    disp('NormX error: dimensions do not match')
else
    out = nan(28,K);
    
    out(1,:)  = x(1,:); % avg log size
    out(2,:)  = x(2,:); % std log size
    out(3,:)  = x(6,:) - x(4,:); % 50 minus 10 log size percentile
    out(4,:)  = x(8,:) - x(6,:); % 90 minus 50 log size percentile
    out(5,:)  = x(10,:); % avg log variance
    out(6,:)  = x(11,:); % std log variance
    out(7,:)  = x(15,:) - x(13,:); % 50 minus 10 log variance percentile
    out(8,:)  = x(17,:) - x(15,:); % 90 minus 50 log variance percentile
    out(9,:)  = x(19,:); % correlation between log size and log variance
    out(10,:) = x(23,:); % 50 percentile of Nout
    out(11,:) = log(x(23,:)) - log(x(21,:)); % 50 minus 10 percentile of Nout
    out(12,:) = log(x(25,:)) - log(x(23,:)); % 90 minus 10 percentile of Nout
    out(13,:) = x(30,:); % 50 percentile of Hout
    out(14,:) = log(x(30,:)) - log(x(28,:)); % 50 minus 10 percentile of Hout
    out(15,:) = log(x(32,:)) - log(x(30,:)); % 90 minus 10 percentile of Hout
    out(16,:) = x(34,:); % correlation between log Nout and log Size
    out(17,:) = x(35,:); % correlation between log Hout and log Size
    out(18,:) = x(36,:); % correlation between log Hout and log Variance
    out(19,:) = x(40,:); % 50 percentile of Nin
    out(20,:) = log(x(40,:)) - log(x(38,:)); % 50 minus 10 percentile of Nin
    out(21,:) = log(x(42,:)) - log(x(40,:)); % 90 minus 10 percentile of Nin
    out(22,:) = x(47,:); % 50 percentile of Hin
    out(23,:) = log(x(47,:)) - log(x(45,:)); % 50 minus 10 percentile of Hin
    out(24,:) = log(x(49,:)) - log(x(47,:)); % 90 minus 10 percentile of Hin
    out(25,:) = x(51,:); % correlation between log Nin and log Size
    out(26,:) = x(52,:); % correlation between log Hin and log Size
    out(27,:) = x(53,:); % correlation between log Hin and log Variance
    out(28,:) = log(x(54,:));
end    