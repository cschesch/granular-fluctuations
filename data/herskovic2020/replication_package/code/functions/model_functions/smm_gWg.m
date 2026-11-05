function [out] = smm_gWg(x,W,N,nloop,data_moments,loc_exclude)

theta = x2theta(x,N);

disp(num2str([theta'; x'],'%.4f   '))

% Using transformed variables and Identify weighting matrix:
g_mean = nanmean( bsxfun(@plus, NormX(simulate_model(theta(1:7),N,theta(8),nloop,loc_exclude)), - data_moments),2);

out    = g_mean' * W * g_mean;

disp(out)
    
end