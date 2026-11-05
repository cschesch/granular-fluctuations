function [out] = smm_gWg_gamma0(x,W,N,nloop,data_moments,loc_exclude)
x = [1e-8;x];
theta = x2theta(x,N);
theta(1) = 0;
disp(num2str([theta'; x'],'%.4f   '))

% Using transformed variables and Identify weighting matrix:
g_mean = nanmean( bsxfun(@plus, NormX(simulate_model_gamma0(theta(2:7),N,theta(8),nloop,loc_exclude)), - data_moments),2);

out    = g_mean' * W * g_mean;

disp(out)
    
end