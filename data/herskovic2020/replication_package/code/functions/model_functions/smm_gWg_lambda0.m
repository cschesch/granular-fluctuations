function [out] = smm_gWg_lambda0(x,W,N,nloop,data_moments,loc_exclude)
x = [x(1:2); 1e-8; x(3:end)];
theta = x2theta(x,N);
theta(3) = 0;
disp(num2str([theta'; x'],'%.4f   '))

% Using transformed variables and Identify weighting matrix:
g_mean = nanmean( bsxfun(@plus, NormX(simulate_model_lambda0(theta([1:2, 4:7]),N,theta(8),nloop,loc_exclude)), - data_moments),2);

out    = g_mean' * W * g_mean;

disp(out)
    
end