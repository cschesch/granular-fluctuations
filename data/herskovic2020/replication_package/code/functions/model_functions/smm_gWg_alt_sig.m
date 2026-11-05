function [out] = smm_gWg_alt_sig(x,W,N,nloop,data_moments,loc_exclude,sigtype)

theta = x2theta(x,N);

disp(num2str([theta'; x'],'%.4f   '))

% Using transformed variables and Identify weighting matrix:
g_mean = nanmean( bsxfun(@plus, NormX(simulate_model_alt_sig(theta(1:7),N,theta(8),nloop,loc_exclude,sigtype)), - data_moments),2);

out    = g_mean' * W * g_mean;

disp(out)
    
end