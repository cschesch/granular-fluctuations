function fit_surface(root, tag, directory)
% Fit the publication package's one-factor model to each Kalman-shock cell.
% The share uses its fitted covariance; fit_limit records optimizer warnings.
% ROOT locates the prepared source package. DIRECTORY defaults to its surface
% exports and can be overridden for a small isolated verification run.
if nargin < 3
    directory = fullfile(root, 'code', '.cache', 'derived', 'atalay2017', 'elasticity_surface');
end
package = fullfile(root, 'code', '.cache', 'derived', 'atalay2017', 'replication_package');
addpath(package);
cleanupPath = onCleanup(@() rmpath(package));
s = load(fullfile(directory, ['atalay_surface_epsD_' tag '.mat']));
out = fullfile(directory, ['one_factor_' tag '.csv']);
if isfile(out)
    tab = readtable(out);
else
    tab = table('Size', [0 5], 'VariableTypes', {'double','double','double','double','logical'}, ...
        'VariableNames', {'eps_D','eps_M','eps_Q','one_factor','fit_limit'});
end
opt = optimset('TolFun', 1e-14, 'TolX', 1e-14, 'MaxIter', 5000);
previousWarnings = warning;
cleanupWarnings = onCleanup(@() warning(previousWarnings));
warning('off', 'all');
for i = 1:length(s.eps_M_grid)
    for j = 1:length(s.eps_Q_grid)
        m = s.eps_M_grid(i); q = s.eps_Q_grid(j);
        if ~s.ok(i,j) || any(abs(tab.eps_M-m)<1e-10 & abs(tab.eps_Q-q)<1e-10)
            continue
        end
        x = squeeze(s.shocks_kf(i,j,:,:)); v = squeeze(s.V(i,j,:,:));
        lastwarn('');
        [l,p] = factoran_(x, 1, 'Nobs', size(x,1)-1, 'Delta', 1e-6, 'optimopts', opt);
        [~,id] = lastwarn;
        a = diag(v)'*p;
        tab(end+1,:) = {s.eps_D, m, q, a/(a+l'*v*l), contains(id,'Limit')};
    end
    checkpoint = [out '.partial'];
    writetable(tab, checkpoint, 'FileType', 'text');
    movefile(checkpoint, out, 'f');
    fprintf('%s: fitted row %d/%d, %d cells\n', tag, i, length(s.eps_M_grid), height(tab));
end
end
