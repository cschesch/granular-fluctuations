% Export shocks and propagation matrices for the three calibration surfaces.
% The original filter's first 395 lines recover shocks before factor fitting.
% A benchmark check below compares this unchanged prefix with filter_full.m.
% One MAT file per eps_D is checkpointed after each eps_M row and resumed.
% Optional ATALAY_REPO_ROOT, ATALAY_SURFACE_OUT and ATALAY_SURFACE_EPS{D,M,Q}
% override the root, destination and grids, including small verification grids.

if exist('ATALAY_REPO_ROOT', 'var')
    surf_root = ATALAY_REPO_ROOT;
else
    surf_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
end
surf_pkg = fullfile(surf_root, 'code', '.cache', 'derived', 'atalay2017', 'replication_package');
if exist('ATALAY_SURFACE_OUT', 'var')
    surf_out = ATALAY_SURFACE_OUT;
else
    surf_out = fullfile(surf_root, 'code', '.cache', 'derived', 'atalay2017', 'elasticity_surface');
end
if ~exist(surf_out, 'dir'); mkdir(surf_out); end
if exist('ATALAY_SURFACE_EPSD', 'var')
    surf_eps_D = ATALAY_SURFACE_EPSD(:)';
else
    surf_eps_D = [2/3, 0.999, 4/3];
end
if exist('ATALAY_SURFACE_EPSM', 'var')
    surf_eps_M = ATALAY_SURFACE_EPSM(:)';
else
    surf_eps_M = [0.005, 0.01, 0.02, 0.03, 0.04:0.03:1.06];
end
if exist('ATALAY_SURFACE_EPSQ', 'var')
    surf_eps_Q = ATALAY_SURFACE_EPSQ(:)';
else
    surf_eps_Q = [0.03:0.03:1.50, 0.999];
end
maxNumCompThreads(1);
% The source model is singular at unity. Preserve exact fractional eps_D values.
unity = @(v) v .* (abs(v - 1) > 1e-9) + 0.999 * (abs(v - 1) <= 1e-9);
surf_eps_D = unity(sort(unique(surf_eps_D)));
surf_eps_M = unity(sort(unique(round(surf_eps_M, 6))));
surf_eps_Q = unity(sort(unique(round(surf_eps_Q, 6))));
nD = numel(surf_eps_D); nM = numel(surf_eps_M); nQ = numel(surf_eps_Q);

CUT_LINE = 395;
surf_lines = strsplit(fileread(fullfile(surf_pkg, 'filter_full.m')), sprintf('\n'));
assert(contains(surf_lines{CUT_LINE}, 'filtered_shocks_D_kf=productivity_shocks_kf_D'));
assert(contains(surf_lines{CUT_LINE + 1}, 'optionsFactoran'));
surf_cut_dir = tempname;
mkdir(surf_cut_dir);
surf_cut_fid = fopen(fullfile(surf_cut_dir, 'atalay_filter_shocks_only.m'), 'w');
fprintf(surf_cut_fid, '%s\n', surf_lines{1:CUT_LINE});
fclose(surf_cut_fid);
addpath(surf_cut_dir); rehash path;
% Call the prefix by name so its relative csvread calls use the source directory.
surf_old_dir = pwd;
cleanupDir = onCleanup(@() cd(surf_old_dir));
cleanupCut = onCleanup(@() rmdir(surf_cut_dir, 's'));
cleanupPath = onCleanup(@() rmpath(surf_cut_dir));
cd(surf_pkg);

results_table = zeros(nM * nQ + 1, 24);
historical_decomposition = zeros(49, 6);
counter = 0;
cov_ind_output_A = 0; cov_ind_output_B = 0; cov_ind_output_D = 0;
cov_model_output_A = 0; cov_model_output_B = 0; cov_model_output_D = 0;
parameters = [0.999, 0.999, 0.100, 0.999, 2, 0, 1, 4];
filter_full;
chk_A = real(filtered_shocks_A); chk_kf = real(filtered_shocks_A_kf);
chk_V = filter_matrix_2_row_2_A' * filter_matrix_2_row_2_A;
chk_S = DR_State_State; chk_E = DR_State_Exo;
chk_r = max(abs(eig(final_filter_3_A))); chk_m = min(abs(eig(filter_matrix_1_A)));
clear filtered_shocks_A filtered_shocks_A_kf
counter = 0;
atalay_filter_shocks_only;
chk_gap = max([max(abs(chk_A(:) - reshape(real(filtered_shocks_A), [], 1))), ...
               max(abs(chk_kf(:) - reshape(real(filtered_shocks_A_kf), [], 1))), ...
               max(abs(chk_V(:) - reshape(filter_matrix_2_row_2_A' * filter_matrix_2_row_2_A, [], 1))), ...
               max(abs(chk_S(:) - DR_State_State(:))), max(abs(chk_E(:) - DR_State_Exo(:))), ...
               abs(chk_r - max(abs(eig(final_filter_3_A)))), ...
               abs(chk_m - min(abs(eig(filter_matrix_1_A))))]);
assert(chk_gap == 0, 'Filter prefix differs from filter_full.m by %.3e.', chk_gap);

T = 49; N = 30;
for iD = 1:nD
    slab_file = fullfile(surf_out, sprintf('atalay_surface_epsD_%03d.mat', round(1000 * surf_eps_D(iD))));
    if isfile(slab_file)
        saved = load(slab_file);
        assert(saved.eps_D == surf_eps_D(iD) && isequal(saved.eps_M_grid, surf_eps_M) ...
            && isequal(saved.eps_Q_grid, surf_eps_Q), 'Existing slab uses a different grid: %s', slab_file);
        shocks = saved.shocks; shocks_kf = saved.shocks_kf; V_all = saved.V;
        ok = saved.ok; rho = saved.rho; min_eig = saved.min_eig;
        years = saved.years; messages = saved.messages; rows_done = saved.rows_done;
    else
        shocks = nan(nM, nQ, T, N); shocks_kf = nan(nM, nQ, T, N);
        V_all = nan(nM, nQ, N, N);
        ok = false(nM, nQ); rho = nan(nM, nQ); min_eig = nan(nM, nQ);
        years = []; messages = repmat({''}, nM, nQ); rows_done = 0;
    end
    for iM = rows_done + 1:nM
        for iQ = 1:nQ
            clear filtered_shocks_A filtered_shocks_A_kf V data_time
            counter = 0;
            parameters = [surf_eps_D(iD), surf_eps_Q(iQ), surf_eps_M(iM), 0.999, 2, 0, 1, 4];
            try
                atalay_filter_shocks_only;
                % Impact response plus horizons 0--25, as in filter_full.m.
                V = filter_matrix_2_row_2_A' * filter_matrix_2_row_2_A;
                power = eye(industries);
                for h = 0:25
                    ir = filter_matrix_1_row_2 * power * DR_State_Exo(:, 1:industries);
                    V = V + ir' * ir;
                    power = power * DR_State_State;
                end
                V = 0.5 * (V + V');
                assert(isequal(size(filtered_shocks_A), [T N]) && isequal(size(V), [N N]), ...
                    'Unexpected shock or propagation dimensions.');
                shocks(iM, iQ, :, :) = real(filtered_shocks_A);
                shocks_kf(iM, iQ, :, :) = real(filtered_shocks_A_kf);
                V_all(iM, iQ, :, :) = V;
                ok(iM, iQ) = true;
                rho(iM, iQ) = max(abs(eig(final_filter_3_A)));
                min_eig(iM, iQ) = min(abs(eig(filter_matrix_1_A)));
                if isempty(years)
                    years = data_time((burn_in + 1):end, industries + 1);
                end
            catch ME
                messages{iM, iQ} = ME.message;
            end
        end
        payload = struct('eps_D', surf_eps_D(iD), 'eps_M_grid', surf_eps_M, ...
            'eps_Q_grid', surf_eps_Q, 'shocks', shocks, 'shocks_kf', shocks_kf, ...
            'V', V_all, 'ok', ok, 'rho', rho, 'min_eig', min_eig, 'years', years, ...
            'rows_done', iM, 'messages', {messages});
        checkpoint = [slab_file '.partial.mat'];
        save(checkpoint, '-struct', 'payload', '-v7');
        movefile(checkpoint, slab_file, 'f');
        fprintf('eps_D=%.6g: row %d/%d, %d/%d cells solved\n', ...
            surf_eps_D(iD), iM, nM, sum(ok(:)), nM * nQ);
    end
end
clear cleanupPath cleanupCut cleanupDir
