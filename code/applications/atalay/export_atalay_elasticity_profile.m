% Export the Atalay (2017) primitive-shock inversion over a grid of one
% elasticity of substitution, holding the others at Atalay's Figure 3 values.
%
% Sweeping eps_M fixes (eps_D, eps_Q) = (1, 1); sweeping eps_Q fixes
% (eps_D, eps_M) = (1, 1); unity is the package's own 0.999 stand-in, since the
% model is singular at exactly one.  These are the two panels of his Figure 3.
% For each grid point this driver calls the DEPOSITED, UNMODIFIED filter_full.m
% and saves the objects our downstream analysis needs:
%
%   filtered_shocks_A     the 49-by-30 raw recursive Equation-(11) filter
%                         productivity innovations for 1965--2013, exactly as
%                         they exist immediately before factoran_ is applied,
%   filtered_shocks_A_kf  the Kalman-filter counterpart, which filter_full.m
%                         computes unconditionally and which the package falls
%                         back on where the Equation-(11) filter fails its own
%                         stability/conditioning screens, and
%   V                     the symmetric matrix with aggregate-output variance
%                         trace(V * Sigma_omega), built from the impact response
%                         plus horizons 0--25 exactly as in filter_full.m.
%
% Nothing in the deposited replication package is edited: filter_full.m is a
% script, so this driver only supplies the workspace that filter_master.m
% would have supplied, and reads its outputs afterwards.
%
% Usage (from any directory):
%   matlab -batch "ATALAY_SWEEP='eps_M'; run('<repo>/code/applications/atalay/export_atalay_elasticity_profile.m')"
%   matlab -batch "ATALAY_SWEEP='eps_Q'; run('<repo>/code/applications/atalay/export_atalay_elasticity_profile.m')"
%
% ATALAY_REPO_ROOT, ATALAY_PROFILE_OUT and ATALAY_PROFILE_GRID may also be set
% in the caller's workspace to override the repository root, the export
% directory and the grid (the test harness uses the last two).

if exist('ATALAY_SWEEP', 'var')
    prof_sweep = ATALAY_SWEEP;
else
    prof_sweep = 'eps_M';
end
% Position of the swept elasticity in filter_full.m's parameter vector
% [eps_D, eps_Q, eps_M, eps_X, eps_LS, durables, data_set, burn_in].
switch prof_sweep
    case 'eps_M'
        prof_slot = 3;
    case 'eps_Q'
        prof_slot = 2;
    otherwise
        error('ATALAY_SWEEP must be ''eps_M'' or ''eps_Q'', got ''%s''.', prof_sweep);
end

if exist('ATALAY_REPO_ROOT', 'var')
    prof_root = ATALAY_REPO_ROOT;
else
    prof_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
end
prof_pkg = fullfile(prof_root, 'code', '.cache', 'derived', 'atalay2017', 'replication_package');
if exist('ATALAY_PROFILE_OUT', 'var')
    prof_out = ATALAY_PROFILE_OUT;
else
    prof_out = fullfile(prof_root, 'code', '.cache', 'derived', 'atalay2017', ...
                        'elasticity_profile', prof_sweep);
end
if ~exist(prof_out, 'dir'); mkdir(prof_out); end

% Grids.  Unity is the package's 0.999 stand-in throughout.  The eps_M grid is
% dense at the low end, where Atalay's own estimates sit and where the inversion
% is most sensitive; the eps_Q grid covers his Figure 3 panel B range.
if exist('ATALAY_PROFILE_GRID', 'var')
    prof_grid = ATALAY_PROFILE_GRID(:)';
elseif strcmp(prof_sweep, 'eps_M')
    prof_grid = [0.001, 0.002, 0.005, 0.0075, ...  % the near-Leontief tail
                 0.01:0.01:0.30, 0.32:0.02:0.50, 0.52:0.02:0.70, ...
                 0.75:0.05:0.95, 0.999, 1.05:0.05:1.50, ...
                 0.052:0.002:0.062];  % pins the explosive-filter boundary
else
    prof_grid = [0.05:0.05:0.95, 0.999, 1.05:0.05:1.55];
end
prof_grid = sort(unique(round(prof_grid, 6)));

% The workspace filter_master.m sets up before its first filter_full call.
results_table = zeros(numel(prof_grid), 24);
historical_decomposition = zeros(49, 6);
counter = 0;
cov_ind_output_A = 0; cov_ind_output_B = 0; cov_ind_output_D = 0;
cov_model_output_A = 0; cov_model_output_B = 0; cov_model_output_D = 0;

prof_old_dir = pwd;
cleanupObj = onCleanup(@() cd(prof_old_dir));
cd(prof_pkg);

prof_status = cell(numel(prof_grid), 1);
prof_files  = cell(numel(prof_grid), 1);

for prof_k = 1:numel(prof_grid)
    % Never let a failed grid point be read as the previous point's output.
    % filter_full.m only assigns psi_A/lambda_A/sigma_Matrix_A inside its own
    % stability guard, and reads the cov_* accumulators unconditionally, so
    % reset both groups the way filter_master.m initialises them.
    clear filtered_shocks_A filtered_shocks_A_kf V years
    psi_A = nan(30, 1); lambda_A = nan(30, 1); sigma_Matrix_A = nan(30);
    cov_ind_output_A = 0; cov_ind_output_B = 0; cov_ind_output_D = 0;
    cov_model_output_A = 0; cov_model_output_B = 0; cov_model_output_D = 0;
    cov_ind_output_A_kf = 0; cov_model_output_A_kf = 0;
    parameters = [0.999, 0.999, 0.999, 0.999, 2, 0, 1, 4];
    parameters(prof_slot) = prof_grid(prof_k);
    prof_t0 = tic;
    prof_ok = true;
    prof_msg = '';
    try
        filter_full;
    catch prof_ME
        prof_ok = false;
        prof_msg = prof_ME.message;
    end

    if prof_ok
        % V: aggregate-output variance matrix, impact response plus horizons
        % 0--25, exactly the sum filter_full.m accumulates for
        % cov_model_output_A.
        V = filter_matrix_2_row_2_A' * filter_matrix_2_row_2_A;
        prof_power = eye(industries);
        for prof_h = 0:25
            prof_ir = filter_matrix_1_row_2 * prof_power * DR_State_Exo(:, 1:industries);
            V = V + prof_ir' * prof_ir;
            prof_power = prof_power * DR_State_State;
        end
        V = 0.5 * (V + V');
        years = data_time((burn_in + 1):end, industries + 1);

        prof_file = fullfile(prof_out, sprintf('atalay_%s_%03d.mat', prof_sweep, prof_k));
        prof_payload = struct( ...
            'sweep', prof_sweep, ...
            'value', prof_grid(prof_k), ...
            'parameters', parameters, ...
            'shocks', real(filtered_shocks_A), ...
            'shocks_kf', real(filtered_shocks_A_kf), ...
            'V', V, ...
            'years', years, ...
            'psi_A', psi_A, ...
            'lambda_A', lambda_A, ...
            'sigma_Matrix_A', sigma_Matrix_A, ...
            'one_factor_share', cov_ind_output_A / cov_model_output_A, ...
            'mean_corr', mean(mean(corr(real(filtered_shocks_A)))), ...
            'max_abs_eig_final_filter_3_A', max(abs(eig(final_filter_3_A))), ...
            'min_abs_eig_filter_matrix_1_A', min(abs(eig(filter_matrix_1_A))), ...
            'unstable_eigs', UnstableEigs, ...
            'max_abs_imag_shocks', max(max(abs(imag(filtered_shocks_A)))), ...
            'max_abs_imag_shocks_kf', max(max(abs(imag(filtered_shocks_A_kf)))), ...
            'one_factor_share_kf', cov_ind_output_A_kf / cov_model_output_A_kf, ...
            'mean_corr_kf', mean(mean(corr(real(filtered_shocks_A_kf)))), ...
            'results_row', results_table(counter, :));
        save(prof_file, '-struct', 'prof_payload', '-v7');
        prof_files{prof_k} = prof_file;
        prof_status{prof_k} = 'ok';
    else
        prof_files{prof_k} = '';
        prof_status{prof_k} = prof_msg;
    end

    fprintf('[%3d/%3d] %s=%.4f  %-6s  %.1fs  %s\n', prof_k, numel(prof_grid), prof_sweep, ...
        prof_grid(prof_k), prof_status{prof_k}(1:min(6, numel(prof_status{prof_k}))), ...
        toc(prof_t0), prof_files{prof_k});
end

% Grid index, so the Python side never has to guess the file order.
prof_fid = fopen(fullfile(prof_out, 'grid_index.csv'), 'w');
fprintf(prof_fid, 'index,sweep,value,status,file\n');
for prof_k = 1:numel(prof_grid)
    [~, prof_base, prof_ext] = fileparts(prof_files{prof_k});
    fprintf(prof_fid, '%d,%s,%.6f,"%s",%s\n', prof_k, prof_sweep, prof_grid(prof_k), ...
        strrep(prof_status{prof_k}, '"', ''''), [prof_base prof_ext]);
end
fclose(prof_fid);

fprintf('wrote %d %s grid points to %s\n', sum(strcmp(prof_status, 'ok')), prof_sweep, prof_out);
