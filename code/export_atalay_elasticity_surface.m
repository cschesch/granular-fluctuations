% Export the Atalay (2017) primitive-shock inversion over a THREE-dimensional
% grid of his production elasticities (eps_D, eps_M, eps_Q).
%
% ``export_atalay_elasticity_profile.m`` sweeps one elasticity at a time, which
% is Atalay's own Figure 3.  This driver instead fixes eps_D at each of a small
% number of values -- one output slab per value -- and, within each slab, walks
% a full (eps_M, eps_Q) rectangle, so the downstream figure can draw the
% granularity ceiling as a surface over the two elasticities Atalay's own
% estimation actually speaks to, one panel per value of the third.
%
% Unity is the package's own 0.999 stand-in, since the model is singular at
% exactly one; any grid value within 1e-9 of 1 is replaced by it.
%
% THE FILTER.  filter_full.m does two things: it inverts the model to recover
% the primitive innovations, and it then runs several maximum-likelihood factor
% analyses (its own TolFun = 1e-14, MaxIter = 5000) to produce Atalay's
% one-factor decomposition.  Only the first half produces anything this grid
% needs, and the second half costs 20 to 4000 times as much -- it converges in
% milliseconds at his own calibration and hits the iteration cap away from it.
% So this driver runs the FIRST CUT_LINE LINES OF THE DEPOSITED FILE, VERBATIM.
% It does not edit them and does not retype them: it reads filter_full.m at run
% time, checks that the cut still falls exactly between the last line that
% assigns a filtered-shock series and the first line of the factor-analysis
% block, writes those lines out unchanged, and runs them as a script in this
% workspace (by name, not via ``run``, which would change the working directory
% the deposited csvread calls depend on).  Before touching the grid it also
% runs the FULL filter_full.m once at Atalay's benchmark and asserts that the
% truncated copy reproduces its shocks, its Kalman shocks and every propagation
% matrix bit for bit.  The Python side re-checks the benchmark cell against the
% separately deposited benchmark export.
%
% What the cut gives up is ``one_factor_share``, Atalay's own decomposition,
% which the one-dimensional profile export still computes with the full file.
%
% Per slab (one .mat file, arrays rather than one file per point, so the whole
% 3-D grid is a handful of files rather than thousands):
%
%   eps_D                scalar, the slab's value
%   eps_M_grid, eps_Q_grid   the two axes, after the unity substitution
%   shocks               nM x nQ x T x N, Equation-(11) recursion innovations
%   shocks_kf            nM x nQ x T x N, Kalman-filter counterpart
%   V                    nM x nQ x N x N, aggregate-output variance matrix
%   ok                   nM x nQ logical, filter_full returned without error
%   rho                  nM x nQ, max|eig| of final_filter_3_A (Atalay's own
%                        explosiveness screen: the recursion is used only if <1)
%   min_eig              nM x nQ, min|eig| of filter_matrix_1_A (his
%                        conditioning screen: used only if > 0.01)
%   mean_corr, mean_corr_kf, unstable_eigs, max_abs_imag, max_abs_imag_kf,
%   seconds              nM x nQ diagnostics, as in the profile driver
%
% With the factor analyses cut a grid point costs about 0.07s everywhere, so a
% dense grid is affordable.  The driver still runs ONE eps_D slab per MATLAB
% process (``make atalay-surface-export`` launches them together), saves the
% slab after every eps_M row so an interrupted run keeps what it computed, and
% stops on ATALAY_SURFACE_MAX_SECONDS.  Each slab writes a flushed
% ``progress_epsD*_*.csv`` as it goes.
%
% ONE HAZARD REMAINS, and it is in the deposited model rather than in the code
% around it.  filter_full.m solves for the steady state with two nested WHILE
% loops that have no iteration cap (its lines 163-184), and in one corner of the
% parameter space that fixed point does not converge, so the grid point is not
% computable with his code at all and no wall-clock budget checked between rows
% can rescue it.  The corner is narrow and easy to state: eps_M above one --
% material inputs as substitutes, the opposite of everything the paper argues,
% and far outside his own estimates -- TOGETHER WITH eps_Q at the very bottom of
% its range.  At eps_M = 1.09 every eps_Q from 0.06 to 1.50 solves in about a
% tenth of a second and eps_Q = 0.03 never returns.
%
% The machinery to push past it is here for anyone who wants to try: set
% ``ATALAY_SURFACE_TAG`` to write a second pass into its own file (the Python side
% globs and concatenates), ``ATALAY_SURFACE_DESCENDING`` to walk eps_M downwards
% so the safest rows finish first, ``ATALAY_SURFACE_SAVE_EVERY=1`` to keep every
% row, and kill the pass on a timer.  It buys a couple of eps_M values per run.
%
% Usage (from any directory):
%   matlab -batch "ATALAY_SURFACE_EPSD=0.9; run('<repo>/code/export_atalay_elasticity_surface.m')"
%   # or all six slabs in one process, slowest to fastest:
%   matlab -batch "run('<repo>/code/export_atalay_elasticity_surface.m')"
%
% ATALAY_REPO_ROOT, ATALAY_SURFACE_OUT, ATALAY_SURFACE_EPSD,
% ATALAY_SURFACE_EPSM and ATALAY_SURFACE_EPSQ override the repository root,
% the export directory and the three grids (the test harness uses the last four
% to run a 2x2x2 grid in seconds).

if exist('ATALAY_REPO_ROOT', 'var')
    surf_root = ATALAY_REPO_ROOT;
else
    surf_root = fileparts(fileparts(mfilename('fullpath')));
end
surf_pkg = fullfile(surf_root, 'code', '.cache', 'derived', 'atalay2017', 'replication_package');
if exist('ATALAY_SURFACE_OUT', 'var')
    surf_out = ATALAY_SURFACE_OUT;
else
    surf_out = fullfile(surf_root, 'code', '.cache', 'derived', 'atalay2017', 'elasticity_surface');
end
if ~exist(surf_out, 'dir'); mkdir(surf_out); end

% eps_D panels.  Atalay identifies neither eps_D nor eps_LS and "will choose a
% wide range, centered around values that have been estimated in previous
% papers": Herrendorf-Rogerson-Valentinyi's 0.9, Oberfield-Raval's 0.8-1.1, and
% his own benchmark 1.  These six span that range and bracket it on both sides.
if exist('ATALAY_SURFACE_EPSD', 'var')
    surf_eps_D = ATALAY_SURFACE_EPSD(:)';
else
    surf_eps_D = [0.25, 0.50, 0.75, 0.90, 1.00, 1.25];
end
% Published (eps_M, eps_Q) rectangle. The eps_M axis stops at 1.06, before
% the unchanged model develops nonconvergent steady states. The eps_M axis
% is refined below 0.3, where the inversion is most sensitive and where Atalay's
% own estimates and the explosive boundary both sit; eps_Q is smooth enough for
% an even grid.
if exist('ATALAY_SURFACE_EPSM', 'var')
    surf_eps_M = ATALAY_SURFACE_EPSM(:)';
else
    % 0.10 is Atalay's benchmark eps_M and must be on the grid.
    surf_eps_M = [0.005, 0.01, 0.02, 0.03, 0.04:0.03:1.06];
end
if exist('ATALAY_SURFACE_EPSQ', 'var')
    surf_eps_Q = ATALAY_SURFACE_EPSQ(:)';
else
    % 0.999 is the package's stand-in for Atalay's benchmark eps_Q = 1; without it
    % his own calibration is not a grid point and the cross-check against the
    % deposited benchmark export has nothing to compare.
    surf_eps_Q = [0.03:0.03:1.50, 0.999];
end
% Wall-clock budget per slab, a backstop rather than a binding constraint now
% that a grid point costs ~0.07s.
if exist('ATALAY_SURFACE_MAX_SECONDS', 'var')
    surf_budget = ATALAY_SURFACE_MAX_SECONDS;
else
    surf_budget = 7200;
end

% One BLAS thread per process by default.  Every matrix here is 30x30 to 90x90,
% far too small for threaded linear algebra to pay for itself, and six MATLABs
% each grabbing all eight cores oversubscribe the machine sixfold -- which costs
% more than an order of magnitude in wall time.
if exist('ATALAY_SURFACE_THREADS', 'var')
    maxNumCompThreads(ATALAY_SURFACE_THREADS);
else
    maxNumCompThreads(1);
end

unity = @(v) v .* (abs(v - 1) > 1e-9) + 0.999 * (abs(v - 1) <= 1e-9);
surf_eps_D = unity(sort(unique(round(surf_eps_D, 6))));
surf_eps_M = unity(sort(unique(round(surf_eps_M, 6))));
% Walk eps_M downwards when asked. The non-convergent steady states sit just
% above eps_M = 1, so a descending pass finishes every row above them before it
% reaches one, and a caller who kills the pass on a timer keeps all of those.
if exist('ATALAY_SURFACE_DESCENDING', 'var') && ATALAY_SURFACE_DESCENDING
    surf_eps_M = fliplr(surf_eps_M);
end
surf_eps_Q = unity(sort(unique(round(surf_eps_Q, 6))));
nD = numel(surf_eps_D); nM = numel(surf_eps_M); nQ = numel(surf_eps_Q);
fprintf('grid: %d eps_D x %d eps_M x %d eps_Q = %d filter runs\n', ...
    nD, nM, nQ, nD * nM * nQ);

% ---- the shocks-only copy of the deposited filter --------------------------
% Read filter_full.m, keep its first CUT_LINE lines verbatim, and refuse to
% proceed unless the cut still falls exactly where it is meant to: after the
% last filtered-shock assignment and before the factor-analysis block.
CUT_LINE = 395;
surf_src   = fileread(fullfile(surf_pkg, 'filter_full.m'));
surf_lines = strsplit(surf_src, sprintf('\n'));
assert(numel(surf_lines) > CUT_LINE, 'filter_full.m is shorter than the cut.');
assert(contains(surf_lines{CUT_LINE}, 'filtered_shocks_D_kf=productivity_shocks_kf_D'), ...
    ['filter_full.m line %d is no longer the last filtered-shock assignment: "%s".' ...
     ' Re-derive CUT_LINE before trusting this export.'], CUT_LINE, strtrim(surf_lines{CUT_LINE}));
assert(contains(surf_lines{CUT_LINE + 1}, 'optionsFactoran'), ...
    ['filter_full.m line %d is no longer the start of the factor-analysis block: "%s".'], ...
    CUT_LINE + 1, strtrim(surf_lines{CUT_LINE + 1}));
surf_cut_dir = fullfile(tempdir, sprintf('atalay_filter_cut_%d', feature('getpid')));
if ~exist(surf_cut_dir, 'dir'); mkdir(surf_cut_dir); end
surf_cut_fid = fopen(fullfile(surf_cut_dir, 'atalay_filter_shocks_only.m'), 'w');
fprintf(surf_cut_fid, '%s\n', surf_lines{1:CUT_LINE});
fclose(surf_cut_fid);
addpath(surf_cut_dir); rehash path;
% Called by name, never through ``run``: ``run`` would cd into surf_cut_dir and
% the deposited csvread calls are relative to the package directory.

% The workspace filter_master.m sets up before its first filter_full call.
results_table = zeros(nM * nQ + 1, 24);
historical_decomposition = zeros(49, 6);
counter = 0;
cov_ind_output_A = 0; cov_ind_output_B = 0; cov_ind_output_D = 0;
cov_model_output_A = 0; cov_model_output_B = 0; cov_model_output_D = 0;

surf_old_dir = pwd;
cleanupObj = onCleanup(@() cd(surf_old_dir));
cd(surf_pkg);

T = 49; N = 30;   % checked against every solved grid point below
if exist('ATALAY_SURFACE_SAVE_EVERY', 'var')
    SAVE_EVERY = ATALAY_SURFACE_SAVE_EVERY;
else
    SAVE_EVERY = 10;   % eps_M rows between slab checkpoints
end
% Filename tag, so a second pass over a different eps_M range lands in its own
% file instead of overwriting the first. The Python side globs and concatenates.
if exist('ATALAY_SURFACE_TAG', 'var')
    surf_tag = ATALAY_SURFACE_TAG;
else
    surf_tag = '';
end

% ---- self-check: the truncated copy against the full deposited file --------
parameters = [0.999, 0.999, 0.100, 0.999, 2, 0, 1, 4];   % Atalay's benchmark
filter_full;
chk_A  = real(filtered_shocks_A);
chk_kf = real(filtered_shocks_A_kf);
chk_V  = filter_matrix_2_row_2_A' * filter_matrix_2_row_2_A;
chk_S  = DR_State_State; chk_E = DR_State_Exo;
chk_r  = max(abs(eig(final_filter_3_A))); chk_m = min(abs(eig(filter_matrix_1_A)));
clear filtered_shocks_A filtered_shocks_A_kf
counter = 0;
atalay_filter_shocks_only;
chk_gap = max([max(abs(chk_A(:)  - reshape(real(filtered_shocks_A), [], 1))), ...
               max(abs(chk_kf(:) - reshape(real(filtered_shocks_A_kf), [], 1))), ...
               max(abs(chk_V(:)  - reshape(filter_matrix_2_row_2_A' * filter_matrix_2_row_2_A, [], 1))), ...
               max(abs(chk_S(:)  - DR_State_State(:))), ...
               max(abs(chk_E(:)  - DR_State_Exo(:))), ...
               abs(chk_r - max(abs(eig(final_filter_3_A)))), ...
               abs(chk_m - min(abs(eig(filter_matrix_1_A))))]);
assert(chk_gap == 0, ...
    'The truncated filter differs from the deposited filter_full.m by %.3e.', chk_gap);
fprintf('shocks-only copy of filter_full.m lines 1-%d reproduces it exactly at the benchmark\n', CUT_LINE);

surf_all_t0 = tic;

for iD = 1:nD
    shocks     = nan(nM, nQ, T, N);
    shocks_kf  = nan(nM, nQ, T, N);
    V_all      = nan(nM, nQ, N, N);
    ok         = false(nM, nQ);
    rho        = nan(nM, nQ);
    min_eig    = nan(nM, nQ);
    mean_corr           = nan(nM, nQ);
    mean_corr_kf        = nan(nM, nQ);
    unstable_eigs       = nan(nM, nQ);
    max_abs_imag        = nan(nM, nQ);
    max_abs_imag_kf     = nan(nM, nQ);
    seconds             = nan(nM, nQ);
    messages = repmat({''}, nM, nQ);
    years = [];

    slab_file = fullfile(surf_out, ...
        sprintf('atalay_surface_epsD%s_%03d.mat', surf_tag, round(1000 * surf_eps_D(iD))));
    slab_t0 = tic;
    for iM = 1:nM
        for iQ = 1:nQ
            % Never let a failed grid point read as the previous point's output.
            clear filtered_shocks_A filtered_shocks_A_kf V data_time
            counter = 0;
            parameters = [surf_eps_D(iD), surf_eps_Q(iQ), surf_eps_M(iM), 0.999, 2, 0, 1, 4];

            point_t0 = tic;
            try
                atalay_filter_shocks_only;
                % V: aggregate-output variance matrix, impact response plus
                % horizons 0--25, exactly the sum filter_full.m accumulates for
                % cov_model_output_A.
                V = filter_matrix_2_row_2_A' * filter_matrix_2_row_2_A;
                power = eye(industries);
                for h = 0:25
                    ir = filter_matrix_1_row_2 * power * DR_State_Exo(:, 1:industries);
                    V = V + ir' * ir;
                    power = power * DR_State_State;
                end
                V = 0.5 * (V + V');
                if ~isequal(size(filtered_shocks_A), [T N]) || ~isequal(size(V), [N N])
                    error('atalay:shape', 'unexpected shapes %s / %s', ...
                        mat2str(size(filtered_shocks_A)), mat2str(size(V)));
                end
                shocks(iM, iQ, :, :)    = real(filtered_shocks_A);
                shocks_kf(iM, iQ, :, :) = real(filtered_shocks_A_kf);
                V_all(iM, iQ, :, :)     = V;
                ok(iM, iQ)      = true;
                rho(iM, iQ)     = max(abs(eig(final_filter_3_A)));
                min_eig(iM, iQ) = min(abs(eig(filter_matrix_1_A)));
                mean_corr(iM, iQ)    = mean(mean(corr(real(filtered_shocks_A))));
                mean_corr_kf(iM, iQ) = mean(mean(corr(real(filtered_shocks_A_kf))));
                unstable_eigs(iM, iQ)   = UnstableEigs;
                max_abs_imag(iM, iQ)    = max(max(abs(imag(filtered_shocks_A))));
                max_abs_imag_kf(iM, iQ) = max(max(abs(imag(filtered_shocks_A_kf))));
                if isempty(years)
                    years = data_time((burn_in + 1):end, industries + 1);
                end
            catch ME
                messages{iM, iQ} = ME.message;
            end
            seconds(iM, iQ) = toc(point_t0);
        end
        % Flushed progress: MATLAB's stdout is block-buffered when redirected to
        % a file, so a long-running batch job is otherwise invisible.
        pfid = fopen(fullfile(surf_out, sprintf('progress_epsD%s_%03d.csv', ...
            surf_tag, round(1000 * surf_eps_D(iD)))), 'a');
        fprintf(pfid, '%.4f,%.4f,%d,%d,%.1f\n', surf_eps_D(iD), surf_eps_M(iM), ...
            sum(ok(iM, :)), nQ, toc(slab_t0));
        fclose(pfid);
        fprintf('  eps_D=%.3f  eps_M=%.3f  %d/%d ok  (%.1fs elapsed)\n', ...
            surf_eps_D(iD), surf_eps_M(iM), sum(ok(iM, :)), nQ, toc(slab_t0));

        % Checkpoint every SAVE_EVERY rows and at the end of the slab.  Not every
        % row: a dense slab's payload is ~80 MB and the export directory is inside
        % a synced Dropbox folder, so saving 53 times per slab in six processes at
        % once costs far more than the ~13 minutes the whole slab takes to compute.
        % Rows not yet reached stay NaN, which the Python side treats as unsolved.
        if mod(iM, SAVE_EVERY) ~= 0 && iM ~= nM && toc(slab_t0) <= surf_budget
            continue
        end
        payload = struct('eps_D', surf_eps_D(iD), ...
            'eps_M_grid', surf_eps_M, 'eps_Q_grid', surf_eps_Q, ...
            'shocks', shocks, 'shocks_kf', shocks_kf, 'V', V_all, ...
            'ok', ok, 'rho', rho, 'min_eig', min_eig, 'mean_corr', mean_corr, 'mean_corr_kf', mean_corr_kf, ...
            'unstable_eigs', unstable_eigs, 'max_abs_imag', max_abs_imag, ...
            'max_abs_imag_kf', max_abs_imag_kf, 'seconds', seconds, ...
            'years', years, 'rows_done', iM, 'messages', {messages});
        save(slab_file, '-struct', 'payload', '-v7');

        if toc(slab_t0) > surf_budget
            fprintf(['eps_D=%.3f: stopping after %d/%d eps_M rows, ' ...
                     '%.0fs budget exhausted\n'], ...
                surf_eps_D(iD), iM, nM, surf_budget);
            break
        end
    end

    fprintf('eps_D=%.3f: %d/%d solved in %.1fs -> %s\n', ...
        surf_eps_D(iD), sum(ok(:)), nM * nQ, toc(slab_t0), slab_file);
end

% Slab index, appended so that one-slab-per-process runs accumulate into it.
% The Python side sorts and de-duplicates on eps_D, so order does not matter.
index_path = fullfile(surf_out, 'slab_index.csv');
fresh = ~exist(index_path, 'file');
fid = fopen(index_path, 'a');
if fresh
    fprintf(fid, 'eps_D,n_eps_M,n_eps_Q,file\n');
end
for iD = 1:nD
    fprintf(fid, '%.6f,%d,%d,%s\n', surf_eps_D(iD), nM, nQ, ...
        sprintf('atalay_surface_epsD%s_%03d.mat', surf_tag, round(1000 * surf_eps_D(iD))));
end
fclose(fid);
fprintf('wrote %d slabs (%d filter runs) in %.1f minutes to %s\n', ...
    nD, nD * nM * nQ, toc(surf_all_t0) / 60, surf_out);
