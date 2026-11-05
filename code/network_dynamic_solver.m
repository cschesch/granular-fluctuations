% Solve the Foerster-Sarte-Watson/Stella multisector model with Sims' gensys.
%
% Usage:
%   octave -qf network_dynamic_solver.m INPUT_DIR OUTPUT_DIR GENSYS_DIR
%
% INPUT_DIR must contain matrix_io.csv, alpha.csv, and matrix_capital.csv.
% The script mirrors Stella's deposited data_input.gss, including the
% 1e-19 replacement for structural zeros and the 25 percent repair share.
% It writes the three matrices used by transformation_data.gss:
% pii_a.csv, xi.csv, and cap_rho.csv.

args = argv();
if numel(args) != 3
  error('Usage: octave -qf network_dynamic_solver.m INPUT_DIR OUTPUT_DIR GENSYS_DIR');
endif
input_dir = args{1};
output_dir = args{2};
gensys_dir = args{3};
addpath(gensys_dir);
if exist('gensys', 'file') != 2
  error('gensys.m not found in %s', gensys_dir);
endif
% Dynare's legacy gensys helper uses Matlab-style scalar boolean operators;
% Octave 9 emits one warning per QZ swap unless these compatibility warnings
% are silenced.  They do not affect the numerical solution (verified below).
warning('off', 'all');
if exist(output_dir, 'dir') != 7
  mkdir(output_dir);
endif

gma = dlmread(fullfile(input_dir, 'matrix_io.csv'), ',');
alpha_d = dlmread(fullfile(input_dir, 'alpha.csv'), ',');
theta_mat = dlmread(fullfile(input_dir, 'matrix_capital.csv'), ',');
N = rows(gma);
if columns(gma) != N || any(size(alpha_d) != [N N]) || any(size(theta_mat) != [N N])
  error('All structural inputs must be square matrices of the same dimension.');
endif

small = 1.0e-19;
gma(gma == 0) = small;
theta_mat(theta_mat == 0) = small;
rm_share = 0.25;
theta = (theta_mat + rm_share * eye(N)) / (1 + rm_share);

sigma = 1;
psi0 = 1;
beta = 0.99;
delta = 0.025;
z = ones(N, 1);

% Steady state: literal translation of Stella's data_input.gss.
gma_tilde = kron(gma', ones(1, N)) .* kron(ones(1, N), eye(N));
phi = eye(N) - alpha_d - diag(sum(gma, 1));
C_l = phi * ones(N, 1);
theta_tilde = kron(theta', ones(1, N)) .* kron(ones(1, N), eye(N));
C_k = theta_tilde * reshape(log(theta'), [], 1) ...
      - log((1 - beta * (1 - delta)) ./ (beta * diag(alpha_d)));
Mat_mlamda = kron(ones(N, 1), eye(N)) - kron(eye(N), ones(N, 1));
temp = alpha_d * (eye(N) - theta') + gma_tilde * Mat_mlamda + phi;
ln_lamda = -temp \ (z + alpha_d * C_k ...
  + gma_tilde * reshape(log(gma'), [], 1) ...
  - phi * (log(psi0) * ones(N, 1)) + phi * log(C_l));
lamda = exp(ln_lamda);
ln_meu = theta' * ln_lamda - theta_tilde * reshape(log(theta'), [], 1);
meu = exp(ln_meu);
Mat_my = gma .* (lamda' ./ lamda);
Mat_xz = theta .* (meu' ./ lamda);
C_exp = diag(exp((eye(N) - theta') * ln_lamda + C_k));
big_y = (eye(N) - Mat_my - delta * Mat_xz * C_exp) \ (lamda .^ (-1 / sigma));
c_shrs = (lamda .^ (-1 / sigma)) ./ big_y;
m_shrs = (Mat_my .* big_y') ./ big_y;
x_shrs = ((delta * Mat_xz * C_exp) .* big_y') ./ big_y;

% Linearized equilibrium system from data_input.gss.
beta_tilde = 1 - beta + beta * delta;
Mat_y = kron(ones(N, 1), eye(N));
Mat_c = sigma * kron(eye(N), ones(N, 1)) ...
        - sigma * kron(ones(N, 1), eye(N));
S_c = diag(c_shrs);
S_m = kron(ones(1, N), m_shrs) .* kron(eye(N), ones(1, N));
S_x = kron(ones(1, N), x_shrs) .* kron(eye(N), ones(1, N));
Mat_z = Mat_y;
Mat_meu = Mat_z;
Mat_xlamda = kron(eye(N), ones(N, 1));
Q_c = gma_tilde * Mat_c - sigma * phi;
ialpha_d = inv(alpha_d);

A = zeros(2 * N);
B = zeros(2 * N);
CF = zeros(2 * N, 2 * N);
A(1:N, 1:N) = -sigma * (beta_tilde * eye(N) ...
  + beta * (1 - delta) * theta') + beta_tilde * ialpha_d * Q_c;
B(1:N, 1:N) = -sigma * theta';
CF(1:N, N+1:2*N) = -beta_tilde * ialpha_d;
A(N+1:2*N, N+1:2*N) = S_x * Mat_z / delta;
B(N+1:2*N, 1:N) = (eye(N) - S_m * Mat_y) * ialpha_d * Q_c ...
  - S_c - S_m * Mat_c + sigma * S_x * (Mat_meu * theta' - Mat_xlamda);
B(N+1:2*N, N+1:2*N) = eye(N) - S_m * Mat_y ...
  + (S_x * Mat_z * (1 - delta)) / delta;
CF(N+1:2*N, 1:N) = (eye(N) - S_m * Mat_y) * ialpha_d;

% Put the random-walk productivity state into Sims' canonical system.
% The first N variables are forward-looking consumption; A(:,1:N) loads
% their forecast errors, matching the lpd=ik' convention in Resolkw.gss.
D = CF(:, 1:N) + CF(:, N+1:2*N);
g0 = blkdiag(A, eye(N));
g1 = [B, D; zeros(N, 2*N), eye(N)];
psimat = [zeros(2*N, N); eye(N)];
pimat = [A(:, 1:N); zeros(N, N)];
[G1, C, impact, fmat, fwt, ywt, gev, eu] = gensys( ...
  g0, g1, zeros(3*N, 1), psimat, pimat, 1.000001);
if any(eu != [1; 1])
  error('gensys did not find a unique stable solution: eu=[%g,%g].', eu(1), eu(2));
endif

% Recover the policy matrices and the observable-output law of motion.
% These are Stella's PII_a, Xi, and cap_rho matrices.
Pa = impact(1:N, :);
u1 = G1 * impact;
u2 = G1 * u1;
k1 = u1(N+1:2*N, :);
a1 = u1(2*N+1:3*N, :);
c1 = u1(1:N, :);
Pk = (c1 - Pa * a1) / k1;
Ma = k1;
Mk = (u2(N+1:2*N, :) - Ma * u1(2*N+1:3*N, :)) / k1;
PII_k = eye(N) + ialpha_d * Q_c * Pk;
PII_a = ialpha_d * (eye(N) + Q_c * Pa);
cap_rho = PII_k * Mk / PII_k;
Xi = PII_k * (Ma - Mk / PII_k * PII_a);

dlmwrite(fullfile(output_dir, 'pii_a.csv'), PII_a, ',', 'precision', '%.17g');
dlmwrite(fullfile(output_dir, 'xi.csv'), Xi, ',', 'precision', '%.17g');
dlmwrite(fullfile(output_dir, 'cap_rho.csv'), cap_rho, ',', 'precision', '%.17g');
dlmwrite(fullfile(output_dir, 'solver_diagnostics.csv'), ...
  [N, eu(1), eu(2), cond(k1), max(abs(ywt(:))), ...
   max(abs(impact(N+1:2*N, :)(:))), ...
   max(abs(impact(2*N+1:3*N, :) - eye(N))(:))], ...
  ',', 'precision', '%.17g');
