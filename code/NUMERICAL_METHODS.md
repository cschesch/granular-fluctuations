# Numerical methods

## Grouped and weighted granularity bounds

The batched solver in `fast_mgp.py` handles the following family of semidefinite
programs:

    max  w'nu   subject to  V'(Sigma - Diag(nu))V >= 0,
                           nu_i >= 0 for i in I.

Here `>= 0` denotes positive semidefiniteness for matrices. The weights `w` are
positive, `V` is an M-by-q matrix with orthonormal columns, and `I` is the mask of
nonnegative coordinates. `basis=None` means `V = I_M`; `weights=None` means
`w = 1`.

With the full basis and all coordinates nonnegative, the grouped bound is

    Gamma_bar(Sigma) = max { tr D : D diagonal >= 0, Sigma - D >= 0 }
                    = min { <Y, Sigma> : Y >= 0, diag(Y) >= 1 }.

Using weights `w_i = S_i^2` gives the full-panel bound

    Gamma(Sigma, S) = max { S'DS : D diagonal >= 0, Sigma - D >= 0 }
                   = min { <Y, Sigma> : Y >= 0, diag(Y) >= S o S }.

The weighted formulation uses the raw covariance and puts size weights in the
objective, avoiding the additional scale dispersion of size-weighted coordinates.
The share divides either value by the corresponding aggregate variance. In the
Monte Carlo populations, the full-panel bound, grouped bound and planted
idiosyncratic share `sum_i S_i^2 psi_i / sigma_y^2` are distinct benchmarks.

## Near-optimal directional derivatives used by the bootstrap

The directional derivative of the grouped SDP value is

    Gamma_bar'(Sigma; H) = min { <Y, H> : Y in Y*(Sigma) },
    Y*(Sigma) = { Y >= 0 : diag(Y) >= 1, <Y, Sigma> = Gamma_bar(Sigma) }.

A selected dual optimizer supplies a linear response `<Y*, H>`. It is the
unique gradient when the optimizer is unique; at a nonregular point, selecting
one optimizer need not give the minimum for each direction.

The current bootstrap in `gradient_bootstrap.py` and `inference_mc.py` uses
`near_optimal.py` to estimate this minimum over

    Y_hat_T = { Y >= 0 : diag(Y) >= 1, <Y, Sigma_hat> <= kappa },
    kappa = (1 + delta_T) Gamma_hat,
    delta_T = c sqrt(log(T) / T).

The objective tolerance is relative to the estimated SDP value: the allowed
excess above the optimum is `delta_T * Gamma_hat`. It is not measured relative
to aggregate variance. Rank and active-set diagnostics do not define this set.

For fixed `c > 0`, `delta_T -> 0` and `sqrt(T) delta_T -> infinity`. Under the
root-T sampling and regularity assumptions in the paper, these rates allow
sampling error in the estimated objective while shrinking the permitted
suboptimality. This is a value-function version of the shrinking-neighbourhood
derivative estimator in Fang and Santos (2019, Section 4.2). The gradient
bootstrap's limit-law argument uses their Theorem 3.2 and its assumptions;
a finite-tolerance numerical solve is an estimate of the derivative.

For positive-definite `Sigma_hat`,

    <Y, Sigma_hat> >= lambda_min(Sigma_hat) tr(Y)   for Y >= 0.

Thus a finite cap bounds the set. When the cap is strictly above the exact
optimum, `Y* + epsilon I_M` is strictly feasible for sufficiently small positive
`epsilon`. This gives a nonempty compact set and strong duality for

    min { <Y, H> : Y in Y_hat_T }
      = max { 1'nu - kappa t : H + t Sigma_hat - Diag(nu) >= 0,
                              nu >= 0, t >= 0 }.

The latter problem has M+1 variables and is solved by `barrier_nearopt` or the
NumPy reference implementation. Its extra scalar multiplier `t` distinguishes
it from the family in the first section. Directions are normalized by their
Frobenius norms; the covariance and cap are scaled together by the covariance's
smallest eigenvalue. Returned values restore the direction's scale.

For each resampling shock, the share response follows the quotient rule:

    H_b = sqrt(T) (Sigma_b* - Sigma_hat),
    a = 1' Sigma_hat 1,
    D_b = estimated_Gamma_bar'(Sigma_hat; H_b) / a
          - g_hat (1' H_b 1) / a.

Independent bootstrap and gradient-bootstrap comparisons share resampling
draws; their moving-block counterparts share block draws. These calculations
use grouped observations. Mathematical boundedness of the near-optimal set
does not ensure that a floating-point solver succeeds.

## Reduced-face calculations implemented by the solver

`fast_mgp.py` also implements a separate reduced-face calculation. Given a
common component `C*`, let `V` span its estimated null space. Complementary
slackness motivates the face

    F = { Y = V Z V' : Z >= 0,
          Y_ii = 1 where D_i > 0, Y_ii >= 1 where D_i = 0 }.

Minimizing `<Y, H>` on this face has the masked dual formulation

    max  1'nu   subject to  V'(H - Diag(nu))V >= 0,
                           nu_i >= 0 where D_i = 0.

The other coordinates of `nu` are free. This is the reduced member of the
batched family, implemented by `barrier_reduced`. The current near-optimal
bootstrap does not use this face construction.

A recession direction of `F` is a positive-semidefinite matrix supported in
`col(V)` whose diagonal vanishes on the equality coordinates. A nonzero such
matrix exists exactly when `col(V)` intersects the coordinate span of the
Heywood indices nontrivially. `face_is_bounded` uses only the conservative
dimension screen `corank + heywood < M`; it rejects equality too. This is a
generic dimension screen, not a test of the actual intersection for every
basis. Numerical rank and activity thresholds can enlarge the estimated face,
even though the exact optimal dual face is bounded when `Sigma > 0`.

`directional_derivative_batch` retries failed reduced solves with a tighter
barrier target and more iterations, then uses a guarded retry for a shared
basis. Remaining failures return `<Y*, H>` and a failure flag. Such a substituted
linear response need not equal the directional derivative at a nonregular
point; it is not the fallback policy of `near_optimal.py`.

## Numerical convergence and feasibility checks

For the batched family, write `G(nu) = V'(Sigma - Diag(nu))V` and
`P(nu) = V G(nu)^{-1} V'`. At barrier parameter `mu > 0`, the objective is

    phi_mu(nu) = -w'nu / mu - log det G(nu) - sum_{i in I} log nu_i.

With `b_i = 1/nu_i` for `i in I` and zero otherwise,

    gradient = -w/mu + diag(P) - b,
    Hessian  = P o P + Diag(b o b).

Only masked coordinates contribute reciprocal terms. The Newton systems use
diagonal jitter. The solver reduces `mu` when its Newton-decrement criterion
passes and marks convergence when the terminal barrier target and final
centering criterion both pass. Iteration limits, factorization failures and
oversized iterates can instead terminate a solve; none alone proves that the
underlying problem is unbounded.

Damped Newton steps follow the self-concordant barrier approach of Nesterov
and Nemirovski. Its exact-arithmetic properties do not certify floating-point
iterates. The compiled `barrier_full`, `barrier_reduced` and `barrier_nearopt`
kernels check trial positive definiteness by Cholesky factorization and halve
unsuccessful steps within finite retry limits. They also limit steps at
nonnegative-coordinate boundaries. The guarded NumPy reduced path caps steps
using the estimated matrix-cone boundary,

    alpha < 1 / lambda_max(G^(-1/2) V'Diag(step)V G^(-1/2)),

when this eigenvalue is positive. The NumPy near-optimal path checks trial
eigenvalues and scalar positivity while halving steps. These checks reduce
numerical failures; they do not establish an unconditional feasibility
invariant.

On the exact central path, `Y = mu P` obeys the appropriate dual diagonal
constraints. The primal-dual gap is `(q + |I|) mu`, since the matrix barrier has
size q and there are `|I|` scalar barriers. For the full basis with all
coordinates nonnegative this is `2 M mu`. `BatchSolution.relative_gap` reports
`2 M mu / |value|` (using denominator one at zero value), even for reduced or
masked problems. Treat it as a barrier-based diagnostic alongside convergence
and explicit feasibility checks, not as an unconditional numerical certificate.

With its default backstop enabled, `near_optimal_derivative` retries unsuccessful
compiled solves using MOSEK when available, otherwise Clarabel. These retries
solve the same near-optimal problem; they do not substitute a selected dual
optimizer. The NumPy path used without Numba returns its own success flags
without these general-solver retries. Returned failure flags must therefore be
considered separately from mathematical boundedness. `tests/test_inference_mc.py`
contains comparisons against CVXPY and analytical benchmarks.
