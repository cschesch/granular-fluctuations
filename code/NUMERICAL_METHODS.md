# Numerical methods

## Batched granularity solver

The routines in `fast_mgp.py` solve many small semidefinite programs together.
The derivation below describes their objective, feasibility safeguards and
convergence certificate. `tests/test_inference_mc.py` compares the implementations
with CVXPY and checks analytical benchmarks.

Every program the paper needs is the same parametric family

    max  w'nu   s.t.   V'(Sigma - Diag(nu)) V >= 0,   nu_i >= 0  (i in a mask),

with objective weights ``w > 0`` and an orthonormal basis ``V`` of R^M
(``basis=None`` means V = I, ``weights=None`` means w = 1):

* ``V = I``, ``w = 1`` and the full nonnegativity mask is the grouped MGP itself,
  ``Gamma_bar(Sigma) = max{tr D : D diagonal >= 0, Sigma - D >= 0}``, whose
  Lagrangian dual is ``min{<Y,Sigma> : Y >= 0, diag(Y) >= 1}``;
* ``V = I`` and ``w = S o S`` is the *weighted* firm-level MGP,
  ``Gamma(Sigma_0, S) = max{S'DS : D diagonal >= 0, Sigma_0 - D >= 0}``. Solving it
  in this form rather than in the size-weighted coordinates used for grouped outcomes keeps the
  variables O(1) when sizes are power-law, where Diag(S) Sigma_0 Diag(S) spans
  eight orders of magnitude and the barrier stalls;
* ``V`` spanning the orthogonal complement of the common subspace col(C*) and an
  empty mask is the Lagrangian dual of the directional derivative
  ``Gamma_bar'(Sigma; H) = min{<Y,H> : Y in Y*(Sigma)}`` on the optimal dual face, the object
  the Fang--Santos gradient bootstrap pushes each resampling direction through.

The method is a long-step path-following barrier method on ``nu``. Writing
G(nu) = V'(Sigma - Diag(nu))V and P(nu) = V G(nu)^{-1} V', the barrier

    phi_t(nu) = -t w'nu - log det G(nu) - sum_{i in mask} log nu_i

has gradient ``-t w + diag(P) - 1/nu`` and Hessian ``P o P + diag(1/nu^2)``: the
Hadamard square of P, so one matrix inverse and one linear solve per Newton step
and nothing else. Damped Newton steps of length 1/(1+lambda) in the Newton
decrement lambda keep every iterate strictly feasible without a line search
(Nesterov--Nemirovski self-concordance), which is what makes the whole step
vectorizable over the stack. Problems that finish drop out of the batch.

That self-concordance guarantee is exact-arithmetic only. On the full-basis
problems it holds numerically too, but the directional-derivative problems are
badly conditioned -- buckets at a Heywood boundary D_i = 0 send the 1/nu_i^2 block
of the Hessian to infinity -- and a damped step computed from an ill-conditioned
Newton system can leave the cone, after which nothing recovers. The reduced-basis
path therefore also caps each step at the exact distance to the boundary,
``alpha < 1/lambda_max(G^{-1} V'Diag(step)V)``, which costs one symmetric
eigendecomposition of a q x q block per iteration and makes strict feasibility a
numerical invariant rather than a theorem.

Some of those reduced problems are genuinely unbounded rather than merely hard.
The dual optimal face is bounded when it is cut out by the exact null space of
C*, but the estimator identifies that null space at a relative tolerance (1e-3 in
``mgp_core.numerical_rank``), and the enlarged face this produces can have
recession directions once some diagonal constraints are slack -- see
``face_is_bounded`` for the exact condition, which is cheap enough to check before
solving anything. Where it is missed, the iterates run off to infinity,
``converged`` is False, and the caller is expected to fall back exactly as
``gradient_bootstrap.ShareDirectionalDerivative`` does on a solver failure.

At the barrier parameter mu = 1/t the matrix ``Y = mu P`` satisfies ``Y >= 0``
and ``diag(Y) >= w`` with duality gap exactly ``2 M mu``, so it converges to the
dual optimizer Y* (the gradient of Gamma_bar, and the influence-function weight
in ``mgp_core``) and the returned value is certified to within 2 M mu of the
optimum.

## Compiled implementation

`mgp_kernels.py` performs the same barrier iteration one problem at a time,
with parallel execution across problems. `barrier_full` solves the full-basis
problem; `barrier_reduced` handles the directional derivative in an estimated
null space. Both accept a coordinate-specific nonnegativity mask and return
the terminal barrier parameter and convergence status. If a Newton step loses
positive definiteness, the compiled implementation halves the step and retries
the Cholesky factorization.
