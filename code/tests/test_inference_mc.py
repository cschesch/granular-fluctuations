#!/usr/bin/env python3
"""Check the Monte Carlo solvers and population calculations.

Compare batched solvers with CVXPY and test feasibility, grouping identities,
directional derivatives, and analytical benchmarks.

Run from code/ with: python -m tests.test_inference_mc
"""

from __future__ import annotations

import numpy as np
from unittest.mock import patch

from mgp_core import (
    choose_solver,
    ledermann_bound,
    mgp_share,
    numerical_rank,
    sample_cov,
    solve_grouped_mgp_dual,
    solve_primal_mtfa,
)
from gradient_bootstrap import ShareDirectionalDerivative
from near_optimal import delta_rate
import fast_mgp
import mgp_kernels
import mosek_face
import inference_mc as mc

TEST_GROUP = "numerical"

SOLVER = choose_solver("auto")


def grouped_population(design):
    """Build the exact grouped inputs without solving an unused full-panel SDP.

    These tests use the grouped covariance and its bound, not the firm-level
    bound. A NaN sentinel prevents that omitted quantity from being used.
    Separate certificate and population-invariant tests check the full bound.
    """
    with patch.object(mc, "_firm_level_truth", return_value=np.nan):
        return mc.build_population(design, use_cache=False)


def _random_covariances(rng, m, r, k, t=200):
    loadings = rng.standard_normal((m, r))
    sigma = loadings @ loadings.T + np.diag(rng.uniform(0.3, 2.0, m))
    half = np.linalg.cholesky(sigma)
    return np.stack([sample_cov(rng.standard_normal((t, m)) @ half.T) for _ in range(k)])


def test_matches_cvxpy():
    rng = np.random.default_rng(20260901)
    for m, r in ((7, 2), (10, 3), (10, 8), (16, 4)):
        cov = _random_covariances(rng, m, r, 12)
        sol = fast_mgp.mgp_batch(cov)
        common = sol.common(cov)
        for k, sigma in enumerate(cov):
            ref = solve_grouped_mgp_dual(sigma, SOLVER)
            rel = abs(sol.value[k] - ref["value"]) / abs(ref["value"])
            assert rel < 1e-5, f"value mismatch at M={m}: {rel:.2e}"
            ref_c, ref_val = solve_primal_mtfa(sigma, SOLVER)
            assert abs(ref_val - sol.value[k]) / abs(ref_val) < 1e-5
            assert np.max(np.abs(common[k] - ref_c)) < 1e-3 * np.abs(ref_c).max()
            ref_rank = numerical_rank(np.linalg.eigvalsh(ref_c)[::-1])
            assert int(fast_mgp.common_rank(common[k][None])[0]) == ref_rank
        assert sol.converged.all(), f"non-convergence at M={m}"
    print("  [ok] fast solver matches mgp_core (value, dual, common part, rank)")


def test_dual_feasible_and_gap():
    rng = np.random.default_rng(11)
    cov = _random_covariances(rng, 12, 4, 20)
    sol = fast_mgp.mgp_batch(cov)
    # Y = mu C^{-1} is exactly feasible on the central path; the slack below is the
    # conditioning of that inverse at mu = 1e-10, not a violated constraint.
    scale = np.linalg.norm(sol.dual, axis=(1, 2))
    eig = np.linalg.eigvalsh(sol.dual)
    assert (eig.min(axis=1) / scale).min() > -1e-8, f"dual not PSD: {eig.min():.2e}"
    diag = np.einsum("kii->ki", sol.dual)
    assert diag.min() > 1.0 - 1e-5, f"diag(Y) < 1: {diag.min():.8f}"
    # The certificate 2 M mu bounds the gap at the exact central-path point; the gap
    # measured from the reconstructed Y also carries the conditioning of inverting a
    # matrix whose smallest eigenvalues are mu = 1e-10, hence the 1e-5 slack.
    dual_value = np.einsum("kij,kij->k", sol.dual, cov)
    gap = dual_value - sol.value
    assert np.all(np.abs(gap) <= 2.0 * 12 * sol.mu + 1e-5 * np.abs(sol.value)), \
        "gap wider than the certified 2 M mu"
    assert np.all(sol.relative_gap < 1e-6), "certificate itself is not tight"
    print("  [ok] returned dual is feasible and the 2 M mu gap certificate holds")


def test_weighted_equals_size_weighted_coordinates():
    """Draft eq. (3) solved with weights == draft eq. (13) solved in size coordinates."""
    rng = np.random.default_rng(5)
    n, r = 40, 3
    sizes = mc.power_law_sizes(n, 1.0)
    loadings = 0.4 * rng.standard_normal((n, r))
    psi = np.exp(0.4 * rng.standard_normal(n))
    sigma = loadings @ loadings.T + np.diag(psi)
    weighted = fast_mgp.firm_level_mgp(sigma[None], sizes).value[0]
    size_coords = (sizes[:, None] * sigma) * sizes[None, :]
    reference = solve_grouped_mgp_dual(size_coords, SOLVER)["value"]
    assert abs(weighted - reference) / reference < 1e-5, f"{weighted} vs {reference}"
    print("  [ok] weighted firm-level MGP == unweighted MGP in size-weighted coordinates")


def _least_squares_certificate(loadings: np.ndarray, sizes: np.ndarray) -> float:
    """Smallest eigenvalue of the explicit dual certificate for U_0 = col(Diag(S)Lambda).

    Y = P Diag(d) P with P the projector onto U_0^perp and d solving
    (P o P) d = 1 satisfies Y U_0 = 0 and diag(Y) = 1 by construction, so Y >= 0
    certifies Gamma_0 = sum_i S_i^2 psi_i exactly (Saunderson et al.; draft
    Proposition ``prop:master``).
    """
    a = sizes[:, None] * loadings
    proj = np.eye(len(sizes)) - a @ np.linalg.solve(a.T @ a, a.T)
    d = np.linalg.solve(proj * proj, np.ones(len(sizes)))
    return float(np.linalg.eigvalsh(proj @ np.diag(d) @ proj).min())


def test_firm_level_truth_matches_certificate():
    rng = np.random.default_rng(7)
    n, r = 200, 3
    sizes = mc.power_law_sizes(n, 1.0)
    loadings = 0.5 * rng.standard_normal((n, r))
    psi = np.exp(0.4 * rng.standard_normal(n))
    psi = psi / psi.mean()
    sigma = loadings @ loadings.T + np.diag(psi)
    margin = _least_squares_certificate(loadings, sizes)
    assert margin > -1e-10, f"no explicit certificate for this draw (min eig {margin:.2e})"
    exact = float(np.sum(sizes**2 * psi))
    solved = fast_mgp.firm_level_mgp(sigma[None], sizes, max_iter=mc.FIRM_MAX_ITER).value[0]
    assert abs(solved - exact) / exact < 1e-5, f"{solved} vs certified {exact}"
    print("  [ok] firm-level MGP equals the certified truth sum_i S_i^2 psi_i")


def test_point_estimates_match_core():
    rng = np.random.default_rng(3)
    design = mc.Design(name="t", label="t", n=200, zeta=1.0, r=3, m=10, t=120)
    pop = grouped_population(design)
    x = mc.simulate_panels(pop, 6, rng)
    est = mc.point_estimates(x, ledermann_bound(design.m))
    for k in range(len(x)):
        ref = mgp_share(x[k], SOLVER)
        assert abs(est.share[k] - ref["mgp_share"]) / ref["mgp_share"] < 1e-5
        assert abs(est.se[k] - ref["se_share"]) / ref["se_share"] < 1e-3
    print("  [ok] point_estimates reproduces mgp_core.mgp_share (share and s.e.)")


def test_directional_derivative_matches_gradient_bootstrap():
    """The MC path and the headline estimator solve the same near-optimal program."""
    rng = np.random.default_rng(13)
    seen = {True: 0, False: 0}
    for r in (3, 8):
        design = mc.Design(name="d", label="d", n=200, zeta=1.0, r=r, m=10, t=300)
        pop = grouped_population(design)
        x = mc.simulate_panels(pop, 4, rng)
        est = mc.point_estimates(x, ledermann_bound(design.m))
        directions = np.sqrt(design.t) * (
            mc.batch_cov(x[:, rng.integers(0, design.t, size=(5, design.t))]) - est.sigma[:, None]
        )
        delta = delta_rate(design.t, design.delta_scale)
        stat, fallbacks = mc._directional_shares(directions, est, np.arange(len(x)), delta)
        assert fallbacks == 0, f"near-optimal solve fell back {fallbacks} times"
        for k in range(len(x)):
            dd = ShareDirectionalDerivative(est.sigma[k], SOLVER, design.t,
                                            delta_scale=design.delta_scale)
            assert dd.regular == bool(est.regular[k]), "regularity classification differs"
            assert abs(dd.delta - delta) < 1e-12
            seen[dd.regular] += 1
            ref = dd.shares(directions[k])
            scale = max(np.abs(ref).max(), 1e-12)
            # The MC path stops the barrier at MC_MU_END (1e-5) where the headline
            # estimator runs to 1e-8, so they agree to the coarser target, not to
            # machine precision.
            assert np.max(np.abs(stat[k] - ref)) / scale < 1e-3, (
                f"derivative mismatch (regular={dd.regular}): "
                f"{np.max(np.abs(stat[k] - ref)) / scale:.2e}"
            )
    assert seen[True] > 0 and seen[False] > 0, "test did not exercise both regimes"
    print(f"  [ok] MC and headline near-optimal derivatives agree "
          f"({seen[True]} regular, {seen[False]} non-regular)")


def test_near_optimal_matches_cvxpy():
    """The headline estimator reproduces a direct cvxpy solve of the near-optimal program."""
    import cvxpy as cp
    rng = np.random.default_rng(21)
    worst = 0.0
    checked = 0
    for (r, t) in ((3, 200), (8, 1000)):
        design = mc.Design(name="d", label="d", n=200, zeta=1.0, r=r, m=10, t=t)
        pop = grouped_population(design)
        x = mc.simulate_panels(pop, 2, rng)
        for k in range(len(x)):
            sigma = mc.batch_cov(x[k][None])[0]
            dd = ShareDirectionalDerivative(sigma, SOLVER, t)
            hs = rng.standard_normal((4, design.m, design.m))
            hs = 0.5 * (hs + np.swapaxes(hs, -1, -2)) * np.abs(sigma).max()
            mine = dd.levels(hs)
            for j, h in enumerate(hs):
                y = cp.Variable((design.m, design.m), PSD=True)
                prob = cp.Problem(
                    cp.Minimize(cp.sum(cp.multiply(y, h))),
                    [cp.diag(y) >= 1, cp.sum(cp.multiply(y, sigma)) <= dd.cap],
                )
                prob.solve(solver=cp.CLARABEL)
                worst = max(worst, abs(mine[j] - prob.value) / max(abs(prob.value), 1e-12))
                checked += 1
        assert dd._face_fallbacks == 0
    assert worst < 1e-5, f"near-optimal derivative differs from cvxpy by {worst:.2e}"
    print(f"  [ok] near-optimal derivative matches cvxpy on {checked} directions "
          f"(max rel {worst:.1e})")


def test_known_optimizer_face():
    """Audit Sigma=11'+I, where D*=I and the exact dual face is known."""
    import cvxpy as cp

    m = 4
    one = np.ones(m)
    sigma = np.outer(one, one) + np.eye(m)
    rng = np.random.default_rng(912)
    directions = rng.standard_normal((12, m, m))
    directions = 0.5 * (directions + directions.transpose(0, 2, 1))
    mine, solved = mc.near_optimal_derivative(
        directions, sigma, m + 1e-8 * m, mu_end=1e-10
    )
    assert solved.all(), "production solver failed on the known bounded face"
    worst_abs = 0.0
    for h, value in zip(directions, mine):
        y = cp.Variable((m, m), symmetric=True)
        problem = cp.Problem(
            cp.Minimize(cp.sum(cp.multiply(y, h))),
            [y >> 0, cp.diag(y) == 1, y @ one == 0],
        )
        problem.solve(solver=cp.CLARABEL)
        worst_abs = max(worst_abs, abs(float(value) - float(problem.value)))
    assert worst_abs < 1e-3, f"known-face cross-check differs by {worst_abs:.2e}"
    print(f"  [ok] known M=4 optimizer face matches independent CVXPY program "
          f"(max abs {worst_abs:.1e})")


def test_unbounded_face_detection():
    """The q + h > M rule really does identify a recession direction of the face."""
    rng = np.random.default_rng(4)
    m = 10
    for corank, heywood in ((6, 1), (5, 4), (5, 7), (7, 4)):
        basis = np.linalg.qr(rng.standard_normal((m, corank)))[0]
        mask = np.zeros(m, dtype=bool)
        mask[rng.choice(m, size=heywood, replace=False)] = True
        bounded = fast_mgp.face_is_bounded(corank, heywood, m)
        assert bounded == (corank + heywood < m)
        # col(V) intersect span{e_i : i Heywood}: a nonzero element is a recession
        # direction R = u u' of the face, so the face is unbounded exactly when it
        # exists. The rule is strict at equality, where the intersection is trivial
        # only by a dimension count that fails in every neighbourhood.
        stacked = np.hstack([basis, np.eye(m)[:, mask]])
        overlap = corank + heywood - np.linalg.matrix_rank(stacked, tol=1e-9)
        assert overlap == max(0, corank + heywood - m), (
            f"corank={corank}, heywood={heywood}: overlap={overlap}"
        )
        assert not (overlap > 0 and bounded), "an unbounded face was called bounded"
        if overlap > 0:
            # V alpha + E_mask beta = 0 puts V alpha in both spans, so R = u u' is a
            # recession direction: PSD, and zero on the equality-constrained diagonal.
            null_space = np.linalg.svd(stacked)[2][-overlap:]
            u = basis @ null_space[0, :corank]
            r_mat = np.outer(u, u)
            assert np.linalg.norm(u) > 1e-8
            assert np.abs(np.diag(r_mat)[~mask]).max() < 1e-8 * np.abs(r_mat).max()
            assert np.linalg.eigvalsh(r_mat).min() > -1e-12
    print("  [ok] unbounded-face rule matches an explicit recession direction")


def test_unbounded_face_matches_solver():
    """Where the rule says unbounded, the batched face program indeed fails to solve."""
    rng = np.random.default_rng(6)
    design = mc.Design(name="u", label="u", n=300, zeta=1.2, r=3, m=10, t=120)
    pop = grouped_population(design)
    x = mc.simulate_panels(pop, 24, rng)
    est = mc.point_estimates(x, ledermann_bound(design.m))
    checked = 0
    for k in range(len(x)):
        if est.regular[k]:
            continue
        corank = design.m - int(est.rank[k])
        mask = mc.heywood_mask(est.nu[k])
        if fast_mgp.face_is_bounded(corank, int(mask.sum()), design.m):
            continue
        directions = np.sqrt(design.t) * (
            mc.batch_cov(x[k][rng.integers(0, design.t, size=(3, design.t))]) - est.sigma[k]
        )
        _, failed = fast_mgp.directional_derivative_batch(
            directions, ystar=est.ystar[k],
            basis=fast_mgp.null_basis(est.common[k], int(est.rank[k])),
            nonneg=mask if mask.any() else False,
        )
        assert failed.all(), "an unbounded face was reported as solved"
        checked += 1
    assert checked > 0, "this design no longer produces an unbounded face to check"
    print(f"  [ok] unbounded faces are refused up front ({checked} replications)")


def test_population_invariants():
    design = mc.Design(name="p", label="p", n=60, zeta=1.0, r=6, m=10, t=200)
    pop = mc.build_population(design, use_cache=False)
    one = np.ones(design.m)
    # Grouping preserves the aggregate: 1'Sigma_tilde 1 = S'Sigma_0 S.
    assert abs(float(one @ pop.sigma_grouped @ one) - pop.sigma_y2) / pop.sigma_y2 < 1e-12
    # Proposition prop:ub: the grouped ceiling dominates the truth.
    assert pop.gamma_bar0 >= pop.gamma0 * (1 - 1e-6), f"{pop.gamma_bar0} < {pop.gamma0}"
    # Diag(psi) is feasible for the firm-level MGP, so the truth dominates the design.
    assert pop.gamma0 >= pop.gamma_designed * (1 - 1e-4)
    assert 0.0 <= pop.gamma_bar0 <= 1.0
    # Closed-form asymptotic variance against a direct simulation of the influence function.
    rng = np.random.default_rng(21)
    z = rng.standard_normal((400000, design.m)) @ pop.half
    quad = np.einsum("ti,ij,tj->t", z, pop.ystar, z)
    agg = z @ one
    u = (quad - pop.gamma_bar0 * agg**2) / pop.sigma_y2
    ratio = u.std(ddof=1) / pop.sigma_u
    assert abs(ratio - 1.0) < 0.01, f"sigma_u off by {ratio - 1.0:+.3%}"
    print(f"  [ok] population invariants (aggregate, prop:ub, sigma_u within {ratio - 1:+.2%})")


def test_sampling_schemes():
    """Grouped Gaussian draws, the AR(1) recursion and t shocks all target Sigma_tilde_0."""
    design = mc.Design(name="s", label="s", n=200, zeta=1.0, r=3, m=10, t=400)
    pop = grouped_population(design)
    for name, alt in (
        ("iid Gaussian", design),
        ("AR(1)", mc.Design(**{**design.__dict__, "rho": 0.5})),
        ("t5", mc.Design(**{**design.__dict__, "shock": "t5"})),
    ):
        local = grouped_population(alt)
        x = mc.simulate_panels(local, 400, np.random.default_rng(9))
        empirical = mc.batch_cov(x).mean(axis=0)
        rel = np.abs(empirical - pop.sigma_grouped).max() / np.abs(pop.sigma_grouped).max()
        assert rel < 0.08, f"{name}: grouped covariance off by {rel:.3f}"
    print("  [ok] all three sampling schemes reproduce Sigma_tilde_0")


def test_lane_kernel_matches_scalar():
    """The lane-blocked kernel is a layout change, so it must agree to machine precision."""
    rng = np.random.default_rng(31)
    for m, r in ((7, 2), (10, 3), (10, 8), (25, 4)):
        cov = _random_covariances(rng, m, r, 6 * mgp_kernels.LANES, t=4 * m)
        big = fast_mgp.mgp_batch(cov, need_dual=False)          # lane path
        small = np.array([fast_mgp.mgp_batch(cov[k : k + 1], need_dual=False).value[0]
                          for k in range(0, 12)])               # scalar path
        assert big.converged.all(), f"lane kernel did not converge at M={m}"
        rel = np.abs(big.value[:12] - small) / np.abs(small)
        assert rel.max() < 1e-12, f"lane vs scalar at M={m}: {rel.max():.2e}"
    print("  [ok] lane-blocked kernel matches the scalar kernel to 1e-12")


def test_lane_face_kernel_matches_scalar():
    """Same check for the reduced-basis kernel, which carries the directional derivatives."""
    rng = np.random.default_rng(41)
    design = mc.Design(name="l", label="l", n=400, zeta=0.85, r=1, m=10, t=120)
    pop = grouped_population(design)
    x = mc.simulate_panels(pop, 20, rng)
    est = mc.point_estimates(x, ledermann_bound(design.m))
    checked = 0
    for k in range(len(x)):
        if est.regular[k] or checked >= 3:
            continue
        basis = fast_mgp.null_basis(est.common[k], int(est.rank[k]))
        heywood = mc.heywood_mask(est.nu[k])
        mask = heywood if heywood.any() else False
        if not fast_mgp.face_is_bounded(basis.shape[1], int(heywood.sum()), design.m):
            continue
        directions = np.sqrt(design.t) * (
            mc.batch_cov(x[k][rng.integers(0, design.t, size=(8 * mgp_kernels.LANES,
                                                             design.t))]) - est.sigma[k]
        )
        lane = fast_mgp.solve_batch(directions, nonneg=mask, basis=basis,
                                    mu_end=mc.MC_MU_END, need_dual=False)
        scalar = fast_mgp.solve_batch(directions[:12], nonneg=mask, basis=basis,
                                      mu_end=mc.MC_MU_END, need_dual=False)
        ok = scalar.converged
        rel = (np.abs(lane.value[:12] - scalar.value)
               / np.maximum(np.abs(scalar.value), 1e-12))
        assert np.all(lane.converged[:12] == scalar.converged), "convergence differs"
        assert rel[ok].max() < 1e-12, f"lane vs scalar face kernel: {rel[ok].max():.2e}"
        checked += 1
    assert checked > 0, "no bounded non-regular face to check"
    print(f"  [ok] lane-blocked face kernel matches the scalar one ({checked} faces)")


def test_counts_resampling_matches_gather():
    """cov_from_counts is an algebraic rewrite of gather-then-covariance, not an approximation."""
    rng = np.random.default_rng(17)
    c, t, m, b = 4, 120, 10, 40
    x = rng.standard_normal((c, t, m))
    idx = rng.integers(0, t, size=(c, b, t))
    gathered = mc.batch_cov(mc._gather(x, idx))
    counted = mc.cov_from_counts(x, mc._resample_counts(idx, t))
    rel = np.abs(gathered - counted).max() / np.abs(gathered).max()
    assert rel < 1e-12, f"counts vs gather: {rel:.2e}"
    # And with contiguous windows of a different length, as subsampling uses.
    ell = 30
    starts = np.arange(t - ell + 1)
    windows = starts[:, None] + np.arange(ell)[None, :]
    counts = np.zeros((len(starts), t))
    np.put_along_axis(counts, windows, 1.0, axis=1)
    ref = mc.batch_cov(x[:, windows])
    got = mc.cov_from_counts(x, np.broadcast_to(counts, (c,) + counts.shape))
    rel = np.abs(ref - got).max() / np.abs(ref).max()
    assert rel < 1e-12, f"counts vs gather (windows): {rel:.2e}"

    # The prefix-sum path, used for both contiguous schemes.
    prefix = mc.window_prefix_sums(x)
    got = mc.cov_from_windows(prefix, starts[:, None], np.full((len(starts), 1), ell), m)
    rel = np.abs(ref - got).max() / np.abs(ref).max()
    assert rel < 1e-12, f"prefix sums vs gather (subsamples): {rel:.2e}"

    # Moving blocks are several runs per draw, with the last one truncated.
    for block_length in (7, 8, 13):
        rng2 = np.random.default_rng(99)
        runs = [mc._block_runs(t, b, block_length, rng2) for _ in range(c)]
        starts_b = np.stack([r[0] for r in runs])
        lengths_b = np.stack([r[1] for r in runs])
        explicit = (starts_b[:, :, :, None]
                    + np.arange(block_length)[None, None, None, :]).reshape(c, b, -1)[:, :, :t]
        ref_b = mc.batch_cov(mc._gather(x, explicit))
        got_b = mc.cov_from_windows(prefix, starts_b, lengths_b, m)
        rel = np.abs(ref_b - got_b).max() / np.abs(ref_b).max()
        assert rel < 1e-12, f"prefix sums vs gather (blocks, l={block_length}): {rel:.2e}"
        assert lengths_b.sum(axis=-1).max() == t, "block runs do not cover exactly T periods"
    print("  [ok] counts and prefix-sum resampling reproduce an explicit gather to 1e-12")


def test_mosek_backstop_matches_cvxpy():
    """The direct-to-MOSEK backstop solves the same near-optimal program cvxpy would build."""
    from unittest import SkipTest
    if not mosek_face.MOSEK_AVAILABLE:
        raise SkipTest("Optional MOSEK solver is not installed")
    import cvxpy as cp
    import mosek
    probe = cp.Variable()
    try:
        cp.Problem(cp.Minimize(probe), [probe >= 1]).solve(solver=cp.MOSEK)
    except (cp.error.SolverError, mosek.Error) as error:
        if "license" in str(error).lower():
            raise SkipTest("Optional MOSEK license is unavailable") from error
        raise

    rng = np.random.default_rng(23)
    design = mc.Design(name="f", label="f", n=300, zeta=1.0, r=3, m=10, t=150)
    pop = grouped_population(design)
    x = mc.simulate_panels(pop, 12, rng)
    est = mc.point_estimates(x, ledermann_bound(design.m))
    delta = delta_rate(design.t)
    checked = 0
    worst = 0.0
    for k in range(len(x)):
        if est.regular[k] or checked >= 12:
            continue
        sigma = est.sigma[k]
        cap = float(est.share[k] * est.aggregate_var[k] + delta * est.aggregate_var[k])
        directions = np.sqrt(design.t) * (
            mc.batch_cov(x[k][rng.integers(0, design.t, size=(3, design.t))]) - sigma
        )
        with mosek_face.NearOptimalSolver(sigma, cap) as solver:
            for h in directions:
                h = 0.5 * (h + h.T)
                mine = solver.value(h)
                y = cp.Variable((design.m, design.m), PSD=True)
                prob = cp.Problem(
                    cp.Minimize(cp.sum(cp.multiply(y, h))),
                    [cp.diag(y) >= 1, cp.sum(cp.multiply(y, sigma)) <= cap],
                )
                prob.solve(solver=cp.CLARABEL)
                assert mine is not None, "MOSEK failed on a bounded near-optimal set"
                worst = max(worst, abs(mine - prob.value) / max(abs(prob.value), 1e-12))
                checked += 1
    assert worst < 1e-5, f"backstop differs from cvxpy by {worst:.2e}"
    print(f"  [ok] direct MOSEK backstop matches cvxpy ({checked} non-regular directions, "
          f"max rel {worst:.1e})")


def test_tightness_grid_reduction():
    """The tightness grid solves in M dimensions what is defined in N; check they agree.

    Under Assumption ``ass:haar`` the grouped loadings are independent Gaussians with
    variance ``theta w_m``, so the grouped ceiling depends on the size distribution only
    through the bucket loads. The grid script uses that; here it is checked against
    building the N-dimensional population and grouping it explicitly.
    """
    from mgp_core import lpt_buckets

    rng = np.random.default_rng(4242)
    for n, exponent, m, r, theta in ((400, 1.0, 8, 3, 0.7), (600, 0.6, 12, 5, 1.0)):
        sizes = mc.power_law_sizes(n, exponent)
        buckets, loads = lpt_buckets(sizes**2, m)
        grouping = np.zeros((m, n))
        for j, bucket in enumerate(buckets):
            grouping[j, bucket] = sizes[bucket]
        for _ in range(3):
            raw = rng.standard_normal((n, r))               # size-rank independent
            grouped_loadings = grouping @ (np.sqrt(theta) * raw)
            sigma = grouped_loadings @ grouped_loadings.T + np.diag((grouping**2) @ np.ones(n))
            explicit = fast_mgp.mgp_batch(sigma[None], mu_end=1e-10, need_dual=False).value[0]
            # Reduced form on the same realisation: Z standardised by the bucket loads.
            z = grouped_loadings / np.sqrt(theta * loads)[:, None]
            reduced = fast_mgp.solve_batch(
                (theta * (z @ z.T) + np.eye(m))[None], weights=loads,
                mu_end=1e-10, need_dual=False,
            ).value[0]
            rel = abs(explicit - reduced) / explicit
            assert rel < 1e-8, f"grid reduction off by {rel:.2e} at N={n}, M={m}, r={r}"
            # d = w is feasible, so the ceiling dominates the truth. The barrier value
            # is a certified lower bound and in an exactly tight cell approaches it from
            # below, hence the tolerance rather than a strict inequality.
            assert explicit >= loads.sum() * (1 - 1e-6), "grouped ceiling below the truth"
    print("  [ok] tightness-grid reduction matches an explicit N-dimensional population")


def main() -> None:
    print("Monte Carlo machinery checks")
    test_matches_cvxpy()
    test_dual_feasible_and_gap()
    test_weighted_equals_size_weighted_coordinates()
    test_firm_level_truth_matches_certificate()
    test_point_estimates_match_core()
    test_directional_derivative_matches_gradient_bootstrap()
    test_near_optimal_matches_cvxpy()
    test_known_optimizer_face()
    test_unbounded_face_detection()
    test_unbounded_face_matches_solver()
    test_population_invariants()
    test_sampling_schemes()
    test_lane_kernel_matches_scalar()
    test_lane_face_kernel_matches_scalar()
    test_counts_resampling_matches_gather()
    test_mosek_backstop_matches_cvxpy()
    test_tightness_grid_reduction()
    print("All checks passed.")


if __name__ == "__main__":
    main()
