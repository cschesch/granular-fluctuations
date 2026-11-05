#!/usr/bin/env python3
"""Fast invariants for the grouped Hahn-GIV implementation."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from grouped_giv import (
    estimate_joint_giv,
    estimated_instrument_space,
    hahn_bic,
    known_instrument_space,
    orthogonal_complement,
    projected_weight,
    scalar_iv,
)
from inference_mc import power_law_sizes
from mgp_core import lpt_buckets


TOL = 2e-10


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_projection_geometry() -> None:
    rng = np.random.default_rng(101)
    m = 8
    b = rng.standard_normal(m)
    loadings = rng.standard_normal((m, 2))
    d = rng.standard_normal(m)
    space = known_instrument_space(b, loadings, d)
    span = np.column_stack((b, loadings))
    check(space.residual_dimension == 5, "oracle residual dimension is wrong")
    check(np.linalg.norm(space.basis.T @ span) < TOL, "oracle instruments are not excluded")
    check(
        np.linalg.norm(space.basis.T @ space.basis - np.eye(5)) < TOL,
        "oracle instrument basis is not orthonormal",
    )
    direct = orthogonal_complement(span)
    check(
        abs(space.information - np.sum((direct.T @ d) ** 2)) < TOL,
        "oracle information is not the projection value",
    )


def test_grouped_whitening() -> None:
    n, m = 1000, 10
    sizes = power_law_sizes(n, 1.0)
    buckets, loads = lpt_buckets(sizes**2, m)
    grouping = np.zeros((m, n))
    for j, members in enumerate(buckets):
        grouping[j, members] = sizes[members]
    whitening = grouping / np.sqrt(loads)[:, None]
    b = whitening @ np.ones(n)
    d = np.sqrt(loads)
    check(np.linalg.norm(whitening @ whitening.T - np.eye(m)) < TOL, "whitening failed")
    check(abs(d @ b - 1.0) < TOL, "grouped system no longer aggregates exactly")
    no_factor = known_instrument_space(b, np.empty((m, 0)), d)
    n_eff = np.sum((grouping @ np.ones(n)) ** 2 / loads)
    formula = sizes @ sizes - 1.0 / n_eff
    check(abs(no_factor.information - formula) < TOL, "no-factor I_G formula failed")


def test_bic_and_estimated_space() -> None:
    rank, values = hahn_bic(np.ones(9), 250)
    check(rank == 1, "BIC should choose no heterogeneous factor for equal eigenvalues")
    spike = np.r_[np.ones(8), 3.0]
    rank, _ = hahn_bic(spike, 250)
    check(rank == 2, "BIC should choose one clearly separated factor")
    check(values.shape == (9,), "BIC candidate set has the wrong dimension")

    rng = np.random.default_rng(202)
    t, m = 4000, 6
    b = np.ones(m)
    d = np.arange(1.0, m + 1.0)
    loading = np.array([1.4, -0.9, 0.5, 0.2, -0.6, -0.6])
    factors = rng.standard_normal(t)
    x = factors[:, None] * loading + rng.standard_normal((t, m))
    space = estimated_instrument_space(x, b, d, combined_rank=2)
    check(space.basis.shape == (m, m - 2), "estimated complement has the wrong shape")
    check(np.linalg.norm(space.basis.T @ b) < TOL, "estimated space misses known direction")
    check(space.eigengap > 0.5, "planted factor did not yield spectral separation")
    check(np.isfinite(space.upsilon).all(), "separated factor produced invalid Upsilon")


def test_estimation_and_failures() -> None:
    rng = np.random.default_rng(404)
    t, m = 20_000, 5
    b = np.array([2.0, 1.4, 0.8, 0.5, 0.3])
    d = np.array([0.10, 0.10, 0.40, 0.50, 0.60])
    d /= d @ b
    loading = np.array([1.0, -0.6, 0.1, 0.3, -0.8])[:, None]
    space = known_instrument_space(b, loading, d)
    factor = rng.standard_normal(t)
    error = rng.standard_normal((t, m))
    structural = 0.2 * rng.standard_normal(t)
    phi, psi = -0.5, 1.5
    aggregate_loading = float(d @ loading[:, 0])
    aggregate_error = error @ d
    aggregate = (phi * structural + aggregate_loading * factor + aggregate_error) / (1 - phi * psi)
    price = psi * aggregate + structural
    x = phi * price[:, None] * b + factor[:, None] * loading[:, 0] + error
    check(np.allclose(x @ d, aggregate), "nonuniform grouped system does not aggregate")
    estimate = estimate_joint_giv(x, price, aggregate, b, d, space)
    check(not estimate.phi.failed and not estimate.psi.failed, "valid oracle GIV failed")
    check(abs(estimate.phi.estimate - phi) < 0.025, "oracle phi estimate misses its truth")
    check(abs(estimate.psi.estimate - psi) < 0.04, "oracle psi estimate misses its truth")

    saturated_loadings = np.eye(m)[:, 1:]
    saturated = known_instrument_space(np.eye(m)[:, 0], saturated_loadings, d)
    check(saturated.residual_dimension == 0, "saturated common space was not detected")
    check(saturated.information == 0.0, "saturated common space has spurious information")
    stopped = estimate_joint_giv(x, price, aggregate, b, d, saturated)
    check(stopped.phi.failed and stopped.psi.failed, "zero-dimensional GIV did not stop")
    check(
        stopped.psi.failure_reason == "no residual instrument dimension",
        "saturated failure reason is not explicit",
    )

    b0 = np.eye(m)[:, 0]
    loading0 = np.eye(m)[:, 1:3]
    d0 = b0 + loading0[:, 0]
    zero_info = known_instrument_space(b0, loading0, d0)
    check(zero_info.residual_dimension == 2, "zero-information example lost its instruments")
    check(zero_info.information < TOL, "zero-information space has spurious relevance")


def reference_moment_covariance(
    x: np.ndarray,
    residual: np.ndarray,
    sigma_hat: np.ndarray,
    upsilon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Oracle and generated-space-corrected influence-function covariances."""

    t = x.shape[0]
    mean_xb = np.mean(x * residual[:, None], axis=0)
    oracle_if = x * residual[:, None] - mean_xb
    direction = upsilon @ mean_xb
    correction = x * (x @ direction)[:, None] - (sigma_hat @ direction)[None, :]
    corrected_if = oracle_if + correction
    oracle_if -= oracle_if.mean(axis=0)
    corrected_if -= corrected_if.mean(axis=0)
    return oracle_if.T @ oracle_if / t, corrected_if.T @ corrected_if / t


def test_scalar_instrument() -> None:
    rng = np.random.default_rng(505)
    t, m = 5000, 6
    b = np.array([1.0, 1.0, 1.0, 1.4, 1.8, 2.2])
    d = np.array([0.5, 0.05, 0.1, 0.6, 0.1, 0.4])
    d /= d @ b
    loading = np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0])[:, None]
    oracle = known_instrument_space(b, loading, d)
    weight = projected_weight(oracle, d)
    check(abs(weight @ weight - oracle.information) < TOL, "P d does not attain I_G")
    check(np.linalg.norm(weight - oracle.basis @ (oracle.basis.T @ d)) < TOL, "P d is not the projection")
    span = np.column_stack((b, loading))
    check(np.linalg.norm(span.T @ weight) < TOL, "scalar oracle instrument is not excluded")

    phi, psi = -0.5, 1.5
    factor = rng.standard_normal(t)
    structural = 0.2 * (0.6 * factor + 0.8 * rng.standard_normal(t))
    error = rng.standard_normal((t, m))
    aggregate = (phi * structural + (d @ loading[:, 0]) * factor + error @ d) / (1 - phi * psi)
    price = psi * aggregate + structural
    x = phi * price[:, None] * b + factor[:, None] * loading[:, 0] + error

    # Oracle: the scalar IV is the ratio of instrument cross-moments.
    est = scalar_iv(x, weight, price, aggregate)
    z = x @ weight
    check(not est.failed, "scalar oracle IV failed")
    check(abs(est.estimate - (z @ price) / (z @ aggregate)) < TOL, "scalar IV is not the ratio estimator")
    check(abs(est.se - est.naive_se) < TOL, "oracle scalar IV should not carry a correction")
    check(abs(est.estimate - psi) < 0.1, "scalar oracle misses its truth")

    # Feasible: se^2 = w' V_corrected w / (T (w'D)^2) with Hahn's xi_t.
    space = estimated_instrument_space(x, b, d, combined_rank=2)
    w_hat = projected_weight(space, d)
    feasible = scalar_iv(x, w_hat, price, aggregate, upsilon=space.upsilon)
    residual = price - feasible.estimate * aggregate
    sigma = x.T @ x / t
    oracle_v, corrected_v = reference_moment_covariance(x, residual, sigma, space.upsilon)
    derivative = np.mean((x @ w_hat) * aggregate)
    check(
        abs(feasible.se - np.sqrt(w_hat @ corrected_v @ w_hat / t) / abs(derivative)) < TOL,
        "generated-space scalar standard error is not the projected sandwich",
    )
    check(
        abs(feasible.naive_se - np.sqrt(w_hat @ oracle_v @ w_hat / t) / abs(derivative)) < TOL,
        "naive scalar standard error is not the fixed-weight sandwich",
    )
    check(abs(feasible.se - feasible.naive_se) > 1e-6, "generated-space term was silently dropped")
    check(abs(feasible.estimate - psi) < 0.1, "feasible scalar estimator misses its truth")

    # A K x 1 instrument matrix with unit weight reduces to ordinary scalar IV.
    single = scalar_iv(z[:, None], np.ones(1), price, aggregate)
    check(abs(single.estimate - est.estimate) < TOL, "single-column reduction changed the estimate")
    check(abs(single.se - est.se) < TOL, "single-column reduction changed the standard error")


def test_full_panel_spectral_equivalence() -> None:
    """Compare the fast full-panel path with the generic Hahn implementation."""
    from grouped_giv_mc import _full_spectral, _pca_loading

    rng = np.random.default_rng(606)
    t, n = 350, 18
    sizes = np.arange(1.0, n + 1.0)
    sizes /= sizes.sum()
    loading = np.r_[3.0, -2.0, np.zeros(n - 2)]
    x = rng.standard_normal((t, n)) + np.outer(rng.standard_normal(t), loading)
    x += rng.standard_normal((t, 1))
    pop = SimpleNamespace(factor_count=1, full_baseline_weight=sizes - 1.0 / n)
    pca, hahn = _full_spectral(pop, x)
    weight, rank, _, _ = hahn
    reference = estimated_instrument_space(x, np.ones(n), sizes)
    check(rank == reference.combined_rank, "fast full-panel Hahn changed the BIC rank")
    check(np.linalg.norm(weight - projected_weight(reference, sizes)) < TOL,
          "fast full-panel Hahn changed the projected instrument")
    pca /= np.linalg.norm(pca)
    pca_reference = _pca_loading(x, 1)
    pca_reference /= np.linalg.norm(pca_reference)
    check(np.linalg.norm(pca @ pca.T - pca_reference @ pca_reference.T) < TOL,
          "shared spectral decomposition changed full-panel PCA")
    _, singular = _full_spectral(pop, x[:10])
    check(singular is None, "original BIC must be undefined for singular covariance")


def test_mc_sample_cells_and_tables() -> None:
    """Guard against pooling the three T blocks with the same N."""
    import pandas as pd
    from grouped_giv_mc import (
        SAMPLES, DESIGNS, METHODS, build_population,
        table_lines, write_combined_table,
    )

    check(SAMPLES == ((1000, 100), (1000, 1000), (1000, 10000)), "MC sample design changed")
    for panel in ("full", "group"):
        pca = METHODS.index(f"{panel}_pca")
        check(METHODS[pca+1:pca+3] == (f"{panel}_bai_ng", f"{panel}_rank_known"),
              "Bai--Ng must sit between oracle PCA and known-rank Hahn")
    populations = [build_population(n, t, design) for n, t in SAMPLES for design in DESIGNS]
    rows = []
    for pop in populations:
        for method in METHODS:
            rank = 1 if pop.t == 100 else 2
            rows.append(dict(N=pop.n, T=pop.t, design=pop.name, method=method,
                             available=not (pop.t == 100 and method == "full_hahn"),
                             rank_correct=rank == 2, selected_rank=rank,
                             eigengap=rank, bic_margin=rank, pca_overlap=rank, factor_leak=rank,
                             psi_bias=pop.t / 1000, psi_sd=0.1, psi_coverage=0.95,
                             phi_bias=-pop.t / 1000, phi_sd=0.2, phi_coverage=0.9))
    frame = pd.DataFrame(rows)
    with TemporaryDirectory() as directory:
        for parameter, expected in (("psi", "0.100"), ("phi", "-0.100")):
            table = "\n".join(table_lines(frame, parameter))
            check(expected in table, "table used the wrong parameter or sample cell")
            check(table.count("---") == 3, "original singular BIC must be labelled undefined")
            check("$T=10000$" in table and "$N=" not in table,
                  "table must show the long sample and omit the common N")
        combined = Path(directory) / "combined.tex"
        write_combined_table(frame, combined)
        check(combined.read_text().count(r"\multicolumn{10}{l}{\textbf{Panel") == 2,
              "both combined-table panel titles must span all ten columns")


def test_hahn_bic_bound() -> None:
    """The directly evaluated BIC obeys its high-dimension rank-one bound."""
    rng = np.random.default_rng(707)
    for m, t in ((18, 20), (1000, 100), (1000, 1000)):
        lower_bound = np.log(t) - t / (2 * (m - 1))
        check(lower_bound > 0, "test must activate the analytic BIC bound")
        for _ in range(5):
            eigenvalues = np.exp(rng.normal(size=m - 1) * 5)
            rank, values = hahn_bic(eigenvalues, t)
            check(rank == 1, "Hahn BIC violated the analytic rank-one bound")
            check(np.all(values[1:] - values[0] >= lower_bound - TOL),
                  "analytic BIC margin bound is invalid")


def test_one_dataset_all_methods_loop() -> None:
    from grouped_giv_mc import (
        DESIGNS, MC_SEED, METHODS, REPS,
        bai_ng_count, build_population, simulate_cell,
    )
    check(bai_ng_count(np.ones(9), 9, 100, 9) == 0, "ICp2 selected a factor in a flat spectrum")
    check(bai_ng_count(np.r_[30., np.ones(8)], 38, 100, 9) == 1,
          "ICp2 missed the single dominant eigenvalue")
    check(REPS == 1000 and DESIGNS == ("core_weak",),
          "barebones run must use 1,000 draws and one design")
    check(MC_SEED == 0, "MC seed must be zero")
    pop = build_population(1000, 100, "core_weak")
    check(np.allclose(pop.loadings[:3, 0], 1 / np.sqrt(3))
          and np.count_nonzero(pop.loadings[3:]) == 0,
          "MC requires exactly three loadings of 1/sqrt(3)")
    rng = np.random.default_rng(909)
    records = simulate_cell(build_population(1000, 100, "core_weak"), 2, rng)
    check(len(records) == 2 * len(METHODS), "a dataset must receive all methods")
    for replication in range(2):
        block = records[replication*len(METHODS):(replication+1)*len(METHODS)]
        check({r["replication"] for r in block} == {replication},
              "next dataset drawn before applying all methods")
        check({r["method"] for r in block} == set(METHODS), "method set changed within a draw")
    expected = np.random.default_rng(909)
    for _ in range(2):
        expected.standard_normal(100)
        expected.standard_normal(100)
        expected.standard_normal((100, 1000))
    check(rng.bit_generator.state == expected.bit_generator.state,
          "estimators consumed random numbers or the simulation redrew a dataset")


def test_joint_algorithm_one() -> None:
    """Compare every joint estimate/SE to a literal dense A.1--A.2 calculation."""
    from dataclasses import replace
    from grouped_giv import estimate_joint_giv, joint_covariance_failure
    rng = np.random.default_rng(1010)
    t, m = 450, 7
    b = np.linspace(1.0, 2.0, m)
    d = np.arange(1, m+1, dtype=float); d /= d @ b
    loading = rng.standard_normal((m, 1))
    eta = rng.standard_normal(t)
    eps = .15 * eta + .3 * rng.standard_normal(t)
    errors = rng.standard_normal((t, m))
    y = (-.5 * eps + (d @ loading[:, 0]) * eta + errors @ d) / 1.75
    price = 1.5 * y + eps
    x = -.5 * price[:, None] * b + eta[:, None] * loading[:, 0] + errors
    space = estimated_instrument_space(x, b, d, combined_rank=2)
    result = estimate_joint_giv(x, price, y, b, d, space)
    check(not result.psi.failed, "identified joint GMM failed")
    a, k = space.basis, space.residual_dimension
    z = x @ a; ref = x @ b / (b @ b)
    derivative = np.zeros((2*k, 2))
    derivative[:k, 0] = z.T @ price / t
    derivative[k:, 1] = z.T @ y / t
    numerator = np.r_[z.T @ ref / t, z.T @ price / t]
    preliminary = np.linalg.inv(derivative.T @ derivative) @ derivative.T @ numerator
    sigma = x.T @ x / t
    xi = []
    for residual in (ref-preliminary[0]*price, price-preliminary[1]*y):
        mean = x.T @ residual / t
        correction = x * (x @ space.upsilon @ mean)[:, None] - sigma @ space.upsilon @ mean
        xi.append(x * residual[:, None] - mean + correction)
    dense = np.column_stack(xi)
    covariance = dense.T @ dense / t - np.outer(dense.mean(0), dense.mean(0))
    projection = np.zeros((2*m, 2*k)); projection[:m, :k] = a; projection[m:, k:] = a
    weight = np.linalg.inv(projection.T @ covariance @ projection)
    gram = derivative.T @ weight @ derivative
    expected = np.linalg.inv(gram) @ derivative.T @ weight @ numerator
    expected_se = np.sqrt(np.diag(np.linalg.inv(gram)) / t)
    actual = np.array([result.phi.estimate, result.psi.estimate])
    actual_se = np.array([result.phi.se, result.psi.se])
    check(np.max(np.abs(actual-expected)) < TOL, "joint point estimates differ from Algorithm 1")
    check(np.max(np.abs(actual_se-expected_se)) < TOL, "joint SE differs from equation 45")
    no_diagnostic = estimate_joint_giv(x, price, y, b, d, space, diagnostics=False)
    check(max(abs(no_diagnostic.phi.estimate-result.phi.estimate),
              abs(no_diagnostic.psi.estimate-result.psi.estimate),
              abs(no_diagnostic.phi.se-result.phi.se),
              abs(no_diagnostic.psi.se-result.psi.se),
              abs(no_diagnostic.phi.naive_se-result.phi.naive_se),
              abs(no_diagnostic.psi.naive_se-result.psi.naive_se)) < TOL,
          "equivalent covariance calculations changed joint GMM")
    rotation, _ = np.linalg.qr(rng.standard_normal((k, k)))
    rotated = estimate_joint_giv(x, price, y, b, d, replace(space, basis=a @ rotation))
    check(abs(rotated.psi.estimate-result.psi.estimate) < TOL
          and abs(rotated.phi.se-result.phi.se) < TOL, "joint estimator depends on basis rotation")
    check(joint_covariance_failure(100, 998) and joint_covariance_failure(1000, 998),
          "structurally singular full-panel joint covariance was not detected")
    small_space = known_instrument_space(np.ones(7), np.empty((7, 0)), np.ones(7)/7)
    stopped = estimate_joint_giv(x[:10], price[:10], y[:10], np.ones(7), np.ones(7)/7, small_space)
    check(stopped.psi.failed and '2K' in stopped.psi.failure_reason,
          "joint GMM silently regularized a structurally singular covariance")


def main() -> None:
    test_joint_algorithm_one()
    test_projection_geometry()
    test_grouped_whitening()
    test_bic_and_estimated_space()
    test_estimation_and_failures()
    test_scalar_instrument()
    test_full_panel_spectral_equivalence()
    test_mc_sample_cells_and_tables()
    test_hahn_bic_bound()
    test_one_dataset_all_methods_loop()
    print("all grouped-GIV invariants passed")


if __name__ == "__main__":
    main()
